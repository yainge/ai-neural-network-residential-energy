"""Named models for the LSTM / RNN experiments.

Every model keeps the platform contract:
    input  -> float32 [batch, 168, 3]   raw kWh, hour 0..23, day of week 0..6 (Mon = 0)
    output -> float32 [batch, 24] in real kWh
All scaling and preprocessing happens inside forward(). mu and sigma are buffers.

Models
------
mlp_max      the tuned MLP in model.py (the team reference), imported unchanged
lstm_hourly  plain LSTM over the 168 hourly steps, linear head to 24
lstm_daily   daily-chunk model with regime gate, LSTM encoder (7 steps of 24 h)
gru_daily    same, GRU encoder
mlp_daily    same, no recurrence (the control)

Use build_model(name, mu, sigma, config) to create one. The config saved in a
checkpoint is enough to rebuild the same model in evaluate.py and export.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model import ConsumptionPredictor as MaxMLP

HOURS = 24
DAYS = 7
N_REGIME = 4


class DailyChunkPredictor(nn.Module):
    """Daily-chunk GRU / LSTM / MLP with a regime gate"""
    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        cell: str = "gru",
        hidden: int = 64,
        num_layers: int = 1,
        head_hidden: int = 64,
        dropout: float = 0.1,
        use_regime: bool = True,
    ):
        super().__init__()
        cell = cell.lower()
        if cell not in ("gru", "lstm", "mlp"):
            raise ValueError("cell must be 'gru', 'lstm' or 'mlp'")

        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float32))
        self.register_buffer("dow_ids", torch.arange(7, dtype=torch.float32))

        self.cell = cell
        self.use_regime = use_regime

        # Each step is one day: 24 standardised values + weekday one-hot
        step_in = HOURS + 7
        if cell == "mlp":
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(DAYS * step_in, hidden), nn.ReLU())
        else:
            rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
            self.encoder = rnn_cls(
                input_size=step_in,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        n_regime = N_REGIME if use_regime else 0
        n_calendar = 7 + 2  # target weekday one-hot + start hour sin/cos

        # Gate: how much to trust yesterday vs the same day last week
        self.gate = nn.Sequential(nn.Linear(n_regime + 7, 16), nn.Tanh(), nn.Linear(16, 1))

        # Correction on top of the persistence blend
        self.head = nn.Sequential(
            nn.Linear(hidden + n_regime + n_calendar, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, HOURS),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # ------------------------------------------------------------------ helpers
    def _one_hot_dow(self, d: torch.Tensor) -> torch.Tensor:
        """d: [...] weekday 0..6 as float -> [..., 7]. Comparison-based, exports cleanly."""
        return (torch.abs(d.unsqueeze(-1) - self.dow_ids) < 0.5).float()

    @staticmethod
    def regime_features(days: torch.Tensor) -> torch.Tensor:
        """days: [B, 7, 24] standardised kWh, oldest first -> [B, 4] (r1, r2, r3, r_flat)."""
        day_mean = days.mean(dim=2)
        ref_mean = day_mean[:, 0:4].mean(dim=1, keepdim=True)
        r = torch.flip(day_mean[:, 4:7] - ref_mean, dims=[1])     # r1 = yesterday first

        centred = days - days.mean(dim=2, keepdim=True)
        day_std = torch.sqrt((centred * centred).mean(dim=2) + 1e-6)
        r_flat = day_std[:, 6:7] - day_std[:, 0:4].mean(dim=1, keepdim=True)
        return torch.cat([r, r_flat], dim=1)

    def _target_dow(self, x: torch.Tensor) -> torch.Tensor:
        """Weekday of the first forecast hour (the window may end at any hour)."""
        t = x[:, -1, 2] + (x[:, -1, 1] > 22.5).float()
        return t - 7.0 * (t > 6.5).float()

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x[:, :, 0] - self.mu) / self.sigma                    # [B, 168]
        days = z.reshape(-1, DAYS, HOURS)                          # [B, 7, 24]

        chunk_dow = x[:, HOURS - 1 :: HOURS, 2]                    # weekday of each chunk
        steps = torch.cat([days, self._one_hot_dow(chunk_dow)], dim=2)

        if self.cell == "mlp":
            h_last = self.encoder(steps)
        else:
            out, _ = self.encoder(steps)
            h_last = out[:, -1, :]

        target_oh = self._one_hot_dow(self._target_dow(x))
        angle = 2.0 * math.pi * (x[:, -1, 1] + 1.0) / 24.0
        hour_feat = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)

        yesterday = days[:, 6, :]
        last_week = days[:, 0, :]

        if self.use_regime:
            regime = self.regime_features(days)
            gate_in = torch.cat([regime, target_oh], dim=1)
            head_in = torch.cat([h_last, regime, target_oh, hour_feat], dim=1)
        else:
            gate_in = target_oh
            head_in = torch.cat([h_last, target_oh, hour_feat], dim=1)

        g = torch.sigmoid(self.gate(gate_in))
        z_out = g * yesterday + (1.0 - g) * last_week + self.head(head_in)
        return z_out * self.sigma + self.mu

    # ------------------------------------------------------------------ report helper
    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> dict:
        """Gate weight on yesterday and regime features per window, for report plots."""
        days = ((x[:, :, 0] - self.mu) / self.sigma).reshape(-1, DAYS, HOURS)
        target_oh = self._one_hot_dow(self._target_dow(x))
        if self.use_regime:
            regime = self.regime_features(days)
            g = torch.sigmoid(self.gate(torch.cat([regime, target_oh], 1))).squeeze(1)
            return {"gate_yesterday": g, "r1": regime[:, 0], "r2": regime[:, 1],
                    "r3": regime[:, 2], "r_flat": regime[:, 3]}
        return {"gate_yesterday": torch.sigmoid(self.gate(target_oh)).squeeze(1)}


class HourlyLSTM(nn.Module):
    """Plain LSTM that reads the 168 hours one step at a time.

    Step inputs: standardised kWh, hour as sin / cos, weekday as sin / cos.
    The last hidden state goes through one linear layer to the 24 outputs.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float32))
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, HOURS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x[..., 0:1] - self.mu) / self.sigma                   # [B, 168, 1]
        hour_angle = x[..., 1:2] * (2.0 * math.pi / 24.0)
        dow_angle = x[..., 2:3] * (2.0 * math.pi / 7.0)
        steps = torch.cat(
            [z, torch.sin(hour_angle), torch.cos(hour_angle),
             torch.sin(dow_angle), torch.cos(dow_angle)],
            dim=-1,
        )                                                          # [B, 168, 5]
        out, _ = self.lstm(steps)
        return self.head(out[:, -1, :]) * self.sigma + self.mu


# ---------------------------------------------------------------------- registry
_DAILY = {"hidden": 64, "num_layers": 1, "head_hidden": 64, "dropout": 0.1, "use_regime": True}

MODEL_CONFIGS: dict[str, dict] = {
    "mlp_max": {},
    "lstm_hourly": {"hidden": 64, "num_layers": 1, "dropout": 0.0},
    "lstm_daily": {"cell": "lstm", **_DAILY},
    "gru_daily": {"cell": "gru", **_DAILY},
    "mlp_daily": {"cell": "mlp", **_DAILY},
}
MODEL_NAMES = list(MODEL_CONFIGS)
DEFAULT_MODEL = "mlp_max"


def default_config(name: str) -> dict:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")
    return dict(MODEL_CONFIGS[name])


def build_model(name: str = DEFAULT_MODEL, mu: float = 0.0, sigma: float = 1.0,
                config: dict | None = None) -> nn.Module:
    """Create a model by name. config overrides the defaults for that name."""
    cfg = default_config(name)
    if config:
        cfg.update(config)
    if name == "mlp_max":
        return MaxMLP(mu=mu, sigma=sigma)
    if name == "lstm_hourly":
        return HourlyLSTM(mu=mu, sigma=sigma, **cfg)
    return DailyChunkPredictor(mu=mu, sigma=sigma, **cfg)


def load_checkpoint_model(checkpoint_path, device: str = "cpu", model_name: str | None = None):
    """Rebuild the model stored in a checkpoint. Returns (model in eval mode, checkpoint, name).

    New checkpoints store the model name and config. Old ones (no name) are
    treated as model_name if given, otherwise as the default mlp_max.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_name = checkpoint.get("model_name")
    if saved_name and model_name and saved_name != model_name:
        raise ValueError(
            f"Checkpoint holds '{saved_name}' but --model {model_name} was given."
        )
    name = saved_name or model_name or DEFAULT_MODEL
    model = build_model(
        name, mu=checkpoint["mu"], sigma=checkpoint["sigma"],
        config=checkpoint.get("model_config"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, name


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.zeros(2, 168, 3, dtype=torch.float32)
    x[:, :, 0] = 60.0
    x[:, :, 1] = torch.arange(168) % 24
    x[:, :, 2] = (torch.arange(168) // 24) % 7
    for name in MODEL_NAMES:
        model = build_model(name, mu=60.0, sigma=25.0).eval()
        y = model(x)
        print(f"{name:<12} {tuple(x.shape)} -> {tuple(y.shape)}  params={count_parameters(model):,}")
