"""Forecasting model: daily-chunk GRU / LSTM / MLP with a regime gate.

Mandatory contract (unchanged):
    input  -> float32 [batch, 168, 3]   raw kWh, hour 0..23, day of week 0..6 (Mon = 0)
    output -> float32 [batch, 24] in real kWh

All preprocessing lives INSIDE forward(), so it is exported into the ONNX file.

How it works
------------
1. Standardise kWh with mu / sigma (buffers, as in the starter).
2. Cut the 168 hours into 7 chunks of 24 hours. Chunk 0 is the oldest day,
   chunk 6 is "yesterday". At a midnight call, chunk 0 is the same weekday as
   the day being forecast.
3. The encoder reads the 7 days:
     cell="gru" / "lstm" : 7 recurrent steps (one per day), not 168
     cell="mlp"          : same inputs flattened, no recurrence (the control)
4. Regime features, computed in forward():
     r1, r2, r3 : mean of yesterday / 2 days ago / 3 days ago minus the mean of
                  the 4 older days, in sigma units
     r_flat     : yesterday's spread minus the older days' spread. A Casa Mila
                  closure day is flat (about 21 kWh every hour), so this drops.
5. Output = gate * yesterday + (1 - gate) * same day last week + correction.
   The gate (0..1) comes from the regime features and the target weekday. The
   correction comes from the encoder. Its last layer starts at zero, so an
   untrained model is already a 50/50 persistence blend.

The constructor keeps the starter signature ConsumptionPredictor(mu, sigma), so
train.py, evaluate.py and export.py only need the extra config passed through.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

HOURS = 24
DAYS = 7
N_REGIME = 4


class ConsumptionPredictor(nn.Module):
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for cell in ("gru", "lstm", "mlp"):
        model = ConsumptionPredictor(mu=60.0, sigma=25.0, cell=cell)
        x = torch.zeros(2, 168, 3, dtype=torch.float32)
        x[:, :, 0] = 60.0
        x[:, :, 1] = torch.arange(168) % 24
        x[:, :, 2] = (torch.arange(168) // 24) % 7
        y = model(x)
        print(f"{cell}: Input {tuple(x.shape)} -> Output {tuple(y.shape)}, {count_parameters(model):,} params")
