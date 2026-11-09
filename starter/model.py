"""Starter forecasting model.

You may redesign this architecture. The only mandatory requirement is:
    input  -> float32 [batch, 168, 3]
    output -> float32 [batch, 24] in real kWh

All preprocessing needed at inference time must live INSIDE the model so it is
exported into the final ONNX file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

class ResBlock(nn.Module):
    """Residual MLP block with LayerNorm, SiLU activation, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ConsumptionPredictor(nn.Module):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, hidden_dim: int = 256):
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float32))

        # Inputs count:
        # 168 (raw kWh)
        # + 24 (lag t-24)
        # + 24 (lag t-48)
        # + 24 (lag t-168)
        # + 6 (stats: mean/std for 6h, 24h, 168h)
        # + 7 (target day one-hot)
        # + 48 (sin/cos for 24 target hours)
        in_dim = 168 + 24 + 24 + 24 + 6 + 7 + 48

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.res_block1 = ResBlock(hidden_dim, dropout=0.1)
        self.res_block2 = ResBlock(hidden_dim, dropout=0.1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 24),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 168, 3] 
        kwh_norm = (x[..., 0] - self.mu) / self.sigma  # [batch, 168]

        lag_24 = kwh_norm[:, -24:]       # Yesterday same 24h
        lag_48 = kwh_norm[:, -48:-24]    # Day before yesterday same 24h
        lag_168 = kwh_norm[:, :24]       # Last week same 24h

        m_6 = kwh_norm[:, -6:].mean(dim=1, keepdim=True)
        s_6 = kwh_norm[:, -6:].std(dim=1, keepdim=True)
        m_24 = kwh_norm[:, -24:].mean(dim=1, keepdim=True)
        s_24 = kwh_norm[:, -24:].std(dim=1, keepdim=True)
        m_168 = kwh_norm.mean(dim=1, keepdim=True)
        s_168 = kwh_norm.std(dim=1, keepdim=True)

        stats = torch.cat([m_6, s_6, m_24, s_24, m_168, s_168], dim=-1)

        last_hour = x[:, -1, 1]
        target_day_idx = torch.remainder((x[:, -1, 2] + (last_hour == 23).float()).long(), 7)
        day_onehot = nn.functional.one_hot(target_day_idx, num_classes=7).float()

        target_hours = torch.arange(24, device=x.device).float()
        sin_target = torch.sin(2 * np.pi * target_hours / 24.0).repeat(x.size(0), 1)
        cos_target = torch.cos(2 * np.pi * target_hours / 24.0).repeat(x.size(0), 1)

        
        features = torch.cat(
            [kwh_norm, lag_24, lag_48, lag_168, stats, day_onehot, sin_target, cos_target],
            dim=-1,
        )

        h = self.input_proj(features)
        h = self.res_block1(h)
        h = self.res_block2(h)
        out_norm = self.head(h)

        return out_norm * self.sigma + self.mu


if __name__ == "__main__":
    model = ConsumptionPredictor()
    x = torch.randn(2, 168, 3, dtype=torch.float32)
    x[:, :, 1] = torch.randint(0, 24, (2, 168)).float()
    x[:, :, 2] = torch.randint(0, 7, (2, 168)).float()
    y = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")