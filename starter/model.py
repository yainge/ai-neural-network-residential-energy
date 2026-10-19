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


class ConsumptionPredictor(nn.Module):
    """Intentionally simple MLP baseline for the assignment."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float32))

        self.network = nn.Sequential(
            nn.Linear(168 * 3 + 1, 256),  # flattened window + weekend flag
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 24),           # [B, 24], normalized kWh
        )
        # one more layer test
        # self.network = nn.Sequential(
        #     nn.Linear(168 * 3 + 1, 256),  # flattened window + weekend flag
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 24),           # [B, 24], normalized kWh
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel 0: standardize kWh using statistics learned from training data.
        kwh = (x[..., 0:1] - self.mu) / self.sigma

        # Calendar channels are also transformed INSIDE the model so the
        # platform can always send the raw contract values 0..23 and 0..6.
        hour = x[..., 1:2] / 23.0
        dayofweek = x[..., 2:3] / 6.0
        x_model = torch.cat([kwh, hour, dayofweek], dim=-1).flatten(1)  # [B, 504]

        # Weekday of the first forecast hour. The platform always forecasts from
        # midnight, so this is the weekday of the output day.
        last_hour = x[:, -1, 1]
        target_dow = torch.remainder(x[:, -1, 2] + (last_hour == 23.0).float(), 7.0)

        # Weekend flag for the output day: 1 for Saturday/Sunday, 0 otherwise.
        target_weekend = (target_dow >= 5.0).float().unsqueeze(1)        # [B, 1]

        y_normalized = self.network(torch.cat([x_model, target_weekend], dim=1))
        return y_normalized * self.sigma + self.mu


if __name__ == "__main__":
    model = ConsumptionPredictor()
    x = torch.randn(2, 168, 3, dtype=torch.float32)
    y = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")
