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

        self.network =

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel 0: standardize kWh using statistics learned from training data.
        kwh = (x[..., 0:1] - self.mu) / self.sigma

        # Calendar channels are also transformed INSIDE the model so the
        # platform can always send the raw contract values 0..23 and 0..6.
        hour = x[..., 1:2] / 23.0
        dayofweek = x[..., 2:3] / 6.0
        x_model = torch.cat([kwh, hour, dayofweek], dim=-1)

        y_normalized = self.network(x_model)
        return y_normalized * self.sigma + self.mu


if __name__ == "__main__":
    model = ConsumptionPredictor()
    x = torch.randn(2, 168, 3, dtype=torch.float32)
    y = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")
