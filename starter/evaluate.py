"""Evaluate model.pt on the validation set and compare it with persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import prepare_datasets
from model import ConsumptionPredictor

CHECKPOINT = Path(__file__).resolve().parent / "model.pt"
BATCH_SIZE = 128


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    error = y_pred - y_true
    return float(np.mean(np.abs(error))), float(np.mean(error**2))


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError("model.pt not found. Run: python starter/train.py")

    _, val_ds = prepare_datasets()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = ConsumptionPredictor(mu=checkpoint["mu"], sigma=checkpoint["sigma"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    targets = []
    persistence = []

    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        offset = 0
        for X, y in loader:
            pred = model(X).cpu().numpy()
            x_np = X.cpu().numpy()
            predictions.append(pred)
            targets.append(y.cpu().numpy())
            # Same 24 hours on the previous day: positions -24..-1 of input.
            persistence.append(x_np[:, -24:, 0])
            offset += len(X)

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    y_persistence = np.concatenate(persistence)

    mae, mse = metrics(y_true, y_pred)
    base_mae, base_mse = metrics(y_true, y_persistence)

    print("Validation results")
    print("------------------")
    print(f"Model       MAE: {mae:.4f} kWh | MSE: {mse:.4f} kWh²")
    print(f"Persistence MAE: {base_mae:.4f} kWh | MSE: {base_mse:.4f} kWh²")
    print(f"MAE skill vs persistence: {(1.0 - mae / base_mae) * 100:+.2f}%")


if __name__ == "__main__":
    main()
