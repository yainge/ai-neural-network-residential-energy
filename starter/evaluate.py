"""Evaluate model.pt on the 4 seasonal Multi-OOT validation sets and compare with persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import prepare_multi_oot_datasets
from model import ConsumptionPredictor

CHECKPOINT = Path(__file__).resolve().parent / "model.pt"
BATCH_SIZE = 128

SEASON_WEIGHTS = {
    "winter": 0.40,
    "spring": 0.20,
    "autumn": 0.40,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    error = y_pred - y_true
    return float(np.mean(np.abs(error))), float(np.mean(error**2))


def evaluate_season(
    model: ConsumptionPredictor, ds, device: str
) -> tuple[float, float, float, float]:
    """Evaluate model and weekly persistence (t-168) on a single dataset block."""
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    predictions = []
    targets = []
    persistence = []

    with torch.no_grad():
        for X, y in loader:
            X_dev = X.to(device)
            pred = model(X_dev).cpu().numpy()
            x_np = X.numpy()

            predictions.append(pred)
            targets.append(y.numpy())
            persistence.append(x_np[:, -24:, 0])

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    y_persistence = np.concatenate(persistence)

    mae, mse = metrics(y_true, y_pred)
    base_mae, base_mse = metrics(y_true, y_persistence)

    return mae, mse, base_mae, base_mse


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError("model.pt not found. Run: python starter/train.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_dict = prepare_multi_oot_datasets()

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model = ConsumptionPredictor(
        mu=checkpoint["mu"], sigma=checkpoint["sigma"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Validation Results ")
    print("---------------------------------------")

    weighted_model_mae = 0.0
    weighted_base_mae = 0.0

    for season, ds in val_dict.items():
        mae, mse, base_mae, base_mse = evaluate_season(model, ds, device)
        weight = SEASON_WEIGHTS[season]

        weighted_model_mae += mae * weight
        weighted_base_mae += base_mae * weight

        skill = (1.0 - mae / base_mae) * 100.0

        print(
            f"Season {season.capitalize():<6} ({weight*100:.0f}%) | "
            f"Model MAE: {mae:.4f} kWh | "
            f"Base MAE: {base_mae:.4f} kWh | "
            f"Skill: {skill:+.2f}%"
        )

    print("---------------------------------------")
    overall_skill = (1.0 - weighted_model_mae / weighted_base_mae) * 100.0
    print(f"Weighted Model MAE      : {weighted_model_mae:.4f} kWh")
    print(f"Weighted Persistence MAE : {weighted_base_mae:.4f} kWh")
    print(f"OVERALL MAE Skill Score  : {overall_skill:+.2f}%")


if __name__ == "__main__":
    main()