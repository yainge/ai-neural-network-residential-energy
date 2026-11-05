"""Train the starter model.

Students may edit the architecture, hyperparameters, optimizer, loss, scheduler,
early stopping, and data split. Preserve only the platform ONNX contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import prepare_multi_datasets
from model import ConsumptionPredictor

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 50
CHECKPOINT = Path(__file__).resolve().parent / "model.pt"

SEASON_WEIGHTS = {
    "winter": 0.40,
    "spring": 0.20,
    "autumn": 0.40,
}


def training_statistics(train_ds) -> tuple[float, float]:
    """Compute kWh normalization statistics from the training dataset only."""
    values = np.concatenate(
        [train_ds.X[:, :, 0].reshape(-1), train_ds.y.reshape(-1)]
    )
    return float(values.mean()), float(values.std() + 1e-6)


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(X)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device) -> float:
    model.eval()
    total = 0.0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        prediction = model(X)
        total += loss_fn(prediction, y).item() * X.size(0)
    return total / len(loader.dataset)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds, val_dict = prepare_multi_datasets()
    mu, sigma = training_statistics(train_ds)
    print(f"Training kWh statistics: mu={mu:+.4f}, sigma={sigma:.4f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loaders = {name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)for name, ds in val_dict.items()}

    model = ConsumptionPredictor(mu=mu, sigma=sigma).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        # Evaluate across all 4 seasonal blocks
        seasonal_maes = {
            name: evaluate_epoch(model, loader, loss_fn, device)
            for name, loader in val_loaders.items()}
        
        weighted_val_mae = sum(
            seasonal_maes[s] * SEASON_WEIGHTS[s] for s in SEASON_WEIGHTS
        )

        scheduler.step()
        marker = ""

        if weighted_val_mae < best_val:
            best_val = weighted_val_mae
            epochs_without_improvement = 0
            torch.save(
                {
                    "contract_version": "1.0",
                    "model_state_dict": model.state_dict(),
                    "mu": mu,
                    "sigma": sigma,
                    "best_val_mae": best_val,
                },
                CHECKPOINT,
            )
            marker = " <- saved"
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | train MAE={train_mae:.4f} kWh "
            f"| weighted val MAE={weighted_val_mae:.4f} kWh "
            f"[W:{seasonal_maes['winter']:.2f}, Sp:{seasonal_maes['spring']:.2f}, "
            f" Au:{seasonal_maes['autumn']:.2f}]{marker}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping after {PATIENCE} epochs without improvement."
            )
            break

    print(f"\n✓ Best weighted validation MAE: {best_val:.4f} kWh")
    print(f"✓ Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()