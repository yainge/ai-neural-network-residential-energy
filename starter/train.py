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

from dataset import prepare_datasets
from model import ConsumptionPredictor

EPOCHS =
BATCH_SIZE =
LEARNING_RATE =
PATIENCE =
CHECKPOINT = Path(__file__).resolve().parent / "model.pt"


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

    train_ds, val_ds = prepare_datasets()
    mu, sigma = training_statistics(train_ds)
    print(f"Training kWh statistics: mu={mu:+.4f}, sigma={sigma:.4f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ConsumptionPredictor(mu=mu, sigma=sigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mae = evaluate_epoch(model, val_loader, loss_fn, device)
        marker = ""

        if val_mae < best_val:
            best_val = val_mae
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
            f"| val MAE={val_mae:.4f} kWh{marker}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without improvement.")
            break

    print(f"\n✓ Best validation MAE: {best_val:.4f} kWh")
    print(f"✓ Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
