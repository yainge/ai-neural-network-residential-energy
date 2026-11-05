"""Train the starter model.

Students may edit the architecture, hyperparameters, optimizer, loss, scheduler,
early stopping, and data split. Preserve only the platform ONNX contract.

Best config so far (see experiments.py, 2 folds x 3 seeds):
    python starter/train.py --cell mlp --all-hours --augment 3
Recurrent version for comparison:
    python starter/train.py --cell gru --all-hours --augment 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import prepare_datasets
from model import ConsumptionPredictor, count_parameters

EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 20
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
CHECKPOINT = Path(__file__).resolve().parent / "model.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the forecasting model.")
    p.add_argument("--cell", default="gru", choices=["gru", "lstm", "mlp"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-regime", action="store_true", help="turn off regime features and gate input")
    p.add_argument("--all-hours", action="store_true", help="train on all hourly windows, not only midnight")
    p.add_argument("--augment", type=int, default=0, help="copies of the training range with synthetic dark days")
    p.add_argument("--augment-blocks", type=int, default=0, help="multi-day dark blocks per copy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds, val_ds = prepare_datasets(
        align_train=not args.all_hours,
        augment_copies=args.augment,
        augment_blocks=args.augment_blocks,
    )
    mu, sigma = training_statistics(train_ds)
    print(f"Training kWh statistics: mu={mu:+.4f}, sigma={sigma:.4f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Saved in the checkpoint so evaluate.py and export.py rebuild the same model
    config = {
        "cell": args.cell,
        "hidden": args.hidden,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "use_regime": not args.no_regime,
    }
    model = ConsumptionPredictor(mu=mu, sigma=sigma, **config).to(device)
    print(f"Model: {config} | {count_parameters(model):,} trainable parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss()  # the platform scores MAE

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
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
                    "config": config,
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
