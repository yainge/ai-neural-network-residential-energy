"""Dataset utilities for the Residential Energy Challenge.

NON-NEGOTIABLE MODEL CONTRACT
-----------------------------
Input:  float32 [batch, 168, 3]
    channel 0 -> hourly net energy kWh
    channel 1 -> hour of day (0..23)
    channel 2 -> day of week (0=Monday .. 6=Sunday)
Output: float32 [batch, 24] in real kWh

Students may change the train/validation strategy, feature treatment inside the
model, filtering, augmentation, etc. The ONNX input/output contract must remain
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

WINDOW = 168
HORIZON = 24
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw.csv"


def load_and_enrich(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Load and validate the platform-downloaded raw.csv directly."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Historical data not found at {csv_path}.\n"
            "Download Historical data from the platform and place the downloaded "
            "raw.csv file at data/raw.csv."
        )

    df = pd.read_csv(csv_path)
    required = {"timestamp", "kwh"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df = df[["timestamp", "kwh"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["kwh"] = pd.to_numeric(df["kwh"], errors="raise").astype(np.float32)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("The historical dataset is empty.")
    if df["timestamp"].duplicated().any():
        duplicated = df.loc[df["timestamp"].duplicated(), "timestamp"].iloc[0]
        raise ValueError(f"Duplicated timestamp found: {duplicated}")
    if df["kwh"].isna().any():
        raise ValueError("Column 'kwh' contains missing values.")

    differences = df["timestamp"].diff().dropna()
    expected = pd.Timedelta(hours=1)
    bad = differences[differences != expected]
    if not bad.empty:
        idx = int(bad.index[0])
        previous = df.loc[idx - 1, "timestamp"]
        current = df.loc[idx, "timestamp"]
        raise ValueError(
            "Historical data must be continuous and hourly. "
            f"Gap found between {previous} and {current}."
        )

    minimum = WINDOW + HORIZON
    if len(df) < minimum:
        raise ValueError(
            f"At least {minimum} hourly samples are required; received {len(df)}."
        )

    df["hour"] = df["timestamp"].dt.hour.astype(np.float32)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(np.float32)
    return df


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = df[["kwh", "hour", "dayofweek"]].to_numpy(dtype=np.float32)
    total = len(values) - WINDOW - HORIZON + 1
    if total <= 0:
        raise ValueError(
            f"Not enough data to create one sample: need at least {WINDOW + HORIZON} rows."
        )

    X = np.empty((total, WINDOW, 3), dtype=np.float32)
    y = np.empty((total, HORIZON), dtype=np.float32)

    for i in range(total):
        X[i] = values[i : i + WINDOW]
        y[i] = values[i + WINDOW : i + WINDOW + HORIZON, 0]
    return X, y


class ConsumptionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


# ---------------------------------------------------------------------------
# STUDENT-EDITABLE SECTION
# ---------------------------------------------------------------------------
def prepare_datasets() -> tuple[ConsumptionDataset, ConsumptionDataset]:
    """Create train and validation datasets.

    Starter strategy: reserve the last 28 complete days for validation.
    Students are encouraged to justify and test alternative time-aware splits.
    """
    df = load_and_enrich()

    validation_hours = 24 * 28
    minimum_train_hours = WINDOW + HORIZON
    if len(df) <= validation_hours + minimum_train_hours:
        raise ValueError(
            "Not enough history for the starter 28-day validation split. "
            "Choose a shorter validation period in dataset.py."
        )

    train_df = df.iloc[:-validation_hours].reset_index(drop=True)
    # Include the preceding input window so the first validation target begins
    # exactly at the validation boundary.
    val_df = df.iloc[-validation_hours - WINDOW :].reset_index(drop=True)

    X_train, y_train = build_windows(train_df)
    X_val, y_val = build_windows(val_df)

    print(f"Train windows: {len(X_train)}")
    print(f"Val windows:   {len(X_val)}")
    return ConsumptionDataset(X_train, y_train), ConsumptionDataset(X_val, y_val)


if __name__ == "__main__":
    train_ds, val_ds = prepare_datasets()
    X, y = train_ds[0]
    print(f"Input sample shape:  {X.shape}")
    print(f"Target sample shape: {y.shape}")
    print("First three input rows:")
    print(X[:3])
