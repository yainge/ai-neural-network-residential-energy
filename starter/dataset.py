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


def build_windows(
    df: pd.DataFrame, align_midnight: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sliding input windows [N, 168, 3] and targets [N, 24]."""
    values = df[["kwh", "hour", "dayofweek"]].to_numpy(dtype=np.float32)
    total = len(values) - WINDOW - HORIZON + 1
    if total <= 0:
        raise ValueError("Not enough rows to generate windows.")

    X = np.empty((total, WINDOW, 3), dtype=np.float32)
    y = np.empty((total, HORIZON), dtype=np.float32)

    for i in range(total):
        X[i] = values[i : i + WINDOW]
        y[i] = values[i + WINDOW : i + WINDOW + HORIZON, 0]

    if align_midnight:
        # Keep only windows ending at 23:00 (forecast target begins at 00:00)
        midnight_mask = X[:, -1, 1] == 23
        X, y = X[midnight_mask], y[midnight_mask]

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
def prepare_multi_datasets() -> (
    tuple[ConsumptionDataset, dict[str, ConsumptionDataset]]
):
    """Prepare training dataset and 4 static midnight-aligned test blocks at different time of the year.

    Static Test Periods (14 days each):
      1. Winter: Feb 1, 2026 -> Feb 15, 2026 (Weight: 30%)
      2. Spring: May 15, 2026 -> May 29, 2026 (Weight: 15%)
      3. Summer: Jul 15, 2026 -> Jul 29, 2026 (Weight: 15%)
      4. Autumn: Oct 15, 2026 -> Oct 29, 2026 (Weight: 40%)

    A 168h purge window before each block is removed from the training set
    to eliminate data leakage.
    """
    df = load_and_enrich()

    test_blocks = {
        "winter": ("2026-02-01 00:00:00", "2026-02-15 00:00:00"),
        "spring": ("2026-04-15 00:00:00", "2026-04-29 00:00:00"),
        "autumn": ("2026-10-15 00:00:00", "2026-10-29 00:00:00"),
    }

    train_mask = pd.Series(True, index=df.index)
    val_datasets: dict[str, ConsumptionDataset] = {}

    for name, (start_str, end_str) in test_blocks.items():
        start_ts = pd.to_datetime(start_str)
        end_ts = pd.to_datetime(end_str)
        purge_start_ts = start_ts - pd.Timedelta(hours=WINDOW)

        #  Mask out (purge_start <= timestamp < end_ts) from training set
        train_mask &= ~(
            (df["timestamp"] >= purge_start_ts) & (df["timestamp"] < end_ts)
        )

        # 2. Extract test slice including the 168h context for evaluation
        block_df = df[
            (df["timestamp"] >= purge_start_ts) & (df["timestamp"] < end_ts)
        ].reset_index(drop=True)
        X_val, y_val = build_windows(block_df, align_midnight=True)
        val_datasets[name] = ConsumptionDataset(X_val, y_val)

    # Build training set (ALIGN_TRAIN = False for maximum samples)
    train_df = df[train_mask].reset_index(drop=True)
    X_train, y_train = build_windows(train_df, align_midnight=False)
    train_ds = ConsumptionDataset(X_train, y_train)

    print(f"Train samples: {len(X_train)}")
    for name, ds in val_datasets.items():
        print(f"Val block '{name}': {len(ds)} midnight-aligned windows")

    return train_ds, val_datasets 


# Default starter function wrapper
def prepare_datasets():
    train_ds, val_dict = prepare_multi_datasets()
    # Returns autumn as default validation set 
    return train_ds, val_dict["autumn"]


if __name__ == "__main__":
    train_ds, val_dict = prepare_multi_datasets()
    X, y = train_ds[0]
    print(f"\nInput sample shape:  {X.shape}")
    print(f"Target sample shape: {y.shape}")
