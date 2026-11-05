"""
Synthetic event augmentation for the regime gate.

The regime gate can only learn "after unusual days, follow yesterday" if it sees
enough unusual days. The real data has about 3 closure days. This module learns
the *shape* of a closure day from the ones you label, then pastes that shape onto
random stretches of normal data to make extra training series.

It works with differences (closure minus a normal reference), not ratios, so it
is safe with negative kWh (net export).

Used by dataset.prepare_datasets(augment_copies=N). Casa Mila's known closure
("dark") days in raw.csv: 2026-07-06 and 2026-09-07 (flat ~21 kWh every hour).
2026-05-18 is a partial dip and is left out of the profile.

Only the training range is augmented. The validation range is never touched.

Caution: January 2026 in raw.csv shows NO dark week, so we do not know that the
Jan 2027 maintenance closure will look like these single dark days. Multi-day
blocks (n_blocks > 0) encode that assumption, so treat them as an experiment.
"""

import numpy as np
import pandas as pd


def _daily_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = calendar dates, columns = hours 0..23, values = kWh. Needs full days."""
    d = df.copy()
    d["date"] = d["timestamp"].dt.normalize()
    d["hour"] = d["timestamp"].dt.hour
    return d.pivot_table(index="date", columns="hour", values="kwh", aggfunc="mean")


def closure_delta(df: pd.DataFrame, event_dates, n_ref_weeks: int = 2) -> np.ndarray:
    """
    Average 24-hour difference between event days and normal days of the same weekday.
    Reference = the same weekday up to n_ref_weeks before and after, excluding other events.
    Returns an array of 24 values in kWh.
    """
    mat = _daily_matrix(df)
    events = pd.to_datetime(pd.Series(event_dates)).dt.normalize()
    deltas = []
    for day in events:
        if day not in mat.index:
            continue
        refs = [day + pd.Timedelta(weeks=k) for k in range(-n_ref_weeks, n_ref_weeks + 1) if k != 0]
        refs = [r for r in refs if r in mat.index and r not in set(events)]
        if not refs:
            continue
        deltas.append(mat.loc[day].values - mat.loc[refs].mean(axis=0).values)
    if not deltas:
        raise ValueError("No usable event days found. Check the dates are inside the data.")
    return np.nanmean(np.vstack(deltas), axis=0)


def inject_events(
    df: pd.DataFrame,
    delta: np.ndarray,
    n_single: int = 8,
    n_blocks: int = 3,
    block_len=(4, 7),
    scale=(0.7, 1.3),
    noise: float = 0.15,
    seed: int = 0,
    end_date=None,
):
    """
    Return (copy of df with synthetic events, list of event dates).

    n_single : one-off low days (holiday-like)
    n_blocks : multi-day blocks (closure-week-like), each block_len days long
    scale    : random strength of each event, as a multiple of delta
    noise    : extra hour-to-hour noise, as a fraction of |delta|
    end_date : only place events before this date (set it to your validation start)
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    dates = out["timestamp"].dt.normalize()
    all_days = pd.Series(dates.unique())
    # skip the first week so every event has a full 168 h history in front of it
    candidates = all_days.iloc[7:]
    if end_date is not None:
        candidates = candidates[candidates < pd.Timestamp(end_date)]
    candidates = list(candidates)

    used, events = set(), []

    def place(start_idx, length):
        span = candidates[start_idx : start_idx + length]
        if len(span) < length or any(d in used for d in span):
            return
        s = rng.uniform(*scale)
        for d in span:
            mask = dates == d
            hours = out.loc[mask, "timestamp"].dt.hour.values
            jitter = rng.normal(0, noise * np.abs(delta[hours]) + 1e-9)
            out.loc[mask, "kwh"] = out.loc[mask, "kwh"].values + s * delta[hours] + jitter
            used.add(d)
            events.append(d)
        # leave a gap after each event so events do not merge
        for g in range(1, 3):
            if start_idx + length - 1 + g < len(candidates):
                used.add(candidates[start_idx + length - 1 + g])

    for _ in range(n_blocks):
        length = int(rng.integers(block_len[0], block_len[1] + 1))
        place(int(rng.integers(0, max(1, len(candidates) - length))), length)
    for _ in range(n_single):
        place(int(rng.integers(0, len(candidates))), 1)

    return out, sorted(events)
