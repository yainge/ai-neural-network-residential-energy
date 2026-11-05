"""Single-factor experiments with a fair split.

For each fold:
    train      : start of data -> early-stop start
    early stop : the 14 days before the fold (picks the best epoch)
    test       : the fold itself, midnight-aligned windows only (as on the platform)
Nothing in the test fold is used to pick epochs, so the scores are honest.

Run:  python starter/experiments.py            (about 10 to 15 min on a laptop CPU)
      python starter/experiments.py --quick    (1 seed, midnight-only variants)
Writes results to starter/experiments_results.csv
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from augment import closure_delta, inject_events
from dataset import DARK_DAYS, WINDOW, build_windows, load_and_enrich
from model import ConsumptionPredictor, count_parameters

HERE = Path(__file__).resolve().parent
FOLDS = {
    # name: (test start, test end exclusive)
    "Aug24-Sep20 (has 7 Sep dark day)": ("2026-08-24", "2026-09-21"),
    "Sep21-Oct18 (starter split)": ("2026-09-21", "2026-10-19"),
}
EARLY_STOP_DAYS = 14


def windows_between(df, start, end, midnight_only):
    """Windows whose first forecast hour is in [start, end)."""
    s = df.index[df["timestamp"] >= pd.Timestamp(start)][0]
    e = df.index[df["timestamp"] < pd.Timestamp(end)][-1]
    part = df.iloc[max(0, s - WINDOW) : e + 24].reset_index(drop=True)
    X, y = build_windows(part)
    first = part["timestamp"].iloc[WINDOW : WINDOW + len(X)].to_numpy()
    keep = (first >= np.datetime64(pd.Timestamp(start))) & (first < np.datetime64(pd.Timestamp(end)))
    if midnight_only:
        keep &= X[:, -1, 1] == 23
    return X[keep], y[keep], first[keep]


def fit(cfg, Xtr, ytr, Xes, yes, seed, epochs=150, patience=20, lr=1e-3, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    vals = np.concatenate([Xtr[:, :, 0].ravel(), ytr.ravel()])
    model = ConsumptionPredictor(mu=float(vals.mean()), sigma=float(vals.std() + 1e-6), **cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.L1Loss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xe, ye = torch.tensor(Xes), torch.tensor(yes)
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for k in range(0, len(Xt), bs):
            idx = perm[k : k + bs]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xe), ye).item()
        if v < best:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)

    seeds = [0] if args.quick else [0, 1, 2]
    variants = [
        # name, model config, train on all hours, augment copies
        ("GRU", dict(cell="gru"), False, 0),
        ("LSTM", dict(cell="lstm"), False, 0),
        ("MLP (no recurrence)", dict(cell="mlp"), False, 0),
        ("GRU, no regime", dict(cell="gru", use_regime=False), False, 0),
        ("GRU + 3 aug copies (single dark days)", dict(cell="gru"), False, 3),
    ]
    if not args.quick:
        variants += [
            ("GRU, all-hour windows", dict(cell="gru"), True, 0),
            ("MLP, all-hour windows", dict(cell="mlp"), True, 0),
            ("GRU, all-hour + 3 aug copies", dict(cell="gru"), True, 3),
            ("MLP, all-hour + 3 aug copies", dict(cell="mlp"), True, 3),
        ]

    df = load_and_enrich()
    dark = set(pd.to_datetime(DARK_DAYS))
    rows = []

    for fold, (t0, t1) in FOLDS.items():
        es0 = (pd.Timestamp(t0) - pd.Timedelta(days=EARLY_STOP_DAYS)).strftime("%Y-%m-%d")
        Xte, yte, tte = windows_between(df, t0, t1, midnight_only=True)
        Xes, yes, _ = windows_between(df, es0, t0, midnight_only=True)
        day = pd.to_datetime(tte).normalize()
        is_dark = np.array([d in dark for d in day])
        is_exit = np.array([(d - pd.Timedelta(days=1)) in dark for d in day])
        weekend = day.dayofweek.to_numpy() >= 5

        def score(pred, name, seed, params=np.nan):
            mae = np.abs(pred - yte).mean(axis=1)
            normal = ~is_dark & ~is_exit
            rows.append({
                "fold": fold, "variant": name, "seed": seed, "params": params,
                "MAE all": mae.mean(),
                "MAE weekday": mae[normal & ~weekend].mean(),
                "MAE weekend": mae[normal & weekend].mean(),
                "MAE dark day": mae[is_dark].mean() if is_dark.any() else np.nan,
                "MAE day after": mae[is_exit].mean() if is_exit.any() else np.nan,
            })

        score(Xte[:, -24:, 0], "Persistence (yesterday)", "-")
        score(Xte[:, :24, 0], "Same day last week", "-")

        train_df = df[df["timestamp"] < pd.Timestamp(es0)].reset_index(drop=True)
        delta = closure_delta(train_df, [d for d in DARK_DAYS if pd.Timestamp(d) < pd.Timestamp(es0)])

        for name, cfg, all_hours, aug in variants:
            Xtr, ytr = build_windows(train_df)
            if aug:
                parts = [(Xtr, ytr)] + [build_windows(inject_events(train_df, delta, seed=100 + s)[0]) for s in range(aug)]
                Xtr = np.concatenate([p[0] for p in parts]); ytr = np.concatenate([p[1] for p in parts])
            if not all_hours:
                keep = Xtr[:, -1, 1] == 23
                Xtr, ytr = Xtr[keep], ytr[keep]
            for seed in seeds:
                bs = 256 if all_hours else 64
                ep = 60 if all_hours else 150
                m = fit(cfg, Xtr, ytr, Xes, yes, seed, epochs=ep, bs=bs, patience=10 if all_hours else 20)
                with torch.no_grad():
                    score(m(torch.tensor(Xte)).numpy(), name, seed, count_parameters(m))
            print(f"  done: {fold} | {name}")

    res = pd.DataFrame(rows)
    res.to_csv(HERE / "experiments_results.csv", index=False)
    cols = ["MAE all", "MAE weekday", "MAE weekend", "MAE dark day", "MAE day after"]
    order = list(dict.fromkeys(res["variant"]))
    for fold in FOLDS:
        r = res[res["fold"] == fold]
        mean = r.groupby("variant")[cols].mean().reindex(order)
        spread = r.groupby("variant")["MAE all"].agg(lambda s: s.max() - s.min()).reindex(order)
        pers = mean.loc["Persistence (yesterday)", "MAE all"]
        mean["skill vs persistence"] = (1 - mean["MAE all"] / pers) * 100
        mean["seed spread"] = spread
        print(f"\n=== {fold}: mean daily MAE in kWh, {len(seeds)} seed(s) ===")
        print(mean.round(2).to_string())


if __name__ == "__main__":
    main()
