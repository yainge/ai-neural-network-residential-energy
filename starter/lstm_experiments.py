"""Compare the named models on the team split.

Every run uses train.main() and evaluate.evaluate_season() unchanged, so the
training settings, epoch selection and scoring are the team standard:
    batch size, learning rate, weight decay, scheduler, epochs, patience,
    all-hour training windows, best epoch = lowest weighted MAE on the blocks.

Two scores are kept for every run:
    best epoch  the team-standard number (epoch picked on the test blocks, optimistic)
    last epoch  the model after the full training budget (no picking on test days)

Run from the repo root:
    python starter/lstm_experiments.py --smoke-test      1 seed, 3 epochs
    python starter/lstm_experiments.py                   all models, 3 seeds
    python starter/lstm_experiments.py --export          also export + validate best seeds
    python starter/lstm_experiments.py --models lstm_hourly gru_daily --seeds 42 43
    python starter/lstm_experiments.py --organize-existing   import legacy runs only
    python starter/lstm_experiments.py --run-id my_run --export

Output:
    results/runs/<run_id>/          per-run CSVs and metadata
    results/lstm_*_history.csv      cumulative results, per-day scores and summaries
    results/lstm_comparison.csv     weighted MAE and skill side by side by run
    starter/checkpoints/<run_id>/   best-epoch and last-epoch checkpoints
    results/onnx/<run_id>/          with --export

Run IDs default to the data hash, run kind and Madrid date/time. Legacy CSVs remain in
place; their checkpoints and ONNX files move into folders named from the
recorded date, run kind and data hash. Legacy start times are unknown.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import re
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

import evaluate
import export
import train
import validate_onnx
from dataset import RAW_CSV, prepare_multi_datasets
from lstm_models import MODEL_NAMES, load_checkpoint_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINT_DIR = HERE / "checkpoints"
RESULTS_DIR = ROOT / "results"
DEFAULT_SEEDS = [42, 43, 44]
FULL_BLOCK_DAYS = 14
RUN_TIMEZONE = ZoneInfo("Europe/Madrid")
SCORE_COLS = [
    "weighted_mae", "mae_winter", "mae_spring", "mae_autumn",
    "winter_spring_mean_mae", "weekend_mae", "skill_pct", "weekend_skill_pct",
]


# ------------------------------------------------------------------------ ledger
def run_ledger() -> dict:
    """Facts that make a result traceable: date, code version, data version."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    raw = RAW_CSV.read_bytes()
    lines = raw.decode().strip().splitlines()
    return {
        "date": datetime.datetime.now(RUN_TIMEZONE).date().isoformat(),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_uncommitted_changes": bool(git("status", "--porcelain")),
        "data_sha256": hashlib.sha256(raw).hexdigest()[:12],
        "data_rows": len(lines) - 1,
        "data_last_timestamp": lines[-1].split(",")[0],
        "torch_version": torch.__version__,
    }


def save_history(frame: pd.DataFrame, kind: str, metadata: dict) -> None:
    """Save this run and retain other runs in the cumulative history."""
    enriched = frame.assign(**metadata)
    path = RESULTS_DIR / f"lstm_{kind}_history.csv"
    if path.exists():
        previous = pd.read_csv(path)
        previous = previous[previous["run_id"] != metadata["run_id"]]
        enriched = pd.concat([previous, enriched], ignore_index=True)
    temporary = path.with_suffix(".csv.tmp")
    enriched.to_csv(temporary, index=False)
    temporary.replace(path)
    if kind == "summary":
        comparison = enriched.pivot(
            index="model", columns="run_id", values=["weighted_mae", "skill_pct"]
        )
        comparison.columns = [f"{run_id}__{metric}" for metric, run_id in comparison.columns]
        comparison.to_csv(RESULTS_DIR / "lstm_comparison.csv")


def rename_existing_runs() -> None:
    """Give saved runs hash-first IDs and explicit experiment names."""
    renamed = {}
    kinds = {"full": "model_comparison", "quick": "smoke_test"}
    moves = []
    manifests = []
    for marker in sorted((RESULTS_DIR / "runs").glob("*/metadata.json")):
        metadata = json.loads(marker.read_text())
        old_kind = metadata.get("run_kind")
        if old_kind not in kinds:
            continue
        old_id = metadata["run_id"]
        kind = kinds[old_kind]
        prefix = metadata.get("run_started_at")
        stamp = (datetime.datetime.fromisoformat(prefix).strftime("%Y%m%d_%H%M%S_%f")
                 if prefix else metadata["date"])
        new_id = f"{metadata['data_sha256']}__{kind}__{stamp}"
        renamed[old_id] = new_id
        metadata.update(run_id=new_id, run_kind=kind)
        for parent in (RESULTS_DIR / "runs", CHECKPOINT_DIR, RESULTS_DIR / "onnx"):
            source = parent / old_id
            if source.exists():
                moves.append((source, parent / new_id))
        manifests.append((RESULTS_DIR / "runs" / new_id / "metadata.json", metadata))
    for parent in (RESULTS_DIR, CHECKPOINT_DIR, RESULTS_DIR / "onnx"):
        for source in parent.rglob("*"):
            # Directory contents are renamed after their parents move.
            if source.is_file() and "_quick" in source.name:
                target = source.with_name(source.name.replace("_quick_test", "_smoke_test").replace("_quick", "_smoke_test"))
                for old_id, new_id in renamed.items():
                    if old_id in target.parts:
                        target = Path(*(new_id if part == old_id else part for part in target.parts))
                        break
                moves.append((source, target))
    full_log = RESULTS_DIR / "lstm_full_run.log"
    if full_log.exists():
        moves.append((full_log, RESULTS_DIR / "lstm_model_comparison.log"))
    # A file can appear in overlapping searches; move it once.
    moves = list(dict.fromkeys(moves))
    for _, destination in moves:
        if destination.exists():
            raise FileExistsError(destination)
    for source, destination in moves:
        for old_id, new_id in renamed.items():
            if old_id in source.parts and not source.exists():
                source = Path(*(new_id if part == old_id else part for part in source.parts))
                break
        source.rename(destination)
    for marker, metadata in manifests:
        marker.write_text(json.dumps(metadata, indent=2) + "\n")
    if renamed:
        for path in RESULTS_DIR.rglob("*.csv"):
            frame = pd.read_csv(path)
            if "run_id" in frame:
                frame["run_id"] = frame["run_id"].replace(renamed)
            if "run_kind" in frame:
                frame["run_kind"] = frame["run_kind"].replace(kinds)
            frame.rename(columns={column: column.replace(old_id, new_id)
                                  for column in frame.columns
                                  for old_id, new_id in renamed.items() if old_id in column}, inplace=True)
            frame.to_csv(path, index=False)
        for old_id, new_id in renamed.items():
            print(f"Renamed {old_id} -> {new_id}")


def organize_existing_runs() -> None:
    """Import legacy results once, using their recorded date and data hash."""
    rename_existing_runs()
    smoke_tag = "_smoke_test" if (RESULTS_DIR / "lstm_results_smoke_test.csv").exists() else "_quick"
    for tag, kind in (("", "model_comparison"), (smoke_tag, "smoke_test")):
        source = RESULTS_DIR / f"lstm_results{tag}.csv"
        if not source.exists():
            continue
        runs = pd.read_csv(source)
        first = runs.iloc[0]
        run_id = f"{first['data_sha256']}__{kind}__{first['date']}"
        run_dir = RESULTS_DIR / "runs" / run_id
        marker = run_dir / "metadata.json"
        if marker.exists():
            continue
        metadata = {key: first[key].item() if hasattr(first[key], "item") else first[key]
                    for key in ("date", "data_sha256", "data_rows", "data_last_timestamp",
                                "n_train_windows", "n_test_winter", "n_test_spring", "n_test_autumn")}
        metadata.update(run_id=run_id, run_started_at="", run_kind=kind)
        checkpoint_dir = CHECKPOINT_DIR / run_id
        onnx_dir = RESULTS_DIR / "onnx" / run_id
        # Check all destinations before moving any artifacts.
        moves = []
        for row in runs.itertuples():
            for suffix in ("", "_last"):
                artifact = CHECKPOINT_DIR / f"{row.model}_seed{row.seed}{tag}{suffix}.pt"
                if artifact.exists():
                    moves.append((artifact, checkpoint_dir / artifact.name))
        for model in runs["model"].unique():
            artifact = RESULTS_DIR / "onnx" / f"{model}{tag}.onnx"
            if artifact.exists():
                moves.append((artifact, onnx_dir / artifact.name))
        for _, destination in moves:
            if destination.exists():
                raise FileExistsError(destination)
        run_dir.mkdir(parents=True, exist_ok=True)
        for artifact, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            artifact.rename(destination)
        for name in ("results", "per_day", "summary"):
            legacy = RESULTS_DIR / f"lstm_{name}{tag}.csv"
            if legacy.exists():
                frame = pd.read_csv(legacy)
                frame.assign(**metadata).to_csv(run_dir / f"lstm_{name}.csv", index=False)
                save_history(frame, name, metadata)
        marker.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Imported {run_id}; checkpoints: {checkpoint_dir}")


# ----------------------------------------------------------------------- scoring
def weighted(per_block: dict[str, float]) -> float:
    return sum(per_block[s] * w for s, w in evaluate.SEASON_WEIGHTS.items())


def block_arrays(ds) -> dict[str, np.ndarray]:
    """Targets, the two naive references and the target weekday for one test block."""
    X, y = ds.X, ds.y
    target_dow = ((X[:, -1, 2] + 1) % 7).astype(int)      # same rule as evaluate.py
    return {
        "y": y,
        "persistence": X[:, -24:, 0],                      # yesterday
        "last_week": X[:, :24, 0],                         # same weekday, 7 days back
        "target_dow": target_dow,
        "is_weekend": (target_dow == 5) | (target_dow == 6),
    }


def summarise_forecast(pred: dict[str, np.ndarray], val_dict: dict) -> tuple[dict, list[dict]]:
    """Score per-block forecasts with the team weights. Returns (scores, per-day rows)."""
    mae, base, mae_we, base_we, days = {}, {}, {}, {}, []
    for season, ds in val_dict.items():
        a = block_arrays(ds)
        day_mae = np.abs(pred[season] - a["y"]).mean(axis=1)
        day_base = np.abs(a["persistence"] - a["y"]).mean(axis=1)
        day_week = np.abs(a["last_week"] - a["y"]).mean(axis=1)
        mae[season], base[season] = float(day_mae.mean()), float(day_base.mean())
        we = a["is_weekend"]
        mae_we[season] = float(day_mae[we].mean()) if we.any() else 0.0
        base_we[season] = float(day_base[we].mean()) if we.any() else 0.0
        for i in range(len(day_mae)):
            days.append({
                "block": season, "day_index": i, "target_weekday": int(a["target_dow"][i]),
                "mae": float(day_mae[i]), "persistence_mae": float(day_base[i]),
                "last_week_mae": float(day_week[i]),
            })

    scores = {f"mae_{s}": v for s, v in mae.items()}
    scores["weighted_mae"] = weighted(mae)
    scores["weighted_persistence_mae"] = weighted(base)
    scores["skill_pct"] = (1.0 - weighted(mae) / weighted(base)) * 100.0
    scores["weekend_mae"] = weighted(mae_we)
    scores["weekend_skill_pct"] = (1.0 - weighted(mae_we) / weighted(base_we)) * 100.0
    scores["winter_spring_mean_mae"] = (mae["winter"] + mae["spring"]) / 2.0
    return scores, days


@torch.no_grad()
def score_checkpoint(checkpoint_path: Path, val_dict: dict, device: str) -> tuple[dict, list[dict]]:
    """Score one checkpoint. The headline numbers are cross-checked against evaluate.py."""
    model, _, _ = load_checkpoint_model(checkpoint_path, device)
    pred = {
        season: model(torch.from_numpy(ds.X).to(device)).cpu().numpy()
        for season, ds in val_dict.items()
    }
    scores, days = summarise_forecast(pred, val_dict)

    # Team-standard scoring code must give the same numbers.
    official = {s: evaluate.evaluate_season(model, ds, device) for s, ds in val_dict.items()}
    official_weighted = weighted({s: r[0] for s, r in official.items()})
    official_weekend = weighted({s: r[4] for s, r in official.items()})
    assert abs(scores["weighted_mae"] - official_weighted) < 1e-4, (scores["weighted_mae"], official_weighted)
    assert abs(scores["weekend_mae"] - official_weekend) < 1e-4, (scores["weekend_mae"], official_weekend)
    return scores, days


def baseline_rows(val_dict: dict) -> pd.DataFrame:
    """Persistence and same-day-last-week on exactly the same test days."""
    rows = []
    for name, key in (("persistence", "persistence"), ("last_week", "last_week")):
        pred = {s: block_arrays(ds)[key] for s, ds in val_dict.items()}
        scores, _ = summarise_forecast(pred, val_dict)
        rows.append({"model": name, "n_seeds": 0, **{c: scores[c] for c in SCORE_COLS}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- summary
def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per model: mean over seeds plus the seed spread of the weighted MAE."""
    rows = []
    for name, g in runs.groupby("model", sort=False):
        best = g.loc[g["weighted_mae"].idxmin()]
        row = {"model": name, "n_seeds": len(g)}
        row.update({c: g[c].mean() for c in SCORE_COLS})
        row["weighted_mae_std"] = g["weighted_mae"].std(ddof=1) if len(g) > 1 else float("nan")
        row["weighted_mae_min"] = g["weighted_mae"].min()
        row["weighted_mae_max"] = g["weighted_mae"].max()
        row["last_weighted_mae"] = g["last_weighted_mae"].mean()
        row["last_weighted_mae_std"] = g["last_weighted_mae"].std(ddof=1) if len(g) > 1 else float("nan")
        row["last_winter_spring_mean_mae"] = g["last_winter_spring_mean_mae"].mean()
        row["last_skill_pct"] = g["last_skill_pct"].mean()
        row["mean_best_epoch"] = g["best_epoch"].mean()
        row["best_seed"] = int(best["seed"])
        row["n_parameters"] = int(best["n_parameters"])
        row["n_train_windows"] = int(best["n_train_windows"])
        row["mean_minutes_per_run"] = g["train_seconds"].mean() / 60.0
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, baselines: pd.DataFrame, block_sizes: dict[str, int]) -> None:
    line = "=" * 124
    print(f"\n{line}\nBEST EPOCH (team standard: epoch picked on the test blocks). Mean over seeds, MAE in kWh.\n{line}")
    print(
        f"{'model':<12} {'seeds':>5} {'weighted':>9} {'std':>6} {'min':>6} {'max':>6} "
        f"{'winter':>7} {'spring':>7} {'autumn':>7} {'W+S avg':>8} {'weekend':>8} "
        f"{'skill %':>8} {'params':>9} {'train win':>9}"
    )
    for _, r in baselines.iterrows():
        print(
            f"{r['model']:<12} {'-':>5} {r['weighted_mae']:>9.4f} {'-':>6} {'-':>6} {'-':>6} "
            f"{r['mae_winter']:>7.3f} {r['mae_spring']:>7.3f} {r['mae_autumn']:>7.3f} "
            f"{r['winter_spring_mean_mae']:>8.3f} {r['weekend_mae']:>8.3f} "
            f"{r['skill_pct']:>+8.2f} {'-':>9} {'-':>9}"
        )
    for _, r in summary.iterrows():
        print(
            f"{r['model']:<12} {int(r['n_seeds']):>5} {r['weighted_mae']:>9.4f} "
            f"{r['weighted_mae_std']:>6.3f} {r['weighted_mae_min']:>6.3f} {r['weighted_mae_max']:>6.3f} "
            f"{r['mae_winter']:>7.3f} {r['mae_spring']:>7.3f} {r['mae_autumn']:>7.3f} "
            f"{r['winter_spring_mean_mae']:>8.3f} {r['weekend_mae']:>8.3f} "
            f"{r['skill_pct']:>+8.2f} {int(r['n_parameters']):>9,} {int(r['n_train_windows']):>9,}"
        )

    print(f"\n{line}\nLAST EPOCH (no picking on test days). Mean over seeds, MAE in kWh.\n{line}")
    print(f"{'model':<12} {'weighted':>9} {'std':>6} {'W+S avg':>8} {'skill %':>8} "
          f"{'gap to best epoch':>18} {'mean best epoch':>16}")
    for _, r in summary.iterrows():
        print(
            f"{r['model']:<12} {r['last_weighted_mae']:>9.4f} {r['last_weighted_mae_std']:>6.3f} "
            f"{r['last_winter_spring_mean_mae']:>8.3f} {r['last_skill_pct']:>+8.2f} "
            f"{r['last_weighted_mae'] - r['weighted_mae']:>+18.3f} {r['mean_best_epoch']:>16.1f}"
        )

    print("-" * 124)
    print("Test windows per block: " + ", ".join(f"{k}={v}" for k, v in block_sizes.items()))
    for name, n in block_sizes.items():
        if n < FULL_BLOCK_DAYS:
            weight = evaluate.SEASON_WEIGHTS[name] * 100
            print(
                f"WARNING: the {name} block has only {n} of {FULL_BLOCK_DAYS} days "
                f"(raw.csv ends early) but carries {weight:.0f}% of the weighted MAE. "
                "Read the weighted MAE and the autumn column with care; "
                "'W+S avg' does not depend on it."
            )


def export_best(summary: pd.DataFrame, tag: str, checkpoint_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Export the best seed of each model and run the platform validator on it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\nONNX export of the best seed per model (best-epoch checkpoint)")
    for _, r in summary.iterrows():
        name, seed = r["model"], int(r["best_seed"])
        checkpoint_path = checkpoint_dir / f"{name}_seed{seed}{tag}.pt"
        onnx_path = out_dir / f"{name}{tag}.onnx"
        with contextlib.redirect_stdout(io.StringIO()):
            export.main(checkpoint_path=checkpoint_path, onnx_out=onnx_path)

        validate_onnx.ONNX_PATH = onnx_path
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                validate_onnx.main()
            status = "PASS"
        except (SystemExit, AssertionError) as exc:
            status = "FAIL"
            print(buffer.getvalue())
            print(exc)
        size_kb = onnx_path.stat().st_size / 1024
        print(f"  {name:<12} seed {seed} | validate_onnx: {status} | {size_kb:.1f} KB | {onnx_path}")
        rows.append({"model": name, "best_seed": seed, "onnx_valid": status, "onnx_kb": size_kb})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the named models on the team split.")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=train.EPOCHS)
    parser.add_argument("--smoke-test", "--quick", dest="smoke_test", action="store_true",
                        help="pipeline check: 1 seed, 3 epochs (--quick is a legacy alias)")
    parser.add_argument("--export", action="store_true", help="export + validate the best seed per model")
    parser.add_argument("--run-id", help="unique folder name; default: Madrid date/time and run kind")
    parser.add_argument("--organize-existing", action="store_true", help="organize legacy runs and exit without training")
    args = parser.parse_args()

    tag = ""
    if args.smoke_test:
        args.seeds, args.epochs, tag = args.seeds[:1], 3, "_smoke_test"

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    organize_existing_runs()
    if args.organize_existing:
        return
    now = datetime.datetime.now(RUN_TIMEZONE)
    ledger = run_ledger()
    run_kind = "smoke_test" if args.smoke_test else "model_comparison"
    run_id = args.run_id or f"{ledger['data_sha256']}__{run_kind}__{now:%Y%m%d_%H%M%S_%f}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        parser.error("--run-id must contain only letters, numbers, underscores, dots or hyphens")
    run_dir = RESULTS_DIR / "runs" / run_id
    checkpoint_dir = CHECKPOINT_DIR / run_id
    onnx_dir = RESULTS_DIR / "onnx" / run_id
    for directory in (run_dir, checkpoint_dir, onnx_dir):
        if directory.exists():
            parser.error(f"Run destination already exists: {directory}")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    runs_csv = run_dir / "lstm_results.csv"
    days_csv = run_dir / "lstm_per_day.csv"
    summary_csv = run_dir / "lstm_summary.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ledger.update(run_id=run_id, run_started_at=now.isoformat(), run_kind=run_kind)
    (run_dir / "metadata.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"Run ID: {run_id}")
    print(f"Models: {args.models}\nSeeds: {args.seeds} | epochs: {args.epochs} | device: {device} "
          f"| torch threads: {torch.get_num_threads()}")
    print(f"Settings from train.py: batch={train.BATCH_SIZE}, lr={train.LEARNING_RATE}, "
          f"patience={train.PATIENCE}, weights={evaluate.SEASON_WEIGHTS}")
    print(f"Ledger: {ledger}\n")

    _, val_dict = prepare_multi_datasets()
    block_sizes = {name: len(ds) for name, ds in val_dict.items()}
    history_metadata = {**ledger, **{f"n_test_{k}": v for k, v in block_sizes.items()}}
    baselines = baseline_rows(val_dict)

    rows, day_rows = [], []
    total = len(args.models) * len(args.seeds)
    started = time.time()
    for name in args.models:
        for seed in args.seeds:
            print(f"\n{'#' * 100}\n# Run {len(rows) + 1}/{total}: {name}, seed {seed}\n{'#' * 100}", flush=True)
            best_path = checkpoint_dir / f"{name}_seed{seed}{tag}.pt"
            last_path = checkpoint_dir / f"{name}_seed{seed}{tag}_last.pt"
            t0 = time.time()
            info = train.main(model_name=name, seed=seed, epochs=args.epochs, checkpoint_path=best_path)
            seconds = time.time() - t0

            # Keep the last-epoch model too, in the same checkpoint format.
            torch.save(
                {
                    "contract_version": "1.0",
                    "model_state_dict": info["final_state_dict"],
                    "mu": info["mu"],
                    "sigma": info["sigma"],
                    "best_val_mae": info["final_val_mae"],
                    "model_name": name,
                    "model_config": info["model_config"],
                    "seed": seed,
                    "epoch": info["final_epoch"],
                },
                last_path,
            )

            best_scores, best_days = score_checkpoint(best_path, val_dict, device)
            last_scores, last_days = score_checkpoint(last_path, val_dict, device)
            # Both must equal what train.py reported for those epochs.
            assert abs(best_scores["weighted_mae"] - info["best_val_mae"]) < 1e-3, (best_scores, info["best_val_mae"])
            assert abs(last_scores["weighted_mae"] - info["final_val_mae"]) < 1e-3, (last_scores, info["final_val_mae"])

            row = {"model": name, "seed": seed, "epochs": args.epochs,
                   "best_epoch": info["best_epoch"], "last_epoch": info["final_epoch"]}
            row.update(best_scores)
            row.update({f"last_{k}": v for k, v in last_scores.items()})
            row.update({
                "n_parameters": info["n_parameters"],
                "n_train_windows": info["n_train_windows"],
                "train_seconds": seconds,
                **{f"n_test_{k}": v for k, v in block_sizes.items()},
                **ledger,
            })
            rows.append(row)
            for which, days in (("best", best_days), ("last", last_days)):
                day_rows += [{"model": name, "seed": seed, "checkpoint": which, **d} for d in days]
            pd.DataFrame(rows).to_csv(runs_csv, index=False)      # saved after every run
            pd.DataFrame(day_rows).to_csv(days_csv, index=False)
            save_history(pd.DataFrame(rows), "results", history_metadata)
            save_history(pd.DataFrame(day_rows), "per_day", history_metadata)
            print(f"-> {name} seed {seed}: best epoch {info['best_epoch']} weighted MAE "
                  f"{row['weighted_mae']:.4f} (skill {row['skill_pct']:+.2f}%) | last epoch "
                  f"{info['final_epoch']} weighted MAE {row['last_weighted_mae']:.4f} | "
                  f"{seconds:.0f} s (elapsed {(time.time() - started) / 60:.1f} min)", flush=True)

    runs = pd.DataFrame(rows)
    summary = summarise(runs)
    if args.export:
        summary = summary.merge(export_best(summary, tag, checkpoint_dir, onnx_dir), on=["model", "best_seed"])
    combined = pd.concat([baselines, summary], ignore_index=True)
    history_metadata["n_train_windows"] = int(runs.iloc[0]["n_train_windows"])
    combined.assign(**history_metadata).to_csv(summary_csv, index=False)
    save_history(combined, "summary", history_metadata)
    print_summary(summary, baselines, block_sizes)

    print("\nTime per run (seconds): "
          + ", ".join(f"{n}={g['train_seconds'].mean():.0f}" for n, g in runs.groupby("model", sort=False)))
    print(f"Total: {(time.time() - started) / 60:.1f} min")
    print(f"Saved: {runs_csv}\n       {days_csv}\n       {summary_csv}")


if __name__ == "__main__":
    main()
