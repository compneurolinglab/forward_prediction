#!/usr/bin/env python3
"""
Ridge regression: sentence-level model embeddings → neural activity.

For one denoising step, fit ridge regression from embedding features to each
(sensor, timepoint) neural response across sentences. Report held-out Pearson r
on a sequential train/test split (default 90% / 10%).

Inputs
------
embeddings : (n_sentences, n_steps, n_features)
neural     : (n_sentences, n_sensors, n_timepoints)

For each sensor s and timepoint t:
  X = embeddings[:, step-1, :]
  y = neural[:, s, t]
  RidgeCV on train sentences → predict test → Pearson r

Outputs one file per sensor:
  {output_dir}/step{step}_{tag}/sensor_{idx:03d}.npy  shape (n_timepoints,)

Example:
  python ridge_regression_sentence.py \\
    --embeddings embeddings.npy \\
    --neural neural.npy \\
    --step 5 \\
    --tag sub-01_comp \\
    --output-dir ./ridge_out
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ConstantInputWarning, NearConstantInputWarning, pearsonr
from sklearn.linear_model import RidgeCV


DEFAULT_TRAIN_FRAC = 0.9
DEFAULT_MIN_TRAIN = 10
DEFAULT_MIN_TEST = 3
DEFAULT_ALPHAS = np.logspace(0, 20, 10)


def sequential_train_test_indices(
    n_sentences: int,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """First train_frac sentences for train; remainder for test."""
    if n_sentences < 2:
        raise ValueError(f"Need at least 2 sentences, got {n_sentences}")
    frac = float(train_frac)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {frac}")
    n_train = max(1, int(frac * n_sentences))
    if n_train >= n_sentences:
        n_train = n_sentences - 1
    train_idx = np.arange(n_train, dtype=int)
    test_idx = np.arange(n_train, n_sentences, dtype=int)
    return train_idx, test_idx


def ridge_test_correlation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alphas: Sequence[float],
    min_train: int = DEFAULT_MIN_TRAIN,
    min_test: int = DEFAULT_MIN_TEST,
) -> float:
    """RidgeCV on train rows; Pearson r between predictions and y on test rows."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    train_ok = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    test_ok = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
    X_tr, y_tr = X_train[train_ok], y_train[train_ok]
    X_te, y_te = X_test[test_ok], y_test[test_ok]

    if len(X_tr) < int(min_train) or len(X_te) < int(min_test):
        return float("nan")
    if np.std(y_tr) == 0.0:
        return float("nan")

    alpha_grid = np.asarray(alphas, dtype=np.float64)
    try:
        ridge = RidgeCV(alphas=alpha_grid, cv=min(5, len(X_tr)))
        ridge.fit(X_tr, y_tr)
        pred = ridge.predict(X_te)
        if np.std(y_te) == 0.0 or np.std(pred) == 0.0:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            warnings.simplefilter("ignore", NearConstantInputWarning)
            corr, _ = pearsonr(y_te, pred)
        return float(corr) if np.isfinite(corr) else float("nan")
    except Exception:
        return float("nan")


def sensor_output_path(
    output_dir: str,
    tag: str,
    step: int,
    sensor_idx: int,
) -> str:
    return os.path.join(output_dir, f"step{int(step)}_{tag}", f"sensor_{int(sensor_idx):03d}.npy")


def process_sensor_timepoints(
    sensor_idx: int,
    neural_data: np.ndarray,
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: Sequence[float],
    output_path: str | None = None,
    skip_existing: bool = True,
) -> tuple[int, np.ndarray, bool]:
    """Compute held-out r for all timepoints at one sensor; optionally save."""
    if output_path and skip_existing and os.path.isfile(output_path):
        return sensor_idx, np.load(output_path), True

    n_timepoints = int(neural_data.shape[2])
    out = np.full(n_timepoints, np.nan, dtype=np.float64)
    for ti in range(n_timepoints):
        y = neural_data[:, sensor_idx, ti]
        out[ti] = ridge_test_correlation(
            X_train,
            y[train_idx],
            X_test,
            y[test_idx],
            alphas=alphas,
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, out)

    return sensor_idx, out, False


def sensor_chunk_bounds(n_sensors: int, n_chunks: int, chunk_id: int) -> tuple[int, int]:
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    if chunk_id < 0 or chunk_id >= n_chunks:
        raise ValueError(f"chunk_id must be in [0, {n_chunks - 1}], got {chunk_id}")
    chunk_size = (n_sensors + n_chunks - 1) // n_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, n_sensors)
    return start, end


def run_ridge_sentence(
    embeddings: np.ndarray,
    neural_data: np.ndarray,
    *,
    step: int,
    tag: str,
    output_dir: str,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    alphas: Sequence[float] | None = None,
    sensor_start: int = 0,
    sensor_end: int | None = None,
    n_jobs: int = 1,
    skip_existing: bool = True,
) -> dict[str, object]:
    """
    Run ridge regression for one embedding step across sensors.

    Returns summary dict including stacked results (n_sensors_chunk, n_timepoints).
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    neu = np.asarray(neural_data, dtype=np.float64)
    if emb.ndim != 3:
        raise ValueError(f"embeddings must be 3D (sentences, steps, features); got {emb.shape}")
    if neu.ndim != 3:
        raise ValueError(f"neural must be 3D (sentences, sensors, timepoints); got {neu.shape}")

    n_sentences_emb, n_steps, _n_feat = emb.shape
    n_sentences_neu, n_sensors, n_timepoints = neu.shape
    if n_sentences_emb != n_sentences_neu:
        raise ValueError(
            f"Sentence count mismatch: embeddings {n_sentences_emb} vs neural {n_sentences_neu}"
        )
    step_i = int(step) - 1
    if step_i < 0 or step_i >= n_steps:
        raise ValueError(f"step must be in [1, {n_steps}], got {step}")

    if sensor_end is None:
        sensor_end = n_sensors
    sensor_start = max(0, min(int(sensor_start), n_sensors))
    sensor_end = max(sensor_start, min(int(sensor_end), n_sensors))
    if sensor_end <= sensor_start:
        raise ValueError("Empty sensor range.")

    alpha_grid = np.asarray(DEFAULT_ALPHAS if alphas is None else alphas, dtype=np.float64)
    train_idx, test_idx = sequential_train_test_indices(n_sentences_emb, train_frac=train_frac)

    X = emb[:, step_i, :]
    X_train, X_test = X[train_idx], X[test_idx]

    n_jobs = max(1, min(int(n_jobs), sensor_end - sensor_start))
    sensor_results = Parallel(n_jobs=n_jobs)(
        delayed(process_sensor_timepoints)(
            sensor,
            neu,
            X_train=X_train,
            X_test=X_test,
            train_idx=train_idx,
            test_idx=test_idx,
            alphas=alpha_grid,
            output_path=sensor_output_path(output_dir, tag, step, sensor),
            skip_existing=skip_existing,
        )
        for sensor in range(sensor_start, sensor_end)
    )

    saved = skipped = 0
    rows: list[np.ndarray] = []
    for _sensor, corrs, was_skipped in sensor_results:
        if was_skipped:
            skipped += 1
        else:
            saved += 1
        rows.append(corrs)

    stacked = np.stack(rows, axis=0) if rows else np.zeros((0, n_timepoints), dtype=np.float64)
    valid = int(np.sum(np.isfinite(stacked)))

    return {
        "tag": tag,
        "step": int(step),
        "n_sentences": int(n_sentences_emb),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_sensors_chunk": int(stacked.shape[0]),
        "n_timepoints": int(n_timepoints),
        "sensor_start": int(sensor_start),
        "sensor_end": int(sensor_end),
        "saved": int(saved),
        "skipped": int(skipped),
        "valid_correlations": valid,
        "total_regressions": int(stacked.size),
        "results": stacked,
        "alphas": alpha_grid.tolist(),
        "train_frac": float(train_frac),
    }


def parse_alphas(text: str) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty alpha list.")
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Numpy array (n_sentences, n_steps, n_features).",
    )
    parser.add_argument(
        "--neural",
        type=str,
        required=True,
        help="Numpy array (n_sentences, n_sensors, n_timepoints).",
    )
    parser.add_argument("--step", type=int, required=True, help="Embedding step index (1-based).")
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Output subfolder tag, e.g. sub-01_comp.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument(
        "--alphas",
        type=str,
        default="",
        help="Comma-separated ridge alphas (default: logspace 1e0..1e20, 10 values).",
    )
    parser.add_argument("--sensor-start", type=int, default=0)
    parser.add_argument("--sensor-end", type=int, default=None)
    parser.add_argument("--n-chunks", type=int, default=None)
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if per-sensor output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.n_chunks is None) ^ (args.chunk_id is None):
        raise SystemExit("Provide both --n-chunks and --chunk-id, or neither.")

    embeddings = np.load(args.embeddings)
    neural = np.load(args.neural)

    sensor_start = int(args.sensor_start)
    sensor_end = args.sensor_end
    if args.n_chunks is not None:
        n_sensors = int(neural.shape[1])
        sensor_start, sensor_end = sensor_chunk_bounds(
            n_sensors, int(args.n_chunks), int(args.chunk_id)
        )

    alphas = parse_alphas(args.alphas) if str(args.alphas).strip() else None

    result = run_ridge_sentence(
        embeddings,
        neural,
        step=int(args.step),
        tag=str(args.tag),
        output_dir=str(args.output_dir),
        train_frac=float(args.train_frac),
        alphas=alphas,
        sensor_start=sensor_start,
        sensor_end=sensor_end,
        n_jobs=int(args.n_jobs),
        skip_existing=not bool(args.overwrite),
    )

    os.makedirs(str(args.output_dir), exist_ok=True)
    summary_path = os.path.join(
        str(args.output_dir),
        f"step{args.step}_{args.tag}_ridge_summary.json",
    )
    summary = {k: v for k, v in result.items() if k != "results"}
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"[{args.tag}] step={args.step} sensors=[{result['sensor_start']}, {result['sensor_end']}) "
        f"saved={result['saved']} skipped={result['skipped']} "
        f"valid_r={result['valid_correlations']}/{result['total_regressions']}"
    )
    print(f"Wrote per-sensor files under {args.output_dir} and {summary_path}")


if __name__ == "__main__":
    main()
