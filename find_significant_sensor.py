#!/usr/bin/env python3
"""
Select significant ECoG sensors from ridge correlation maps (downstream analysis).

Input:
  - Observed r map: (n_sensors, n_timepoints, n_steps)
  - Pre-computed null map stack from upstream: (n_null, n_sensors, n_timepoints, n_steps)

The null stack must be generated upstream using a permutation scheme appropriate for
the experimental design. This script assumes that the provided null maps are valid
and exchangeable under the null hypothesis; it does not generate or validate them.

Pipeline:
  1. Load observed and null maps.
  2. At each (sensor, time, step), rank observed r against the null distribution
     (non-finite null draws are excluded per cell).
  3. Apply Benjamini-Hochberg FDR across all cells (default).
  4. Select sensors meeting the significance threshold (--selection-mode).

Example:
  python significant_sensor_selection.py \\
    --r-map path/to/cluster_r.npy \\
    --null-map path/to/cluster_null_stack.npy \\
    --output-dir path/to/out \\
    --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_STEPS = [1, 2, 3, 4, 5]
DEFAULT_ALPHA = 0.05


def load_null_map_stack(
    path: str,
    *,
    n_null: int | None = None,
    mmap: bool = False,
) -> np.ndarray:
    """
    Load upstream null map stack.

    Expected shape: (n_null, n_sensors, n_timepoints, n_steps).
    """
    mode = "r" if mmap else None
    null = np.load(path, mmap_mode=mode)
    null = np.asarray(null, dtype=np.float64)
    if null.ndim != 4:
        raise ValueError(
            f"null_map must be 4D (n_null, sensors, time, steps); got shape {null.shape}"
        )
    if n_null is not None:
        n_use = min(int(n_null), int(null.shape[0]))
        null = null[:n_use]
    return null


def rank_pvalues_upper(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """
    One-sided p: P(null >= obs) at each cell.

    Non-finite null draws are excluded from the rank count and denominator.
    Cells with no finite null draws or non-finite observed values receive NaN.
    """
    obs = np.asarray(obs, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    if null.ndim != 4:
        raise ValueError(f"null must be 4D (n_null, S, T, K); got {null.shape}")
    if null.shape[1:] != obs.shape:
        raise ValueError(
            f"null trailing dims {null.shape[1:]} != observed shape {obs.shape}"
        )

    out = np.full(obs.shape, np.nan, dtype=np.float64)
    obs_finite = np.isfinite(obs)
    null_finite = np.isfinite(null)
    n_finite = np.sum(null_finite, axis=0)
    ge = null >= obs[np.newaxis, ...]
    counts = np.sum(ge & null_finite, axis=0)

    valid = obs_finite & (n_finite > 0)
    out[valid] = (counts[valid] + 1.0) / (n_finite[valid] + 1.0)
    return out


def benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values; NaN p-values remain NaN."""
    shape = p_values.shape
    flat = np.asarray(p_values, dtype=np.float64).reshape(-1)
    q_flat = np.full(flat.size, np.nan, dtype=np.float64)
    finite = np.isfinite(flat)
    if not bool(finite.any()):
        return q_flat.reshape(shape)

    p = flat[finite]
    m = int(p.size)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / float(rank)
        prev = min(prev, val)
        adj[i] = prev
    q_sorted = np.empty(m, dtype=np.float64)
    q_sorted[order] = np.clip(adj, 0.0, 1.0)
    q_flat[finite] = q_sorted
    return q_flat.reshape(shape)


def build_significance_mask(
    data: np.ndarray,
    test_p: np.ndarray,
    *,
    alpha: float,
    require_positive_r: bool = False,
) -> np.ndarray:
    """Boolean mask of significant (sensor, time, step) cells."""
    arr = np.asarray(data, dtype=np.float64)
    pv = np.asarray(test_p, dtype=np.float64)
    sig = (pv <= float(alpha)) & np.isfinite(arr) & np.isfinite(pv)
    if require_positive_r:
        sig &= arr > 0.0
    return sig


def select_sensors_from_mask(
    sig_mask: np.ndarray,
    *,
    selection_mode: str = "any_cell",
) -> np.ndarray:
    """Return boolean vector (n_sensors,) indicating selected sensors."""
    mask = np.asarray(sig_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D sig mask, got shape {mask.shape}")
    flat = mask.reshape(mask.shape[0], -1)
    if selection_mode == "any_cell":
        return np.any(flat, axis=1)
    if selection_mode == "all_cells":
        return np.all(flat, axis=1)
    raise ValueError(f"Unknown selection_mode: {selection_mode!r}")


def run_sensor_selection(
    r_map: np.ndarray,
    null_map: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    selection_mode: str = "any_cell",
    use_fdr: bool = True,
    require_positive_r: bool = False,
) -> dict[str, object]:
    """Rank observed r against upstream null stack and select significant sensors."""
    X = np.asarray(r_map, dtype=np.float64)
    null = np.asarray(null_map, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"r_map must be 3D (sensors, time, steps); got {X.shape}")
    if null.ndim != 4:
        raise ValueError(
            f"null_map must be 4D (n_null, sensors, time, steps); got {null.shape}"
        )
    if null.shape[1:] != X.shape:
        raise ValueError(
            f"null_map trailing dims {null.shape[1:]} != r_map shape {X.shape}"
        )

    p_values = rank_pvalues_upper(X, null)
    q_values = benjamini_hochberg_fdr(p_values)
    test_p = q_values if use_fdr else p_values

    sig_mask = build_significance_mask(
        X,
        test_p,
        alpha=alpha,
        require_positive_r=require_positive_r,
    )
    selected = select_sensors_from_mask(sig_mask, selection_mode=selection_mode)
    selected_rows = np.flatnonzero(selected).astype(int)

    p_sensor_min = np.nanmin(p_values.reshape(p_values.shape[0], -1), axis=1)
    q_sensor_min = np.nanmin(q_values.reshape(q_values.shape[0], -1), axis=1)

    n_sig_cells = int(np.sum(sig_mask))
    n_total_cells = int(np.prod(sig_mask.shape))
    n_sig_sensors = int(selected.sum())

    return {
        "r_map_shape": tuple(int(x) for x in X.shape),
        "null_map_shape": tuple(int(x) for x in null.shape),
        "p_values": p_values,
        "p_sensor_min": p_sensor_min,
        "q_values": q_values,
        "q_sensor_min": q_sensor_min,
        "sig_mask": sig_mask,
        "selected": selected,
        "selected_rows": selected_rows,
        "n_sig_cells": n_sig_cells,
        "n_total_cells": n_total_cells,
        "n_sig_sensors": n_sig_sensors,
        "n_null": int(null.shape[0]),
        "alpha": float(alpha),
        "selection_mode": selection_mode,
        "use_fdr": bool(use_fdr),
        "require_positive_r": bool(require_positive_r),
    }


def build_selected_sensors_table(
    result: dict[str, object],
    *,
    steps: Sequence[int],
    sensor_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Summarize significant cells for each selected sensor."""
    sig_mask = np.asarray(result["sig_mask"], dtype=bool)
    selected_rows = np.asarray(result["selected_rows"], dtype=int)
    p_values = np.asarray(result["p_values"], dtype=np.float64)
    q_values = np.asarray(result["q_values"], dtype=np.float64)
    use_fdr = bool(result["use_fdr"])
    rows: list[dict[str, object]] = []

    for local_row in selected_rows:
        sid = int(sensor_ids[local_row]) if sensor_ids is not None else int(local_row)
        for ti in range(sig_mask.shape[1]):
            for ki, step in enumerate(steps):
                if not sig_mask[local_row, ti, ki]:
                    continue
                row: dict[str, object] = {
                    "sensor_row": int(local_row),
                    "sensor_id": sid,
                    "time_idx": int(ti),
                    "step": int(step),
                    "p_value": float(p_values[local_row, ti, ki]),
                    "q_value": float(q_values[local_row, ti, ki]),
                }
                if use_fdr:
                    row["test_stat"] = float(q_values[local_row, ti, ki])
                else:
                    row["test_stat"] = float(p_values[local_row, ti, ki])
                rows.append(row)
    return pd.DataFrame(rows)


def save_selection_outputs(
    result: dict[str, object],
    *,
    output_dir: str,
    tag: str,
    steps: Sequence[int],
    sensor_ids: Sequence[int] | None = None,
    null_map_path: str = "",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{tag}_p_values.npy"), result["p_values"])
    np.save(os.path.join(output_dir, f"{tag}_p_sensor_min.npy"), result["p_sensor_min"])
    np.save(os.path.join(output_dir, f"{tag}_q_values.npy"), result["q_values"])
    np.save(os.path.join(output_dir, f"{tag}_q_sensor_min.npy"), result["q_sensor_min"])
    np.save(os.path.join(output_dir, f"{tag}_sig_mask.npy"), result["sig_mask"])
    np.save(os.path.join(output_dir, f"{tag}_selected_rows.npy"), result["selected_rows"])

    summary = {
        "tag": tag,
        "r_map_shape": list(result["r_map_shape"]),
        "null_map_shape": list(result["null_map_shape"]),
        "null_map_path": null_map_path,
        "n_null": result["n_null"],
        "n_sig_cells": result["n_sig_cells"],
        "n_total_cells": result["n_total_cells"],
        "n_sig_sensors": result["n_sig_sensors"],
        "alpha": result["alpha"],
        "selection_mode": result["selection_mode"],
        "use_fdr": result["use_fdr"],
        "require_positive_r": result["require_positive_r"],
        "method": "rank_against_upstream_null",
        "p_value_type": "fdr" if result["use_fdr"] else "uncorrected",
    }
    with open(os.path.join(output_dir, f"{tag}_selection_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    counts = pd.DataFrame(
        [
            {
                "tag": tag,
                "n_sensors_total": int(result["r_map_shape"][0]),
                "n_selected_sensors": int(result["n_sig_sensors"]),
            }
        ]
    )
    counts.to_csv(os.path.join(output_dir, f"{tag}_selection_counts.csv"), index=False)

    detail = build_selected_sensors_table(result, steps=steps, sensor_ids=sensor_ids)
    detail.to_csv(os.path.join(output_dir, f"{tag}_selected_cells.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r-map",
        type=str,
        required=True,
        help="Observed r map .npy with shape (n_sensors, n_timepoints, n_steps).",
    )
    parser.add_argument(
        "--null-map",
        type=str,
        required=True,
        help="Upstream null stack .npy with shape (n_null, n_sensors, n_timepoints, n_steps).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for p-values, masks, and summary CSV/JSON.",
    )
    parser.add_argument("--tag", type=str, default="selection", help="Output filename prefix.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Threshold on FDR q-values by default; on raw p if --no-fdr.",
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=None,
        help="Use only the first N null maps from the stack (default: all).",
    )
    parser.add_argument(
        "--null-mmap",
        action="store_true",
        help="Memory-map the null stack .npy (useful for large stacks).",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("any_cell", "all_cells"),
        default="any_cell",
        help="Include sensor if any cell (default) or all cells pass the threshold.",
    )
    parser.add_argument(
        "--no-fdr",
        action="store_true",
        help="Use uncorrected p-values for thresholding (default: BH-FDR q-values).",
    )
    parser.add_argument(
        "--require-positive-r",
        action="store_true",
        help="Only count cells with r > 0 as significant.",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument(
        "--sensor-ids",
        type=str,
        default="",
        help="Optional .npy or comma-separated global sensor IDs aligned to r-map rows.",
    )
    return parser.parse_args()


def load_sensor_ids(path_or_csv: str, n_rows: int) -> np.ndarray | None:
    text = str(path_or_csv).strip()
    if not text:
        return None
    if os.path.isfile(text):
        ids = np.load(text)
    else:
        ids = np.asarray([int(x) for x in text.split(",")], dtype=int)
    if int(ids.size) != int(n_rows):
        raise ValueError(f"sensor_ids length {ids.size} != r_map rows {n_rows}")
    return ids


def main() -> None:
    args = parse_args()
    r_map = np.load(args.r_map)
    null_map = load_null_map_stack(
        args.null_map,
        n_null=args.n_null,
        mmap=bool(args.null_mmap),
    )
    sensor_ids = load_sensor_ids(args.sensor_ids, int(r_map.shape[0]))

    result = run_sensor_selection(
        r_map,
        null_map,
        alpha=float(args.alpha),
        selection_mode=str(args.selection_mode),
        use_fdr=not bool(args.no_fdr),
        require_positive_r=bool(args.require_positive_r),
    )

    save_selection_outputs(
        result,
        output_dir=str(args.output_dir),
        tag=str(args.tag),
        steps=[int(s) for s in args.steps],
        sensor_ids=sensor_ids,
        null_map_path=str(args.null_map),
    )

    correction = "FDR" if result["use_fdr"] else "uncorrected p"
    print(
        f"Selected {result['n_sig_sensors']}/{result['r_map_shape'][0]} sensors "
        f"({result['n_sig_cells']}/{result['n_total_cells']} significant cells, "
        f"n_null={result['n_null']}, alpha={result['alpha']}, correction={correction})"
    )
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
