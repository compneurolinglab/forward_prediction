#!/usr/bin/env python3
"""
Compute step forwardness and infer cross-condition significant time windows.

For each sensor and timepoint, given ridge r values at denoising steps 1..5:

  forwardness = sum(step_k * r_k) / sum(r_k)

This is the weighted step index (typically between 1 and 5). It is not rescaled
to [0, 1]. Cross-sensor SEM uses std(ddof=0) / sqrt(n_valid_sensors).

Single-condition mode:
  Compute per-sensor forwardness and optional mean ± SEM over selected sensors.

Multi-condition mode (requires --null-map-by-condition):
  1. Compute forwardness per condition on observed and upstream null r stacks.
  2. Build observed and null cross-condition F timecourses.
  3. Rank observed F against null F; temporal cluster permutation for sig windows.
  4. Welch t-tests between condition pairs within each significant window (BH-FDR).

Null stacks must be generated upstream using a permutation scheme appropriate
for the experimental design. This script assumes they are valid and
exchangeable under the null hypothesis; it does not generate or validate them.

Examples:
  python compute_forwardness.py \\
    --r-map comp_r.npy --selected-rows selected_rows.npy \\
    --output-dir out/comp

  python compute_forwardness.py \\
    --r-map-by-condition comp=comp_r.npy prod=prod_r.npy podcast=pod_r.npy \\
    --null-map-by-condition comp=comp_null.npy prod=prod_null.npy podcast=pod_null.npy \\
    --selected-rows-by-condition comp=comp_sel.npy prod=prod_sel.npy podcast=pod_sel.npy \\
    --output-dir out/inference --tag cluster1
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind


DEFAULT_STEPS = [1, 2, 3, 4, 5]
DEFAULT_ALPHA = 0.05
ALPHA_TIME_DEFAULT = DEFAULT_ALPHA
ALPHA_CLUSTER_FORM_DEFAULT = DEFAULT_ALPHA
ALPHA_CLUSTER_DEFAULT = DEFAULT_ALPHA
TTEST_ALPHA_DEFAULT = DEFAULT_ALPHA
MIN_WINDOW_POINTS_DEFAULT = 2
WINDOW_METHOD_DEFAULT = "cluster"


def load_null_map_stack(
    path: str,
    *,
    n_null: int | None = None,
    mmap: bool = False,
) -> np.ndarray:
    """Load upstream null stack (n_null, sensors, time, steps)."""
    mode = "r" if mmap else None
    null = np.load(path, mmap_mode=mode)
    null = np.asarray(null, dtype=np.float64)
    if null.ndim != 4:
        raise ValueError(
            f"null_map must be 4D (n_null, sensors, time, steps); got {null.shape}"
        )
    if n_null is not None:
        n_use = min(int(n_null), int(null.shape[0]))
        null = null[:n_use]
    return null


def compute_forwardness(
    r_map: np.ndarray,
    *,
    step_weights: np.ndarray,
    clip_negative: bool = True,
) -> np.ndarray:
    """Weighted step index: sum(step_k * r_k) / sum(r_k)."""
    arr = np.asarray(r_map, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"r_map must be 3D (sensors, time, steps); got {arr.shape}")

    weights = np.asarray(step_weights, dtype=np.float64)
    if weights.shape != (arr.shape[2],):
        raise ValueError(
            f"step_weights length {weights.shape[0]} != n_steps {arr.shape[2]}"
        )

    if clip_negative:
        arr = np.where(arr < 0.0, 0.0, arr)

    num = np.tensordot(arr, weights, axes=([2], [0]))
    den = np.sum(arr, axis=2)

    out = np.full_like(num, np.nan, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    out[valid] = num[valid] / den[valid]
    return out


def sensor_forwardness_from_r_map(
    r_map: np.ndarray,
    *,
    steps: Sequence[int] = DEFAULT_STEPS,
    clip_negative: bool = True,
) -> np.ndarray:
    step_weights = np.asarray(steps, dtype=np.float64)
    return compute_forwardness(
        r_map,
        step_weights=step_weights,
        clip_negative=clip_negative,
    )


def finite_group_values(
    forwardness: np.ndarray,
    sensor_rows: np.ndarray,
    time_idx: int,
) -> np.ndarray:
    if sensor_rows.size == 0:
        return np.array([], dtype=np.float64)
    vals = forwardness[sensor_rows, int(time_idx)]
    return vals[np.isfinite(vals)]


def sem_from_std(std_tc: np.ndarray, n_valid: np.ndarray) -> np.ndarray:
    std_arr = np.asarray(std_tc, dtype=np.float64)
    n_arr = np.asarray(n_valid, dtype=np.float64)
    return np.divide(
        std_arr,
        np.sqrt(np.maximum(n_arr, 1.0)),
        out=np.full_like(std_arr, np.nan, dtype=np.float64),
        where=n_arr > 0,
    )


def sensor_level_timecourse_stats(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(data, dtype=np.float64)
    mean_tc = np.nanmean(arr, axis=0)
    std_tc = np.nanstd(arr, axis=0, ddof=0)
    n_valid = np.sum(np.isfinite(arr), axis=0).astype(int)
    return mean_tc, std_tc, n_valid


def mean_forwardness_timecourse(
    forwardness: np.ndarray,
    sensor_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(sensor_rows, dtype=int)
    arr = np.asarray(forwardness, dtype=np.float64)
    if rows.size == 0:
        n_times = int(arr.shape[1])
        nan = np.full(n_times, np.nan, dtype=np.float64)
        return nan.copy(), nan.copy(), np.zeros(n_times, dtype=int)

    mean_tc, std_tc, n_valid = sensor_level_timecourse_stats(arr[rows, :])
    sem_tc = sem_from_std(std_tc, n_valid)
    return mean_tc, sem_tc, n_valid


def oneway_f_stat(*groups: np.ndarray) -> float:
    clean = [np.asarray(g, dtype=np.float64).ravel() for g in groups]
    clean = [g[np.isfinite(g)] for g in clean]
    nonempty = [g for g in clean if g.size > 0]
    if len(nonempty) < 2:
        return float("nan")
    try:
        f_stat, _ = f_oneway(*nonempty)
    except Exception:
        return float("nan")
    return float(f_stat) if np.isfinite(f_stat) else float("nan")


def f_timecourse_across_conditions(
    forwardness_by_condition: dict[str, np.ndarray],
    sensor_rows_by_condition: dict[str, np.ndarray],
    *,
    conditions: Sequence[str],
) -> np.ndarray:
    conds = [str(c) for c in conditions]
    n_times = int(next(iter(forwardness_by_condition.values())).shape[1])
    out = np.full(n_times, np.nan, dtype=np.float64)
    for ti in range(n_times):
        groups = [
            finite_group_values(
                forwardness_by_condition[c],
                sensor_rows_by_condition[c],
                ti,
            )
            for c in conds
        ]
        out[ti] = oneway_f_stat(*groups)
    return out


def rank_pvalues_upper_finite(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided p: P(null >= obs); non-finite null draws excluded per timepoint."""
    obs = np.asarray(obs, dtype=np.float64).ravel()
    null = np.asarray(null, dtype=np.float64)
    if null.ndim == 1:
        null = null.reshape(-1, 1)
    n_times = int(obs.size)
    out = np.full(n_times, np.nan, dtype=np.float64)
    for ti in range(n_times):
        o = float(obs[ti])
        if not np.isfinite(o):
            continue
        draws = null[:, ti]
        finite = draws[np.isfinite(draws)]
        if finite.size == 0:
            continue
        counts = int(np.sum(finite >= o))
        out[ti] = (counts + 1.0) / float(finite.size + 1)
    return out


def contiguous_time_windows(sig: np.ndarray) -> list[tuple[int, int]]:
    sig = np.asarray(sig, dtype=bool).ravel()
    if sig.size == 0 or not bool(np.any(sig)):
        return []
    idx = np.flatnonzero(sig)
    windows: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for j in idx[1:]:
        j = int(j)
        if j == prev + 1:
            prev = j
        else:
            windows.append((start, prev))
            start = prev = j
    windows.append((start, prev))
    return windows


def cluster_mass_f(f_time: np.ndarray, start: int, end: int) -> float:
    seg = np.asarray(f_time, dtype=np.float64).ravel()[int(start) : int(end) + 1]
    finite = seg[np.isfinite(seg)]
    if finite.size == 0:
        return float("nan")
    return float(np.sum(finite))


def temporal_cluster_perm_on_f(
    obs_f: np.ndarray,
    null_f: np.ndarray,
    *,
    alpha_form: float,
    alpha_cluster: float,
) -> dict[str, object]:
    obs_f = np.asarray(obs_f, dtype=np.float64).ravel()
    null_f = np.asarray(null_f, dtype=np.float64)
    if null_f.ndim != 2:
        raise ValueError(f"null_f must be 2D (n_null, n_times), got {null_f.shape}")
    n_sims, n_times = null_f.shape
    if obs_f.size != n_times:
        raise ValueError(f"obs_f length {obs_f.size} != n_times {n_times}")

    p_obs = rank_pvalues_upper_finite(obs_f, null_f)
    form_obs = np.isfinite(p_obs) & np.isfinite(obs_f) & (p_obs <= float(alpha_form))
    obs_clusters = contiguous_time_windows(form_obs)
    obs_masses = [cluster_mass_f(obs_f, a, b) for a, b in obs_clusters]

    null_max_masses = np.zeros(n_sims, dtype=np.float64)
    for si in range(n_sims):
        f_si = null_f[si]
        p_si = rank_pvalues_upper_finite(f_si, null_f)
        form_si = np.isfinite(p_si) & np.isfinite(f_si) & (p_si <= float(alpha_form))
        clusters_si = contiguous_time_windows(form_si)
        null_max_masses[si] = (
            0.0
            if not clusters_si
            else max(cluster_mass_f(f_si, a, b) for a, b in clusters_si)
        )

    cluster_records: list[dict[str, object]] = []
    sig_cluster_windows: list[tuple[int, int]] = []
    p_cluster_by_time = np.full(n_times, np.nan, dtype=np.float64)
    sig_cluster_time = np.zeros(n_times, dtype=bool)
    cluster_id = np.full(n_times, -1, dtype=int)

    for cid, ((start, end), mass) in enumerate(zip(obs_clusters, obs_masses)):
        if not np.isfinite(mass):
            p_cluster = float("nan")
            sig = False
        else:
            counts = int(np.sum(null_max_masses >= mass))
            p_cluster = (counts + 1.0) / float(n_sims + 1)
            sig = bool(np.isfinite(p_cluster) and p_cluster <= float(alpha_cluster))
        cluster_records.append(
            {
                "time_start_idx": int(start),
                "time_end_idx": int(end),
                "window_label": format_window(start, end),
                "n_timepoints": window_length(start, end),
                "cluster_mass": float(mass) if np.isfinite(mass) else np.nan,
                "p_cluster": float(p_cluster) if np.isfinite(p_cluster) else np.nan,
                "sig_cluster": bool(sig),
                "alpha_cluster_form": float(alpha_form),
                "alpha_cluster": float(alpha_cluster),
            }
        )
        if sig:
            sig_cluster_windows.append((int(start), int(end)))
            for ti in range(int(start), int(end) + 1):
                p_cluster_by_time[ti] = float(p_cluster)
                sig_cluster_time[ti] = True
                cluster_id[ti] = int(cid)

    return {
        "cluster_records": cluster_records,
        "sig_cluster_windows": sig_cluster_windows,
        "in_cluster_form": form_obs,
        "cluster_id": cluster_id,
        "p_cluster_by_time": p_cluster_by_time,
        "sig_cluster_time": sig_cluster_time,
        "null_max_cluster_mass": null_max_masses,
    }


def infer_fwd_time_significance(
    obs_f: np.ndarray,
    null_f: np.ndarray,
    *,
    alpha_time: float,
    alpha_cluster_form: float,
    alpha_cluster: float,
) -> dict[str, object]:
    p_time = rank_pvalues_upper_finite(obs_f, null_f)
    sig_time = np.isfinite(p_time) & np.isfinite(obs_f) & (p_time <= float(alpha_time))
    windows = contiguous_time_windows(sig_time)
    cluster_res = temporal_cluster_perm_on_f(
        obs_f,
        null_f,
        alpha_form=float(alpha_cluster_form),
        alpha_cluster=float(alpha_cluster),
    )
    return {
        "p_time": p_time,
        "sig_time": sig_time,
        "sig_windows": windows,
        **cluster_res,
    }


def format_window(start: int, end: int) -> str:
    return f"{int(start)}..{int(end)}"


def window_length(start: int, end: int) -> int:
    return int(end) - int(start) + 1


def filter_windows_by_min_points(
    windows: Sequence[tuple[int, int]],
    *,
    min_points: int,
) -> list[tuple[int, int]]:
    min_pts = int(min_points)
    return [
        (int(a), int(b))
        for a, b in windows
        if window_length(int(a), int(b)) >= min_pts
    ]


def resolve_inference_windows(
    infer: dict[str, object],
    *,
    window_method: str,
    min_window_points: int,
) -> list[tuple[int, int]]:
    if str(window_method) == "cluster":
        base = list(infer.get("sig_cluster_windows", ()))
    else:
        base = list(infer.get("sig_windows", ()))
    return filter_windows_by_min_points(base, min_points=min_window_points)


def window_avg_sensor_fwd(
    fwd: np.ndarray,
    rows: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    if rows.size == 0:
        return np.array([], dtype=np.float64)
    block = fwd[rows, int(start) : int(end) + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(block, axis=1)


def welch_df(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    n1 = float(a.size)
    n2 = float(b.size)
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1.0) + (v2 / n2) ** 2 / (n2 - 1.0)
    if den <= 0.0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


def cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / float(a.size + b.size - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def benjamini_hochberg_qvalues(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not np.any(finite):
        return q
    p_finite = p[finite]
    n = int(p_finite.size)
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    q_ranked = ranked * n / (np.arange(1, n + 1, dtype=float))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_finite = np.empty(n, dtype=float)
    q_finite[order] = q_ranked
    q[finite] = q_finite
    return q


def pairwise_comparisons(conditions: Sequence[str]) -> list[tuple[str, str]]:
    conds = [str(c) for c in conditions]
    return [(a, b) for a, b in combinations(conds, 2)]


def run_pairwise_ttests_for_window(
    *,
    tag: str,
    window: tuple[int, int],
    fwd_by_cond: dict[str, np.ndarray],
    rows_by_cond: dict[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    start, end = int(window[0]), int(window[1])
    out: list[dict[str, object]] = []
    for cond_a, cond_b in comparisons:
        vals_a = window_avg_sensor_fwd(
            fwd_by_cond[str(cond_a)], rows_by_cond[str(cond_a)], start, end
        )
        vals_b = window_avg_sensor_fwd(
            fwd_by_cond[str(cond_b)], rows_by_cond[str(cond_b)], start, end
        )
        vals_a = vals_a[np.isfinite(vals_a)]
        vals_b = vals_b[np.isfinite(vals_b)]
        if vals_a.size == 0 or vals_b.size == 0:
            t_stat = p_unc = cohens_d = df = float("nan")
        else:
            t_res = ttest_ind(vals_a, vals_b, equal_var=False, nan_policy="omit")
            t_stat = float(t_res.statistic) if t_res.statistic is not None else float("nan")
            p_unc = float(t_res.pvalue) if t_res.pvalue is not None else float("nan")
            cohens_d = cohens_d_independent(vals_a, vals_b)
            df_attr = getattr(t_res, "df", None)
            df = (
                float(df_attr)
                if df_attr is not None and np.isfinite(df_attr)
                else welch_df(vals_a, vals_b)
            )
        out.append(
            {
                "tag": tag,
                "window_label": format_window(start, end),
                "time_start_idx": start,
                "time_end_idx": end,
                "comparison": f"{cond_a}_vs_{cond_b}",
                "cond_a": cond_a,
                "cond_b": cond_b,
                "n_a": int(vals_a.size),
                "n_b": int(vals_b.size),
                "mean_a": float(np.mean(vals_a)) if vals_a.size else np.nan,
                "mean_b": float(np.mean(vals_b)) if vals_b.size else np.nan,
                "std_a": float(np.std(vals_a, ddof=1)) if vals_a.size > 1 else np.nan,
                "std_b": float(np.std(vals_b, ddof=1)) if vals_b.size > 1 else np.nan,
                "t_stat": t_stat,
                "df": df,
                "p_unc": p_unc,
                "cohens_d": cohens_d,
            }
        )
    return out


def apply_fdr_to_ttest_rows(
    rows: list[dict[str, object]],
    *,
    alpha: float,
) -> None:
    if not rows:
        return
    p_unc = np.asarray([float(row.get("p_unc", np.nan)) for row in rows], dtype=float)
    q = benjamini_hochberg_qvalues(p_unc)
    for row, qval in zip(rows, q):
        row["p_fdr"] = float(qval) if np.isfinite(qval) else float("nan")
        row["sig_fdr"] = bool(np.isfinite(qval) and qval <= float(alpha))


def build_null_f_timecourse(
    null_stacks: dict[str, np.ndarray],
    rows_by_cond: dict[str, np.ndarray],
    *,
    conditions: Sequence[str],
    steps: Sequence[int],
    clip_negative: bool,
    n_null: int,
) -> np.ndarray:
    n_times = int(null_stacks[str(conditions[0])].shape[2])
    null_f = np.full((int(n_null), n_times), np.nan, dtype=np.float64)
    for si in range(int(n_null)):
        fwd_sim: dict[str, np.ndarray] = {}
        for cond in conditions:
            c = str(cond)
            fwd_sim[c] = sensor_forwardness_from_r_map(
                null_stacks[c][si],
                steps=steps,
                clip_negative=clip_negative,
            )
        null_f[si] = f_timecourse_across_conditions(
            fwd_sim,
            rows_by_cond,
            conditions=conditions,
        )
    return null_f


def analyze_forwardness_inference(
    *,
    tag: str,
    r_maps: dict[str, np.ndarray],
    null_stacks: dict[str, np.ndarray],
    rows_by_cond: dict[str, np.ndarray],
    conditions: Sequence[str],
    steps: Sequence[int],
    clip_negative: bool,
    n_null: int | None,
    alpha_time: float,
    alpha_cluster_form: float,
    alpha_cluster: float,
    window_method: str,
    min_window_points: int,
    ttest_alpha: float,
) -> dict[str, object]:
    conds = [str(c) for c in conditions]
    n_times: int | None = None
    fwd_by_cond: dict[str, np.ndarray] = {}

    for cond in conds:
        X = np.asarray(r_maps[cond], dtype=np.float64)
        if n_times is None:
            n_times = int(X.shape[1])
        elif int(X.shape[1]) != int(n_times):
            raise ValueError(f"Time mismatch for condition {cond}: {X.shape}")
        if rows_by_cond[cond].size == 0:
            raise ValueError(f"No selected sensors for condition {cond!r}")
        fwd_by_cond[cond] = sensor_forwardness_from_r_map(
            X,
            steps=steps,
            clip_negative=clip_negative,
        )

    assert n_times is not None
    obs_f = f_timecourse_across_conditions(fwd_by_cond, rows_by_cond, conditions=conds)
    n_use = int(n_null) if n_null is not None else int(null_stacks[conds[0]].shape[0])
    null_f = build_null_f_timecourse(
        null_stacks,
        rows_by_cond,
        conditions=conds,
        steps=steps,
        clip_negative=clip_negative,
        n_null=n_use,
    )

    infer = infer_fwd_time_significance(
        obs_f,
        null_f,
        alpha_time=float(alpha_time),
        alpha_cluster_form=float(alpha_cluster_form),
        alpha_cluster=float(alpha_cluster),
    )
    inference_windows = resolve_inference_windows(
        infer,
        window_method=str(window_method),
        min_window_points=int(min_window_points),
    )

    ftimecourse_rows: list[dict[str, object]] = []
    for ti in range(n_times):
        row: dict[str, object] = {
            "tag": tag,
            "time_idx": int(ti),
            "obs_F": float(obs_f[ti]) if np.isfinite(obs_f[ti]) else np.nan,
            "p_perm": float(infer["p_time"][ti]) if np.isfinite(infer["p_time"][ti]) else np.nan,
            "sig_time": bool(infer["sig_time"][ti]),
            "in_cluster_form": bool(infer["in_cluster_form"][ti]),
            "cluster_id": int(infer["cluster_id"][ti]),
            "p_cluster": (
                float(infer["p_cluster_by_time"][ti])
                if np.isfinite(infer["p_cluster_by_time"][ti])
                else np.nan
            ),
            "sig_cluster_time": bool(infer["sig_cluster_time"][ti]),
        }
        for cond in conds:
            row[f"n_sensors_{cond}"] = int(rows_by_cond[cond].size)
            row[f"mean_fwd_{cond}"] = (
                float(np.nanmean(finite_group_values(fwd_by_cond[cond], rows_by_cond[cond], ti)))
                if rows_by_cond[cond].size > 0
                else np.nan
            )
        ftimecourse_rows.append(row)

    ttest_rows: list[dict[str, object]] = []
    for window in inference_windows:
        ttest_rows.extend(
            run_pairwise_ttests_for_window(
                tag=tag,
                window=window,
                fwd_by_cond=fwd_by_cond,
                rows_by_cond=rows_by_cond,
                comparisons=pairwise_comparisons(conds),
            )
        )
    apply_fdr_to_ttest_rows(ttest_rows, alpha=float(ttest_alpha))

    cluster_window_rows = [
        {"tag": tag, **dict(rec)}
        for rec in infer["cluster_records"]
        if bool(rec.get("sig_cluster"))
    ]

    return {
        "tag": tag,
        "obs_F": obs_f,
        "null_F": null_f,
        "infer": infer,
        "inference_windows": inference_windows,
        "fwd_by_cond": fwd_by_cond,
        "ftimecourse_table": pd.DataFrame(ftimecourse_rows),
        "cluster_windows_table": pd.DataFrame(cluster_window_rows),
        "pairwise_ttest_table": pd.DataFrame(ttest_rows),
        "n_null": n_use,
    }


def save_inference_outputs(result: dict[str, object], *, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tag = str(result["tag"])
    result["ftimecourse_table"].to_csv(
        os.path.join(output_dir, f"{tag}_ftimecourse.csv"), index=False
    )
    np.save(os.path.join(output_dir, f"{tag}_null_F.npy"), result["null_F"])
    np.save(os.path.join(output_dir, f"{tag}_obs_F_timecourse.npy"), result["obs_F"])
    result["cluster_windows_table"].to_csv(
        os.path.join(output_dir, f"{tag}_sig_cluster_windows.csv"), index=False
    )
    result["pairwise_ttest_table"].to_csv(
        os.path.join(output_dir, f"{tag}_pairwise_ttest.csv"), index=False
    )

    infer = result["infer"]
    for cond, fwd in result["fwd_by_cond"].items():
        np.save(os.path.join(output_dir, f"{tag}_{cond}_forwardness.npy"), fwd)

    summary = {
        "tag": tag,
        "n_null": result["n_null"],
        "n_sig_timepoints": int(np.sum(infer["sig_time"])),
        "n_sig_cluster_timepoints": int(np.sum(infer["sig_cluster_time"])),
        "n_inference_windows": len(result["inference_windows"]),
        "n_pairwise_tests": len(result["pairwise_ttest_table"]),
        "inference_windows": [
            {"start_idx": int(a), "end_idx": int(b), "label": format_window(a, b)}
            for a, b in result["inference_windows"]
        ],
    }
    with open(os.path.join(output_dir, f"{tag}_inference_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def parse_key_value_pairs(pairs: Sequence[str], flag_name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"{flag_name} entries must look like key=path, got {item!r}")
        key, path = item.split("=", 1)
        out[str(key).strip()] = str(path).strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-map", type=str, default="")
    parser.add_argument(
        "--r-map-by-condition",
        nargs="+",
        default=[],
        metavar="COND=PATH",
    )
    parser.add_argument(
        "--null-map-by-condition",
        nargs="+",
        default=[],
        metavar="COND=PATH",
        help="Required for cluster inference; upstream null stacks per condition.",
    )
    parser.add_argument("--selected-rows", type=str, default="")
    parser.add_argument(
        "--selected-rows-by-condition",
        nargs="+",
        default=[],
        metavar="COND=PATH",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="forwardness")
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--n-null", type=int, default=None)
    parser.add_argument("--null-mmap", action="store_true")
    parser.add_argument("--alpha-time", type=float, default=ALPHA_TIME_DEFAULT)
    parser.add_argument("--alpha-cluster-form", type=float, default=ALPHA_CLUSTER_FORM_DEFAULT)
    parser.add_argument("--alpha-cluster", type=float, default=ALPHA_CLUSTER_DEFAULT)
    parser.add_argument(
        "--window-method",
        choices=("cluster", "pointwise"),
        default=WINDOW_METHOD_DEFAULT,
    )
    parser.add_argument(
        "--min-window-points",
        type=int,
        default=MIN_WINDOW_POINTS_DEFAULT,
    )
    parser.add_argument("--ttest-alpha", type=float, default=TTEST_ALPHA_DEFAULT)
    parser.add_argument("--no-clip-negative", action="store_true")
    return parser.parse_args()


def load_rows(path: str, n_sensors: int) -> np.ndarray:
    rows = np.load(path).astype(int)
    if rows.size and (rows.min() < 0 or rows.max() >= n_sensors):
        raise ValueError(
            f"selected rows out of range for n_sensors={n_sensors}: {path}"
        )
    return rows


def save_single_condition_outputs(
    *,
    tag: str,
    output_dir: str,
    r_map: np.ndarray,
    selected_rows: np.ndarray | None,
    steps: Sequence[int],
    clip_negative: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fwd = compute_forwardness(
        r_map,
        step_weights=np.asarray(steps, dtype=np.float64),
        clip_negative=clip_negative,
    )
    np.save(os.path.join(output_dir, f"{tag}_forwardness.npy"), fwd)

    summary: dict[str, object] = {
        "tag": tag,
        "r_map_shape": [int(x) for x in r_map.shape],
        "steps": [int(s) for s in steps],
        "clip_negative": clip_negative,
        "definition": "sum(step_k * r_k) / sum(r_k)",
    }

    if selected_rows is not None and selected_rows.size > 0:
        mean_tc, sem_tc, n_valid = mean_forwardness_timecourse(fwd, selected_rows)
        np.save(os.path.join(output_dir, f"{tag}_mean_forwardness.npy"), mean_tc)
        np.save(os.path.join(output_dir, f"{tag}_sem_forwardness.npy"), sem_tc)
        pd.DataFrame(
            {
                "time_idx": np.arange(mean_tc.size, dtype=int),
                "mean_forwardness": mean_tc,
                "sem_forwardness": sem_tc,
                "n_sensors": n_valid,
            }
        ).to_csv(os.path.join(output_dir, f"{tag}_mean_forwardness.csv"), index=False)
        summary["n_selected_sensors"] = int(selected_rows.size)
    else:
        summary["n_selected_sensors"] = int(r_map.shape[0])

    with open(os.path.join(output_dir, f"{tag}_forwardness_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    args = parse_args()
    clip_negative = not bool(args.no_clip_negative)
    steps = [int(s) for s in args.steps]
    output_dir = str(args.output_dir)
    tag = str(args.tag)

    if args.r_map_by_condition:
        r_maps = parse_key_value_pairs(args.r_map_by_condition, "--r-map-by-condition")
        row_maps = parse_key_value_pairs(
            args.selected_rows_by_condition, "--selected-rows-by-condition"
        )
        conditions = list(r_maps.keys())
        if set(row_maps.keys()) != set(conditions):
            raise ValueError(
                "selected-rows-by-condition must match r-map-by-condition keys"
            )

        rows_by_cond: dict[str, np.ndarray] = {}
        r_maps_loaded: dict[str, np.ndarray] = {}
        for cond in conditions:
            r_maps_loaded[cond] = np.load(r_maps[cond])
            rows_by_cond[cond] = load_rows(row_maps[cond], int(r_maps_loaded[cond].shape[0]))

        if args.null_map_by_condition:
            null_paths = parse_key_value_pairs(
                args.null_map_by_condition, "--null-map-by-condition"
            )
            if set(null_paths.keys()) != set(conditions):
                raise ValueError(
                    "null-map-by-condition must match r-map-by-condition keys"
                )
            null_stacks = {
                cond: load_null_map_stack(
                    null_paths[cond],
                    n_null=args.n_null,
                    mmap=bool(args.null_mmap),
                )
                for cond in conditions
            }
            for cond in conditions:
                if null_stacks[cond].shape[1:] != r_maps_loaded[cond].shape:
                    raise ValueError(
                        f"Shape mismatch for {cond}: null vs observed r map"
                    )

            result = analyze_forwardness_inference(
                tag=tag,
                r_maps=r_maps_loaded,
                null_stacks=null_stacks,
                rows_by_cond=rows_by_cond,
                conditions=conditions,
                steps=steps,
                clip_negative=clip_negative,
                n_null=args.n_null,
                alpha_time=float(args.alpha_time),
                alpha_cluster_form=float(args.alpha_cluster_form),
                alpha_cluster=float(args.alpha_cluster),
                window_method=str(args.window_method),
                min_window_points=int(args.min_window_points),
                ttest_alpha=float(args.ttest_alpha),
            )
            save_inference_outputs(result, output_dir=output_dir)
            print(
                f"[{tag}] inference windows={len(result['inference_windows'])}, "
                f"pairwise_tests={len(result['pairwise_ttest_table'])} -> {output_dir}"
            )
            return

        # Multi-condition without null: per-condition forwardness + observed F only.
        fwd_by_cond: dict[str, np.ndarray] = {}
        for cond in conditions:
            fwd = sensor_forwardness_from_r_map(
                r_maps_loaded[cond],
                steps=steps,
                clip_negative=clip_negative,
            )
            fwd_by_cond[cond] = fwd
            save_single_condition_outputs(
                tag=f"{tag}_{cond}",
                output_dir=output_dir,
                r_map=r_maps_loaded[cond],
                selected_rows=rows_by_cond[cond],
                steps=steps,
                clip_negative=clip_negative,
            )

        f_tc = f_timecourse_across_conditions(
            fwd_by_cond, rows_by_cond, conditions=conditions
        )
        np.save(os.path.join(output_dir, f"{tag}_obs_F_timecourse.npy"), f_tc)
        pd.DataFrame(
            {
                "time_idx": np.arange(f_tc.size, dtype=int),
                "obs_F": f_tc,
                **{f"n_sensors_{c}": int(rows_by_cond[c].size) for c in conditions},
            }
        ).to_csv(os.path.join(output_dir, f"{tag}_obs_F_timecourse.csv"), index=False)
        print(f"[{tag}] observed F only (pass --null-map-by-condition for inference)")
        return

    if not args.r_map:
        raise SystemExit(
            "Provide --r-map (single condition) or --r-map-by-condition (multi condition)."
        )

    r_map = np.load(args.r_map)
    selected_rows = load_rows(args.selected_rows, int(r_map.shape[0])) if args.selected_rows else None
    save_single_condition_outputs(
        tag=tag,
        output_dir=output_dir,
        r_map=r_map,
        selected_rows=selected_rows,
        steps=steps,
        clip_negative=clip_negative,
    )
    print(f"[{tag}] saved forwardness to {output_dir}")


if __name__ == "__main__":
    main()
