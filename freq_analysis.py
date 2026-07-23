#!/usr/bin/env python3
"""
Word-frequency analysis for autoregressive (allm) token order.

Subcommands
-----------
extract  Look up log word frequencies (1-gram corpus) for allm token sequences
plot     Mean log-frequency by 5 position groups + bar plots

Example:
  python freq_analysis.py extract \\
    --pos-table model_pos.csv \\
    --freq-corpus 1gram_en_lower.csv \\
    --output model_freq.csv

  python freq_analysis.py plot \\
    --input model_freq.csv \\
    --output-dir ./plots/freq
"""

from __future__ import annotations

import argparse
import ast
import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_frequency_dict(freq_path: str) -> dict[str, float]:
    freq_df = pd.read_csv(
        freq_path,
        header=None,
        names=["word", "freq_count"],
        sep=" ",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    freq_df = freq_df.dropna()
    return dict(zip(freq_df["word"].astype(str), freq_df["freq_count"].astype(float)))


def word_log_frequency(word: str, freq_dict: dict[str, float], default_freq: float = 1.0) -> float:
    freq = freq_dict.get(str(word).lower(), default_freq)
    try:
        freq = float(freq)
    except (TypeError, ValueError):
        freq = default_freq
    if freq <= 0:
        freq = default_freq
    return float(np.log(freq))


def sentence_log_frequencies(words: Sequence[str], freq_dict: dict[str, float]) -> list[float]:
    return [word_log_frequency(w, freq_dict) for w in words]


def divide_into_five_group_means(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0] * 5
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    base = n // 5
    rem = n % 5
    avgs: list[float] = []
    start = 0
    for i in range(5):
        size = base + (1 if i < rem else 0)
        end = start + size
        chunk = arr[start:end]
        avgs.append(float(np.mean(chunk)) if len(chunk) else 0.0)
        start = end
    return avgs


def extract_frequency_table(
    pos_table: pd.DataFrame,
    freq_dict: dict[str, float],
) -> pd.DataFrame:
    if "spacy_tokenization" not in pos_table.columns:
        raise ValueError("POS table missing column: spacy_tokenization")

    rows: list[dict] = []
    for _, row in pos_table.iterrows():
        try:
            spacy_tokens = ast.literal_eval(row["spacy_tokenization"])
        except Exception:
            continue

        freq_allm = sentence_log_frequencies(spacy_tokens, freq_dict)
        rows.append(
            {
                "condition": row.get("condition", ""),
                "subject": row.get("subject", row.get("subj", "")),
                "spacy_tokenization": str(spacy_tokens),
                "freq_allm": str([float(x) for x in freq_allm]),
            }
        )

    if not rows:
        raise ValueError("No frequency rows extracted.")
    return pd.DataFrame(rows)


def freq_table_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, row in df.iterrows():
        try:
            freq_allm = ast.literal_eval(row["freq_allm"])
        except Exception:
            continue

        group_avgs = divide_into_five_group_means(freq_allm)
        for gi in range(5):
            rows.append(
                {
                    "condition": row.get("condition", ""),
                    "subject": row.get("subject", row.get("subj", "")),
                    "sentence_id": idx,
                    "group": f"group_{gi + 1}",
                    "allm_freq": group_avgs[gi],
                }
            )
    return pd.DataFrame(rows)


def remove_outliers_3std(data: np.ndarray) -> np.ndarray:
    if len(data) == 0:
        return data
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data
    return data[np.abs(data - mean) <= 3 * std]


def compute_plot_stats(processed_df: pd.DataFrame) -> pd.DataFrame:
    stats: list[dict] = []
    for group in [f"group_{i}" for i in range(1, 6)]:
        raw = processed_df[processed_df["group"] == group]["allm_freq"].values
        filtered = remove_outliers_3std(raw)
        stats.append(
            {
                "Model": "allm",
                "Group": group,
                "Mean_Frequency": float(np.mean(filtered)) if len(filtered) else 0.0,
                "STD": float(np.std(filtered)) if len(filtered) else 0.0,
                "N_original": int(len(raw)),
                "N_filtered": int(len(filtered)),
            }
        )
    stats_df = pd.DataFrame(stats)
    stats_df["SEM"] = stats_df["STD"] / np.sqrt(np.maximum(stats_df["N_filtered"], 1))
    return stats_df


def create_frequency_plot(stats_df: pd.DataFrame, out_path: str) -> None:
    groups = [1, 2, 3, 4, 5]
    group_spacing = 0.72
    x_positions = 0.8 + np.arange(5) * group_spacing
    width = 0.4

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.bar(
        x_positions,
        stats_df["Mean_Frequency"].values,
        width=width,
        label="allm",
        color="#800026",
        yerr=stats_df["STD"].values,
        capsize=0,
        error_kw={"linewidth": 0.8, "ecolor": "grey"},
    )

    ax.set_title("Log word frequency (allm)", fontsize=13, pad=8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(i) for i in groups], fontsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout(pad=0.15)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_frequency_summaries(df: pd.DataFrame, output_dir: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    os.makedirs(output_dir, exist_ok=True)

    processed = freq_table_to_long(df)
    subsets = [
        ("", processed),
        ("_prod", processed[processed["condition"] == "prod"].copy()),
        ("_comp", processed[processed["condition"] == "comp"].copy()),
    ]

    for suffix, sub_df in subsets:
        if len(sub_df) == 0 and suffix:
            continue
        processed_path = os.path.join(output_dir, f"model_freq_processed{suffix or ''}.csv")
        sub_df.to_csv(processed_path, index=False)

        stats_df = compute_plot_stats(sub_df)
        stats_path = os.path.join(output_dir, f"model_freq_plot_stats{suffix or ''}.csv")
        stats_df.to_csv(stats_path, index=False)

        plot_path = os.path.join(output_dir, f"model_freq_plot_allm{suffix or ''}.png")
        create_frequency_plot(stats_df, plot_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Word-frequency analysis (allm).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Build frequency table from POS table.")
    p_extract.add_argument("--pos-table", required=True)
    p_extract.add_argument("--freq-corpus", required=True, help="1-gram word count file.")
    p_extract.add_argument("--output", required=True)

    p_plot = sub.add_parser("plot", help="Group means and bar plots.")
    p_plot.add_argument("--input", required=True)
    p_plot.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.command == "extract":
        pos_df = pd.read_csv(args.pos_table)
        freq_dict = load_frequency_dict(args.freq_corpus)
        out = extract_frequency_table(pos_df, freq_dict)
        out.to_csv(args.output, index=False)
        print(f"Wrote {len(out)} rows to {args.output}")
        return

    if args.command == "plot":
        df = pd.read_csv(args.input)
        plot_frequency_summaries(df, args.output_dir)
        print(f"Wrote frequency plots and stats to {args.output_dir}")
        return


if __name__ == "__main__":
    main()
