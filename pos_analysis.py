#!/usr/bin/env python3
"""
POS composition analysis for autoregressive (allm) token order.

Subcommands
-----------
extract  Tag spaCy tokens and write a POS table
analyze  Split sequences into 5 position groups; compute category fractions
plot     Stacked-bar summaries by sentence position (allm / original order)

POS categories: NOUN, VERB, ADJ/ADV, FUNC (mapped from spaCy coarse tags).

Example:
  python pos_analysis.py extract \\
    --input-glob "data/sentence_filtered_*.csv" \\
    --output model_pos.csv

  python pos_analysis.py analyze --input model_pos.csv --output model_pos.csv
  python pos_analysis.py plot --input model_pos.csv --output-dir ./plots/pos
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
from collections import defaultdict
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import spacy
except ImportError as exc:  # pragma: no cover
    raise SystemExit("spaCy is required for pos_analysis extract.") from exc


POS_CATEGORIES = ("NOUN", "VERB", "ADJ/ADV", "FUNC")
STACK_ORDER = ("FUNC", "ADJ/ADV", "VERB", "NOUN")
CAT_COLORS = {
    "NOUN": "#7b3294",
    "VERB": "#c2a5cf",
    "ADJ/ADV": "#a6dba0",
    "FUNC": "#008837",
}
CAT_COL_KEY = {
    "NOUN": "noun",
    "VERB": "verb",
    "ADJ/ADV": "adj_adv",
    "FUNC": "func",
}

SPACY_POS_MAP = {
    "NOUN": "NOUN",
    "PROPN": "NOUN",
    "PRON": "NOUN",
    "VERB": "VERB",
    "AUX": "VERB",
    "ADJ": "ADJ/ADV",
    "ADV": "ADJ/ADV",
    "DET": "FUNC",
    "ADP": "FUNC",
    "CONJ": "FUNC",
    "CCONJ": "FUNC",
    "SCONJ": "FUNC",
    "PART": "FUNC",
    "INTJ": "FUNC",
    "NUM": "FUNC",
    "SYM": "FUNC",
    "PUNCT": "FUNC",
    "X": "FUNC",
    "SPACE": "FUNC",
}


def load_spacy(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError:
        return spacy.load("en_core_web_sm")


def assign_subject_ids(csv_files: Sequence[str]) -> dict[str, str]:
    """Map each input file to anonymized subject IDs: sub-01, sub-02, ..."""
    return {
        path: f"sub-{index:02d}"
        for index, path in enumerate(sorted(csv_files), start=1)
    }


def divide_into_five_groups(items: Sequence) -> dict[int, list]:
    n = len(items)
    base = n // 5
    rem = n % 5
    groups = {i: [] for i in range(5)}
    start = 0
    for i in range(5):
        size = base + (1 if i < rem else 0)
        end = start + size
        if start < n:
            groups[i] = list(items[start:end])
        start = end
    return groups


def divide_tokens_into_five_lists(tokens: Sequence) -> list[list]:
    groups = divide_into_five_groups(tokens)
    return [groups[i] for i in range(5)]


def pos_tags_for_tokens(spacy_tokens: Sequence[str], nlp) -> list[str]:
    if not spacy_tokens:
        return []
    doc = nlp(" ".join(str(t) for t in spacy_tokens))
    return [SPACY_POS_MAP.get(token.pos_, "FUNC") for token in doc]


def infer_condition(path: str) -> str | None:
    name = os.path.basename(path).lower()
    if "comp" in name:
        return "comp"
    if "prod" in name:
        return "prod"
    return None


def calculate_pos_percentages(
    sequence_tags: Sequence[str],
    sentence_tags: Sequence[str],
) -> dict[str, list[float]]:
    categories = list(POS_CATEGORIES)
    if len(sequence_tags) == 0:
        return {cat: [0.0] * 5 for cat in categories}

    groups = divide_into_five_groups(sequence_tags)
    totals = defaultdict(int)
    for tag in sentence_tags:
        totals[tag] += 1

    results = {cat: [] for cat in categories}
    for gi in range(5):
        group_counts = defaultdict(int)
        for tag in groups[gi]:
            group_counts[tag] += 1
        for cat in categories:
            total_cat = totals[cat]
            results[cat].append(
                (group_counts[cat] / total_cat) if total_cat > 0 else 0.0
            )
    return results


def extract_pos_table(
    csv_files: Sequence[str],
    *,
    spacy_model: str,
) -> pd.DataFrame:
    nlp = load_spacy(spacy_model)
    subject_by_file = assign_subject_ids(csv_files)
    rows: list[dict] = []

    for csv_path in csv_files:
        condition = infer_condition(csv_path)
        if condition is None:
            print(f"Skipping {csv_path}: cannot infer comp/prod from filename.")
            continue

        df = pd.read_csv(csv_path)
        subject = subject_by_file[csv_path]
        spacy_col = f"{condition}_spacy"
        if spacy_col not in df.columns:
            print(f"Skipping {csv_path}: missing column {spacy_col}")
            continue

        for _, row in df.iterrows():
            spacy_raw = row[spacy_col]
            if pd.isna(spacy_raw) or not str(spacy_raw).strip():
                continue

            spacy_tokens = ast.literal_eval(str(spacy_raw))
            pos_tags = pos_tags_for_tokens(spacy_tokens, nlp)
            rows.append(
                {
                    "condition": condition,
                    "subject": subject,
                    "spacy_tokenization": str(spacy_tokens),
                    "pos": str(pos_tags),
                }
            )

    if not rows:
        raise ValueError("No rows extracted; check input files and column names.")

    return pd.DataFrame(rows).sort_values(["condition", "subject"]).reset_index(drop=True)


def analyze_pos_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"spacy_tokenization", "pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input table missing columns: {sorted(missing)}")

    new_cols: dict[str, list] = {
        "word_groups": [],
        **{f"allm_{CAT_COL_KEY[c]}": [] for c in POS_CATEGORIES},
    }

    for _, row in df.iterrows():
        spacy_tokens = ast.literal_eval(row["spacy_tokenization"])
        pos_tags = ast.literal_eval(row["pos"])
        allm_pct = calculate_pos_percentages(pos_tags, pos_tags)

        new_cols["word_groups"].append(divide_tokens_into_five_lists(spacy_tokens))
        for cat in POS_CATEGORIES:
            new_cols[f"allm_{CAT_COL_KEY[cat]}"].append(allm_pct[cat])

    out = df.copy()
    if "word_groups" in out.columns:
        out = out.drop(columns=["word_groups"])
    insert_at = out.columns.get_loc("spacy_tokenization") + 1
    out.insert(insert_at, "word_groups", new_cols.pop("word_groups"))
    for name, values in new_cols.items():
        out[name] = values
    return out


def pos_table_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, row in df.iterrows():
        condition = row["condition"]
        subject = row.get("subject", row.get("subj", ""))
        try:
            allm = {
                cat: ast.literal_eval(row[f"allm_{CAT_COL_KEY[cat]}"])
                for cat in POS_CATEGORIES
            }
        except Exception:
            continue

        for gi in range(5):
            rec = {
                "condition": condition,
                "subject": subject,
                "sentence_id": idx,
                "group": f"group_{gi + 1}",
                "model": "allm",
            }
            for cat in POS_CATEGORIES:
                rec[f"{cat}_percentage"] = allm[cat][gi]
            rows.append(rec)
    return pd.DataFrame(rows)


def compute_stacked_means(sub_df: pd.DataFrame) -> dict[str, np.ndarray]:
    cat_means: dict[str, np.ndarray] = {}
    for cat in POS_CATEGORIES:
        vals = []
        for group_num in range(1, 6):
            arr = sub_df[sub_df["group"] == f"group_{group_num}"][f"{cat}_percentage"].values
            vals.append(float(np.mean(arr)) if len(arr) else 0.0)
        cat_means[cat] = np.array(vals)

    step_totals = np.zeros(5)
    for cat in POS_CATEGORIES:
        step_totals += cat_means[cat]
    for cat in POS_CATEGORIES:
        cat_means[cat] = np.where(step_totals > 0, cat_means[cat] / step_totals, 0.0)
    return cat_means


def create_stacked_plot(cat_means: dict[str, np.ndarray], title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(1.9, 1.47))
    group_spacing = 0.72
    x = 0.8 + np.arange(5) * group_spacing
    w = 0.4
    bottoms = np.zeros(5)
    for cat in STACK_ORDER:
        vals = cat_means[cat]
        ax.bar(x, vals, w, bottom=bottoms, color=CAT_COLORS[cat], edgecolor="white", linewidth=0.3)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=14)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50", "100"], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title(title, fontsize=13, pad=8)
    plt.tight_layout(pad=0.15)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_pos_summaries(df: pd.DataFrame, output_dir: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    os.makedirs(output_dir, exist_ok=True)

    processed = pos_table_to_long(df)
    processed.to_csv(os.path.join(output_dir, "model_pos_processed.csv"), index=False)

    means_rows: list[dict] = []
    subsets = [
        ("all", processed),
        ("prod", processed[processed["condition"] == "prod"].copy()),
        ("comp", processed[processed["condition"] == "comp"].copy()),
    ]

    for cond_label, sub_df in subsets:
        if len(sub_df) == 0 and cond_label != "all":
            continue
        cat_means = compute_stacked_means(sub_df)
        suffix = "" if cond_label == "all" else f"_{cond_label}"

        for cat in POS_CATEGORIES:
            rec = {"Condition": cond_label, "Model": "allm", "Category": cat}
            for gi in range(5):
                rec[f"Step{gi + 1}"] = float(cat_means[cat][gi])
            means_rows.append(rec)

        out_png = os.path.join(output_dir, f"model_pos_stacked_allm{suffix}.png")
        create_stacked_plot(cat_means, "allm", out_png)

    pd.DataFrame(means_rows).to_csv(
        os.path.join(output_dir, "model_pos_stacked_means.csv"),
        index=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="POS composition analysis (allm).")
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
        help="spaCy model for POS tagging during extract.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Build pos table from sentence CSVs.")
    p_extract.add_argument("--input-glob", required=True, help="Glob for input CSVs.")
    p_extract.add_argument("--output", required=True, help="Output pos table CSV.")

    p_analyze = sub.add_parser("analyze", help="Add 5-group POS fraction columns.")
    p_analyze.add_argument("--input", required=True)
    p_analyze.add_argument("--output", required=True)

    p_plot = sub.add_parser("plot", help="Stacked-bar plots for allm token order.")
    p_plot.add_argument("--input", required=True)
    p_plot.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.command == "extract":
        files = sorted(glob.glob(args.input_glob))
        if not files:
            raise SystemExit(f"No files matched: {args.input_glob}")
        out = extract_pos_table(files, spacy_model=args.spacy_model)
        out.to_csv(args.output, index=False)
        print(f"Wrote {len(out)} rows to {args.output}")
        return

    if args.command == "analyze":
        df = pd.read_csv(args.input)
        out = analyze_pos_table(df)
        out.to_csv(args.output, index=False)
        print(f"Wrote analyzed table ({len(out)} rows) to {args.output}")
        return

    if args.command == "plot":
        df = pd.read_csv(args.input)
        plot_pos_summaries(df, args.output_dir)
        print(f"Wrote POS plots and stats to {args.output_dir}")
        return


if __name__ == "__main__":
    main()
