#!/usr/bin/env python3
"""
spaCy tokenization utilities for sentence-level text columns.

Tokenize text with spaCy, lower-case tokens, and expand common English
contractions (e.g. "don't" → "do not"). Optionally map original word lists
to 1-based spaCy token indices.

This script covers the spaCy preprocessing step only. Downstream POS/frequency
analysis for autoregressive (allm) token order is handled by separate utilities.

Example (add tokens column to a CSV):
  python spacy_tokenize.py \\
    --input sentences.csv \\
    --text-column comprehension_str \\
    --output-column comp_spacy \\
    --output sentences_spacy.csv

Example (tokenize one string):
  python spacy_tokenize.py --text "Let's go home."
"""

from __future__ import annotations

import argparse
import ast
from typing import Iterable, Sequence

import pandas as pd

try:
    import spacy
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "spaCy is required. Install with: pip install spacy && "
        "python -m spacy download en_core_web_lg"
    ) from exc


DEFAULT_SPACY_MODEL = "en_core_web_lg"
FALLBACK_SPACY_MODEL = "en_core_web_sm"

CONTRACTIONS: tuple[tuple[str, str], ...] = (
    ("n't", " not"),
    ("'re", " are"),
    ("'ve", " have"),
    ("'ll", " will"),
    ("'d", " would"),
    ("'m", " am"),
    ("'s", " is"),
    ("let's", "let us"),
    ("won't", "will not"),
    ("can't", "cannot"),
    ("don't", "do not"),
    ("doesn't", "does not"),
    ("didn't", "did not"),
    ("shouldn't", "should not"),
    ("wouldn't", "would not"),
    ("couldn't", "could not"),
    ("mustn't", "must not"),
    ("haven't", "have not"),
    ("hasn't", "has not"),
    ("hadn't", "had not"),
    ("isn't", "is not"),
    ("aren't", "are not"),
    ("wasn't", "was not"),
    ("weren't", "were not"),
    ("i'm", "i am"),
    ("you're", "you are"),
    ("he's", "he is"),
    ("she's", "she is"),
    ("it's", "it is"),
    ("we're", "we are"),
    ("they're", "they are"),
    ("that's", "that is"),
    ("what's", "what is"),
    ("where's", "where is"),
    ("gonna", "going to"),
    ("gotta", "got to"),
    ("wanna", "want to"),
)


def expand_contractions(text: str) -> str:
    """Expand common English contractions in lower-case text."""
    expanded = str(text).lower()
    for contraction, expansion in CONTRACTIONS:
        expanded = expanded.replace(contraction, expansion)
    return expanded


def tokenize_with_spacy(text: str, nlp) -> list[str]:
    """Tokenize with spaCy and expand contractions token-wise."""
    doc = nlp(str(text))
    tokens: list[str] = []
    for token in doc:
        expanded = expand_contractions(token.text.lower())
        if " " in expanded:
            tokens.extend(expanded.split())
        else:
            tokens.append(expanded)
    return tokens


def map_words_to_spacy_ids(
    original_words: Sequence[str],
    spacy_tokens: Sequence[str],
) -> list[list[int]]:
    """Map each original word to one or more 1-based spaCy token indices."""
    if not original_words or not spacy_tokens:
        return []

    mapping: list[list[int]] = []
    spacy_idx = 0

    for word in original_words:
        word_clean = str(word).strip().lower()
        spacy_token_ids: list[int] = []

        if "'" in word_clean:
            expanded_parts = expand_contractions(word_clean).strip().split()
            for part in expanded_parts:
                for search_idx in range(spacy_idx, min(len(spacy_tokens), spacy_idx + 5)):
                    if part == str(spacy_tokens[search_idx]).strip().lower():
                        spacy_token_ids.append(search_idx + 1)
                        spacy_idx = search_idx + 1
                        break
        elif " " in word_clean:
            for part in word_clean.split():
                for search_idx in range(spacy_idx, min(len(spacy_tokens), spacy_idx + 3)):
                    if part == str(spacy_tokens[search_idx]).strip().lower():
                        spacy_token_ids.append(search_idx + 1)
                        spacy_idx = search_idx + 1
                        break
        else:
            if spacy_idx < len(spacy_tokens):
                spacy_token = str(spacy_tokens[spacy_idx]).strip().lower()
                if word_clean == spacy_token:
                    spacy_token_ids.append(spacy_idx + 1)
                    spacy_idx += 1
                else:
                    combined = ""
                    temp_ids: list[int] = []
                    temp_idx = spacy_idx
                    while temp_idx < len(spacy_tokens) and len(combined) < len(word_clean) * 2:
                        token = str(spacy_tokens[temp_idx]).strip().lower()
                        combined += token
                        temp_ids.append(temp_idx + 1)
                        temp_idx += 1
                        if combined == word_clean:
                            spacy_token_ids = temp_ids
                            spacy_idx = temp_idx
                            break
                    if not spacy_token_ids and spacy_idx < len(spacy_tokens):
                        spacy_token_ids.append(spacy_idx + 1)
                        spacy_idx += 1

        if spacy_token_ids:
            mapping.append(spacy_token_ids)

    return mapping


def load_spacy(model_name: str = DEFAULT_SPACY_MODEL):
    """Load spaCy model, falling back to en_core_web_sm."""
    try:
        return spacy.load(model_name)
    except OSError:
        if model_name == FALLBACK_SPACY_MODEL:
            raise
        return spacy.load(FALLBACK_SPACY_MODEL)


def add_spacy_column(
    df: pd.DataFrame,
    *,
    text_column: str,
    output_column: str,
    nlp,
) -> pd.DataFrame:
    """Return a copy of df with a stringified token list column added."""
    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    out = df.copy()
    tokens_col: list[str] = []
    for _, row in out.iterrows():
        text = row[text_column]
        if pd.isna(text) or not str(text).strip():
            tokens_col.append(str([]))
            continue
        tokens_col.append(str(tokenize_with_spacy(str(text), nlp)))
    out[output_column] = tokens_col
    return out


def add_word_to_spacy_mapping_column(
    df: pd.DataFrame,
    *,
    words_column: str,
    spacy_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Return a copy of df with word→spaCy-id mapping column added."""
    for col in (words_column, spacy_column):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    out = df.copy()
    mappings: list[str] = []
    for _, row in out.iterrows():
        try:
            words = ast.literal_eval(row[words_column])
            spacy_tokens = ast.literal_eval(row[spacy_column])
            mappings.append(str(map_words_to_spacy_ids(words, spacy_tokens)))
        except Exception:
            mappings.append(str([]))
    out[output_column] = mappings
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="spaCy tokenization with contraction expansion.",
    )
    parser.add_argument(
        "--spacy-model",
        default=DEFAULT_SPACY_MODEL,
        help=f"spaCy model name (default: {DEFAULT_SPACY_MODEL})",
    )
    parser.add_argument(
        "--input",
        help="Input CSV path. If omitted, use --text for a single string.",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path (required with --input).",
    )
    parser.add_argument(
        "--text-column",
        help="Column containing raw sentence text.",
    )
    parser.add_argument(
        "--output-column",
        default="spacy_tokens",
        help="Column name for stringified token lists (default: spacy_tokens).",
    )
    parser.add_argument(
        "--words-column",
        help="Optional column with stringified original word lists.",
    )
    parser.add_argument(
        "--spacy-column",
        help="Existing spaCy token column (required with --words-column).",
    )
    parser.add_argument(
        "--mapping-column",
        default="spacy_id_group",
        help="Output column for word→spaCy mapping (default: spacy_id_group).",
    )
    parser.add_argument(
        "--text",
        help="Tokenize a single string and print the token list.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.text:
        nlp = load_spacy(args.spacy_model)
        print(tokenize_with_spacy(args.text, nlp))
        return

    if not args.input:
        raise SystemExit("Provide --input CSV or --text.")
    if not args.output:
        raise SystemExit("--output is required with --input.")
    if not args.text_column:
        raise SystemExit("--text-column is required with --input.")

    nlp = load_spacy(args.spacy_model)
    df = pd.read_csv(args.input)
    df = add_spacy_column(
        df,
        text_column=args.text_column,
        output_column=args.output_column,
        nlp=nlp,
    )

    if args.words_column:
        if not args.spacy_column:
            raise SystemExit("--spacy-column is required with --words-column.")
        df = add_word_to_spacy_mapping_column(
            df,
            words_column=args.words_column,
            spacy_column=args.spacy_column,
            output_column=args.mapping_column,
        )

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
