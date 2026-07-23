#!/usr/bin/env python3
"""
Extract progressive 5-step sentence embeddings from a causal LLM.

For each sentence, token prefixes at 20/40/60/80/100% of tokens are fed to the
model after a conversational context prompt. Hidden states at a chosen layer are
mean-pooled over response tokens to produce one vector per step.

Output array shape: (n_sentences, 5, hidden_dim)

Example (LLaMA-style CSV columns):
  python extract_progressive_llm_embeddings.py \\
    --model-path ./models/Llama-3.1-8B \\
    --input sentences_comp.csv \\
    --output ./embs/sub-01_comp.npy \\
    --mode comp \\
    --layer 15

Example (Qwen):
  python extract_progressive_llm_embeddings.py \\
    --model-path ./models/Qwen2.5-7B \\
    --input sentences_prod.csv \\
    --output ./embs/sub-01_prod.npy \\
    --mode prod \\
    --layer 14
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import warnings
from typing import Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

Mode = Literal["comp", "prod"]
N_PROGRESSIVE_STEPS = 5

MODE_COLUMNS: dict[Mode, tuple[str, str]] = {
    "comp": ("comprehension_context", "comprehension"),
    "prod": ("production_context", "production"),
}


def build_context_prompt(mode: Mode, context_sentence: str) -> str:
    if mode == "comp":
        return (
            f"In a casual conversation, you said '{context_sentence}' "
            "and you heard the response:"
        )
    return (
        f"In a casual conversation, you heard '{context_sentence}' "
        "and you responded:"
    )


def load_model(model_path: str, *, dtype: torch.dtype):
    print(f"Loading model from {model_path} ...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    print(f"Model ready (device={device}, hidden_size={hidden_size})")
    return model, tokenizer, device, hidden_size


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def progressive_token_steps(
    sentence_text: str,
    tokenizer,
    device: torch.device,
) -> list[torch.Tensor]:
    token_ids = tokenizer.encode(sentence_text, add_special_tokens=False)
    if not token_ids:
        return []

    sentence_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    total_tokens = len(token_ids)
    steps: list[torch.Tensor] = []

    for step in range(1, N_PROGRESSIVE_STEPS + 1):
        tokens_to_include = max(1, int(total_tokens * step / N_PROGRESSIVE_STEPS))
        if step == N_PROGRESSIVE_STEPS:
            tokens_to_include = total_tokens
        steps.append(sentence_tensor[:tokens_to_include])

    return steps


def mean_pooled_layer_embedding(
    model,
    tokenizer,
    device: torch.device,
    *,
    context_prompt: str,
    token_sequence: torch.Tensor,
    layer: int,
) -> np.ndarray:
    prompt_inputs = tokenizer(context_prompt, return_tensors="pt", return_attention_mask=True)
    prompt_inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in prompt_inputs.items()
    }

    full_input = torch.cat(
        [prompt_inputs["input_ids"], token_sequence.unsqueeze(0)],
        dim=1,
    )
    response_attention = torch.ones((1, len(token_sequence)), dtype=torch.bool, device=device)
    if "attention_mask" in prompt_inputs:
        prompt_attention = prompt_inputs["attention_mask"].bool()
        full_attention = torch.cat([prompt_attention, response_attention], dim=1)
    else:
        full_attention = torch.ones_like(full_input, dtype=torch.bool)

    with torch.no_grad():
        outputs = model(
            input_ids=full_input,
            attention_mask=full_attention,
            output_hidden_states=True,
            return_dict=True,
        )

    response_start = prompt_inputs["input_ids"].shape[1]
    response_end = response_start + len(token_sequence)

    if len(outputs.hidden_states) <= layer:
        raise ValueError(
            f"Requested layer {layer}, but model has {len(outputs.hidden_states)} hidden states."
        )

    layer_hidden = outputs.hidden_states[layer]
    response_hidden = layer_hidden[0, response_start:response_end, :]
    sentence_embedding = torch.mean(response_hidden, dim=0)
    return sentence_embedding.float().cpu().numpy()


def extract_sentence_embeddings(
    model,
    tokenizer,
    device: torch.device,
    *,
    target_sentence: str,
    context_sentence: str,
    mode: Mode,
    layer: int,
    hidden_size: int,
) -> np.ndarray:
    steps = progressive_token_steps(target_sentence, tokenizer, device)
    if not steps:
        return np.zeros((N_PROGRESSIVE_STEPS, hidden_size), dtype=np.float32)

    context_prompt = build_context_prompt(mode, context_sentence)
    step_embeddings: list[np.ndarray] = []

    for step_tokens in steps:
        embedding = mean_pooled_layer_embedding(
            model,
            tokenizer,
            device,
            context_prompt=context_prompt,
            token_sequence=step_tokens,
            layer=layer,
        )
        step_embeddings.append(embedding)
        clear_gpu_memory()

    return np.stack(step_embeddings, axis=0).astype(np.float32)


def row_is_valid(row: dict, valid_column: str | None) -> bool:
    if not valid_column or valid_column not in row:
        return True
    return str(row[valid_column]).strip().lower() == "true"


def read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_from_csv(
    model,
    tokenizer,
    device: torch.device,
    *,
    input_csv: str,
    mode: Mode,
    layer: int,
    hidden_size: int,
    max_sentences: int | None = None,
    valid_column: str | None = "valid",
    context_column: str | None = None,
    target_column: str | None = None,
) -> tuple[np.ndarray, dict]:
    default_context, default_target = MODE_COLUMNS[mode]
    context_col = context_column or default_context
    target_col = target_column or default_target

    rows = read_csv_rows(input_csv)
    if max_sentences is not None:
        rows = rows[: max(0, int(max_sentences))]

    embeddings: list[np.ndarray] = []
    kept_indices: list[int] = []
    skipped = 0

    for row_idx, row in enumerate(rows):
        if not row_is_valid(row, valid_column):
            skipped += 1
            continue

        context_sentence = str(row[context_col])
        target_sentence = str(row[target_col])
        sentence_emb = extract_sentence_embeddings(
            model,
            tokenizer,
            device,
            target_sentence=target_sentence,
            context_sentence=context_sentence,
            mode=mode,
            layer=layer,
            hidden_size=hidden_size,
        )
        embeddings.append(sentence_emb)
        kept_indices.append(row_idx)

        if len(embeddings) % 25 == 0:
            print(f"Processed {len(embeddings)} sentences ({skipped} skipped)")

    if not embeddings:
        raise ValueError("No valid sentences processed; check CSV columns and valid filter.")

    array = np.stack(embeddings, axis=0)
    summary = {
        "input_csv": os.path.abspath(input_csv),
        "mode": mode,
        "layer": layer,
        "n_rows_read": len(rows),
        "n_sentences_kept": len(embeddings),
        "n_sentences_skipped": skipped,
        "kept_row_indices": kept_indices,
        "output_shape": list(array.shape),
        "context_column": context_col,
        "target_column": target_col,
        "valid_column": valid_column,
    }
    return array, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract progressive 5-step LLM sentence embeddings.",
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace model directory.")
    parser.add_argument("--input", required=True, help="Input sentence CSV.")
    parser.add_argument("--output", required=True, help="Output .npy path.")
    parser.add_argument(
        "--mode",
        choices=["comp", "prod"],
        required=True,
        help="comp: heard response; prod: produced response.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="0-indexed transformer layer for hidden-state extraction.",
    )
    parser.add_argument(
        "--context-column",
        help="Override context column (default depends on --mode).",
    )
    parser.add_argument(
        "--target-column",
        help="Override target sentence column (default depends on --mode).",
    )
    parser.add_argument(
        "--valid-column",
        default="valid",
        help="Boolean validity column; set to '' to disable filtering.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        help="Process only the first N CSV rows (testing).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model load dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path for run metadata JSON (defaults next to --output).",
    )
    return parser


def parse_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = build_parser().parse_args()
    valid_column = args.valid_column.strip() or None

    model, tokenizer, device, hidden_size = load_model(
        args.model_path,
        dtype=parse_dtype(args.dtype),
    )

    embeddings, summary = extract_from_csv(
        model,
        tokenizer,
        device,
        input_csv=args.input,
        mode=args.mode,
        layer=int(args.layer),
        hidden_size=hidden_size,
        max_sentences=args.max_sentences,
        valid_column=valid_column,
        context_column=args.context_column,
        target_column=args.target_column,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.save(args.output, embeddings)

    summary_path = args.summary_json
    if not summary_path:
        base, _ = os.path.splitext(args.output)
        summary_path = f"{base}_summary.json"
    summary["output_npy"] = os.path.abspath(args.output)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved embeddings: {embeddings.shape} -> {args.output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
