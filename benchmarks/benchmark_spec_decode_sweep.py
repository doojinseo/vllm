# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep num_spec_tokens × batch_size × quantization variant.

For each value of num_spec_tokens, prints a comparison table identical in
layout to benchmark_spec_decode_quant.py.  Run all spec-token values with
all three variants to find the optimal k per batch size per quantization.

Usage:
    python benchmarks/benchmark_spec_decode_sweep.py \\
        --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --num-prompts 500 \\
        --spec-tokens 3 5 7 9 \\
        --batch-sizes 1 4 8 16 32 64 128
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tabulate import tabulate


# ── shared helpers ────────────────────────────────────────────────────────────

def load_sharegpt(
    dataset_path: str,
    num_samples: int,
    max_model_len: int,
    tokenizer,
    seed: int,
) -> list[tuple[list[int], int]]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    data = [d for d in data if len(d.get("conversations", [])) >= 2]
    random.seed(seed)
    random.shuffle(data)
    results: list[tuple[list[int], int]] = []
    for entry in data:
        if len(results) >= num_samples:
            break
        prompt_ids: list[int] = (
            tokenizer(entry["conversations"][0]["value"]).input_ids
        )
        completion_ids: list[int] = (
            tokenizer(entry["conversations"][1]["value"]).input_ids
        )
        if len(prompt_ids) < 4 or len(completion_ids) < 4:
            continue
        if len(prompt_ids) + len(completion_ids) > max_model_len:
            continue
        results.append((prompt_ids, len(completion_ids)))
    return results


@dataclass
class VariantResult:
    output_tok_per_sec: float
    total_output_tokens: int
    wall_time_sec: float


def _clear_vllm_compile_cache() -> None:
    cache_dir = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def run_variant(
    target_model: str,
    draft_model: str | None,
    max_num_seqs: int,
    prompts: list[tuple[list[int], int]],
    num_spec_tokens: int,
    max_model_len: int,
) -> VariantResult:
    from vllm import LLM, SamplingParams

    llm: LLM | None = None
    try:
        kwargs: dict[str, Any] = dict(
            model=target_model,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
        )
        if draft_model is not None:
            kwargs["speculative_config"] = {
                "method": "draft_model",
                "model": draft_model,
                "num_speculative_tokens": num_spec_tokens,
            }
        llm = LLM(**kwargs)
        vllm_prompts = [{"prompt_token_ids": ids} for ids, _ in prompts]
        sampling_params_list = [
            SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
            for _, out_len in prompts
        ]
        start = time.perf_counter()
        outputs = llm.generate(vllm_prompts, sampling_params=sampling_params_list)
        elapsed = time.perf_counter() - start

        total_output = sum(
            sum(len(o.token_ids) for o in out.outputs) for out in outputs
        )
        return VariantResult(
            output_tok_per_sec=total_output / elapsed if elapsed > 0 else 0.0,
            total_output_tokens=total_output,
            wall_time_sec=elapsed,
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        torch.cuda.empty_cache()


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt(r: VariantResult | None) -> str:
    return f"{r.output_tok_per_sec:.1f}" if r is not None else "N/A"


def print_per_k_table(
    results: dict[int, dict[int, dict[str, VariantResult | None]]],
    batch_sizes: list[int],
    spec_tokens_list: list[int],
    variant_labels: list[str],
) -> None:
    """For each variant: one table with rows=batch_sizes, cols=num_spec_tokens."""
    for label in variant_labels:
        print(f"\n--- {label} ---")
        headers = ["batch_size"] + [f"k={k}" for k in spec_tokens_list]
        rows = []
        for bs in batch_sizes:
            row: list[Any] = [bs]
            for k in spec_tokens_list:
                row.append(_fmt(results[k][bs].get(label)))
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="simple",
                       disable_numparse=True))


def print_per_bs_table(
    results: dict[int, dict[int, dict[str, VariantResult | None]]],
    batch_sizes: list[int],
    spec_tokens_list: list[int],
    variant_labels: list[str],
) -> None:
    """For each num_spec_tokens: one table with rows=batch_sizes, cols=variants."""
    for k in spec_tokens_list:
        print(f"\n=== num_spec_tokens = {k} ===")
        headers = ["batch_size"] + [f"{lbl} (tok/s)" for lbl in variant_labels]
        rows = []
        for bs in batch_sizes:
            row: list[Any] = [bs]
            for label in variant_labels:
                row.append(_fmt(results[k][bs].get(label)))
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="simple",
                       disable_numparse=True))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep num_spec_tokens × batch_size × quantization variant."
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model-base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-model-fp8",  default="Qwen/Qwen3-1.7B-FP8")
    parser.add_argument("--draft-model-int8", default="Qwen/Qwen3-1.7B-GPTQ-Int8")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument(
        "--spec-tokens", type=int, nargs="+", default=[3, 5, 7, 9],
        help="List of num_speculative_tokens values to sweep.",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    from transformers import AutoTokenizer
    print(f"Loading tokenizer for {args.target_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    print(f"Sampling {args.num_prompts} prompts ...")
    prompts = load_sharegpt(
        dataset_path=args.dataset,
        num_samples=args.num_prompts,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    print(f"  → {len(prompts)} prompts after filtering.")

    variants: list[tuple[str, str | None]] = [
        ("base", None),
        ("fp8",  args.draft_model_fp8),
        ("int8", args.draft_model_int8),
    ]
    variant_labels = [lbl for lbl, _ in variants]

    # results[k][batch_size][label] = VariantResult | None
    results: dict[int, dict[int, dict[str, VariantResult | None]]] = {}
    for k in args.spec_tokens:
        results[k] = {bs: {} for bs in args.batch_sizes}

    total_runs = len(args.spec_tokens) * len(variants) * len(args.batch_sizes)
    run_num = 0

    # Loop: spec_tokens → variant → batch_size
    # Cache is cleared at each (spec_tokens, variant) transition to avoid
    # kernel-arity collisions across different speculative decoding settings,
    # and to avoid CUDA-graph shape mismatches across different k values.
    for k in args.spec_tokens:
        for label, draft_model in variants:
            _clear_vllm_compile_cache()
            for batch_size in args.batch_sizes:
                run_num += 1
                print(
                    f"\n[{run_num}/{total_runs}] "
                    f"k={k}, variant={label}, batch_size={batch_size}"
                )
                try:
                    result = run_variant(
                        target_model=args.target_model,
                        draft_model=draft_model,
                        max_num_seqs=batch_size,
                        prompts=prompts,
                        num_spec_tokens=k,
                        max_model_len=args.max_model_len,
                    )
                    results[k][batch_size][label] = result
                    print(
                        f"  output tok/s: {result.output_tok_per_sec:.1f}  "
                        f"wall time: {result.wall_time_sec:.1f}s"
                    )
                except Exception as exc:
                    print(
                        f"  WARNING: run failed "
                        f"({exc.__class__.__name__}: {exc})"
                    )
                    results[k][batch_size][label] = None

    print("\n\n" + "=" * 70)
    print("Per-variant tables  (rows = batch size, cols = num_spec_tokens)")
    print("=" * 70)
    print_per_k_table(results, args.batch_sizes, args.spec_tokens, variant_labels)

    print("\n\n" + "=" * 70)
    print("Per-k tables  (rows = batch size, cols = variant)")
    print("=" * 70)
    print_per_bs_table(results, args.batch_sizes, args.spec_tokens, variant_labels)


if __name__ == "__main__":
    main()
