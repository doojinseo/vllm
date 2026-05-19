# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark speculative decoding across draft model quantization variants.

Sweeps Qwen3-1.7B draft (bf16 / fp8 / int8) against a Qwen3-8B target over
batch sizes [1, 4, 8, 16, 32, 64, 128].  Primary metric: accepted tokens/s.

Usage:
    python benchmarks/benchmark_spec_decode_quant.py \\
        --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --num-prompts 500 \\
        --num-spec-tokens 5 \\
        --batch-sizes 1 4 8 16 32 64 128
"""

from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import dataclass

import torch
from tabulate import tabulate


def load_sharegpt(
    dataset_path: str,
    num_samples: int,
    max_model_len: int,
    tokenizer,
    seed: int,
) -> list[tuple[list[int], int]]:
    """Load and tokenize ShareGPT conversations.

    Returns a list of (prompt_token_ids, output_len) pairs, capped at
    num_samples entries.  Filters out conversations with < 2 turns, sequences
    with < 4 tokens in either prompt or completion, and sequences whose
    combined length exceeds max_model_len.
    """
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    # Keep only entries with at least two conversation turns.
    data = [d for d in data if len(d.get("conversations", [])) >= 2]

    random.seed(seed)
    random.shuffle(data)

    results: list[tuple[list[int], int]] = []
    for entry in data:
        if len(results) >= num_samples:
            break
        prompt_text = entry["conversations"][0]["value"]
        completion_text = entry["conversations"][1]["value"]

        prompt_ids: list[int] = tokenizer(prompt_text).input_ids
        completion_ids: list[int] = tokenizer(completion_text).input_ids

        prompt_len = len(prompt_ids)
        output_len = len(completion_ids)

        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len + output_len > max_model_len:
            continue

        results.append((prompt_ids, output_len))

    return results


@dataclass
class VariantResult:
    accepted_tok_per_sec: float
    total_output_tokens: int
    wall_time_sec: float


def format_results_table(
    results: dict[int, dict[str | None, "VariantResult | None"]],
    batch_sizes: list[int],
) -> str:
    """Return a tabulated string: rows = batch sizes, cols = quant variants."""
    headers = ["Batch size", "base (tok/s)", "fp8 (tok/s)", "int8 (tok/s)"]
    quant_keys: list[str | None] = [None, "fp8", "int8"]
    rows = []
    for bs in batch_sizes:
        row: list = [bs]
        for q in quant_keys:
            r = results.get(bs, {}).get(q)
            row.append(f"{r.accepted_tok_per_sec:.1f}" if r is not None else "N/A")
        rows.append(row)
    return tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)


def run_variant(
    target_model: str,
    draft_model: str,
    quantization: str | None,
    max_num_seqs: int,
    prompts: list[tuple[list[int], int]],
    num_spec_tokens: int,
    max_model_len: int,
) -> VariantResult:
    """Run one (batch_size, quantization) cell and return metrics.

    Constructs a fresh LLM, runs all prompts, reads accepted token counts from
    Prometheus via llm.get_metrics(), then tears down the engine.
    """
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter as VllmCounter

    llm: LLM | None = None
    try:
        # disable_log_stats=False is required: get_metrics() raises AssertionError
        # if log stats are disabled (vllm/entrypoints/llm.py line 1122).
        llm = LLM(
            model=target_model,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            disable_log_stats=False,
            speculative_config={
                "method": "draft_model",
                "model": draft_model,
                "num_speculative_tokens": num_spec_tokens,
                "quantization": quantization,
            },
        )

        vllm_prompts = [{"prompt_token_ids": ids} for ids, _ in prompts]
        sampling_params_list = [
            SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
            for _, out_len in prompts
        ]

        start = time.perf_counter()
        outputs = llm.generate(vllm_prompts, sampling_params=sampling_params_list)
        elapsed = time.perf_counter() - start

        # Extract accepted token count from Prometheus counters.
        accepted_count = 0
        for metric in llm.get_metrics():
            if (metric.name == "vllm:spec_decode_num_accepted_tokens"
                    and isinstance(metric, VllmCounter)):
                accepted_count += metric.value

        total_output = sum(
            sum(len(o.token_ids) for o in out.outputs) for out in outputs
        )

        return VariantResult(
            accepted_tok_per_sec=accepted_count / elapsed if elapsed > 0 else 0.0,
            total_output_tokens=total_output,
            wall_time_sec=elapsed,
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        torch.cuda.empty_cache()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark spec decode draft model quantization variants."
    )
    parser.add_argument(
        "--target-model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID for the target model.",
    )
    parser.add_argument(
        "--draft-model",
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model ID for the draft model.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to ShareGPT JSON file.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=500,
        help="Number of prompts to sample from the dataset.",
    )
    parser.add_argument(
        "--num-spec-tokens",
        type=int,
        default=5,
        help="Number of speculative tokens per draft step.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128],
        help="Space-separated list of max_num_seqs values to sweep.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length (prompt + output).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not __import__("os").path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {args.target_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    print(f"Sampling {args.num_prompts} prompts from {args.dataset} ...")
    prompts = load_sharegpt(
        dataset_path=args.dataset,
        num_samples=args.num_prompts,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    print(f"  → {len(prompts)} prompts after filtering.")

    quant_variants: list[str | None] = [None, "fp8", "int8"]
    results: dict[int, dict[str | None, VariantResult | None]] = {}

    total_runs = len(args.batch_sizes) * len(quant_variants)
    run_num = 0

    for batch_size in args.batch_sizes:
        results[batch_size] = {}
        for quantization in quant_variants:
            run_num += 1
            label = f"base (bf16)" if quantization is None else quantization
            print(
                f"\n[{run_num}/{total_runs}] batch_size={batch_size}, quant={label}"
            )
            try:
                result = run_variant(
                    target_model=args.target_model,
                    draft_model=args.draft_model,
                    quantization=quantization,
                    max_num_seqs=batch_size,
                    prompts=prompts,
                    num_spec_tokens=args.num_spec_tokens,
                    max_model_len=args.max_model_len,
                )
                results[batch_size][quantization] = result
                print(
                    f"  accepted tok/s: {result.accepted_tok_per_sec:.1f}  "
                    f"wall time: {result.wall_time_sec:.1f}s"
                )
            except Exception as exc:
                print(f"  WARNING: run failed ({exc.__class__.__name__}: {exc})")
                results[batch_size][quantization] = None

    print("\n" + "=" * 60)
    print("Results — Accepted tokens/second")
    print("=" * 60)
    print(format_results_table(results, args.batch_sizes))


if __name__ == "__main__":
    main()
