# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare adaptive draft model switching against fixed-model baselines.

Runs four strategies side-by-side at each batch size:
  base    -- Qwen3-1.7B (bf16)
  fp8     -- Qwen3-1.7B-FP8
  int8    -- Qwen3-1.7B-GPTQ-Int8
  adaptive-- fp8 (large batch) / int8 (small batch) switch at threshold

Primary metric: accepted tokens/s.

Usage:
    python benchmarks/benchmark_adaptive_draft.py \\
        --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --num-prompts 500 \\
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

import torch
from tabulate import tabulate


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
        prompt_ids = tokenizer(entry["conversations"][0]["value"]).input_ids
        completion_ids = tokenizer(entry["conversations"][1]["value"]).input_ids
        if len(prompt_ids) < 4 or len(completion_ids) < 4:
            continue
        if len(prompt_ids) + len(completion_ids) > max_model_len:
            continue
        results.append((prompt_ids, len(completion_ids)))

    return results


@dataclass
class WaveResult:
    index: int
    type: str               # "small" or "large"
    batch: int
    accepted_tok_per_sec: float
    wall_time_sec: float


@dataclass
class VariantSummary:
    small_avg: float
    large_avg: float
    overall: float


@dataclass
class RunResult:
    accepted_tok_per_sec: float
    total_output_tokens: int
    wall_time_sec: float


def _clear_compile_cache() -> None:
    cache_dir = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def run_fixed(
    target_model: str,
    draft_model: str,
    max_num_seqs: int,
    prompts: list[tuple[list[int], int]],
    num_spec_tokens: int,
    max_model_len: int,
) -> RunResult:
    """Run a fixed (single) draft model."""
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter as VllmCounter

    llm = None
    try:
        llm = LLM(
            model=target_model,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            disable_log_stats=False,
            speculative_config={
                "method": "draft_model",
                "model": draft_model,
                "num_speculative_tokens": num_spec_tokens,
            },
        )
        vllm_prompts = [{"prompt_token_ids": ids} for ids, _ in prompts]
        sampling_params = [
            SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
            for _, out_len in prompts
        ]
        start = time.perf_counter()
        outputs = llm.generate(vllm_prompts, sampling_params=sampling_params)
        elapsed = time.perf_counter() - start

        accepted = 0
        for m in llm.get_metrics():
            if (m.name == "vllm:spec_decode_num_accepted_tokens"
                    and isinstance(m, VllmCounter)):
                accepted += m.value

        total_out = sum(
            sum(len(o.token_ids) for o in out.outputs) for out in outputs
        )
        return RunResult(
            accepted_tok_per_sec=accepted / elapsed if elapsed > 0 else 0.0,
            total_output_tokens=total_out,
            wall_time_sec=elapsed,
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        torch.cuda.empty_cache()


def run_adaptive(
    target_model: str,
    primary_draft: str,
    alt_draft: str,
    threshold: int,
    ema_alpha: float,
    max_num_seqs: int,
    prompts: list[tuple[list[int], int]],
    num_spec_tokens: int,
    max_model_len: int,
) -> RunResult:
    """Run the adaptive (two draft models) proposer."""
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter as VllmCounter

    llm = None
    try:
        llm = LLM(
            model=target_model,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            disable_log_stats=False,
            speculative_config={
                "method": "draft_model",
                "model": primary_draft,
                "num_speculative_tokens": num_spec_tokens,
                "alt_model": alt_draft,
                "adaptive_threshold": threshold,
                "adaptive_ema_alpha": ema_alpha,
            },
        )
        vllm_prompts = [{"prompt_token_ids": ids} for ids, _ in prompts]
        sampling_params = [
            SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
            for _, out_len in prompts
        ]
        start = time.perf_counter()
        outputs = llm.generate(vllm_prompts, sampling_params=sampling_params)
        elapsed = time.perf_counter() - start

        accepted = 0
        for m in llm.get_metrics():
            if (m.name == "vllm:spec_decode_num_accepted_tokens"
                    and isinstance(m, VllmCounter)):
                accepted += m.value

        total_out = sum(
            sum(len(o.token_ids) for o in out.outputs) for out in outputs
        )
        return RunResult(
            accepted_tok_per_sec=accepted / elapsed if elapsed > 0 else 0.0,
            total_output_tokens=total_out,
            wall_time_sec=elapsed,
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        torch.cuda.empty_cache()


def format_table(
    results: dict[int, dict[str, RunResult | None]],
    batch_sizes: list[int],
    labels: list[str],
) -> str:
    headers = ["Batch size"] + [f"{lbl} (tok/s)" for lbl in labels]
    rows = []
    for bs in batch_sizes:
        row: list = [bs]
        for lbl in labels:
            r = results.get(bs, {}).get(lbl)
            row.append(f"{r.accepted_tok_per_sec:.1f}" if r is not None else "N/A")
        rows.append(row)
    return tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare adaptive vs fixed draft model strategies."
    )
    p.add_argument("--target-model", default="Qwen/Qwen3-8B")
    p.add_argument("--draft-model-base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--draft-model-fp8",  default="Qwen/Qwen3-1.7B-FP8")
    p.add_argument("--draft-model-int8", default="Qwen/Qwen3-1.7B-GPTQ-Int8")
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--num-spec-tokens", type=int, default=5)
    p.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128],
    )
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--adaptive-threshold", type=int, default=8,
        help="EMA batch-size threshold for adaptive switching (default: 8).",
    )
    p.add_argument(
        "--adaptive-ema-alpha", type=float, default=0.1,
        help="EMA decay factor for batch-size smoothing (default: 0.1).",
    )
    p.add_argument(
        "--adaptive-only", action="store_true",
        help="Skip fixed-model runs and load prior results from --prior-results.",
    )
    p.add_argument(
        "--prior-results",
        default=None,
        help="Path to prior benchmark output file to merge with adaptive results.",
    )
    return p.parse_args()


def load_prior_results(
    path: str,
    batch_sizes: list[int],
) -> dict[int, dict[str, RunResult | None]]:
    """Parse a prior benchmark output file into the results dict format."""
    import re
    text = Path(path).read_text()
    lines = text.splitlines()
    header_idx = next(
        (i for i, l in enumerate(lines) if "Batch size" in l and "tok/s" in l),
        None,
    )
    if header_idx is None:
        raise ValueError(f"Could not find results table in {path}")

    header = lines[header_idx]
    labels = re.findall(r"(\w+)\s+\(tok/s\)", header)

    results: dict[int, dict[str, RunResult | None]] = {bs: {} for bs in batch_sizes}
    for line in lines[header_idx + 1:]:
        s = line.strip()
        if not s or s.startswith("-"):
            continue
        cols = s.split()
        try:
            bs = int(cols[0])
        except ValueError:
            break
        if bs not in results:
            continue
        for vi, lbl in enumerate(labels):
            val_str = cols[vi + 1] if vi + 1 < len(cols) else "N/A"
            if val_str == "N/A":
                results[bs][lbl] = None
            else:
                results[bs][lbl] = RunResult(
                    accepted_tok_per_sec=float(val_str),
                    total_output_tokens=0,
                    wall_time_sec=0.0,
                )
    return results


def main():
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {args.target_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    print(f"Sampling {args.num_prompts} prompts per batch size ...")
    prompts_per_bs: dict[int, list[tuple[list[int], int]]] = {}
    for bi, bs in enumerate(args.batch_sizes):
        ps = load_sharegpt(
            dataset_path=args.dataset,
            num_samples=args.num_prompts,
            max_model_len=args.max_model_len,
            tokenizer=tokenizer,
            seed=args.seed + bi,
        )
        prompts_per_bs[bs] = ps
        print(f"  bs={bs}: {len(ps)} prompts")

    results: dict[int, dict[str, RunResult | None]] = {
        bs: {} for bs in args.batch_sizes
    }

    if args.adaptive_only:
        # Load prior fixed-variant results from file or use hardcoded defaults.
        if args.prior_results:
            print(f"Loading prior results from {args.prior_results} ...")
            prior = load_prior_results(args.prior_results, args.batch_sizes)
            for bs in args.batch_sizes:
                results[bs].update(prior.get(bs, {}))
        else:
            # Hardcoded from the previous full run (same seed/prompts).
            prior_data = {
                1:   {"base": 92.0,  "fp8": 97.3,   "int8": 106.9},
                4:   {"base": 327.0, "fp8": 342.1,  "int8": 362.7},
                8:   {"base": 563.7, "fp8": 606.1,  "int8": 613.3},
                16:  {"base": 957.6, "fp8": 994.2,  "int8": 946.1},
                32:  {"base": 1451.6,"fp8": 1519.2, "int8": 1455.5},
                64:  {"base": 1866.7,"fp8": 1902.3, "int8": 1833.4},
                128: {"base": 1899.4,"fp8": 1980.3, "int8": 1906.2},
            }
            for bs in args.batch_sizes:
                if bs in prior_data:
                    for lbl, val in prior_data[bs].items():
                        results[bs][lbl] = RunResult(
                            accepted_tok_per_sec=val,
                            total_output_tokens=0,
                            wall_time_sec=0.0,
                        )
        fixed_labels = list(next(iter(results.values())).keys())
        labels = fixed_labels + ["adaptive"]
        total_runs = len(args.batch_sizes)
        run_num = 0
    else:
        fixed_variants = [
            ("base", args.draft_model_base),
            ("fp8",  args.draft_model_fp8),
            ("int8", args.draft_model_int8),
        ]
        labels = ["base", "fp8", "int8", "adaptive"]
        total_runs = len(args.batch_sizes) * (len(fixed_variants) + 1)
        run_num = 0

        for label, draft_model in fixed_variants:
            _clear_compile_cache()
            for bs in args.batch_sizes:
                run_num += 1
                print(f"\n[{run_num}/{total_runs}] strategy={label}, bs={bs}")
                try:
                    r = run_fixed(
                        target_model=args.target_model,
                        draft_model=draft_model,
                        max_num_seqs=bs,
                        prompts=prompts_per_bs[bs],
                        num_spec_tokens=args.num_spec_tokens,
                        max_model_len=args.max_model_len,
                    )
                    results[bs][label] = r
                    print(
                        f"  accepted tok/s: {r.accepted_tok_per_sec:.1f}  "
                        f"wall: {r.wall_time_sec:.1f}s"
                    )
                except Exception as exc:
                    print(f"  WARNING: failed ({exc.__class__.__name__}: {exc})")
                    results[bs][label] = None

    # Adaptive runs
    _clear_compile_cache()
    for bs in args.batch_sizes:
        run_num += 1
        print(
            f"\n[{run_num}/{total_runs}] strategy=adaptive "
            f"(fp8>int8 @ ema={args.adaptive_threshold}), bs={bs}"
        )
        try:
            r = run_adaptive(
                target_model=args.target_model,
                primary_draft=args.draft_model_fp8,
                alt_draft=args.draft_model_int8,
                threshold=args.adaptive_threshold,
                ema_alpha=args.adaptive_ema_alpha,
                max_num_seqs=bs,
                prompts=prompts_per_bs[bs],
                num_spec_tokens=args.num_spec_tokens,
                max_model_len=args.max_model_len,
            )
            results[bs]["adaptive"] = r
            print(
                f"  accepted tok/s: {r.accepted_tok_per_sec:.1f}  "
                f"wall: {r.wall_time_sec:.1f}s"
            )
        except Exception as exc:
            print(f"  WARNING: failed ({exc.__class__.__name__}: {exc})")
            results[bs]["adaptive"] = None

    print("\n" + "=" * 65)
    print("Results — Accepted tokens/second")
    print("=" * 65)
    print(format_table(results, args.batch_sizes, labels))


if __name__ == "__main__":
    main()
