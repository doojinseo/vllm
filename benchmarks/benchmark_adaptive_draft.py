# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wave-based benchmark for adaptive draft model switching.

Runs four draft-model variants (base, int8, fp8, adaptive) through alternating
small/large request waves on a single long-lived LLM instance per variant.
All variants receive identical pre-sampled prompts per wave for a fair comparison.

Usage:
    python benchmarks/benchmark_adaptive_draft.py \\
        --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --small-batch 4 --large-batch 32 --num-wave-pairs 4
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
        prompt_ids: list[int] = tokenizer(entry["conversations"][0]["value"]).input_ids
        completion_ids: list[int] = tokenizer(entry["conversations"][1]["value"]).input_ids
        if len(prompt_ids) < 4 or len(completion_ids) < 4:
            continue
        if len(prompt_ids) + len(completion_ids) > max_model_len:
            continue
        results.append((prompt_ids, len(completion_ids)))
    return results


def pre_sample_waves(
    dataset_path: str,
    small_batch: int,
    large_batch: int,
    num_wave_pairs: int,
    max_model_len: int,
    tokenizer,
    seed: int,
) -> list[list[tuple[list[int], int]]]:
    """Pre-sample one prompt list per wave; all variants share these lists."""
    waves: list[list[tuple[list[int], int]]] = []
    for i in range(num_wave_pairs * 2):
        batch = small_batch if i % 2 == 0 else large_batch
        prompts = load_sharegpt(
            dataset_path=dataset_path,
            num_samples=batch,
            max_model_len=max_model_len,
            tokenizer=tokenizer,
            seed=seed + i,
        )
        waves.append(prompts)
    return waves


def compute_summary(wave_results: list[WaveResult]) -> VariantSummary:
    """Compute summary statistics across all waves."""
    small = [r.accepted_tok_per_sec for r in wave_results if r.type == "small"]
    large = [r.accepted_tok_per_sec for r in wave_results if r.type == "large"]
    all_vals = [r.accepted_tok_per_sec for r in wave_results]
    return VariantSummary(
        small_avg=sum(small) / len(small) if small else 0.0,
        large_avg=sum(large) / len(large) if large else 0.0,
        overall=sum(all_vals) / len(all_vals) if all_vals else 0.0,
    )


def format_wave_table(
    all_wave_results: dict[str, list[WaveResult]],
    variant_labels: list[str],
) -> str:
    """Format per-wave results as a table."""
    from tabulate import tabulate
    headers = ["Wave", "Type", "Batch"] + [f"{lbl} (tok/s)" for lbl in variant_labels]
    first = next(iter(all_wave_results.values()))
    # Build per-variant lookup by WaveResult.index so formatting is order-independent.
    by_index: dict[str, dict[int, WaveResult]] = {
        lbl: {r.index: r for r in all_wave_results.get(lbl, [])}
        for lbl in variant_labels
    }
    rows = []
    for wave in first:
        row: list = [wave.index, wave.type, wave.batch]
        for lbl in variant_labels:
            r = by_index[lbl].get(wave.index)
            row.append(f"{r.accepted_tok_per_sec:.1f}" if r is not None else "N/A")
        rows.append(row)
    return tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)


def format_summary_table(
    summaries: dict[str, VariantSummary],
    variant_labels: list[str],
) -> str:
    """Format summary statistics as a table."""
    from tabulate import tabulate
    headers = ["Variant", "Small-wave avg", "Large-wave avg", "Overall avg"]
    rows = [
        [lbl, f"{summaries[lbl].small_avg:.1f}",
         f"{summaries[lbl].large_avg:.1f}",
         f"{summaries[lbl].overall:.1f}"]
        for lbl in variant_labels
    ]
    return tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)


def save_results(
    output_path: str,
    config: dict,
    all_wave_results: dict[str, list[WaveResult]],
    summaries: dict[str, VariantSummary],
    variant_labels: list[str],
) -> None:
    """Save results to JSON file."""
    first = next(iter(all_wave_results.values()))
    by_index: dict[str, dict[int, WaveResult]] = {
        lbl: {r.index: r for r in all_wave_results.get(lbl, [])}
        for lbl in variant_labels
    }
    waves = []
    for wave in first:
        entry: dict = {"index": wave.index, "type": wave.type, "batch": wave.batch}
        for lbl in variant_labels:
            r = by_index[lbl].get(wave.index)
            entry[lbl] = r.accepted_tok_per_sec if r is not None else None
        waves.append(entry)

    summary_dict = {
        lbl: {
            "small_avg": summaries[lbl].small_avg,
            "large_avg": summaries[lbl].large_avg,
            "overall":   summaries[lbl].overall,
        }
        for lbl in variant_labels
    }

    with open(output_path, "w") as f:
        json.dump({"config": config, "waves": waves, "summary": summary_dict},
                  f, indent=2)


def plot_results(
    plot_path: str,
    all_wave_results: dict[str, list[WaveResult]],
    summaries: dict[str, VariantSummary],
    variant_labels: list[str],
) -> None:
    """Plot per-wave results and summary statistics."""
    import matplotlib.pyplot as plt
    import numpy as np

    colours = {"base": "C0", "int8": "C1", "fp8": "C2", "adaptive": "C3"}
    first = next(iter(all_wave_results.values()))
    x = [w.index for w in first]
    x_labels = [f"{'S' if w.type == 'small' else 'L'}{w.index}" for w in first]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top panel: per-wave line chart
    for lbl in variant_labels:
        ys = [r.accepted_tok_per_sec for r in all_wave_results[lbl]]
        ax1.plot(x, ys, marker="o", label=lbl, color=colours.get(lbl, None))

    for w in first:
        shade = "#d0e8ff" if w.type == "small" else "#ffe8d0"
        ax1.axvspan(w.index - 0.5, w.index + 0.5, color=shade, alpha=0.3, zorder=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("Wave")
    ax1.set_ylabel("Accepted tok/s")
    ax1.set_title("Per-wave accepted tok/s by variant")
    ax1.legend()

    # Bottom panel: grouped bar chart (small vs large avg)
    bar_w = 0.35
    x2 = np.arange(len(variant_labels))
    small_vals = [summaries[lbl].small_avg for lbl in variant_labels]
    large_vals = [summaries[lbl].large_avg for lbl in variant_labels]
    ax2.bar(x2 - bar_w / 2, small_vals, bar_w, label="small-wave avg", color="#4c9be8")
    ax2.bar(x2 + bar_w / 2, large_vals, bar_w, label="large-wave avg", color="#e87c4c")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(variant_labels)
    ax2.set_ylabel("Accepted tok/s")
    ax2.set_title("Small vs large wave average by variant")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Wave benchmark for adaptive draft model switching."
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model-base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-model-fp8",  default="Qwen/Qwen3-1.7B-FP8")
    parser.add_argument("--draft-model-int8", default="Qwen/Qwen3-1.7B-GPTQ-Int8")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--small-batch",    type=int,   default=4)
    parser.add_argument("--large-batch",    type=int,   default=32)
    parser.add_argument("--num-wave-pairs", type=int,   default=8)
    parser.add_argument("--num-spec-tokens",type=int,   default=5)
    parser.add_argument("--threshold",      type=int,   default=8)
    parser.add_argument("--ema-alpha",      type=float, default=0.3)
    parser.add_argument("--max-model-len",  type=int,   default=4096)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output", default="results/adaptive_draft_wave_results.json")
    parser.add_argument("--plot",   default=None,
                        help="Plot path (default: --output stem + .png)")
    return parser.parse_args(argv)


def _clear_vllm_compile_cache() -> None:
    cache_dir = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def build_llm(
    variant: str,
    target_model: str,
    draft_model_base: str,
    draft_model_fp8: str,
    draft_model_int8: str,
    large_batch: int,
    num_spec_tokens: int,
    max_model_len: int,
    threshold: int,
    ema_alpha: float,
) -> "LLM":
    from vllm import LLM

    draft_model = {
        "base":     draft_model_base,
        "int8":     draft_model_int8,
        "fp8":      draft_model_fp8,
        "adaptive": draft_model_fp8,
    }[variant]

    spec_config: dict = {
        "method": "draft_model",
        "model": draft_model,
        "num_speculative_tokens": num_spec_tokens,
    }
    if variant == "adaptive":
        spec_config["alt_model"] = draft_model_int8
        spec_config["adaptive_threshold"] = threshold
        spec_config["adaptive_ema_alpha"] = ema_alpha

    return LLM(
        model=target_model,
        max_num_seqs=large_batch,
        max_model_len=max_model_len,
        disable_log_stats=False,
        speculative_config=spec_config,
    )


def _warmup(llm: "LLM", wave_prompts: list[tuple[list[int], int]]) -> None:
    """Run one short request to trigger JIT kernel compilation before timing.

    The adaptive model disables CUDA graphs so it cannot pre-compile kernels
    during init the way fixed-quant models do. Without this, wave 0 pays the
    full JIT cost and appears artificially slow.
    """
    from vllm import SamplingParams
    prompt_ids, out_len = wave_prompts[0]
    llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        [SamplingParams(max_tokens=min(out_len, 32), temperature=0.0)],
    )


def run_variant_waves(
    llm: "LLM",
    waves: list[list[tuple[list[int], int]]],
    variant: str,
) -> list[WaveResult]:
    from vllm import SamplingParams
    from vllm.v1.metrics.reader import Counter as VllmCounter

    if variant == "adaptive":
        print(f"  [{variant}] warming up JIT kernels ...")
        _warmup(llm, waves[0])

    results: list[WaveResult] = []

    # Capture the counter baseline after any warm-up so wave 0 delta is clean.
    prev_accepted = 0
    for metric in llm.get_metrics():
        if (metric.name == "vllm:spec_decode_num_accepted_tokens"
                and isinstance(metric, VllmCounter)):
            prev_accepted = metric.value

    for i, wave_prompts in enumerate(waves):
        wave_type = "small" if i % 2 == 0 else "large"
        vllm_prompts = [{"prompt_token_ids": ids} for ids, _ in wave_prompts]
        sampling_params = [
            SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
            for _, out_len in wave_prompts
        ]

        start = time.perf_counter()
        llm.generate(vllm_prompts, sampling_params)
        elapsed = time.perf_counter() - start

        accepted_count = 0
        for metric in llm.get_metrics():
            if (metric.name == "vllm:spec_decode_num_accepted_tokens"
                    and isinstance(metric, VllmCounter)):
                accepted_count += metric.value

        delta = accepted_count - prev_accepted
        prev_accepted = accepted_count

        if delta == 0:
            import warnings
            warnings.warn(
                f"Wave {i} ({wave_type}): accepted token count is 0. "
                "This may indicate Prometheus multiprocess mode is active "
                "or speculative decoding is not functioning.",
                stacklevel=2,
            )

        tok_per_sec = delta / elapsed if elapsed > 0 else 0.0
        results.append(WaveResult(
            index=i, type=wave_type, batch=len(wave_prompts),
            accepted_tok_per_sec=tok_per_sec, wall_time_sec=elapsed,
        ))
        print(f"  [{variant}] wave {i} ({wave_type}, bs={len(wave_prompts)}): "
              f"{tok_per_sec:.1f} tok/s  ({elapsed:.1f}s)")

    return results


def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {args.target_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    print(f"Pre-sampling waves (small={args.small_batch}, large={args.large_batch}, "
          f"pairs={args.num_wave_pairs}) ...")
    waves = pre_sample_waves(
        dataset_path=args.dataset,
        small_batch=args.small_batch,
        large_batch=args.large_batch,
        num_wave_pairs=args.num_wave_pairs,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    for i, w in enumerate(waves):
        wtype = "small" if i % 2 == 0 else "large"
        print(f"  wave {i} ({wtype}): {len(w)} prompts")

    variant_labels = ["base", "int8", "fp8", "adaptive"]
    all_wave_results: dict[str, list[WaveResult]] = {}
    summaries: dict[str, VariantSummary] = {}

    for variant in variant_labels:
        print(f"\n{'=' * 60}")
        print(f"Variant: {variant}")
        print(f"{'=' * 60}")
        _clear_vllm_compile_cache()
        llm = build_llm(
            variant=variant,
            target_model=args.target_model,
            draft_model_base=args.draft_model_base,
            draft_model_fp8=args.draft_model_fp8,
            draft_model_int8=args.draft_model_int8,
            large_batch=args.large_batch,
            num_spec_tokens=args.num_spec_tokens,
            max_model_len=args.max_model_len,
            threshold=args.threshold,
            ema_alpha=args.ema_alpha,
        )
        try:
            wave_results = run_variant_waves(llm, waves, variant)
        finally:
            del llm
            gc.collect()
            torch.cuda.empty_cache()

        all_wave_results[variant] = wave_results
        summaries[variant] = compute_summary(wave_results)

    print("\n" + "=" * 60)
    print("Per-wave results (accepted tok/s)")
    print("=" * 60)
    print(format_wave_table(all_wave_results, variant_labels))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(format_summary_table(summaries, variant_labels))

    config = {
        "small_batch":       args.small_batch,
        "large_batch":       args.large_batch,
        "num_wave_pairs":    args.num_wave_pairs,
        "num_spec_tokens":   args.num_spec_tokens,
        "threshold":         args.threshold,
        "ema_alpha":         args.ema_alpha,
        "target_model":      args.target_model,
        "draft_model_base":  args.draft_model_base,
        "draft_model_fp8":   args.draft_model_fp8,
        "draft_model_int8":  args.draft_model_int8,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_results(args.output, config, all_wave_results, summaries, variant_labels)
    print(f"\nResults saved to {args.output}")

    plot_path = args.plot or str(Path(args.output).with_suffix(".png"))
    plot_results(plot_path, all_wave_results, summaries, variant_labels)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
