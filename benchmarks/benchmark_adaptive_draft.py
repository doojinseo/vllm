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
    rows = []
    for wave in first:
        row: list = [wave.index, wave.type, wave.batch]
        for lbl in variant_labels:
            results = all_wave_results.get(lbl, [])
            row.append(
                f"{results[wave.index].accepted_tok_per_sec:.1f}"
                if wave.index < len(results) else "N/A"
            )
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
    waves = []
    for wave in first:
        entry: dict = {"index": wave.index, "type": wave.type, "batch": wave.batch}
        for lbl in variant_labels:
            results = all_wave_results.get(lbl, [])
            entry[lbl] = (
                results[wave.index].accepted_tok_per_sec
                if wave.index < len(results) else None
            )
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
