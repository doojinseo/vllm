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
