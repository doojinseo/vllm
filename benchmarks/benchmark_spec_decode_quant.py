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
