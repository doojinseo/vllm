#!/usr/bin/env python3
"""Sweep draft-model quantization variants for speculative decoding throughput.

Calls benchmarks/benchmark_throughput.py as a subprocess for each
(variant, batch_size) pair and aggregates results into a CSV and plot.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Sweep configuration — edit model IDs to match your HF cache / local paths
# ---------------------------------------------------------------------------
TARGET_MODEL = "Qwen/Qwen3-8B"

VARIANTS: dict[str, dict] = {
    "base": {"model": "Qwen/Qwen3-1.7B",          "quant": None},
    "awq":  {"model": "Qwen/Qwen3-1.7B-AWQ",       "quant": "awq"},
    "gptq": {"model": "Qwen/Qwen3-1.7B-GPTQ-Int4", "quant": "gptq"},
}

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
NUM_SPECULATIVE_TOKENS = 5
INPUT_LEN = 128
OUTPUT_LEN = 256

CSV_COLUMNS = [
    "variant", "batch_size", "num_prompts",
    "accepted_tokens_per_sec", "elapsed_time",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep draft-model quantization for speculative decoding.")
    p.add_argument(
        "--output-dir", default="./spec_decode_quant_results",
        help="Directory to write results.csv and results.png (created if absent).")
    p.add_argument(
        "--benchmark-script",
        default=str(Path(__file__).parent / "benchmark_throughput.py"),
        help="Path to benchmark_throughput.py.")
    p.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="--tensor-parallel-size passed to each benchmark run.")
    p.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
        help="Subset of variants to run (default: all three).")
    p.add_argument(
        "--batch-sizes", nargs="+", type=int, default=BATCH_SIZES,
        help="Batch sizes to sweep.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Output dir : {args.output_dir}")
    print(f"Variants   : {args.variants}")
    print(f"Batch sizes: {args.batch_sizes}")
