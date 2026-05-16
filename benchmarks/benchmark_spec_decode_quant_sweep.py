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
MAX_STDERR_CHARS = 2000

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


def build_cmd(
    benchmark_script: str,
    target_model: str,
    draft_model: str,
    quant: Optional[str],
    batch_size: int,
    num_prompts: int,
    tp: int,
    output_json: str,
) -> list[str]:
    cmd = [
        sys.executable, benchmark_script,
        "--backend", "vllm",
        "--model", target_model,
        "--speculative-model", draft_model,
        "--num-speculative-tokens", str(NUM_SPECULATIVE_TOKENS),
        "--max-num-seqs", str(batch_size),
        "--num-prompts", str(num_prompts),
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--tensor-parallel-size", str(tp),
        "--output-json", output_json,
    ]
    if quant is not None:
        cmd += ["--speculative-model-quantization", quant]
    return cmd


def run_one(
    benchmark_script: str,
    target_model: str,
    variant_name: str,
    variant_cfg: dict,
    batch_size: int,
    tp: int,
    tmp_dir: str,
) -> dict:
    num_prompts = max(256, batch_size * 4)
    output_json = str(Path(tmp_dir) / f"run_{variant_name}_{batch_size}.json")

    cmd = build_cmd(
        benchmark_script=benchmark_script,
        target_model=target_model,
        draft_model=variant_cfg["model"],
        quant=variant_cfg["quant"],
        batch_size=batch_size,
        num_prompts=num_prompts,
        tp=tp,
        output_json=output_json,
    )

    print(f"\n[{variant_name}] batch_size={batch_size}  num_prompts={num_prompts}")
    print("  CMD:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    row = {
        "variant": variant_name,
        "batch_size": batch_size,
        "num_prompts": num_prompts,
        "accepted_tokens_per_sec": float("nan"),
        "elapsed_time": float("nan"),
    }

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        print(result.stderr[-MAX_STDERR_CHARS:])
        return row

    json_path = Path(output_json)
    if not json_path.exists():
        print("  FAILED: output JSON not written")
        return row

    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  FAILED: invalid JSON in {output_json}: {e}")
        return row

    required_keys = ["elapsed_time", "num_requests"]
    if not all(k in data for k in required_keys):
        print(f"  FAILED: missing keys in JSON. Expected {required_keys}")
        return row

    elapsed = data["elapsed_time"]
    num_requests = data["num_requests"]
    accepted_tps = (num_requests * OUTPUT_LEN) / elapsed

    row["elapsed_time"] = elapsed
    row["accepted_tokens_per_sec"] = accepted_tps
    print(f"  OK  accepted_tokens/s={accepted_tps:.1f}")
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = {k: VARIANTS[k] for k in args.variants}

    total = len(selected_variants) * len(args.batch_sizes)
    done = 0
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for variant_name, variant_cfg in selected_variants.items():
            for batch_size in args.batch_sizes:
                done += 1
                print(f"\n=== Run {done}/{total} ===")
                row = run_one(
                    benchmark_script=args.benchmark_script,
                    target_model=TARGET_MODEL,
                    variant_name=variant_name,
                    variant_cfg=variant_cfg,
                    batch_size=batch_size,
                    tp=args.tensor_parallel_size,
                    tmp_dir=tmp_dir,
                )
                rows.append(row)

    csv_path = out_dir / "results.csv"
    write_csv(rows, csv_path)
    print(f"\nCSV written to {csv_path}")
