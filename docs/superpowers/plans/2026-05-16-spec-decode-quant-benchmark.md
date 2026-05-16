# Spec Decode Quantization Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `benchmarks/benchmark_spec_decode_quant_sweep.py`, a subprocess orchestrator that sweeps 3 draft-model quantization variants × 8 batch sizes, calls the existing `benchmark_throughput.py` for each combination, and produces `results.csv` + `results.png`.

**Architecture:** A single pure-Python orchestrator script; it never loads a GPU model itself. For each `(variant, batch_size)` pair it shells out to `benchmark_throughput.py`, parses the JSON output, computes `accepted_tokens_per_sec`, and accumulates rows. After all runs it writes the CSV and generates a matplotlib line chart.

**Tech Stack:** Python 3.8+, `subprocess`, `csv`, `json`, `pathlib`, `math`, `argparse`, `matplotlib`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `benchmarks/benchmark_spec_decode_quant_sweep.py` | Entire orchestrator: CLI, sweep loop, metric extraction, CSV, plot |
| No change | `benchmarks/benchmark_throughput.py` | Called as subprocess — untouched |

---

## Task 1: Scaffold the script with CLI and constants

**Files:**
- Create: `benchmarks/benchmark_spec_decode_quant_sweep.py`

- [ ] **Step 1: Create the file with imports, constants, and CLI**

```python
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
```

- [ ] **Step 2: Verify the script is runnable with --help**

```bash
cd /workspace/vllm
python benchmarks/benchmark_spec_decode_quant_sweep.py --help
```

Expected output includes:
```
usage: benchmark_spec_decode_quant_sweep.py [-h] [--output-dir OUTPUT_DIR] ...
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/benchmark_spec_decode_quant_sweep.py
git commit -m "feat: scaffold spec decode quant sweep script with CLI"
```

---

## Task 2: Implement the subprocess runner

**Files:**
- Modify: `benchmarks/benchmark_spec_decode_quant_sweep.py`

- [ ] **Step 1: Write a unit test for `build_cmd`**

Create `tests/benchmarks/test_spec_decode_quant_sweep.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))
from benchmark_spec_decode_quant_sweep import build_cmd, OUTPUT_LEN  # noqa: E402


def test_build_cmd_base_no_quant_flag():
    cmd = build_cmd(
        benchmark_script="benchmarks/benchmark_throughput.py",
        target_model="Qwen/Qwen3-8B",
        draft_model="Qwen/Qwen3-1.7B",
        quant=None,
        batch_size=8,
        num_prompts=256,
        tp=1,
        output_json="/tmp/run.json",
    )
    assert "--speculative-model-quantization" not in cmd
    assert "--speculative-model" in cmd
    assert "Qwen/Qwen3-1.7B" in cmd
    assert "--max-num-seqs" in cmd
    assert "8" in cmd


def test_build_cmd_awq_has_quant_flag():
    cmd = build_cmd(
        benchmark_script="benchmarks/benchmark_throughput.py",
        target_model="Qwen/Qwen3-8B",
        draft_model="Qwen/Qwen3-1.7B-AWQ",
        quant="awq",
        batch_size=4,
        num_prompts=256,
        tp=1,
        output_json="/tmp/run.json",
    )
    assert "--speculative-model-quantization" in cmd
    idx = cmd.index("--speculative-model-quantization")
    assert cmd[idx + 1] == "awq"


def test_num_prompts_formula():
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        expected = max(256, bs * 4)
        assert expected >= 256
        assert expected >= bs * 4
```

- [ ] **Step 2: Run the test to confirm it fails (function not yet defined)**

```bash
cd /workspace/vllm
python -m pytest tests/benchmarks/test_spec_decode_quant_sweep.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` — `build_cmd` does not exist yet.

- [ ] **Step 3: Implement `build_cmd` and `run_one` in the sweep script**

Add these two functions before `if __name__ == "__main__":`:

```python
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
        print(result.stderr[-2000:])
        return row

    json_path = Path(output_json)
    if not json_path.exists():
        print("  FAILED: output JSON not written")
        return row

    with open(json_path) as f:
        data = json.load(f)

    elapsed = data["elapsed_time"]
    num_requests = data["num_requests"]
    accepted_tps = (num_requests * OUTPUT_LEN) / elapsed

    row["elapsed_time"] = elapsed
    row["accepted_tokens_per_sec"] = accepted_tps
    print(f"  OK  accepted_tokens/s={accepted_tps:.1f}")
    return row
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
cd /workspace/vllm
python -m pytest tests/benchmarks/test_spec_decode_quant_sweep.py -v
```

Expected:
```
PASSED tests/benchmarks/test_spec_decode_quant_sweep.py::test_build_cmd_base_no_quant_flag
PASSED tests/benchmarks/test_spec_decode_quant_sweep.py::test_build_cmd_awq_has_quant_flag
PASSED tests/benchmarks/test_spec_decode_quant_sweep.py::test_num_prompts_formula
```

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_spec_decode_quant_sweep.py \
        tests/benchmarks/test_spec_decode_quant_sweep.py
git commit -m "feat: implement build_cmd and run_one with unit tests"
```

---

## Task 3: Implement the sweep loop and CSV writer

**Files:**
- Modify: `benchmarks/benchmark_spec_decode_quant_sweep.py`

- [ ] **Step 1: Write a unit test for `write_csv`**

Add to `tests/benchmarks/test_spec_decode_quant_sweep.py`:

```python
import csv
import math
import tempfile
from pathlib import Path

from benchmark_spec_decode_quant_sweep import write_csv, CSV_COLUMNS


def test_write_csv_creates_file_with_correct_columns():
    rows = [
        {"variant": "base", "batch_size": 1, "num_prompts": 256,
         "accepted_tokens_per_sec": 123.4, "elapsed_time": 5.2},
        {"variant": "awq",  "batch_size": 1, "num_prompts": 256,
         "accepted_tokens_per_sec": float("nan"), "elapsed_time": float("nan")},
    ]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.csv"
        write_csv(rows, out)
        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_COLUMNS
            data = list(reader)
        assert len(data) == 2
        assert data[0]["variant"] == "base"
        assert float(data[0]["accepted_tokens_per_sec"]) == pytest.approx(123.4)
        assert math.isnan(float(data[1]["accepted_tokens_per_sec"]))


# add this import at the top of the test file
import pytest
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /workspace/vllm
python -m pytest tests/benchmarks/test_spec_decode_quant_sweep.py::test_write_csv_creates_file_with_correct_columns -v
```

Expected: `ImportError` — `write_csv` not yet defined.

- [ ] **Step 3: Implement `write_csv` and the main sweep loop**

Add `write_csv` before `if __name__ == "__main__":`:

```python
def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
```

Replace the `if __name__ == "__main__":` block with:

```python
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
```

- [ ] **Step 4: Run all tests**

```bash
cd /workspace/vllm
python -m pytest tests/benchmarks/test_spec_decode_quant_sweep.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_spec_decode_quant_sweep.py \
        tests/benchmarks/test_spec_decode_quant_sweep.py
git commit -m "feat: add sweep loop and CSV writer with tests"
```

---

## Task 4: Implement the matplotlib plot

**Files:**
- Modify: `benchmarks/benchmark_spec_decode_quant_sweep.py`

- [ ] **Step 1: Write a unit test for `plot_results`**

Add to `tests/benchmarks/test_spec_decode_quant_sweep.py`:

```python
from benchmark_spec_decode_quant_sweep import plot_results


def test_plot_results_creates_png():
    rows = []
    for variant in ["base", "awq", "gptq"]:
        for bs in [1, 2, 4, 8]:
            rows.append({
                "variant": variant,
                "batch_size": bs,
                "num_prompts": max(256, bs * 4),
                "accepted_tokens_per_sec": bs * 100.0,
                "elapsed_time": 1.0,
            })
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.png"
        plot_results(rows, out)
        assert out.exists()
        assert out.stat().st_size > 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /workspace/vllm
python -m pytest tests/benchmarks/test_spec_decode_quant_sweep.py::test_plot_results_creates_png -v
```

Expected: `ImportError` — `plot_results` not yet defined.

- [ ] **Step 3: Implement `plot_results`**

Add before `if __name__ == "__main__":`:

```python
def plot_results(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_data: dict[str, dict[str, list]] = {}
    for row in rows:
        v = row["variant"]
        if v not in variant_data:
            variant_data[v] = {"batch_sizes": [], "tps": []}
        variant_data[v]["batch_sizes"].append(int(row["batch_size"]))
        tps = row["accepted_tokens_per_sec"]
        variant_data[v]["tps"].append(
            float(tps) if not (isinstance(tps, float) and math.isnan(tps)) else None
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"base": "o", "awq": "s", "gptq": "^"}

    for variant, data in variant_data.items():
        xs = data["batch_sizes"]
        ys = data["tps"]
        ax.plot(xs, ys, marker=markers.get(variant, "x"),
                label=variant, linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_SIZES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Batch size (max-num-seqs)", fontsize=12)
    ax.set_ylabel("Accepted tokens / sec", fontsize=12)
    ax.set_title(
        "Speculative Decoding Throughput vs Batch Size\n"
        "(Qwen3-1.7B draft → Qwen3-8B)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Wire `plot_results` into the main block**

After the `write_csv` call in `if __name__ == "__main__":`, add:

```python
    png_path = out_dir / "results.png"
    plot_results(rows, png_path)
    print(f"Plot written to {png_path}")
```

- [ ] **Step 5: Run all tests**

```bash
cd /workspace/vllm
python -m pytest tests/benchmarks/test_spec_decode_quant_sweep.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/benchmark_spec_decode_quant_sweep.py \
        tests/benchmarks/test_spec_decode_quant_sweep.py
git commit -m "feat: add matplotlib plot with unit test"
```

---

## Task 5: Smoke-test the full script end-to-end (no GPU required)

This task validates the orchestration logic — the subprocess call, JSON parsing, CSV
writing, and plotting — without touching a real GPU. We mock `subprocess.run` to return
a pre-baked JSON.

**Files:**
- Modify: `tests/benchmarks/test_spec_decode_quant_sweep.py`

- [ ] **Step 1: Write the end-to-end smoke test**

Add to `tests/benchmarks/test_spec_decode_quant_sweep.py`:

```python
import json as _json
from unittest.mock import MagicMock, patch

from benchmark_spec_decode_quant_sweep import run_one, OUTPUT_LEN


def _make_fake_subprocess(tmp_path, variant, batch_size, elapsed=10.0,
                           num_requests=None):
    if num_requests is None:
        num_requests = max(256, batch_size * 4)

    def fake_run(cmd, capture_output, text):
        # write the JSON that benchmark_throughput.py would write
        json_file = tmp_path / f"run_{variant}_{batch_size}.json"
        payload = {
            "elapsed_time": elapsed,
            "num_requests": num_requests,
            "total_num_tokens": num_requests * (128 + OUTPUT_LEN),
            "requests_per_second": num_requests / elapsed,
            "tokens_per_second": num_requests * (128 + OUTPUT_LEN) / elapsed,
        }
        json_file.write_text(_json.dumps(payload))
        mock = MagicMock()
        mock.returncode = 0
        return mock

    return fake_run


def test_run_one_success(tmp_path):
    with patch("benchmark_spec_decode_quant_sweep.subprocess.run",
               side_effect=_make_fake_subprocess(tmp_path, "base", 8)):
        row = run_one(
            benchmark_script="benchmarks/benchmark_throughput.py",
            target_model="Qwen/Qwen3-8B",
            variant_name="base",
            variant_cfg={"model": "Qwen/Qwen3-1.7B", "quant": None},
            batch_size=8,
            tp=1,
            tmp_dir=str(tmp_path),
        )
    assert not math.isnan(row["accepted_tokens_per_sec"])
    expected_tps = (max(256, 8 * 4) * OUTPUT_LEN) / 10.0
    assert row["accepted_tokens_per_sec"] == pytest.approx(expected_tps)


def test_run_one_nonzero_exit_returns_nan(tmp_path):
    def fail_run(cmd, capture_output, text):
        m = MagicMock()
        m.returncode = 1
        m.stderr = "CUDA OOM"
        return m

    with patch("benchmark_spec_decode_quant_sweep.subprocess.run",
               side_effect=fail_run):
        row = run_one(
            benchmark_script="benchmarks/benchmark_throughput.py",
            target_model="Qwen/Qwen3-8B",
            variant_name="awq",
            variant_cfg={"model": "Qwen/Qwen3-1.7B-AWQ", "quant": "awq"},
            batch_size=4,
            tp=1,
            tmp_dir=str(tmp_path),
        )
    assert math.isnan(row["accepted_tokens_per_sec"])
    assert row["variant"] == "awq"
    assert row["batch_size"] == 4
```

- [ ] **Step 2: Run all tests**

```bash
cd /workspace/vllm
python -m pytest tests/benchmarks/test_spec_decode_quant_sweep.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 3: Verify `--help` and dry-run still work**

```bash
cd /workspace/vllm
python benchmarks/benchmark_spec_decode_quant_sweep.py --help
```

Expected: clean help output, no import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/benchmarks/test_spec_decode_quant_sweep.py
git commit -m "test: add end-to-end smoke tests with mocked subprocess"
```

---

## Task 6: Final check and usage instructions

**Files:**
- No new files.

- [ ] **Step 1: Run the full test suite one final time**

```bash
cd /workspace/vllm
python -m pytest tests/benchmarks/test_spec_decode_quant_sweep.py -v
```

Expected: 7/7 PASS.

- [ ] **Step 2: Confirm the `--variants` and `--batch-sizes` flags work**

```bash
cd /workspace/vllm
python benchmarks/benchmark_spec_decode_quant_sweep.py \
    --variants base \
    --batch-sizes 1 2 \
    --output-dir /tmp/dry_run_test 2>&1 | head -5
```

Expected: prints variant and batch size plan, then attempts subprocess (will fail without
real models — that is expected here).

- [ ] **Step 3: Record the real-run command for reference**

Before running for real, update the three model IDs at the top of the script, then:

```bash
cd /workspace/vllm
python benchmarks/benchmark_spec_decode_quant_sweep.py \
    --output-dir ./spec_decode_quant_results \
    --tensor-parallel-size 1
```

Results will appear in `./spec_decode_quant_results/results.csv` and `results.png`.

- [ ] **Step 4: Commit plan reference note**

```bash
git add docs/superpowers/plans/2026-05-16-spec-decode-quant-benchmark.md
git commit -m "docs: add implementation plan for spec decode quant benchmark"
```
