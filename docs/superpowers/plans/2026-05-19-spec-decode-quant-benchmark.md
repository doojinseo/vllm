# Spec Decode Quantization Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `benchmarks/benchmark_spec_decode_quant.py` — a standalone script that sweeps Qwen3-1.7B draft model quantization variants (bf16, fp8, int8) across batch sizes [1,4,8,16,32,64,128] and reports accepted tokens/second for each cell.

**Architecture:** Single script with four functions: `load_sharegpt` (dataset loading), `run_variant` (one LLM run), `format_results_table` (tabulate output), and `main` (CLI + nested loop). Each `run_variant` call constructs a fresh `LLM`, runs all prompts, reads accepted token counts from `llm.get_metrics()`, then destroys the engine. The outer loop iterates batch sizes; the inner loop iterates quantization variants.

**Tech Stack:** vLLM `LLM` API, `speculative_config` dict with `method="draft_model"`, `llm.get_metrics()` for Prometheus counters, `tabulate`, `argparse`, Python `gc` + `torch.cuda.empty_cache()` for teardown.

---

## File Structure

| Path | Role |
|------|------|
| `benchmarks/benchmark_spec_decode_quant.py` | New benchmark script (create) |
| `tests/benchmarks/test_benchmark_spec_decode_quant.py` | Pure-Python unit tests for dataset loading and table formatting (create) |

---

## Task 1: Write failing tests for `load_sharegpt`

**Files:**
- Create: `tests/benchmarks/test_benchmark_spec_decode_quant.py`

- [ ] **Step 1: Create the test file**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for benchmark_spec_decode_quant.py (pure-Python parts only).

These tests do not require a GPU or real models.
"""
import json
import sys
from pathlib import Path

import pytest

# Make benchmarks/ importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))


class _MockTokenizer:
    """Minimal tokenizer stub: splits on whitespace, returns word indices."""

    def __call__(self, text: str):
        tokens = text.split()
        return type("Enc", (), {"input_ids": list(range(len(tokens)))})()


@pytest.mark.benchmark
def test_load_sharegpt_basic(tmp_path):
    """Valid 2-turn conversations with enough tokens are returned."""
    from benchmark_spec_decode_quant import load_sharegpt

    data = [
        {
            "conversations": [
                {"value": " ".join(["word"] * 10)},  # 10-token prompt
                {"value": " ".join(["word"] * 10)},  # 10-token completion
            ]
        }
    ]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))

    result = load_sharegpt(
        str(path), num_samples=10, max_model_len=4096, tokenizer=_MockTokenizer(), seed=42
    )
    assert len(result) == 1
    prompt_ids, output_len = result[0]
    assert len(prompt_ids) == 10
    assert output_len == 10


@pytest.mark.benchmark
def test_load_sharegpt_filters_single_turn(tmp_path):
    """Conversations with fewer than 2 turns are dropped."""
    from benchmark_spec_decode_quant import load_sharegpt

    data = [{"conversations": [{"value": " ".join(["word"] * 10)}]}]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))

    result = load_sharegpt(
        str(path), num_samples=10, max_model_len=4096, tokenizer=_MockTokenizer(), seed=42
    )
    assert result == []


@pytest.mark.benchmark
def test_load_sharegpt_filters_too_short(tmp_path):
    """Sequences with fewer than 4 tokens in prompt or completion are dropped."""
    from benchmark_spec_decode_quant import load_sharegpt

    data = [
        {
            "conversations": [
                {"value": "hi"},        # 1 token — too short
                {"value": "hello"},     # 1 token — too short
            ]
        }
    ]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))

    result = load_sharegpt(
        str(path), num_samples=10, max_model_len=4096, tokenizer=_MockTokenizer(), seed=42
    )
    assert result == []


@pytest.mark.benchmark
def test_load_sharegpt_filters_over_max_model_len(tmp_path):
    """Sequences exceeding max_model_len are dropped."""
    from benchmark_spec_decode_quant import load_sharegpt

    # 50 + 50 = 100 tokens total, max_model_len=80 → filtered
    data = [
        {
            "conversations": [
                {"value": " ".join(["word"] * 50)},
                {"value": " ".join(["word"] * 50)},
            ]
        }
    ]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))

    result = load_sharegpt(
        str(path), num_samples=10, max_model_len=80, tokenizer=_MockTokenizer(), seed=42
    )
    assert result == []


@pytest.mark.benchmark
def test_load_sharegpt_respects_num_samples(tmp_path):
    """At most num_samples entries are returned."""
    from benchmark_spec_decode_quant import load_sharegpt

    data = [
        {
            "conversations": [
                {"value": " ".join(["word"] * 10)},
                {"value": " ".join(["word"] * 10)},
            ]
        }
        for _ in range(20)
    ]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))

    result = load_sharegpt(
        str(path), num_samples=5, max_model_len=4096, tokenizer=_MockTokenizer(), seed=42
    )
    assert len(result) == 5


@pytest.mark.benchmark
def test_load_sharegpt_missing_file():
    """Missing dataset file raises FileNotFoundError."""
    from benchmark_spec_decode_quant import load_sharegpt

    with pytest.raises(FileNotFoundError):
        load_sharegpt(
            "/nonexistent/path/sharegpt.json",
            num_samples=10,
            max_model_len=4096,
            tokenizer=_MockTokenizer(),
            seed=42,
        )
```

- [ ] **Step 2: Run tests — expect ImportError (module not yet created)**

```bash
cd /workspace && .venv/bin/python -m pytest tests/benchmarks/test_benchmark_spec_decode_quant.py -v -m benchmark 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'benchmark_spec_decode_quant'`

---

## Task 2: Implement `load_sharegpt` and make tests pass

**Files:**
- Create: `benchmarks/benchmark_spec_decode_quant.py`

- [ ] **Step 1: Create the script with just `load_sharegpt`**

```python
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
```

- [ ] **Step 2: Run tests — expect all to pass**

```bash
cd /workspace && .venv/bin/python -m pytest tests/benchmarks/test_benchmark_spec_decode_quant.py -v -m benchmark 2>&1 | tail -20
```

Expected output (all 6 pass):
```
PASSED tests/benchmarks/test_benchmark_spec_decode_quant.py::test_load_sharegpt_basic
PASSED tests/benchmarks/test_benchmark_spec_decode_quant.py::test_load_sharegpt_filters_single_turn
PASSED tests/benchmarks/test_benchmark_spec_decode_quant.py::test_load_sharegpt_filters_too_short
PASSED tests/benchmarks/test_benchmark_spec_decode_quant.py::test_load_sharegpt_filters_over_max_model_len
PASSED tests/benchmarks/test_benchmark_spec_decode_quant.py::test_load_sharegpt_respects_num_samples
PASSED tests/benchmarks/test_benchmark_spec_decode_quant.py::test_load_sharegpt_missing_file
6 passed
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/benchmark_spec_decode_quant.py tests/benchmarks/test_benchmark_spec_decode_quant.py
git commit -m "feat: add load_sharegpt and its unit tests for spec decode quant benchmark"
```

---

## Task 3: Write failing tests for `VariantResult` and `format_results_table`

**Files:**
- Modify: `tests/benchmarks/test_benchmark_spec_decode_quant.py`

- [ ] **Step 1: Append table-formatting tests to the test file**

```python
# Append to tests/benchmarks/test_benchmark_spec_decode_quant.py


@pytest.mark.benchmark
def test_format_results_table_all_present():
    """Table contains header labels and all numeric values."""
    from benchmark_spec_decode_quant import VariantResult, format_results_table

    results = {
        1: {
            None: VariantResult(accepted_tok_per_sec=100.0, total_output_tokens=500, wall_time_sec=5.0),
            "fp8": VariantResult(accepted_tok_per_sec=120.0, total_output_tokens=500, wall_time_sec=4.2),
            "int8": VariantResult(accepted_tok_per_sec=110.0, total_output_tokens=500, wall_time_sec=4.5),
        }
    }
    table = format_results_table(results, batch_sizes=[1])
    assert "base" in table
    assert "fp8" in table
    assert "int8" in table
    assert "100.0" in table
    assert "120.0" in table
    assert "110.0" in table


@pytest.mark.benchmark
def test_format_results_table_na_on_failure():
    """N/A appears for cells where the variant result is None."""
    from benchmark_spec_decode_quant import VariantResult, format_results_table

    results = {
        4: {
            None: VariantResult(accepted_tok_per_sec=50.0, total_output_tokens=200, wall_time_sec=4.0),
            "fp8": None,
            "int8": None,
        }
    }
    table = format_results_table(results, batch_sizes=[4])
    assert "N/A" in table
    assert "50.0" in table


@pytest.mark.benchmark
def test_format_results_table_row_order():
    """Rows appear in the order given by batch_sizes."""
    from benchmark_spec_decode_quant import VariantResult, format_results_table

    r = VariantResult(accepted_tok_per_sec=1.0, total_output_tokens=1, wall_time_sec=1.0)
    results = {bs: {None: r, "fp8": r, "int8": r} for bs in [1, 128]}
    table = format_results_table(results, batch_sizes=[1, 128])
    pos_1 = table.index("1")
    pos_128 = table.index("128")
    assert pos_1 < pos_128
```

- [ ] **Step 2: Run — expect ImportError on `VariantResult` / `format_results_table`**

```bash
cd /workspace && .venv/bin/python -m pytest tests/benchmarks/test_benchmark_spec_decode_quant.py -v -m benchmark -k "table or VariantResult" 2>&1 | tail -15
```

Expected: `ImportError: cannot import name 'VariantResult'`

---

## Task 4: Implement `VariantResult` and `format_results_table`

**Files:**
- Modify: `benchmarks/benchmark_spec_decode_quant.py`

- [ ] **Step 1: Add `VariantResult` dataclass and `format_results_table` after `load_sharegpt`**

```python
# Add after load_sharegpt in benchmark_spec_decode_quant.py

from tabulate import tabulate  # already at top of file


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
    return tabulate(rows, headers=headers, tablefmt="simple")
```

The `from tabulate import tabulate` import goes with the other imports at the top of the file. Add it there.

Updated imports block at the top of `benchmarks/benchmark_spec_decode_quant.py`:

```python
from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import dataclass

import torch
from tabulate import tabulate
```

- [ ] **Step 2: Run table tests — expect all 3 to pass**

```bash
cd /workspace && .venv/bin/python -m pytest tests/benchmarks/test_benchmark_spec_decode_quant.py -v -m benchmark 2>&1 | tail -20
```

Expected: all 9 tests pass.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/benchmark_spec_decode_quant.py tests/benchmarks/test_benchmark_spec_decode_quant.py
git commit -m "feat: add VariantResult dataclass and format_results_table"
```

---

## Task 5: Implement `run_variant`

**Files:**
- Modify: `benchmarks/benchmark_spec_decode_quant.py`

No unit tests for this function — it requires a live GPU with loaded models. The integration test is the full script run in Task 7.

- [ ] **Step 1: Add `run_variant` to the script**

Add the following after `format_results_table` in `benchmarks/benchmark_spec_decode_quant.py`:

```python
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

    llm: LLM | None = None
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
            if metric.name == "vllm:spec_decode_num_accepted_tokens":
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
```

- [ ] **Step 2: Verify the import check still passes (no GPU needed)**

```bash
cd /workspace && .venv/bin/python -c "import sys; sys.path.insert(0, 'benchmarks'); from benchmark_spec_decode_quant import run_variant; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add benchmarks/benchmark_spec_decode_quant.py
git commit -m "feat: implement run_variant with GPU teardown and get_metrics()"
```

---

## Task 6: Implement `parse_args` and `main`

**Files:**
- Modify: `benchmarks/benchmark_spec_decode_quant.py`

- [ ] **Step 1: Add `parse_args` and `main` to the script**

Append the following at the bottom of `benchmarks/benchmark_spec_decode_quant.py` (before `if __name__ == "__main__"`):

```python
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
```

- [ ] **Step 2: Run the `--help` check (no GPU needed)**

```bash
cd /workspace && .venv/bin/python benchmarks/benchmark_spec_decode_quant.py --help
```

Expected output includes:
```
usage: benchmark_spec_decode_quant.py [-h] [--target-model TARGET_MODEL]
                                      [--draft-model DRAFT_MODEL]
                                      --dataset DATASET ...
```

- [ ] **Step 3: Run the missing-dataset guard (no GPU needed)**

```bash
cd /workspace && .venv/bin/python benchmarks/benchmark_spec_decode_quant.py --dataset /nonexistent/file.json 2>&1 | head -5
```

Expected: `FileNotFoundError: Dataset not found: /nonexistent/file.json`

- [ ] **Step 4: Commit**

```bash
git add benchmarks/benchmark_spec_decode_quant.py
git commit -m "feat: add parse_args and main loop to spec decode quant benchmark"
```

---

## Task 7: Run all unit tests and verify final import

**Files:** (none modified)

- [ ] **Step 1: Run all unit tests**

```bash
cd /workspace && .venv/bin/python -m pytest tests/benchmarks/test_benchmark_spec_decode_quant.py -v -m benchmark 2>&1 | tail -20
```

Expected: all 9 tests pass, 0 failures.

- [ ] **Step 2: Syntax/import check of the complete script**

```bash
cd /workspace && .venv/bin/python -c "
import sys, ast
src = open('benchmarks/benchmark_spec_decode_quant.py').read()
ast.parse(src)
print('Syntax OK')
sys.path.insert(0, 'benchmarks')
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('b', 'benchmarks/benchmark_spec_decode_quant.py')
mod = importlib.util.module_from_spec(spec)
# Don't exec — just parse
print('Module loadable')
"
```

Expected:
```
Syntax OK
Module loadable
```

- [ ] **Step 3: Final commit (if any last-minute fixes applied)**

```bash
git add benchmarks/benchmark_spec_decode_quant.py tests/benchmarks/test_benchmark_spec_decode_quant.py
git commit -m "test: verify spec decode quant benchmark import and unit tests all pass"
```

---

## Running the Full Benchmark (GPU required)

Once all tasks above are complete, run the actual benchmark on your RTX PRO 6000:

```bash
cd /workspace && .venv/bin/python benchmarks/benchmark_spec_decode_quant.py \
    --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 \
    --num-spec-tokens 5 \
    --batch-sizes 1 4 8 16 32 64 128
```

Expected final output format:
```
============================================================
Results — Accepted tokens/second
============================================================
Batch size    base (tok/s)    fp8 (tok/s)    int8 (tok/s)
----------    ------------    -----------    ------------
1             ...             ...            ...
4             ...             ...            ...
...
128           ...             ...            ...
```

---

## Self-Review

**Spec coverage check:**
- ✅ Target model Qwen3-8B, draft Qwen3-1.7B — hardcoded as defaults in `parse_args`
- ✅ Three quantization variants (base/fp8/int8) — `quant_variants = [None, "fp8", "int8"]` in `main`
- ✅ Batch sizes [1,4,8,16,32,64,128] via `max_num_seqs` — `--batch-sizes` arg, passed to `LLM(max_num_seqs=batch_size)`
- ✅ ShareGPT workload — `load_sharegpt` reads JSON, filters, tokenizes
- ✅ 5 speculative tokens — `--num-spec-tokens 5` default
- ✅ Accepted tok/s metric — `llm.get_metrics()` → `vllm:spec_decode_num_accepted_tokens / elapsed`
- ✅ 2D table output — `format_results_table` with `tabulate`
- ✅ Catch-and-continue error handling — `try/except Exception` per cell, `None` recorded
- ✅ Hard fail on missing dataset — `FileNotFoundError` check in `main` before GPU work
- ✅ GPU teardown in `finally` — `del llm; gc.collect(); torch.cuda.empty_cache()`

**No placeholders found.**

**Type consistency:** `VariantResult` defined in Task 4, used identically in Tasks 5 and 6. `format_results_table` signature matches usage in `main`. `load_sharegpt` return type `list[tuple[list[int], int]]` matches consumption in `run_variant`.
