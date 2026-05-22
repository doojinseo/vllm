# Adaptive Draft Wave Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `benchmarks/benchmark_adaptive_draft.py` — a wave-based benchmark that runs four draft-model variants (base/int8/fp8/adaptive) through alternating small/large request waves, writes JSON results, and generates a two-panel plot.

**Architecture:** A single long-lived LLM instance per variant runs `num_wave_pairs * 2` alternating waves (even=small, odd=large). All variants receive identical pre-sampled prompts per wave. After all variants complete, results are written to JSON and a two-panel matplotlib PNG is saved.

**Tech Stack:** Python, vLLM (`LLM`, `SamplingParams`, `get_metrics()`), tabulate, matplotlib, pytest

---

## File Map

| Path | Action | Responsibility |
|------|--------|----------------|
| `benchmarks/benchmark_adaptive_draft.py` | Create | All benchmark logic |
| `tests/benchmarks/test_benchmark_adaptive_draft.py` | Create | Unit tests for pure-Python functions |

---

## Task 1: Test scaffold + WaveResult + VariantSummary dataclasses

**Files:**
- Create: `benchmarks/benchmark_adaptive_draft.py`
- Create: `tests/benchmarks/test_benchmark_adaptive_draft.py`

- [ ] **Step 1: Create the test file scaffold**

```python
# tests/benchmarks/test_benchmark_adaptive_draft.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for benchmark_adaptive_draft.py (pure-Python parts only).

These tests do not require a GPU or real models.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))


class _MockTokenizer:
    """Minimal tokenizer stub: splits on whitespace, returns word indices."""

    def __call__(self, text: str):
        tokens = text.split()
        return type("Enc", (), {"input_ids": list(range(len(tokens)))})()


@pytest.mark.benchmark
def test_wave_result_fields():
    from benchmark_adaptive_draft import WaveResult
    r = WaveResult(index=0, type="small", batch=4,
                   accepted_tok_per_sec=142.3, wall_time_sec=1.5)
    assert r.index == 0
    assert r.type == "small"
    assert r.batch == 4
    assert r.accepted_tok_per_sec == 142.3
    assert r.wall_time_sec == 1.5


@pytest.mark.benchmark
def test_variant_summary_fields():
    from benchmark_adaptive_draft import VariantSummary
    s = VariantSummary(small_avg=100.0, large_avg=300.0, overall=200.0)
    assert s.small_avg == 100.0
    assert s.large_avg == 300.0
    assert s.overall == 200.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark_adaptive_draft'`

- [ ] **Step 3: Create the benchmark file with just the dataclasses**

```python
# benchmarks/benchmark_adaptive_draft.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py tests/benchmarks/test_benchmark_adaptive_draft.py
git commit -m "feat: scaffold wave benchmark with WaveResult and VariantSummary"
```

---

## Task 2: load_sharegpt

**Files:**
- Modify: `benchmarks/benchmark_adaptive_draft.py`
- Modify: `tests/benchmarks/test_benchmark_adaptive_draft.py`

- [ ] **Step 1: Add the test**

Append to `tests/benchmarks/test_benchmark_adaptive_draft.py`:

```python
@pytest.mark.benchmark
def test_load_sharegpt_basic(tmp_path):
    import json as _json
    from benchmark_adaptive_draft import load_sharegpt
    data = [{"conversations": [
        {"value": " ".join(["word"] * 10)},
        {"value": " ".join(["word"] * 10)},
    ]}]
    path = tmp_path / "sg.json"
    path.write_text(_json.dumps(data))
    result = load_sharegpt(str(path), num_samples=10, max_model_len=4096,
                           tokenizer=_MockTokenizer(), seed=42)
    assert len(result) == 1
    prompt_ids, output_len = result[0]
    assert len(prompt_ids) == 10
    assert output_len == 10


@pytest.mark.benchmark
def test_load_sharegpt_filters_short(tmp_path):
    import json as _json
    from benchmark_adaptive_draft import load_sharegpt
    data = [{"conversations": [{"value": "hi"}, {"value": "ok"}]}]
    path = tmp_path / "sg.json"
    path.write_text(_json.dumps(data))
    result = load_sharegpt(str(path), num_samples=10, max_model_len=4096,
                           tokenizer=_MockTokenizer(), seed=42)
    assert result == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py::test_load_sharegpt_basic tests/benchmarks/test_benchmark_adaptive_draft.py::test_load_sharegpt_filters_short -v
```

Expected: `ImportError` or `AttributeError` — `load_sharegpt` not defined.

- [ ] **Step 3: Add load_sharegpt to the benchmark file**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
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
```

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py tests/benchmarks/test_benchmark_adaptive_draft.py
git commit -m "feat: add load_sharegpt to wave benchmark"
```

---

## Task 3: pre_sample_waves

**Files:**
- Modify: `benchmarks/benchmark_adaptive_draft.py`
- Modify: `tests/benchmarks/test_benchmark_adaptive_draft.py`

- [ ] **Step 1: Add the test**

Append to `tests/benchmarks/test_benchmark_adaptive_draft.py`:

```python
@pytest.mark.benchmark
def test_pre_sample_waves_count_and_sizes(tmp_path):
    import json as _json
    from benchmark_adaptive_draft import pre_sample_waves

    # 100 entries, each with 10 tokens prompt + 10 tokens completion
    data = [{"conversations": [
        {"value": " ".join(["word"] * 10)},
        {"value": " ".join(["word"] * 10)},
    ]} for _ in range(100)]
    path = tmp_path / "sg.json"
    path.write_text(_json.dumps(data))

    waves = pre_sample_waves(
        dataset_path=str(path),
        small_batch=4,
        large_batch=16,
        num_wave_pairs=3,
        max_model_len=4096,
        tokenizer=_MockTokenizer(),
        seed=42,
    )
    # 3 pairs = 6 waves total
    assert len(waves) == 6
    # Even indices → small (up to 4 prompts), odd → large (up to 16 prompts)
    for i, wave in enumerate(waves):
        expected_max = 4 if i % 2 == 0 else 16
        assert len(wave) <= expected_max
        assert len(wave) > 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py::test_pre_sample_waves_count_and_sizes -v
```

Expected: `ImportError` — `pre_sample_waves` not defined.

- [ ] **Step 3: Add pre_sample_waves to the benchmark file**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py tests/benchmarks/test_benchmark_adaptive_draft.py
git commit -m "feat: add pre_sample_waves to wave benchmark"
```

---

## Task 4: compute_summary

**Files:**
- Modify: `benchmarks/benchmark_adaptive_draft.py`
- Modify: `tests/benchmarks/test_benchmark_adaptive_draft.py`

- [ ] **Step 1: Add the test**

Append to `tests/benchmarks/test_benchmark_adaptive_draft.py`:

```python
@pytest.mark.benchmark
def test_compute_summary():
    from benchmark_adaptive_draft import WaveResult, compute_summary

    waves = [
        WaveResult(0, "small", 4,  100.0, 1.0),
        WaveResult(1, "large", 32, 300.0, 2.0),
        WaveResult(2, "small", 4,  120.0, 1.0),
        WaveResult(3, "large", 32, 280.0, 2.0),
    ]
    s = compute_summary(waves)
    assert s.small_avg == pytest.approx(110.0)
    assert s.large_avg == pytest.approx(290.0)
    assert s.overall  == pytest.approx(200.0)


@pytest.mark.benchmark
def test_compute_summary_only_small():
    from benchmark_adaptive_draft import WaveResult, compute_summary

    waves = [WaveResult(0, "small", 4, 100.0, 1.0)]
    s = compute_summary(waves)
    assert s.small_avg == 100.0
    assert s.large_avg == 0.0
    assert s.overall   == 100.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py::test_compute_summary tests/benchmarks/test_benchmark_adaptive_draft.py::test_compute_summary_only_small -v
```

Expected: `ImportError` — `compute_summary` not defined.

- [ ] **Step 3: Add compute_summary to the benchmark file**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
def compute_summary(wave_results: list[WaveResult]) -> VariantSummary:
    small = [r.accepted_tok_per_sec for r in wave_results if r.type == "small"]
    large = [r.accepted_tok_per_sec for r in wave_results if r.type == "large"]
    all_vals = [r.accepted_tok_per_sec for r in wave_results]
    return VariantSummary(
        small_avg=sum(small) / len(small) if small else 0.0,
        large_avg=sum(large) / len(large) if large else 0.0,
        overall=sum(all_vals) / len(all_vals) if all_vals else 0.0,
    )
```

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py tests/benchmarks/test_benchmark_adaptive_draft.py
git commit -m "feat: add compute_summary to wave benchmark"
```

---

## Task 5: format_wave_table + format_summary_table

**Files:**
- Modify: `benchmarks/benchmark_adaptive_draft.py`
- Modify: `tests/benchmarks/test_benchmark_adaptive_draft.py`

- [ ] **Step 1: Add the tests**

Append to `tests/benchmarks/test_benchmark_adaptive_draft.py`:

```python
LABELS = ["fp8", "int8", "adaptive"]


@pytest.mark.benchmark
def test_format_wave_table_headers_and_values():
    from benchmark_adaptive_draft import WaveResult, format_wave_table

    all_results = {
        "fp8":      [WaveResult(0, "small", 4, 130.0, 1.0),
                     WaveResult(1, "large", 32, 330.0, 2.0)],
        "int8":     [WaveResult(0, "small", 4, 155.0, 1.0),
                     WaveResult(1, "large", 32, 275.0, 2.0)],
        "adaptive": [WaveResult(0, "small", 4, 154.0, 1.0),
                     WaveResult(1, "large", 32, 329.0, 2.0)],
    }
    table = format_wave_table(all_results, LABELS)
    assert "small" in table
    assert "large" in table
    assert "130.0" in table
    assert "329.0" in table


@pytest.mark.benchmark
def test_format_wave_table_row_order():
    from benchmark_adaptive_draft import WaveResult, format_wave_table

    r = WaveResult(0, "small", 4, 1.0, 1.0)
    r2 = WaveResult(1, "large", 32, 2.0, 1.0)
    all_results = {"fp8": [r, r2], "int8": [r, r2], "adaptive": [r, r2]}
    table = format_wave_table(all_results, LABELS)
    lines = [l for l in table.splitlines() if l.strip() and not l.strip().startswith("-")]
    idx_col = [l.split()[0] for l in lines if l.split()[0].isdigit()]
    assert idx_col.index("0") < idx_col.index("1")


@pytest.mark.benchmark
def test_format_summary_table_contains_all_variants():
    from benchmark_adaptive_draft import VariantSummary, format_summary_table

    summaries = {
        "fp8":      VariantSummary(130.0, 330.0, 230.0),
        "int8":     VariantSummary(155.0, 275.0, 215.0),
        "adaptive": VariantSummary(154.0, 329.0, 241.5),
    }
    table = format_summary_table(summaries, LABELS)
    for lbl in LABELS:
        assert lbl in table
    assert "130.0" in table
    assert "241.5" in table
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py::test_format_wave_table_headers_and_values tests/benchmarks/test_benchmark_adaptive_draft.py::test_format_wave_table_row_order tests/benchmarks/test_benchmark_adaptive_draft.py::test_format_summary_table_contains_all_variants -v
```

Expected: `ImportError` — functions not defined.

- [ ] **Step 3: Add both formatting functions to the benchmark file**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
def format_wave_table(
    all_wave_results: dict[str, list[WaveResult]],
    variant_labels: list[str],
) -> str:
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
    from tabulate import tabulate
    headers = ["Variant", "Small-wave avg", "Large-wave avg", "Overall avg"]
    rows = [
        [lbl, f"{summaries[lbl].small_avg:.1f}",
         f"{summaries[lbl].large_avg:.1f}",
         f"{summaries[lbl].overall:.1f}"]
        for lbl in variant_labels
    ]
    return tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)
```

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py tests/benchmarks/test_benchmark_adaptive_draft.py
git commit -m "feat: add format_wave_table and format_summary_table to wave benchmark"
```

---

## Task 6: save_results

**Files:**
- Modify: `benchmarks/benchmark_adaptive_draft.py`
- Modify: `tests/benchmarks/test_benchmark_adaptive_draft.py`

- [ ] **Step 1: Add the test**

Append to `tests/benchmarks/test_benchmark_adaptive_draft.py`:

```python
@pytest.mark.benchmark
def test_save_results_json_structure(tmp_path):
    import json as _json
    from benchmark_adaptive_draft import WaveResult, VariantSummary, save_results

    all_wave_results = {
        "fp8": [WaveResult(0, "small", 4, 130.0, 1.0),
                WaveResult(1, "large", 32, 330.0, 2.0)],
        "int8": [WaveResult(0, "small", 4, 155.0, 1.0),
                 WaveResult(1, "large", 32, 275.0, 2.0)],
    }
    summaries = {
        "fp8":  VariantSummary(130.0, 330.0, 230.0),
        "int8": VariantSummary(155.0, 275.0, 215.0),
    }
    config = {"small_batch": 4, "large_batch": 32, "num_wave_pairs": 1}
    out = tmp_path / "results.json"

    save_results(str(out), config, all_wave_results, summaries, ["fp8", "int8"])

    data = _json.loads(out.read_text())
    assert data["config"]["small_batch"] == 4
    assert len(data["waves"]) == 2
    assert data["waves"][0]["type"] == "small"
    assert data["waves"][0]["fp8"] == pytest.approx(130.0)
    assert data["waves"][1]["int8"] == pytest.approx(275.0)
    assert data["summary"]["fp8"]["small_avg"] == pytest.approx(130.0)
    assert data["summary"]["int8"]["overall"] == pytest.approx(215.0)
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py::test_save_results_json_structure -v
```

Expected: `ImportError` — `save_results` not defined.

- [ ] **Step 3: Add save_results to the benchmark file**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
def save_results(
    output_path: str,
    config: dict,
    all_wave_results: dict[str, list[WaveResult]],
    summaries: dict[str, VariantSummary],
    variant_labels: list[str],
) -> None:
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py tests/benchmarks/test_benchmark_adaptive_draft.py
git commit -m "feat: add save_results to wave benchmark"
```

---

## Task 7: plot_results

**Files:**
- Modify: `benchmarks/benchmark_adaptive_draft.py`
- Modify: `tests/benchmarks/test_benchmark_adaptive_draft.py`

- [ ] **Step 1: Add the smoke test**

Append to `tests/benchmarks/test_benchmark_adaptive_draft.py`:

```python
@pytest.mark.benchmark
def test_plot_results_creates_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")  # non-interactive, no display required
    from benchmark_adaptive_draft import WaveResult, VariantSummary, plot_results

    labels = ["fp8", "int8", "adaptive"]
    all_wave_results = {
        lbl: [
            WaveResult(0, "small", 4,  130.0, 1.0),
            WaveResult(1, "large", 32, 330.0, 2.0),
        ]
        for lbl in labels
    }
    summaries = {lbl: VariantSummary(130.0, 330.0, 230.0) for lbl in labels}
    out = tmp_path / "plot.png"

    plot_results(str(out), all_wave_results, summaries, labels)

    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py::test_plot_results_creates_file -v
```

Expected: `ImportError` — `plot_results` not defined.

- [ ] **Step 3: Add plot_results to the benchmark file**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
def plot_results(
    plot_path: str,
    all_wave_results: dict[str, list[WaveResult]],
    summaries: dict[str, VariantSummary],
    variant_labels: list[str],
) -> None:
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py tests/benchmarks/test_benchmark_adaptive_draft.py
git commit -m "feat: add plot_results to wave benchmark"
```

---

## Task 8: parse_args

**Files:**
- Modify: `benchmarks/benchmark_adaptive_draft.py`
- Modify: `tests/benchmarks/test_benchmark_adaptive_draft.py`

- [ ] **Step 1: Add the test**

Append to `tests/benchmarks/test_benchmark_adaptive_draft.py`:

```python
@pytest.mark.benchmark
def test_parse_args_defaults(tmp_path):
    from benchmark_adaptive_draft import parse_args

    fake_dataset = tmp_path / "sg.json"
    fake_dataset.write_text("[]")

    args = parse_args(["--dataset", str(fake_dataset)])
    assert args.small_batch == 4
    assert args.large_batch == 32
    assert args.num_wave_pairs == 4
    assert args.num_spec_tokens == 5
    assert args.threshold == 8
    assert args.ema_alpha == pytest.approx(0.1)
    assert args.seed == 42
    assert args.output == "adaptive_draft_wave_results.json"
    assert args.plot is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py::test_parse_args_defaults -v
```

Expected: `ImportError` — `parse_args` not defined.

- [ ] **Step 3: Add parse_args to the benchmark file**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--num-wave-pairs", type=int,   default=4)
    parser.add_argument("--num-spec-tokens",type=int,   default=5)
    parser.add_argument("--threshold",      type=int,   default=8)
    parser.add_argument("--ema-alpha",      type=float, default=0.1)
    parser.add_argument("--max-model-len",  type=int,   default=4096)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output", default="adaptive_draft_wave_results.json")
    parser.add_argument("--plot",   default=None,
                        help="Plot path (default: --output stem + .png)")
    return parser.parse_args(argv)
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py tests/benchmarks/test_benchmark_adaptive_draft.py
git commit -m "feat: add parse_args to wave benchmark"
```

---

## Task 9: build_llm + run_variant_waves + main (integration)

These functions require a GPU and real model checkpoints; no unit tests. They are validated by running the benchmark end-to-end.

**Files:**
- Modify: `benchmarks/benchmark_adaptive_draft.py`

- [ ] **Step 1: Add _clear_vllm_compile_cache**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
def _clear_vllm_compile_cache() -> None:
    cache_dir = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
```

- [ ] **Step 2: Add build_llm**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
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
```

- [ ] **Step 3: Add run_variant_waves**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
def run_variant_waves(
    llm: "LLM",
    waves: list[list[tuple[list[int], int]]],
    variant: str,
) -> list[WaveResult]:
    from vllm import SamplingParams
    from vllm.v1.metrics.reader import Counter as VllmCounter

    results: list[WaveResult] = []
    prev_accepted = 0

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

        tok_per_sec = delta / elapsed if elapsed > 0 else 0.0
        results.append(WaveResult(
            index=i, type=wave_type, batch=len(wave_prompts),
            accepted_tok_per_sec=tok_per_sec, wall_time_sec=elapsed,
        ))
        print(f"  [{variant}] wave {i} ({wave_type}, bs={len(wave_prompts)}): "
              f"{tok_per_sec:.1f} tok/s  ({elapsed:.1f}s)")

    return results
```

- [ ] **Step 4: Add main**

Append to `benchmarks/benchmark_adaptive_draft.py`:

```python
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
        "small_batch":    args.small_batch,
        "large_batch":    args.large_batch,
        "num_wave_pairs": args.num_wave_pairs,
        "num_spec_tokens":args.num_spec_tokens,
        "threshold":      args.threshold,
        "ema_alpha":      args.ema_alpha,
        "target_model":   args.target_model,
    }
    save_results(args.output, config, all_wave_results, summaries, variant_labels)
    print(f"\nResults saved to {args.output}")

    plot_path = args.plot or str(Path(args.output).with_suffix(".png"))
    plot_results(plot_path, all_wave_results, summaries, variant_labels)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run unit tests one final time**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_benchmark_adaptive_draft.py -v
```

Expected: `13 passed`

- [ ] **Step 6: Commit**

```bash
git add benchmarks/benchmark_adaptive_draft.py
git commit -m "feat: complete benchmark_adaptive_draft.py with GPU integration path"
```

---

## Running the Benchmark

```bash
.venv/bin/python benchmarks/benchmark_adaptive_draft.py \
    --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --small-batch 4 \
    --large-batch 32 \
    --num-wave-pairs 4 \
    --output results/adaptive_wave_results.json
```

The plot is saved automatically to `results/adaptive_wave_results.png`.

**Success criteria to verify manually:**
- `adaptive` accepted tok/s on small waves ≈ `int8` values
- `adaptive` accepted tok/s on large waves ≈ `fp8` values
- `adaptive` overall average exceeds both `int8` and `fp8` overall averages
