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
    lines = [line for line in table.splitlines() if line.strip()]
    batch_size_col = [line.split()[0] for line in lines if line.split()]
    assert "1" in batch_size_col
    assert "128" in batch_size_col
    assert batch_size_col.index("1") < batch_size_col.index("128")
