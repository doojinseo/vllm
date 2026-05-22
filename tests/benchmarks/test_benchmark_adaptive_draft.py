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
