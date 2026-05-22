# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for benchmark_spec_decode_sweep.py (pure-Python parts only).

These tests do not require a GPU or real models.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))


@pytest.mark.benchmark
def test_variant_result_fields():
    from benchmark_spec_decode_sweep import VariantResult
    r = VariantResult(output_tok_per_sec=234.5, total_output_tokens=1000,
                      wall_time_sec=4.26)
    assert r.output_tok_per_sec == pytest.approx(234.5)
    assert r.total_output_tokens == 1000
    assert r.wall_time_sec == pytest.approx(4.26)
