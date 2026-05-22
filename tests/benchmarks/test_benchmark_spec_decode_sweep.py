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


@pytest.mark.benchmark
def test_find_crossover_fp8_wins_at_bs32():
    from benchmark_spec_decode_sweep import VariantResult, find_crossover
    def r(tok):
        return VariantResult(output_tok_per_sec=tok,
                             total_output_tokens=100, wall_time_sec=1.0)
    results = {
        5: {
            4:  {"int8": r(200.0), "fp8": r(190.0)},
            16: {"int8": r(300.0), "fp8": r(295.0)},
            32: {"int8": r(400.0), "fp8": r(410.0)},
            64: {"int8": r(500.0), "fp8": r(530.0)},
        }
    }
    crossover = find_crossover(results, [4, 16, 32, 64], [5])
    assert crossover[5] == 32


@pytest.mark.benchmark
def test_find_crossover_never_crosses():
    from benchmark_spec_decode_sweep import VariantResult, find_crossover
    def r(tok):
        return VariantResult(output_tok_per_sec=tok,
                             total_output_tokens=100, wall_time_sec=1.0)
    results = {
        5: {
            4:  {"int8": r(200.0), "fp8": r(190.0)},
            64: {"int8": r(400.0), "fp8": r(390.0)},
        }
    }
    crossover = find_crossover(results, [4, 64], [5])
    assert crossover[5] is None


@pytest.mark.benchmark
def test_find_crossover_missing_variant():
    from benchmark_spec_decode_sweep import VariantResult, find_crossover
    def r(tok):
        return VariantResult(output_tok_per_sec=tok,
                             total_output_tokens=100, wall_time_sec=1.0)
    # fp8 result is None (run failed) — should skip that batch size
    results = {
        5: {
            4:  {"int8": r(200.0), "fp8": None},
            32: {"int8": r(400.0), "fp8": r(410.0)},
        }
    }
    crossover = find_crossover(results, [4, 32], [5])
    assert crossover[5] == 32
