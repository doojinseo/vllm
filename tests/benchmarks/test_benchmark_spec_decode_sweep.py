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


@pytest.mark.benchmark
def test_save_results_json_structure(tmp_path):
    import json as _json
    from benchmark_spec_decode_sweep import VariantResult, save_results
    def r(tok):
        return VariantResult(output_tok_per_sec=tok,
                             total_output_tokens=100, wall_time_sec=1.0)
    results = {
        5: {
            4:  {"int8": r(200.0), "fp8": r(190.0), "base": r(150.0), "draft_base": r(180.0)},
            32: {"int8": r(400.0), "fp8": r(410.0), "base": r(300.0), "draft_base": r(380.0)},
        }
    }
    crossover = {5: 32}
    config = {"target_model": "M/T", "spec_tokens": [5], "batch_sizes": [4, 32]}
    out = tmp_path / "out.json"

    save_results(
        str(out), config, results,
        ["base", "draft_base", "int8", "fp8"], [5], [4, 32], crossover,
    )

    data = _json.loads(out.read_text())
    assert data["config"]["target_model"] == "M/T"
    assert data["crossover"]["5"] == 32
    assert data["results"]["5"]["4"]["fp8"]["output_tok_per_sec"] == pytest.approx(190.0)
    assert data["results"]["5"]["32"]["int8"]["total_output_tokens"] == 100


@pytest.mark.benchmark
def test_save_results_none_variant(tmp_path):
    import json as _json
    from benchmark_spec_decode_sweep import VariantResult, save_results
    results = {
        5: {4: {"int8": VariantResult(200.0, 100, 1.0), "fp8": None}},
    }
    config = {"spec_tokens": [5], "batch_sizes": [4]}
    out = tmp_path / "out.json"
    save_results(str(out), config, results, ["int8", "fp8"], [5], [4], {5: None})
    data = _json.loads(out.read_text())
    assert data["results"]["5"]["4"]["fp8"] is None


@pytest.mark.benchmark
def test_plot_results_creates_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from benchmark_spec_decode_sweep import VariantResult, plot_results
    def r(tok):
        return VariantResult(output_tok_per_sec=tok,
                             total_output_tokens=100, wall_time_sec=1.0)
    results = {
        5: {
            4:  {"base": r(150.0), "draft_base": r(180.0),
                 "int8": r(200.0), "fp8": r(190.0)},
            32: {"base": r(300.0), "draft_base": r(360.0),
                 "int8": r(400.0), "fp8": r(410.0)},
        }
    }
    out = tmp_path / "plot.png"
    plot_results(
        str(out), results,
        ["base", "draft_base", "int8", "fp8"],
        [5], [4, 32], crossover={5: 32},
    )
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.benchmark
def test_parse_args_defaults(tmp_path):
    from benchmark_spec_decode_sweep import parse_args
    fake = tmp_path / "sg.json"
    fake.write_text("[]")
    args = parse_args(["--dataset", str(fake)])
    assert args.spec_tokens == [3, 5, 7, 9]
    assert args.batch_sizes == [1, 4, 8, 16, 32, 64, 128]
    assert args.num_prompts == 500
    assert args.max_model_len == 4096
    assert args.seed == 42
