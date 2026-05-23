# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for benchmark_adaptive_online.py (pure-Python parts only).

These tests do not require a GPU, a real vLLM server, or network access.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))


class _MockTokenizer:
    def __call__(self, text: str):
        tokens = text.split()
        return type("Enc", (), {"input_ids": list(range(len(tokens)))})()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_request_result_fields():
    from benchmark_adaptive_online import RequestResult
    r = RequestResult(
        prompt_len=10, output_tokens=20,
        ttft=0.15, itl=[0.05, 0.04], e2el=1.2,
        success=True,
    )
    assert r.prompt_len == 10
    assert r.output_tokens == 20
    assert r.ttft == pytest.approx(0.15)
    assert r.itl == [0.05, 0.04]
    assert r.e2el == pytest.approx(1.2)
    assert r.success is True
    assert r.error == ""


@pytest.mark.benchmark
def test_online_metrics_fields():
    from benchmark_adaptive_online import OnlineMetrics
    m = OnlineMetrics(
        completed=100, failed=2,
        output_throughput=500.0, request_goodput=0.83,
        ttft_p50_ms=120.0, ttft_p99_ms=400.0,
        itl_p50_ms=30.0, itl_p99_ms=80.0,
        e2el_p50_ms=2000.0, e2el_p99_ms=6000.0,
    )
    assert m.completed == 100
    assert m.failed == 2
    assert m.ttft_p50_ms == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# make_spec_config
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_make_spec_config_base_is_none():
    from benchmark_adaptive_online import make_spec_config
    cfg = make_spec_config(
        variant="base",
        draft_model_base="M/base", draft_model_fp8="M/fp8", draft_model_int8="M/int8",
        num_spec_tokens=5, threshold=8, low_threshold=4, ema_alpha=0.3,
    )
    assert cfg is None


@pytest.mark.benchmark
def test_make_spec_config_draft_base():
    from benchmark_adaptive_online import make_spec_config
    cfg = make_spec_config(
        variant="draft_base",
        draft_model_base="M/base", draft_model_fp8="M/fp8", draft_model_int8="M/int8",
        num_spec_tokens=5, threshold=8, low_threshold=4, ema_alpha=0.3,
    )
    assert cfg is not None
    assert cfg["model"] == "M/base"
    assert cfg["num_speculative_tokens"] == 5
    assert "alt_model" not in cfg


@pytest.mark.benchmark
def test_make_spec_config_fp8():
    from benchmark_adaptive_online import make_spec_config
    cfg = make_spec_config(
        variant="fp8",
        draft_model_base="M/base", draft_model_fp8="M/fp8", draft_model_int8="M/int8",
        num_spec_tokens=5, threshold=8, low_threshold=4, ema_alpha=0.3,
    )
    assert cfg is not None
    assert cfg["model"] == "M/fp8"
    assert cfg["num_speculative_tokens"] == 5
    assert "alt_model" not in cfg


@pytest.mark.benchmark
def test_make_spec_config_int8():
    from benchmark_adaptive_online import make_spec_config
    cfg = make_spec_config(
        variant="int8",
        draft_model_base="M/base", draft_model_fp8="M/fp8", draft_model_int8="M/int8",
        num_spec_tokens=5, threshold=8, low_threshold=4, ema_alpha=0.3,
    )
    assert cfg is not None
    assert cfg["model"] == "M/int8"
    assert "alt_model" not in cfg


@pytest.mark.benchmark
def test_make_spec_config_adaptive():
    from benchmark_adaptive_online import make_spec_config
    cfg = make_spec_config(
        variant="adaptive",
        draft_model_base="M/base", draft_model_fp8="M/fp8", draft_model_int8="M/int8",
        num_spec_tokens=5, threshold=8, low_threshold=4, ema_alpha=0.3,
    )
    assert cfg is not None
    assert cfg["model"] == "M/fp8"
    assert cfg["alt_model"] == "M/int8"
    assert cfg["adaptive_threshold"] == 8
    assert cfg["adaptive_low_threshold"] == 4
    assert cfg["adaptive_ema_alpha"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# build_serve_cmd
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_build_serve_cmd_base():
    from benchmark_adaptive_online import build_serve_cmd
    cmd = build_serve_cmd(
        target_model="M/target",
        max_num_seqs=64,
        max_model_len=4096,
        port=8001,
        spec_config=None,
    )
    assert "--model" in cmd
    assert "M/target" in cmd
    assert "--port" in cmd
    assert "8001" in cmd
    assert "--speculative-config" not in cmd


@pytest.mark.benchmark
def test_build_serve_cmd_with_spec():
    from benchmark_adaptive_online import build_serve_cmd
    spec = {"method": "draft_model", "model": "M/fp8", "num_speculative_tokens": 5}
    cmd = build_serve_cmd(
        target_model="M/target",
        max_num_seqs=64,
        max_model_len=4096,
        port=8000,
        spec_config=spec,
    )
    assert "--speculative-config" in cmd
    idx = cmd.index("--speculative-config")
    parsed = json.loads(cmd[idx + 1])
    assert parsed["model"] == "M/fp8"


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_compute_metrics_empty():
    from benchmark_adaptive_online import compute_metrics
    m = compute_metrics([], duration=120.0)
    assert m.completed == 0
    assert m.failed == 0
    assert m.output_throughput == 0.0
    assert m.ttft_p50_ms == 0.0


@pytest.mark.benchmark
def test_compute_metrics_all_failed():
    from benchmark_adaptive_online import RequestResult, compute_metrics
    results = [
        RequestResult(0, 0, 0, [], 1.0, success=False, error="timeout"),
        RequestResult(0, 0, 0, [], 1.0, success=False, error="timeout"),
    ]
    m = compute_metrics(results, duration=10.0)
    assert m.completed == 0
    assert m.failed == 2
    assert m.output_throughput == 0.0


@pytest.mark.benchmark
def test_compute_metrics_basic():
    from benchmark_adaptive_online import RequestResult, compute_metrics
    results = [
        RequestResult(prompt_len=10, output_tokens=50,
                      ttft=0.1, itl=[0.05, 0.05, 0.05], e2el=1.0, success=True),
        RequestResult(prompt_len=10, output_tokens=100,
                      ttft=0.2, itl=[0.04, 0.04], e2el=2.0, success=True),
    ]
    m = compute_metrics(results, duration=10.0)
    assert m.completed == 2
    assert m.failed == 0
    assert m.output_throughput == pytest.approx(15.0)   # 150 tok / 10s
    assert m.request_goodput == pytest.approx(0.2)       # 2 req / 10s
    assert 100.0 <= m.ttft_p50_ms <= 200.0
    assert m.itl_p50_ms > 0


@pytest.mark.benchmark
def test_compute_metrics_mixed_success():
    from benchmark_adaptive_online import RequestResult, compute_metrics
    results = [
        RequestResult(0, 50, 0.1, [0.05], 1.0, success=True),
        RequestResult(0, 0, 0, [], 0.5, success=False, error="err"),
    ]
    m = compute_metrics(results, duration=5.0)
    assert m.completed == 1
    assert m.failed == 1
    assert m.output_throughput == pytest.approx(10.0)   # 50 tok / 5s


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_format_results_table_contains_variants():
    from benchmark_adaptive_online import OnlineMetrics, format_results_table
    metrics = {
        "fp8": OnlineMetrics(100, 0, 500.0, 0.83, 120.0, 400.0, 30.0, 80.0, 2000.0, 6000.0),
        "adaptive": OnlineMetrics(98, 2, 510.0, 0.81, 115.0, 380.0, 28.0, 75.0, 1900.0, 5500.0),
    }
    table = format_results_table(metrics, ["fp8", "adaptive"])
    assert "fp8" in table
    assert "adaptive" in table
    assert "120.0" in table


@pytest.mark.benchmark
def test_format_results_table_missing_variant():
    from benchmark_adaptive_online import OnlineMetrics, format_results_table
    metrics = {
        "fp8": OnlineMetrics(50, 0, 200.0, 0.4, 100.0, 300.0, 20.0, 60.0, 1500.0, 4000.0),
    }
    table = format_results_table(metrics, ["fp8", "int8"])
    assert "int8" in table
    assert "N/A" in table


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_save_results_json_structure(tmp_path):
    from benchmark_adaptive_online import OnlineMetrics, save_results
    metrics = {
        "fp8": OnlineMetrics(100, 0, 500.0, 0.83, 120.0, 400.0, 30.0, 80.0, 2000.0, 6000.0),
        "adaptive": OnlineMetrics(98, 2, 510.0, 0.81, 115.0, 380.0, 28.0, 75.0, 1900.0, 5500.0),
    }
    config = {"request_rate": 4.0, "duration": 120.0}
    out = tmp_path / "results.json"

    save_results(str(out), config, metrics, ["fp8", "adaptive"])

    data = json.loads(out.read_text())
    assert data["config"]["request_rate"] == 4.0
    assert "fp8" in data["results"]
    assert "adaptive" in data["results"]
    assert data["results"]["fp8"]["completed"] == 100
    assert data["results"]["adaptive"]["failed"] == 2
    assert data["results"]["fp8"]["ttft_p50_ms"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# load_sharegpt
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_load_sharegpt_basic(tmp_path):
    data = [{"conversations": [
        {"value": " ".join(["word"] * 10)},
        {"value": " ".join(["word"] * 10)},
    ]}]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))
    from benchmark_adaptive_online import load_sharegpt
    result = load_sharegpt(str(path), num_samples=10, max_model_len=4096,
                           tokenizer=_MockTokenizer(), seed=42)
    assert len(result) == 1
    prompt_text, prompt_len, output_len = result[0]
    assert isinstance(prompt_text, str)
    assert prompt_len == 10
    assert output_len == 10


@pytest.mark.benchmark
def test_load_sharegpt_filters_short(tmp_path):
    data = [{"conversations": [{"value": "hi"}, {"value": "ok"}]}]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))
    from benchmark_adaptive_online import load_sharegpt
    result = load_sharegpt(str(path), num_samples=10, max_model_len=4096,
                           tokenizer=_MockTokenizer(), seed=42)
    assert result == []


@pytest.mark.benchmark
def test_load_sharegpt_filters_overlong(tmp_path):
    data = [{"conversations": [
        {"value": " ".join(["word"] * 50)},
        {"value": " ".join(["word"] * 50)},
    ]}]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))
    from benchmark_adaptive_online import load_sharegpt
    result = load_sharegpt(str(path), num_samples=10, max_model_len=80,
                           tokenizer=_MockTokenizer(), seed=42)
    assert result == []


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_parse_args_defaults(tmp_path):
    from benchmark_adaptive_online import parse_args
    fake = tmp_path / "sg.json"
    fake.write_text("[]")
    args = parse_args(["--dataset", str(fake)])
    assert args.request_rate == pytest.approx(4.0)
    assert args.duration == pytest.approx(120.0)
    assert args.warmup == pytest.approx(30.0)
    assert args.num_prompts == 500
    assert args.max_num_seqs == 128
    assert args.threshold == 16
    assert args.low_threshold == 8
    assert args.ema_alpha == pytest.approx(0.3)
    assert args.seed == 42
    assert args.port == 8000
    assert args.output == "results/adaptive_online_results.json"
    assert set(args.variants) == {"base", "draft_base", "int8", "fp8", "adaptive"}


@pytest.mark.benchmark
def test_parse_args_custom_rate(tmp_path):
    from benchmark_adaptive_online import parse_args
    fake = tmp_path / "sg.json"
    fake.write_text("[]")
    args = parse_args(["--dataset", str(fake), "--request-rate", "8.0",
                       "--duration", "60", "--variants", "fp8", "adaptive"])
    assert args.request_rate == pytest.approx(8.0)
    assert args.duration == pytest.approx(60.0)
    assert args.variants == ["fp8", "adaptive"]
