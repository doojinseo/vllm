import sys
from pathlib import Path

import pytest
import csv
import math
import tempfile
import json as _json
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))
from benchmark_spec_decode_quant_sweep import build_cmd, OUTPUT_LEN, write_csv, CSV_COLUMNS, plot_results, run_one  # noqa: E402


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


def _make_fake_subprocess(tmp_path, variant, batch_size, elapsed=10.0,
                           num_requests=None):
    if num_requests is None:
        num_requests = max(256, batch_size * 4)

    def fake_run(cmd, capture_output, text):
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
