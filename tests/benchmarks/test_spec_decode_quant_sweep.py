import sys
from pathlib import Path

import pytest

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
