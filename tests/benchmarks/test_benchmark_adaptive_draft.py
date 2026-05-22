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


@pytest.mark.benchmark
def test_pre_sample_waves_count_and_sizes(tmp_path):
    import json as _json
    from benchmark_adaptive_draft import pre_sample_waves

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
    assert len(waves) == 6
    for i, wave in enumerate(waves):
        expected_max = 4 if i % 2 == 0 else 16
        assert len(wave) <= expected_max
        assert len(wave) > 0


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
