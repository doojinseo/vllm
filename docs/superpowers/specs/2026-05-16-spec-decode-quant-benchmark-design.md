# Speculative Decoding Draft-Model Quantization Benchmark — Design Spec

**Date:** 2026-05-16
**Status:** Approved

## Goal

Measure how quantizing the Qwen3-1.7B draft model (base fp16, AWQ 4-bit, GPTQ 4-bit)
affects speculative decoding throughput (accepted tokens/sec) as batch size scales from
1 to 128 on a 4× Tesla V100-32GB cluster.

## Hardware Context

| Item | Value |
|---|---|
| GPU | Tesla V100-PCIE-32GB × 4 (Volta, sm_70) |
| CUDA | 12.8 |
| Driver | 570.211.01 |

**Why AWQ/GPTQ instead of int8/fp8:** V100 has no INT8 Tensor Cores (requires Turing,
sm_75+) and no fp8 support (requires Ada/Hopper, sm_89+). Weight-only 4-bit formats
(AWQ, GPTQ) reduce memory bandwidth pressure and work on V100, making them the correct
comparison on this hardware.

## Models

| Role | Model ID |
|---|---|
| Target | `Qwen/Qwen3-8B` |
| Draft — base | `Qwen/Qwen3-1.7B` |
| Draft — AWQ | `Qwen/Qwen3-1.7B-AWQ` *(confirm HF ID or substitute local path)* |
| Draft — GPTQ | `Qwen/Qwen3-1.7B-GPTQ-Int4` *(confirm HF ID or substitute local path)* |

AWQ and GPTQ require **pre-quantized checkpoints**; vLLM reads already-quantized weights
and does not quantize on the fly. Update the model IDs before running if the above HF
slugs do not exist.

## Sweep Parameters

```python
VARIANTS = {
    "base": {"model": "Qwen/Qwen3-1.7B",          "quant": None},
    "awq":  {"model": "Qwen/Qwen3-1.7B-AWQ",       "quant": "awq"},
    "gptq": {"model": "Qwen/Qwen3-1.7B-GPTQ-Int4", "quant": "gptq"},
}

BATCH_SIZES             = [1, 2, 4, 8, 16, 32, 64, 128]
NUM_SPECULATIVE_TOKENS  = 5
INPUT_LEN               = 128   # tokens
OUTPUT_LEN              = 256   # tokens
NUM_PROMPTS_PER_RUN     = max(256, batch_size * 4)  # guarantees ≥4 full scheduler cycles
```

Total runs: 3 variants × 8 batch sizes = 24 subprocess invocations.

## Architecture

### File

`benchmarks/benchmark_spec_decode_quant_sweep.py` — a pure orchestrator; never loads a
GPU model itself.

### Flow

```
for variant in [base, awq, gptq]:
    for batch_size in BATCH_SIZES:
        1. Build subprocess args for benchmark_throughput.py
        2. Run subprocess, capture stdout/stderr
        3. On success: parse JSON, compute accepted_tokens_per_sec
        4. On failure (non-zero exit, missing JSON, OOM): log, write NaN row, continue
collect all rows → write results.csv → generate results.png
```

### Subprocess call

```
python benchmarks/benchmark_throughput.py
  --backend vllm
  --model                          Qwen/Qwen3-8B
  --speculative-model              <draft_model_path>
  --speculative-model-quantization <quant>          # omitted when None (base)
  --num-speculative-tokens         5
  --max-num-seqs                   <batch_size>
  --num-prompts                    <max(256, batch_size*4)>
  --input-len                      128
  --output-len                     256
  --output-json                    <tmp_dir>/run_<variant>_<batch_size>.json
```

### Metric extraction

```python
accepted_tokens_per_sec = (result["num_requests"] * OUTPUT_LEN) / result["elapsed_time"]
```

`output_tokens = num_requests × OUTPUT_LEN` because all prompts use a fixed output length.
This equals accepted tokens for speculative decoding (only verified draft tokens become
output tokens).

## Outputs

### CSV — `<output_dir>/results.csv`

Columns: `variant, batch_size, num_prompts, accepted_tokens_per_sec, elapsed_time`

One row per run. Failed runs written with `accepted_tokens_per_sec = NaN`.

### Plot — `<output_dir>/results.png`

- X-axis: batch size (log₂ scale, ticks at each power of 2)
- Y-axis: accepted tokens/sec
- One line per variant (`base`, `awq`, `gptq`), with point markers
- Legend upper-left
- Title: "Speculative Decoding Throughput vs Batch Size (Qwen3-1.7B draft → Qwen3-8B)"

### Output directory

Controlled by `--output-dir` (default: `./spec_decode_quant_results/`). Created
automatically if it does not exist.

## Prerequisites

1. vLLM installed and importable in the Python environment.
2. `benchmark_throughput.py` present at `benchmarks/benchmark_throughput.py` (unchanged).
3. All three draft model checkpoints downloaded and accessible (HF cache or local path).
4. Enough GPU VRAM: Qwen3-8B (target) ~16 GB fp16 + Qwen3-1.7B (draft) ~3.4 GB fp16.
   With 4× V100-32GB this is comfortable; single-GPU may OOM at large batch sizes.

## Open Questions

- Confirm AWQ and GPTQ HuggingFace model IDs for Qwen3-1.7B before running.
- Decide tensor-parallel degree (`--tensor-parallel-size`); default is 1. Adjust if
  Qwen3-8B does not fit on a single V100.
