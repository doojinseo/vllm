# Adaptive Draft Model Wave Benchmark — Design Spec

**Date:** 2026-05-22  
**Status:** Approved

---

## Problem

The existing `benchmark_spec_decode_quant.py` tests each quantization variant at a **fixed `max_num_seqs`** — one isolated cell per (batch_size, variant) combination. This cannot validate the adaptive draft model (`AdaptiveDraftModelProposer`), which is designed to switch between fp8 and int8 draft models as batch size fluctuates. Exercising the adaptive path requires traffic that **crosses the switching threshold during a single run**.

---

## Goal

A dedicated benchmark that sends alternating small/large request waves through a single long-lived LLM instance, compares four variants (base, int8, fp8, adaptive) under identical wave sequences, writes structured results to JSON, and produces a two-panel plot.

---

## New File

`benchmarks/benchmark_adaptive_draft.py`

---

## Variants

| Label      | Draft model             | Notes                                      |
|------------|-------------------------|--------------------------------------------|
| `base`     | Qwen/Qwen3-1.7B         | bf16, no quantization — baseline           |
| `int8`     | Qwen/Qwen3-1.7B-GPTQ-Int8 | Fixed int8                              |
| `fp8`      | Qwen/Qwen3-1.7B-FP8     | Fixed fp8                                  |
| `adaptive` | fp8 primary + int8 alt  | `alt_model` set, EMA-driven switching      |

---

## Architecture

### Outer loop

```
for variant in [base, int8, fp8, adaptive]:
    clear torch compile cache
    spin up LLM (max_num_seqs = large_batch)
    run wave sequence
    collect per-wave metrics
    tear down LLM (del + gc + cuda.empty_cache)
```

Compile cache is cleared between variants to prevent kernel-arity collisions between quantization methods (GPTQ adds scale/zero-point tensors that change kernel argument counts).

### Wave sequence

```
wave 0: small_batch prompts   → small regime, adaptive should use int8
wave 1: large_batch prompts   → large regime, adaptive should switch to fp8
wave 2: small_batch prompts   → adaptive switches back to int8
wave 3: large_batch prompts   → ...
...  (num_wave_pairs × 2 total waves)
```

Starting small lets the adaptive model's EMA settle into the int8 regime before the first large-wave transition, exercising the switch in the interesting direction.

Prompts are **pre-sampled once** before any variant runs. All four variants receive the same prompt at the same wave index — ensures fair comparison. Each wave gets its own slice of the dataset (distinct seed per wave index).

`max_num_seqs` is set to `large_batch` for all variants so large waves are not artificially throttled.

### LLM construction

- Fixed variants: standard `speculative_config` with `method="draft_model"` and the appropriate draft model path. Quantization is auto-detected from the checkpoint's HF config (no explicit `quantization=` arg needed).
- Adaptive variant: same as fp8 fixed, plus `alt_model=<int8 path>`, `adaptive_threshold`, and `adaptive_ema_alpha` in `speculative_config`.
- All variants: `disable_log_stats=False` (required for `get_metrics()` to function).

---

## Metrics

**Primary:** accepted tokens/second per wave.

Accepted token count is read from the Prometheus counter `vllm:spec_decode_num_accepted_tokens` via `llm.get_metrics()` after each `generate()` call. Because the counter is cumulative, the per-wave count is computed as a delta from the previous read.

Wall time is measured with `time.perf_counter()` around each `generate()` call.

```
accepted_tok_per_sec = delta_accepted_tokens / wave_wall_time
```

---

## CLI Arguments

| Argument           | Default                              | Description                                   |
|--------------------|--------------------------------------|-----------------------------------------------|
| `--target-model`   | `Qwen/Qwen3-8B`                      | Target model HF ID                            |
| `--draft-model-base` | `Qwen/Qwen3-1.7B`                  | bf16 draft model                              |
| `--draft-model-fp8`  | `Qwen/Qwen3-1.7B-FP8`              | FP8 draft model (adaptive primary)            |
| `--draft-model-int8` | `Qwen/Qwen3-1.7B-GPTQ-Int8`        | GPTQ-Int8 draft model (adaptive alt)          |
| `--dataset`        | *(required)*                         | Path to ShareGPT JSON                         |
| `--small-batch`    | `4`                                  | Prompts per small wave                        |
| `--large-batch`    | `32`                                 | Prompts per large wave                        |
| `--num-wave-pairs` | `4`                                  | Number of small+large pairs (8 total waves)   |
| `--num-spec-tokens`| `5`                                  | Speculative tokens per draft step             |
| `--threshold`      | `8`                                  | Adaptive model batch-size switching threshold |
| `--ema-alpha`      | `0.1`                                | Adaptive model EMA decay factor               |
| `--max-model-len`  | `4096`                               | Max sequence length                           |
| `--seed`           | `42`                                 | Base random seed                              |
| `--output`         | `adaptive_draft_wave_results.json`   | JSON results file path                        |
| `--plot`           | *(same stem as --output, .png)*      | Plot output path                              |

---

## Output

### JSON results file

```json
{
  "config": {
    "small_batch": 4, "large_batch": 32, "num_wave_pairs": 4,
    "num_spec_tokens": 5, "threshold": 8, "ema_alpha": 0.1
  },
  "waves": [
    {
      "index": 0, "type": "small", "batch": 4,
      "base": 142.3, "int8": 158.1, "fp8": 131.4, "adaptive": 157.9
    },
    {
      "index": 1, "type": "large", "batch": 32,
      "base": 301.2, "int8": 278.4, "fp8": 334.6, "adaptive": 332.1
    }
  ],
  "summary": {
    "base":     { "small_avg": 141.0, "large_avg": 299.9, "overall": 220.5 },
    "int8":     { "small_avg": 157.2, "large_avg": 277.2, "overall": 217.2 },
    "fp8":      { "small_avg": 130.8, "large_avg": 332.9, "overall": 231.9 },
    "adaptive": { "small_avg": 156.7, "large_avg": 331.5, "overall": 244.1 }
  }
}
```

### Console tables

**Per-wave table** (printed live as waves complete):

```
Wave  Type   Batch  base    int8    fp8     adaptive
0     small  4      142.3   158.1   131.4   157.9
1     large  32     301.2   278.4   334.6   332.1
...
```

**Summary table** (printed at end):

```
Variant   Small-wave avg  Large-wave avg  Overall avg
base      141.0           299.9           220.5
int8      157.2           277.2           217.2
fp8       130.8           332.9           231.9
adaptive  156.7           331.5           244.1
```

### Plot (two-panel PNG)

**Top panel — per-wave line chart:**
- X axis: wave index, labeled `S0 L1 S2 L3 ...` (S=small, L=large)
- Y axis: accepted tok/s
- One line per variant, consistent colours with existing benchmark plots
- Background shading: light blue for small waves, light orange for large waves

**Bottom panel — grouped bar chart:**
- X axis: variants (base, int8, fp8, adaptive)
- Grouped bars: small-wave average (blue) and large-wave average (orange)
- Shows regime-level performance at a glance

Saved to the path given by `--plot` using `matplotlib`, same DPI/style as `plot_spec_decode_quant.py`.

---

## Key Invariants

- All variants receive identical prompts at each wave index.
- `max_num_seqs` equals `large_batch` for all variants — no artificial throttling on large waves.
- Compile cache is cleared between variants, not between waves of the same variant.
- The adaptive variant accumulates EMA state across all waves within its run (no reset between waves) — this is intentional; it mirrors production behaviour.
- Accepted token delta is computed per wave; the cumulative counter is not reset between waves.

---

## Success Criteria

- On small waves: `adaptive` accepted tok/s ≈ `int8` accepted tok/s.
- On large waves: `adaptive` accepted tok/s ≈ `fp8` accepted tok/s.
- Overall `adaptive` avg exceeds both `int8` and `fp8` overall avg under a balanced alternating pattern.
