# Machete Adaptive Kernel Scheduling for Draft Model

## Goal

Add batch-size-aware Machete schedule selection to the int8 GPTQ draft model, reducing token-generation latency at small batch sizes without dual-model memory overhead.

## Architecture

Machete (CUTLASS-based) pre-compiles multiple kernel variants per operation signature. The `schedule` parameter to `ops.machete_mm` selects which compiled variant runs; `schedule=None` uses a static default tuned for large batch. We profile the compiled variants at load time for two representative batch sizes (`small_bs=1`, `large_bs=16`), cache the best schedule strings, and dispatch at inference time based on `x.shape[0]`.

**No new CUDA code. Single model. Zero memory overhead vs plain int8.**

## Components

### 1. `_profile_machete_schedules(layer, kernel, small_bs, large_bs) → tuple[str, str]`

Location: `vllm/v1/spec_decode/adaptive_draft_model.py`

- Retrieves `in_features` and `out_features` from `kernel.config.partition_weight_shape`
- Calls `ops.machete_supported_schedules(a_type, b_type, group_scales_type)` to get all compiled schedule strings
- For each schedule and each of the two batch sizes: creates a dummy activation tensor `[bs, in_features]`, runs 3 warmup + 10 timed `ops.machete_mm` calls, measures median GPU time via `torch.cuda.synchronize()`
- Returns `(best_small_schedule, best_large_schedule)`
- Falls back to `(None, None)` if only one schedule exists or profiling fails

### 2. `_install_adaptive_machete_schedules(model, threshold, small_bs=1, large_bs=16)`

Location: `vllm/v1/spec_decode/adaptive_draft_model.py`

- Walks `model.modules()`
- For each module with `module.quant_method` and `module.quant_method.kernel` being a `MacheteLinearKernel` instance:
  - Calls `_profile_machete_schedules` to get `(small_sched, large_sched)`
  - If both are identical, skips patching (no benefit)
  - Otherwise replaces `kernel.apply_weights` with a closure:
    ```python
    def adaptive_apply(layer, x, bias=None):
        n_tokens = x.reshape(-1, x.shape[-1]).shape[0]
        schedule = small_sched if n_tokens < threshold else large_sched
        # call machete_mm with explicit schedule instead of None
    ```
- Logs the count of patched layers and the profiled schedule strings

### 3. Hook in `DraftModelProposer.load_model()`

Location: `vllm/v1/spec_decode/draft_model.py`

At the end of `load_model()`, reads `VLLM_ADAPTIVE_MACHETE_THRESHOLD` from the environment. If set and non-empty, parses the integer and calls `_install_adaptive_machete_schedules(self.model, threshold)`. Default behavior (env var absent) is unchanged.

### 4. Benchmark variant `int8_machete`

Location: `benchmarks/benchmark_adaptive_online.py`

New entry in the variants list. Identical server configuration to `int8` except the server subprocess environment includes `VLLM_ADAPTIVE_MACHETE_THRESHOLD=<threshold>` (defaulting to the same `--threshold` value used by the adaptive dual-model variant). Results land in the same JSON output as other variants.

## Data Flow

```
Server startup
  └── DraftModelProposer.load_model()
        └── [if VLLM_ADAPTIVE_MACHETE_THRESHOLD set]
              └── _install_adaptive_machete_schedules(model, threshold)
                    └── for each MacheteLinearKernel layer:
                          └── _profile_machete_schedules() → (small_sched, large_sched)
                          └── monkey-patch kernel.apply_weights

Inference (per draft propose() call)
  └── model forward pass
        └── each linear layer apply_weights(x, ...)
              └── n_tokens = x.reshape(-1, in_features).shape[0]
              └── schedule = small_sched if n_tokens < threshold else large_sched
              └── ops.machete_mm(..., schedule=schedule)
```

## Error Handling

- If `ops.machete_supported_schedules` returns an empty list or raises: log a warning, skip patching, fall back to `schedule=None` (original behavior).
- If profiling a schedule raises (e.g., shape mismatch): skip that schedule, continue.
- If `small_sched == large_sched`: skip patching, log info.

## Testing

Two unit tests in `tests/benchmarks/test_benchmark_adaptive_online.py` (or a new `tests/spec_decode/test_machete_adaptive.py`):

1. **Profiler test**: mock `ops.machete_supported_schedules` returning `["sched_A", "sched_B"]` and `ops.machete_mm` where `sched_A` is faster for small batch and `sched_B` for large batch. Assert `_profile_machete_schedules` returns `("sched_A", "sched_B")`.

2. **Installer test**: mock a minimal module with a fake `MacheteLinearKernel` (duck-typed). Assert `kernel.apply_weights` is replaced; assert calls with `x.shape[0] < threshold` pass `schedule=small_sched` and calls with `x.shape[0] >= threshold` pass `schedule=large_sched`.

## Success Criteria

- Server logs confirm schedule profiling ran and N layers were patched.
- `int8_machete` variant appears in benchmark JSON output with all standard metrics.
- ITL p50/p99 for `int8_machete` is ≤ `int8` at request rates where average concurrency < threshold.
- No server crashes or silent fallback failures.

## Tech Stack

- Python 3.12, PyTorch 2.x
- `vllm._custom_ops.machete_mm` / `machete_supported_schedules`
- `vllm.model_executor.kernels.linear.mixed_precision.machete.MacheteLinearKernel`
- `vllm.v1.spec_decode.draft_model.DraftModelProposer`
