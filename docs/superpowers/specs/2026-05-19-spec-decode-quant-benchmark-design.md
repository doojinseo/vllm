# Speculative Decoding Draft Model Quantization Benchmark

**Date:** 2026-05-19
**Status:** Approved

## Goal

Compare accepted tokens/second across three draft model quantization variants in vLLM's speculative decoding pipeline, sweeping over batch sizes to reveal how quantization interacts with the compute/memory-bandwidth tradeoff.

| Role | Model |
|------|-------|
| Target | Qwen3-8B (bf16) |
| Draft — base | Qwen3-1.7B (bf16, `quantization=None`) |
| Draft — fp8 | Qwen3-1.7B (`quantization="fp8"`) |
| Draft — int8 | Qwen3-1.7B (`quantization="int8"`) |

- **Workload:** ShareGPT dataset (offline batch, all prompts submitted at once)
- **Speculative tokens:** 5 per draft step
- **GPU:** Single GPU
- **Batch sizes swept:** `[1, 4, 8, 16, 32, 64, 128]` (controlled via `max_num_seqs`)
- **Primary metric:** accepted tokens/second

---

## Architecture

Three phases run sequentially in a single script:

1. **Setup** — parse CLI args, load and sample the ShareGPT dataset once (shared across all runs for a fair comparison), initialize the tokenizer.
2. **Benchmark loop** — outer loop over batch sizes `[1, 4, 8, 16, 32, 64, 128]`; inner loop over quantization variants `[base, fp8, int8]`. For each `(batch_size, quantization)` combination, construct `LLM` with `max_num_seqs=batch_size`, run `generate()` on all prompts, capture metrics, destroy the engine and free GPU memory before the next combination.
3. **Output** — print a 2D comparison table (rows = batch sizes, columns = quantization variants).

---

## Components

### `load_sharegpt(path, num_samples, max_prompt_tokens, max_output_tokens, tokenizer)`

Reads the ShareGPT JSON file, filters conversations that have at least one assistant turn, tokenizes the human prompt, truncates to `max_prompt_tokens`, caps requested output length at `max_output_tokens`. Returns `list[tuple[list[int], int]]` — `(prompt_token_ids, output_len)` pairs. Logic mirrors `benchmark_throughput.py` so results are directly comparable. Called once; the same list is reused across all 21 benchmark runs.

### `run_variant(target_model, draft_model, quantization, max_num_seqs, prompts, num_spec_tokens) -> VariantResult`

Constructs `LLM` with:

```python
LLM(
    model=target_model,
    max_num_seqs=max_num_seqs,        # controls effective batch size
    speculative_config=SpeculativeConfig(
        model=draft_model,
        num_speculative_tokens=num_spec_tokens,
        quantization=quantization,    # None for base, "fp8", or "int8"
    ),
)
```

Wraps `llm.generate()` with wall-clock timing. Captures accepted tok/s by attaching a `logging.StreamHandler` backed by a `StringIO` buffer to the `vllm.v1.spec_decode.metrics` logger, then parsing `Accepted throughput: X.XX tokens/s` from the `SpecDecoding metrics:` log line emitted after generation completes. Returns `VariantResult(accepted_tok_per_sec, total_output_tokens, wall_time_sec)`.

### `main()`

Orchestrates setup → benchmark loop → table print. CLI args:

| Arg | Default | Description |
|-----|---------|-------------|
| `--target-model` | `Qwen/Qwen3-8B` | HuggingFace target model ID |
| `--draft-model` | `Qwen/Qwen3-1.7B` | HuggingFace draft model ID |
| `--dataset` | *(required)* | Path to ShareGPT JSON file |
| `--num-prompts` | `500` | Number of prompts to sample |
| `--num-spec-tokens` | `5` | Speculative tokens per step |
| `--batch-sizes` | `1 4 8 16 32 64 128` | Space-separated list of `max_num_seqs` values to sweep |
| `--max-model-len` | `4096` | Max sequence length |
| `--seed` | `42` | Sampling seed |

### Result table

Printed via `tabulate` (a test/benchmark environment dep; `pip install tabulate` if missing). Rows are batch sizes; columns are the three variants plus a wall-time column per variant:

```
Batch size    base (tok/s)    fp8 (tok/s)    int8 (tok/s)
----------    ------------    -----------    ------------
1             ...             ...            ...
4             ...             ...            ...
8             ...             ...            ...
16            ...             ...            ...
32            ...             ...            ...
64            ...             ...            ...
128           ...             ...            ...
```

---

## Data Flow

```
ShareGPT JSON
      │
      ▼
load_sharegpt() ──► [(prompt_token_ids, output_len), ...] ← shared across all 21 runs
                                        │
              for batch_size in [1, 4, 8, 16, 32, 64, 128]:
                for quantization in [None, "fp8", "int8"]:
                        │
                        ▼
              LLM(Qwen3-8B,
                  max_num_seqs=batch_size,
                  speculative_config={
                    model=Qwen3-1.7B,
                    num_spec_tokens=5,
                    quantization=quantization,
                  })
                        │
                        ▼
              llm.generate(prompts)
                        │
                        ▼
              parse "Accepted throughput: X.XX tokens/s"
              from vllm.v1.spec_decode.metrics logger
                        │
                        ▼
              del llm; gc.collect(); cuda.empty_cache()
                        │
                        ▼
              record VariantResult in results[batch_size][quantization]

      │
      ▼
tabulate & print 2D results table
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Variant raises exception (e.g., fp8 unsupported on GPU) | Caught, warning printed, `N/A` recorded in that cell, continue to next combination |
| CUDA OOM (likely at large batch sizes) | Same catch-and-continue; CUDA error message included in warning |
| Missing ShareGPT dataset file | Hard fail with clear error message before any GPU work begins |
| Generation fails mid-run | `finally` block always runs `del llm; gc.collect(); torch.cuda.empty_cache()` |

---

## File Location

`benchmarks/benchmark_spec_decode_quant.py`

No existing files are modified.

---

## Usage Example

```bash
python benchmarks/benchmark_spec_decode_quant.py \
    --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 \
    --num-spec-tokens 5 \
    --batch-sizes 1 4 8 16 32 64 128
```
