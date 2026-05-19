# Speculative Decoding Draft Model Quantization Benchmark

**Date:** 2026-05-19
**Status:** Approved

## Goal

Compare accepted tokens/second across three draft model quantization variants in vLLM's speculative decoding pipeline, using a fixed target model and a shared realistic workload.

| Role | Model |
|------|-------|
| Target | Qwen3-8B (bf16) |
| Draft — base | Qwen3-1.7B (bf16, `quantization=None`) |
| Draft — fp8 | Qwen3-1.7B (`quantization="fp8"`) |
| Draft — int8 | Qwen3-1.7B (`quantization="int8"`) |

- **Workload:** ShareGPT dataset (offline batch, all prompts submitted at once)
- **Speculative tokens:** 5 per draft step
- **GPU:** Single GPU
- **Primary metric:** accepted tokens/second

---

## Architecture

Three phases run sequentially in a single script:

1. **Setup** — parse CLI args, load and sample the ShareGPT dataset once (shared across all runs for a fair comparison), initialize the tokenizer.
2. **Benchmark loop** — iterate over `[base, fp8, int8]`. For each variant, construct `LLM`, run `generate()` on all prompts, capture metrics, destroy the engine and free GPU memory before the next variant.
3. **Output** — print a formatted comparison table.

---

## Components

### `load_sharegpt(path, num_samples, max_prompt_tokens, max_output_tokens, tokenizer)`

Reads the ShareGPT JSON file, filters conversations that have at least one assistant turn, tokenizes the human prompt, truncates to `max_prompt_tokens`, caps requested output length at `max_output_tokens`. Returns `list[tuple[list[int], int]]` — `(prompt_token_ids, output_len)` pairs. Logic mirrors `benchmark_throughput.py` so results are directly comparable.

### `run_variant(target_model, draft_model, quantization, prompts, num_spec_tokens) -> VariantResult`

Constructs `LLM` with:

```python
speculative_config=SpeculativeConfig(
    model=draft_model,
    num_speculative_tokens=num_spec_tokens,
    quantization=quantization,   # None for base, "fp8", or "int8"
)
```

Wraps `llm.generate()` with wall-clock timing. Captures `accepted_tokens/s` by attaching a string log handler to the `vllm.v1.spec_decode.metrics` logger and parsing its output line. Returns `VariantResult(accepted_tok_per_sec, total_output_tokens, wall_time_sec)`.

### `main()`

Orchestrates setup → benchmark loop → table print. CLI args:

| Arg | Default | Description |
|-----|---------|-------------|
| `--target-model` | `Qwen/Qwen3-8B` | HuggingFace target model ID |
| `--draft-model` | `Qwen/Qwen3-1.7B` | HuggingFace draft model ID |
| `--dataset` | *(required)* | Path to ShareGPT JSON file |
| `--num-prompts` | `500` | Number of prompts to sample |
| `--num-spec-tokens` | `5` | Speculative tokens per step |
| `--max-model-len` | `4096` | Max sequence length |
| `--seed` | `42` | Sampling seed |

### Result table

Printed via `tabulate` (a test/benchmark environment dep; `pip install tabulate` if missing):

```
Draft variant    Accepted tok/s    Output tok/s    Wall time (s)
---------------  ----------------  --------------  ---------------
base (bf16)      ...               ...             ...
fp8              ...               ...             ...
int8             ...               ...             ...
```

---

## Data Flow

```
ShareGPT JSON
      │
      ▼
load_sharegpt() ──► [(prompt_token_ids, output_len), ...] ← shared across all 3 runs
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
  variant: base                   variant: fp8                   variant: int8
  quantization=None               quantization="fp8"             quantization="int8"
        │                               │                               │
        ▼                               ▼                               ▼
  LLM(Qwen3-8B +             LLM(Qwen3-8B +                 LLM(Qwen3-8B +
   draft Qwen3-1.7B)          draft Qwen3-1.7B fp8)          draft Qwen3-1.7B int8)
        │                               │                               │
        └───────────────────────────────┴───────────────────────────────┘
                                        │
                                        ▼
                             tabulate & print results
```

Each LLM instance is destroyed (with `del llm; gc.collect(); torch.cuda.empty_cache()`) in a `finally` block before the next variant is constructed.

Accepted tok/s is extracted by intercepting the `vllm.v1.spec_decode.metrics` logger with a `logging.StreamHandler` pointed at a `StringIO` buffer, then parsing `Accepted throughput: X.XX tokens/s` from the `SpecDecoding metrics:` log line emitted at the end of generation.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Variant raises exception (e.g., fp8 unsupported on GPU) | Caught, warning printed, `N/A` recorded in table, continue to next variant |
| CUDA OOM during `LLM` construction | Same catch-and-continue; CUDA error message included in warning |
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
    --num-spec-tokens 5
```
