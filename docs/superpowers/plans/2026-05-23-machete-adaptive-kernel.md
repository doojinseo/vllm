# Machete Adaptive Kernel Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Patch the int8 GPTQ draft model at load time so each Machete linear layer dispatches to the best-profiled CUTLASS schedule for the current token count instead of using the static default.

**Architecture:** At model load, walk all modules with a `MacheteLinearKernel`, call `ops.machete_supported_schedules` to get compiled CUTLASS variants, time each at `small_bs=1` and `large_bs=16`, then replace `kernel.apply_weights` with a closure that picks the right schedule based on `x.shape[0]`. The hook fires from a new `DraftModelProposer.load_model` override when `VLLM_ADAPTIVE_MACHETE_THRESHOLD` env var is set. A new benchmark variant `int8_machete` passes that env var to the server subprocess.

**Tech Stack:** Python 3.12, PyTorch 2.x, `vllm._custom_ops.machete_mm` / `machete_supported_schedules`, `vllm.model_executor.kernels.linear.mixed_precision.machete.MacheteLinearKernel`, `vllm.v1.spec_decode.draft_model.DraftModelProposer`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `vllm/v1/spec_decode/adaptive_draft_model.py` | Modify | Add `_profile_machete_schedules`, `_make_adaptive_apply`, `_install_adaptive_machete_schedules` |
| `vllm/v1/spec_decode/draft_model.py` | Modify | Add `load_model` override that checks `VLLM_ADAPTIVE_MACHETE_THRESHOLD` |
| `benchmarks/benchmark_adaptive_online.py` | Modify | Add `int8_machete` variant; add `env` param to `start_server` |
| `tests/v1/spec_decode/test_machete_adaptive.py` | Create | Unit tests for profiler and installer |

---

### Task 1: Schedule profiler and installer in `adaptive_draft_model.py`

**Files:**
- Modify: `vllm/v1/spec_decode/adaptive_draft_model.py`
- Create: `tests/v1/spec_decode/test_machete_adaptive.py`

- [ ] **Step 1: Write the failing profiler test**

```python
# tests/v1/spec_decode/test_machete_adaptive.py
# SPDX-License-Identifier: Apache-2.0
import time
from unittest.mock import MagicMock, patch

import torch
import pytest


def _make_fake_kernel(in_f=4096, out_f=4096, group_size=128, has_g_idx=False):
    """Return a duck-typed MacheteLinearKernel with controllable config."""
    config = MagicMock()
    config.partition_weight_shape = (in_f, out_f)
    config.act_type = torch.bfloat16
    config.weight_type = MagicMock()
    config.group_size = group_size
    config.zero_points = False
    config.has_g_idx = has_g_idx

    w_q = torch.zeros(1, dtype=torch.int32)
    w_s = torch.zeros(1, dtype=torch.bfloat16)

    kernel = MagicMock()
    kernel.config = config
    kernel._get_weight_params.return_value = (w_q, w_s, None, None)
    return kernel


def _make_fake_layer(in_f=4096):
    layer = MagicMock()
    # parameters() is called to discover device
    layer.parameters.return_value = iter([
        torch.nn.Parameter(torch.zeros(1))
    ])
    return layer


@patch("torch.cuda.synchronize")
@patch("vllm.v1.spec_decode.adaptive_draft_model.ops")
def test_profile_schedules_returns_two_strings_from_list(mock_ops, mock_sync):
    """_profile_machete_schedules returns (str, str) both from the schedule list."""
    from vllm.v1.spec_decode.adaptive_draft_model import _profile_machete_schedules

    mock_ops.machete_supported_schedules.return_value = ["sched_A", "sched_B"]
    mock_ops.machete_mm.return_value = torch.zeros(1, 4096, dtype=torch.bfloat16)

    kernel = _make_fake_kernel()
    layer = _make_fake_layer()

    small, large = _profile_machete_schedules(layer, kernel, small_bs=1, large_bs=16)

    assert small in ["sched_A", "sched_B"]
    assert large in ["sched_A", "sched_B"]


@patch("torch.cuda.synchronize")
@patch("vllm.v1.spec_decode.adaptive_draft_model.ops")
def test_profile_schedules_single_schedule_returns_same_for_both(mock_ops, mock_sync):
    """When only one schedule exists, both slots return that schedule."""
    from vllm.v1.spec_decode.adaptive_draft_model import _profile_machete_schedules

    mock_ops.machete_supported_schedules.return_value = ["only_one"]
    mock_ops.machete_mm.return_value = torch.zeros(1, 4096, dtype=torch.bfloat16)

    kernel = _make_fake_kernel()
    layer = _make_fake_layer()

    small, large = _profile_machete_schedules(layer, kernel, small_bs=1, large_bs=16)

    assert small == "only_one"
    assert large == "only_one"


@patch("torch.cuda.synchronize")
@patch("vllm.v1.spec_decode.adaptive_draft_model._profile_machete_schedules")
@patch("vllm.v1.spec_decode.adaptive_draft_model.ops")
def test_install_dispatches_small_sched_below_threshold(mock_ops, mock_profile, mock_sync):
    """After install, apply_weights uses small_sched when n_tokens < threshold."""
    from vllm.v1.spec_decode.adaptive_draft_model import _install_adaptive_machete_schedules
    from vllm.model_executor.kernels.linear.mixed_precision.machete import MacheteLinearKernel

    mock_profile.return_value = ("sched_small", "sched_large")
    mock_ops.machete_mm.return_value = torch.zeros(4, 4096, dtype=torch.bfloat16)

    kernel = _make_fake_kernel()
    # Make isinstance check pass by patching the class
    with patch(
        "vllm.v1.spec_decode.adaptive_draft_model.MacheteLinearKernel",
        type(kernel),
    ):
        module = MagicMock()
        module.quant_method = MagicMock()
        module.quant_method.kernel = kernel

        model = MagicMock()
        model.modules.return_value = [module]

        _install_adaptive_machete_schedules(model, threshold=8)

    # Call with n_tokens=4 < 8
    layer = MagicMock()
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    kernel.apply_weights(layer, x)

    call_kwargs = mock_ops.machete_mm.call_args.kwargs
    assert call_kwargs["schedule"] == "sched_small"


@patch("torch.cuda.synchronize")
@patch("vllm.v1.spec_decode.adaptive_draft_model._profile_machete_schedules")
@patch("vllm.v1.spec_decode.adaptive_draft_model.ops")
def test_install_dispatches_large_sched_at_or_above_threshold(mock_ops, mock_profile, mock_sync):
    """After install, apply_weights uses large_sched when n_tokens >= threshold."""
    from vllm.v1.spec_decode.adaptive_draft_model import _install_adaptive_machete_schedules

    mock_profile.return_value = ("sched_small", "sched_large")
    mock_ops.machete_mm.return_value = torch.zeros(16, 4096, dtype=torch.bfloat16)

    kernel = _make_fake_kernel()
    with patch(
        "vllm.v1.spec_decode.adaptive_draft_model.MacheteLinearKernel",
        type(kernel),
    ):
        module = MagicMock()
        module.quant_method = MagicMock()
        module.quant_method.kernel = kernel

        model = MagicMock()
        model.modules.return_value = [module]

        _install_adaptive_machete_schedules(model, threshold=8)

    layer = MagicMock()
    x = torch.zeros(16, 4096, dtype=torch.bfloat16)
    kernel.apply_weights(layer, x)

    call_kwargs = mock_ops.machete_mm.call_args.kwargs
    assert call_kwargs["schedule"] == "sched_large"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace
.venv/bin/python -m pytest tests/v1/spec_decode/test_machete_adaptive.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `AttributeError` — `_profile_machete_schedules` and `_install_adaptive_machete_schedules` do not exist yet.

- [ ] **Step 3: Add imports and helper functions to `adaptive_draft_model.py`**

At the top of `vllm/v1/spec_decode/adaptive_draft_model.py`, after existing imports, add:

```python
import os
import time
```

After the existing `logger = init_logger(__name__)` line, add:

```python
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.machete import (
    MacheteLinearKernel,
)
```

- [ ] **Step 4: Implement `_profile_machete_schedules`**

Add this function directly before the `AdaptiveDraftModelProposer` class definition in `vllm/v1/spec_decode/adaptive_draft_model.py`:

```python
def _profile_machete_schedules(
    layer: nn.Module,
    kernel: MacheteLinearKernel,
    small_bs: int,
    large_bs: int,
    n_warmup: int = 3,
    n_timed: int = 10,
) -> tuple[str | None, str | None]:
    """Profile Machete CUTLASS schedule strings at two batch sizes.

    Returns (best_small_schedule, best_large_schedule).
    Returns (None, None) on failure or when only one schedule exists.
    """
    c = kernel.config
    try:
        schedules: list[str] = ops.machete_supported_schedules(
            a_type=c.act_type,
            b_type=c.weight_type,
            group_scales_type=c.act_type,
        )
    except Exception as e:
        logger.warning("machete_supported_schedules failed: %s; skipping adaptive scheduling", e)
        return None, None

    if not schedules:
        return None, None
    if len(schedules) == 1:
        return schedules[0], schedules[0]

    w_q, w_s, w_zp, _ = kernel._get_weight_params(layer)
    if not c.zero_points:
        w_zp = None
    device = w_q.device
    in_features = c.partition_weight_shape[0]

    def _time_one(bs: int, sched: str) -> float:
        x = torch.zeros(bs, in_features, dtype=c.act_type, device=device)
        if c.has_g_idx:
            x = kernel.act_perm(x)
        try:
            for _ in range(n_warmup):
                ops.machete_mm(
                    a=x, b_q=w_q, b_type=c.weight_type,
                    b_group_zeros=w_zp, b_group_scales=w_s,
                    b_group_size=c.group_size, schedule=sched,
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_timed):
                ops.machete_mm(
                    a=x, b_q=w_q, b_type=c.weight_type,
                    b_group_zeros=w_zp, b_group_scales=w_s,
                    b_group_size=c.group_size, schedule=sched,
                )
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n_timed
        except Exception:
            return float("inf")

    best_small = min(schedules, key=lambda s: _time_one(small_bs, s))
    best_large = min(schedules, key=lambda s: _time_one(large_bs, s))
    logger.debug(
        "Machete schedule profiling: small_bs=%d -> %s, large_bs=%d -> %s",
        small_bs, best_small, large_bs, best_large,
    )
    return best_small, best_large
```

- [ ] **Step 5: Implement `_make_adaptive_apply` and `_install_adaptive_machete_schedules`**

Add these two functions right after `_profile_machete_schedules` in the same file:

```python
def _make_adaptive_apply(
    kernel: MacheteLinearKernel,
    small_sched: str | None,
    large_sched: str | None,
    threshold: int,
):
    """Return a replacement for kernel.apply_weights that dispatches by token count."""
    c = kernel.config

    def adaptive_apply(
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, w_zp, _ = kernel._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        if c.has_g_idx:
            x_2d = kernel.act_perm(x_2d)

        if not c.zero_points:
            w_zp = None

        n_tokens = x_2d.shape[0]
        schedule = small_sched if n_tokens < threshold else large_sched

        output = ops.machete_mm(
            a=x_2d,
            b_q=w_q,
            b_type=c.weight_type,
            b_group_zeros=w_zp,
            b_group_scales=w_s,
            b_group_size=c.group_size,
            schedule=schedule,
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)

    return adaptive_apply


def _install_adaptive_machete_schedules(
    model: nn.Module,
    threshold: int,
    small_bs: int = 1,
    large_bs: int = 16,
) -> None:
    """Monkey-patch all MacheteLinearKernel layers in model with adaptive schedule dispatch."""
    patched = 0
    skipped_same = 0
    for module in model.modules():
        qm = getattr(module, "quant_method", None)
        if qm is None:
            continue
        kernel = getattr(qm, "kernel", None)
        if not isinstance(kernel, MacheteLinearKernel):
            continue

        small_sched, large_sched = _profile_machete_schedules(
            module, kernel, small_bs, large_bs
        )
        if small_sched is None and large_sched is None:
            continue
        if small_sched == large_sched:
            skipped_same += 1
            continue

        kernel.apply_weights = _make_adaptive_apply(kernel, small_sched, large_sched, threshold)
        patched += 1

    logger.info(
        "Adaptive Machete scheduling: patched=%d, skipped_same_schedule=%d, threshold=%d",
        patched, skipped_same, threshold,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /workspace
.venv/bin/python -m pytest tests/v1/spec_decode/test_machete_adaptive.py -v 2>&1 | tail -20
```

Expected: all 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add vllm/v1/spec_decode/adaptive_draft_model.py \
        tests/v1/spec_decode/test_machete_adaptive.py
git commit -m "feat: add Machete adaptive schedule profiler and installer"
```

---

### Task 2: Hook `_install_adaptive_machete_schedules` into `DraftModelProposer`

**Files:**
- Modify: `vllm/v1/spec_decode/draft_model.py`

- [ ] **Step 1: Add the `load_model` override**

`DraftModelProposer` currently has no `load_model` method — it inherits directly from `SpecDecodeBaseProposer` (defined in `vllm/v1/spec_decode/llm_base_proposer.py`). Add the override at the end of the `DraftModelProposer` class body in `vllm/v1/spec_decode/draft_model.py`, after the `_maybe_share_lm_head` method (line 88):

```python
    @override
    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        threshold_str = os.environ.get("VLLM_ADAPTIVE_MACHETE_THRESHOLD")
        if threshold_str:
            from vllm.v1.spec_decode.adaptive_draft_model import (
                _install_adaptive_machete_schedules,
            )
            threshold = int(threshold_str)
            logger.info(
                "VLLM_ADAPTIVE_MACHETE_THRESHOLD=%d: "
                "installing adaptive Machete schedules on draft model",
                threshold,
            )
            _install_adaptive_machete_schedules(self.model, threshold)
```

- [ ] **Step 2: Add `import os` to `draft_model.py`**

The file already has `logger = init_logger(__name__)` and `override`. It is missing `import os`. Add it after `import torch` on line 4:

```python
import os
import torch
import torch.nn as nn
from typing_extensions import override
```

- [ ] **Step 3: Verify the hook triggers via a smoke test**

Set the env var and start a minimal Python check (no GPU needed — just verify the import chain works):

```bash
cd /workspace
VLLM_ADAPTIVE_MACHETE_THRESHOLD=8 .venv/bin/python -c "
from vllm.v1.spec_decode.draft_model import DraftModelProposer
import inspect
src = inspect.getsource(DraftModelProposer.load_model)
assert 'VLLM_ADAPTIVE_MACHETE_THRESHOLD' in src, 'env var check not found in load_model'
print('OK: load_model override is present')
"
```

Expected output: `OK: load_model override is present`

- [ ] **Step 4: Commit**

```bash
git add vllm/v1/spec_decode/draft_model.py
git commit -m "feat: hook adaptive Machete scheduling into DraftModelProposer.load_model"
```

---

### Task 3: Add `int8_machete` benchmark variant

**Files:**
- Modify: `benchmarks/benchmark_adaptive_online.py`

Context on existing code:
- `make_spec_config(variant, ...)` builds the `--speculative-config` dict; `int8_machete` needs the same spec config as `int8` but the env var `VLLM_ADAPTIVE_MACHETE_THRESHOLD` set on the server.
- `start_server(cmd, log_path)` creates a `subprocess.Popen` with `start_new_session=True` but no `env=` argument currently.
- The main loop in `main()` calls `start_server(cmd, log_path)` at around line 581.

- [ ] **Step 1: Add `int8_machete` to `make_spec_config`**

In `make_spec_config`, the draft model lookup dict is:
```python
draft = {
    "draft_base": draft_model_base,
    "int8":       draft_model_int8,
    "fp8":        draft_model_fp8,
    "adaptive":   draft_model_fp8,
}[variant]
```

Add `"int8_machete"` to this dict. Change the dict to:
```python
draft = {
    "draft_base":   draft_model_base,
    "int8":         draft_model_int8,
    "int8_machete": draft_model_int8,
    "fp8":          draft_model_fp8,
    "adaptive":     draft_model_fp8,
}[variant]
```

No other change to `make_spec_config` — `int8_machete` uses a plain `draft_model` spec config identical to `int8`.

- [ ] **Step 2: Add `env` parameter to `start_server`**

Change `start_server` from:
```python
def start_server(cmd: list[str], log_path: str) -> subprocess.Popen:
    log_file = open(log_path, "w")  # noqa: SIM115 — kept open for subprocess lifetime
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
```

To:
```python
def start_server(
    cmd: list[str],
    log_path: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    log_file = open(log_path, "w")  # noqa: SIM115 — kept open for subprocess lifetime
    process_env = {**os.environ, **(extra_env or {})}
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        env=process_env,
    )
```

- [ ] **Step 3: Thread `extra_env` through the main loop**

In `main()`, find the `start_server(cmd, log_path)` call (around line 581). Replace it with:

```python
        extra_env: dict[str, str] | None = None
        if variant == "int8_machete":
            extra_env = {"VLLM_ADAPTIVE_MACHETE_THRESHOLD": str(args.threshold)}

        log_path = f"/tmp/vllm_serve_{variant}.log"
        proc = start_server(cmd, log_path, extra_env=extra_env)
```

- [ ] **Step 4: Run a quick parse-only check**

```bash
cd /workspace
.venv/bin/python -c "
import ast, pathlib
src = pathlib.Path('benchmarks/benchmark_adaptive_online.py').read_text()
ast.parse(src)
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_adaptive_online.py
git commit -m "feat: add int8_machete benchmark variant with Machete adaptive scheduling"
```

---

### Task 4: Run the benchmark

- [ ] **Step 1: Run with just `int8` and `int8_machete` first to validate**

```bash
cd /workspace
.venv/bin/python benchmarks/benchmark_adaptive_online.py \
  --target-model Qwen/Qwen3-8B \
  --draft-model-base Qwen/Qwen3-1.7B \
  --draft-model-fp8 Qwen/Qwen3-1.7B-FP8 \
  --draft-model-int8 Qwen/Qwen3-1.7B-GPTQ-Int8 \
  --variants int8 int8_machete \
  --request-rate 4.0 \
  --duration 120.0 \
  --warmup 30.0 \
  --threshold 8
```

Expected: both variants complete, JSON results saved to `results/`, server log for `int8_machete` contains lines like `"Machete schedule profiling"` and `"Adaptive Machete scheduling: patched=N"`.

- [ ] **Step 2: Check server log for schedule patching confirmation**

```bash
grep -i "machete\|adaptive" /tmp/vllm_serve_int8_machete.log | head -20
```

Expected lines like:
```
INFO ... VLLM_ADAPTIVE_MACHETE_THRESHOLD=8: installing adaptive Machete schedules on draft model
INFO ... Adaptive Machete scheduling: patched=28, skipped_same_schedule=0, threshold=8
```

- [ ] **Step 3: Run full comparison if Step 1 succeeds**

```bash
cd /workspace
.venv/bin/python benchmarks/benchmark_adaptive_online.py \
  --target-model Qwen/Qwen3-8B \
  --draft-model-base Qwen/Qwen3-1.7B \
  --draft-model-fp8 Qwen/Qwen3-1.7B-FP8 \
  --draft-model-int8 Qwen/Qwen3-1.7B-GPTQ-Int8 \
  --variants base draft_base int8 int8_machete \
  --request-rate 4.0 \
  --duration 180.0 \
  --warmup 60.0 \
  --threshold 8
```
