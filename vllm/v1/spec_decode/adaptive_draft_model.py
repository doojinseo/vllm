# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive draft model proposer.

Loads two draft model checkpoints (primary and alt) and switches between them
at inference time based on the observed batch size, smoothed with an EMA.
Intended use: primary=fp8 (fast at large batch), alt=int8/GPTQ (fast at
small batch), switching around the batch-size crossover (~bs=8).

CUDA graphs are disabled for the draft model so the active model reference
can be swapped freely between calls without needing separate graph pools.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from typing_extensions import override

from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mixed_precision.machete import (
    MacheteLinearKernel,
)
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.draft_model import DraftModelProposer

logger = init_logger(__name__)


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
    Returns (None, None) on failure or when no schedules exist.
    Returns the same schedule twice when only one exists.
    """
    c = kernel.config
    try:
        schedules: list[str] = ops.machete_supported_schedules(
            a_type=c.act_type,
            b_type=c.weight_type,
            group_scales_type=c.act_type,
            group_zeros_type=c.act_type if c.zero_points else None,
        )
    except Exception as e:
        logger.warning(
            "machete_supported_schedules failed: %s; skipping adaptive scheduling", e
        )
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

        if c.zero_points:
            assert w_zp is not None
        else:
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
    skipped_failed = 0
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
            skipped_failed += 1
            continue
        if small_sched == large_sched:
            skipped_same += 1
            continue

        kernel.apply_weights = _make_adaptive_apply(kernel, small_sched, large_sched, threshold)
        patched += 1

    logger.info(
        "Adaptive Machete scheduling: patched=%d, skipped_same_schedule=%d, skipped_failed=%d, threshold=%d",
        patched, skipped_same, skipped_failed, threshold,
    )


class AdaptiveDraftModelProposer(DraftModelProposer):
    """Switches between two draft models based on EMA-smoothed batch size."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        spec = vllm_config.speculative_config
        self._high_threshold: int = spec.adaptive_threshold
        self._low_threshold: int = spec.adaptive_low_threshold
        self._alpha: float = spec.adaptive_ema_alpha
        self._min_dwell: int = spec.adaptive_min_dwell_steps
        self._ema: float = 0.0
        self._prev_batch_size: int = 0
        self._using_primary: bool = False   # start on alt (int8) until EMA rises
        self._steps_since_switch: int = self._min_dwell  # allow first switch immediately
        self._primary_model: nn.Module | None = None
        self._alt_model_module: nn.Module | None = None
        # Attention layer objects keyed by "draft_model.*" names, for each model.
        self._primary_attn_context: dict = {}
        self._alt_attn_context: dict = {}
        self._alt_kv_cache_bound: bool = False

    def _load_alt_model(self) -> nn.Module:
        spec = self.speculative_config
        from vllm.compilation.backends import set_model_tag

        # Derive quantization config for the alt checkpoint.
        # GPTQ checkpoints require the correct quant_config so the model is
        # built with quantized linear layers (g_idx, qweight, scales) before
        # weights are loaded.  alt_model_config.quantization may be None if
        # auto-detection in ModelConfig.__post_init__ did not fire correctly
        # (the runner="draft" path has reduced init context). Fall back to
        # reading hf_config.quantization_config directly and running the same
        # override-detection loop that _verify_quantization() uses.
        alt_model_config = spec.alt_model_config
        if alt_model_config.quantization is None:
            self._detect_alt_quantization(alt_model_config)

        alt_quant_config = None
        if alt_model_config.quantization is not None:
            alt_quant_config = VllmConfig._get_quantization_config(
                alt_model_config, self.vllm_config.load_config
            )

        alt_vllm_config = replace(
            self._create_draft_vllm_config(),
            model_config=alt_model_config,
            quant_config=alt_quant_config,
        )
        with set_model_tag("draft_model_alt"):
            model = get_model(
                vllm_config=alt_vllm_config,
                prefix="draft_model_alt",
            )

        # The alt model's attention layers self-registered under
        # "draft_model_alt.*" in static_forward_context.  Rename them to
        # "draft_model.*" so they find their attention metadata (which is
        # keyed by the primary model's layer names) during forward passes.
        # Remove the alt-prefixed entries from static_forward_context and
        # store the renamed objects in _alt_attn_context for manual swapping.
        sfc = self.vllm_config.compilation_config.static_forward_context
        prefix_alt = "draft_model_alt."
        prefix_primary = "draft_model."
        for name in list(sfc.keys()):
            if name.startswith(prefix_alt):
                attn_obj = sfc.pop(name)
                new_name = prefix_primary + name[len(prefix_alt):]
                attn_obj.layer_name = new_name
                self._alt_attn_context[new_name] = attn_obj

        logger.info(
            "Loaded alt draft model %s (quant=%s) for adaptive switching.",
            spec.alt_model,
            alt_model_config.quantization,
        )
        return model

    def _detect_alt_quantization(self, alt_model_config) -> None:
        """Force-detect the quantization method for the alt model config.

        Called when alt_model_config.quantization is None despite the checkpoint
        being quantized.  Reads hf_config.quantization_config and runs the same
        override-detection order as ModelConfig._verify_quantization() so that
        the correct vLLM method name (e.g. "auto_gptq") is set in-place.
        """
        from vllm.model_executor.layers.quantization import get_quantization_config

        hf_cfg = alt_model_config.hf_config
        hf_quant_cfg = getattr(hf_cfg, "quantization_config", None)
        if hf_quant_cfg is None:
            text_cfg = getattr(hf_cfg, "text_config", None)
            if text_cfg is not None:
                hf_quant_cfg = getattr(text_cfg, "quantization_config", None)
        if hf_quant_cfg is None:
            return

        # Mirror the override priority from ModelConfig._verify_quantization().
        overrides = [
            "auto_gptq", "gptq", "gptq_marlin",
            "awq_marlin", "inc", "moe_wna16",
            "modelopt", "modelopt_fp4", "modelopt_mxfp8",
        ]
        for name in overrides:
            try:
                quant_cls = get_quantization_config(name)
                override = quant_cls.override_quantization_method(
                    hf_quant_cfg, None, hf_config=hf_cfg
                )
                if override is not None:
                    alt_model_config.quantization = override
                    logger.debug(
                        "Alt model quantization detected via override: %s -> %s",
                        name, override,
                    )
                    return
            except Exception:
                continue

        # No override matched; use the raw quant_method string.
        quant_method = hf_quant_cfg.get("quant_method", "").lower()
        if quant_method:
            alt_model_config.quantization = quant_method
            logger.debug(
                "Alt model quantization detected from hf_config: %s", quant_method
            )

    @override
    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._primary_model = self.model

        # Capture primary attention objects BEFORE loading the alt model.
        sfc = self.vllm_config.compilation_config.static_forward_context
        self._primary_attn_context = {
            name: attn for name, attn in sfc.items()
            if name.startswith("draft_model.")
        }

        self._alt_model_module = self._load_alt_model()
        logger.info(
            "AdaptiveDraftModelProposer ready: "
            "primary=fp8, alt=int8, high_threshold=%d, low_threshold=%d, "
            "ema_alpha=%.2f, min_dwell_steps=%d",
            self._high_threshold,
            self._low_threshold,
            self._alpha,
            self._min_dwell,
        )

    def _ensure_alt_kv_cache_bound(self) -> None:
        """Copy KV cache bindings from primary to alt attention layers.

        Called lazily on the first propose() after bind_kv_cache() has run.
        Both models share the same KV cache memory: they have identical
        architectures and alternate (never run concurrently), so sharing is
        safe and avoids allocating a second KV cache.
        """
        if self._alt_kv_cache_bound:
            return
        for name, alt_attn in self._alt_attn_context.items():
            primary_attn = self._primary_attn_context.get(name)
            if primary_attn is not None:
                alt_attn.kv_cache = primary_attn.kv_cache
        self._alt_kv_cache_bound = True

    @override
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Force eager mode so we can swap self.model freely between calls.
        self.cudagraph_dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)

    @override
    def propose(self, *args, **kwargs) -> torch.Tensor:
        # Locate common_attn_metadata from positional or keyword arguments.
        # Signature (from SpecDecodeBaseProposer.propose):
        #   propose(target_token_ids, target_positions, target_hidden_states,
        #           next_token_ids, token_indices_to_sample,
        #           common_attn_metadata, sampling_metadata, ...)
        if len(args) >= 6:
            common_attn_metadata = args[5]
        else:
            common_attn_metadata = kwargs["common_attn_metadata"]

        batch_size: int = common_attn_metadata.batch_size()
        if batch_size > self._prev_batch_size:
            # Batch grew → new large batch started.  Reset EMA so model
            # selection reflects the current load immediately instead of
            # lagging behind the previous wave's decaying batch size.
            self._ema = float(batch_size)
        else:
            self._ema = self._alpha * batch_size + (1.0 - self._alpha) * self._ema
        self._prev_batch_size = batch_size

        self._steps_since_switch += 1

        # Hysteresis: switch TO primary (fp8) only when EMA rises above
        # high_threshold; switch BACK to alt (int8) only when EMA falls below
        # low_threshold.  The dead-band between the two thresholds prevents
        # thrashing as the batch shrinks during the tail of a large wave.
        # Min-dwell guard prevents rapid back-and-forth switching that causes
        # ITL tail-latency spikes when EMA hovers near a threshold boundary.
        if self._steps_since_switch >= self._min_dwell:
            if not self._using_primary and self._ema > self._high_threshold:
                self._using_primary = True
                self._steps_since_switch = 0
            elif self._using_primary and self._ema < self._low_threshold:
                self._using_primary = False
                self._steps_since_switch = 0

        if self._using_primary:
            self.model = self._primary_model       # fp8: good for large batch
            active_attn_context = self._primary_attn_context
        else:
            self.model = self._alt_model_module    # int8: good for small batch
            active_attn_context = self._alt_attn_context

        # Share KV cache memory on first use (bind_kv_cache has run by now).
        self._ensure_alt_kv_cache_bound()

        # Swap attention layer objects in static_forward_context so that
        # attn_metadata_raw lookups and no_compile_layers lookups both resolve
        # to the active model's Attention instances.
        sfc = self.vllm_config.compilation_config.static_forward_context
        sfc.update(active_attn_context)

        return super().propose(*args, **kwargs)
