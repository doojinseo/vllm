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

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.draft_model import DraftModelProposer

logger = init_logger(__name__)


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
        self._ema: float = 0.0
        self._using_primary: bool = False   # start on alt (int8) until EMA rises
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
            "primary=fp8, alt=int8, high_threshold=%d, low_threshold=%d, ema_alpha=%.2f",
            self._high_threshold,
            self._low_threshold,
            self._alpha,
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
        self._ema = self._alpha * batch_size + (1.0 - self._alpha) * self._ema

        # Hysteresis: switch TO primary (fp8) only when EMA rises above
        # high_threshold; switch BACK to alt (int8) only when EMA falls below
        # low_threshold.  The dead-band between the two thresholds prevents
        # thrashing as the batch shrinks during the tail of a large wave.
        if not self._using_primary and self._ema > self._high_threshold:
            self._using_primary = True
        elif self._using_primary and self._ema < self._low_threshold:
            self._using_primary = False

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
