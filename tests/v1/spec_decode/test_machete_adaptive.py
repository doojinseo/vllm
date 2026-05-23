# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import torch


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

    # Call with n_tokens=4 < 8; verify exactly one new machete_mm call is made.
    layer = MagicMock()
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    before_count = mock_ops.machete_mm.call_count
    kernel.apply_weights(layer, x)
    assert mock_ops.machete_mm.call_count == before_count + 1
    assert mock_ops.machete_mm.call_args.kwargs["schedule"] == "sched_small"


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
    before_count = mock_ops.machete_mm.call_count
    kernel.apply_weights(layer, x)
    assert mock_ops.machete_mm.call_count == before_count + 1
    assert mock_ops.machete_mm.call_args.kwargs["schedule"] == "sched_large"


@patch("torch.cuda.synchronize")
@patch("vllm.v1.spec_decode.adaptive_draft_model._profile_machete_schedules")
@patch("vllm.v1.spec_decode.adaptive_draft_model.ops")
def test_install_skips_failed_profile_and_does_not_crash(mock_ops, mock_profile, mock_sync):
    """When _profile_machete_schedules returns (None, None), install does not crash
    and does not replace kernel.apply_weights."""
    from vllm.v1.spec_decode.adaptive_draft_model import _install_adaptive_machete_schedules

    mock_profile.return_value = (None, None)

    kernel = _make_fake_kernel()
    original_apply_weights = kernel.apply_weights

    with patch(
        "vllm.v1.spec_decode.adaptive_draft_model.MacheteLinearKernel",
        type(kernel),
    ):
        module = MagicMock()
        module.quant_method = MagicMock()
        module.quant_method.kernel = kernel

        model = MagicMock()
        model.modules.return_value = [module]

        # Must not raise.
        _install_adaptive_machete_schedules(model, threshold=8)

    # apply_weights must not have been replaced.
    assert kernel.apply_weights is original_apply_weights
