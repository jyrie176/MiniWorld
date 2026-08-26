import pytest
import torch

from miniworld.amp import backward_and_step
from miniworld.compatibility import resolve_attention_backend
from miniworld.miniworld import Attention


pytestmark = pytest.mark.cuda


def _require_target_v100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability != (7, 0):
        pytest.skip(f"target smoke test requires V100/SM70, got {capability}")
    return torch.device("cuda:0"), capability


def _attention(device):
    torch.manual_seed(17)
    return Attention(
        dim=64,
        num_heads=2,
        qkv_bias=True,
        qk_norm=False,
        attention_backend="sdpa",
    ).to(device)


def test_v100_auto_resolves_to_sdpa():
    _, capability = _require_target_v100()
    assert resolve_attention_backend(
        "auto",
        cuda_available=True,
        capability=capability,
        flash_available=False,
    ) == "sdpa"


def test_v100_fp16_sdpa_backward_and_scaled_adamw_step_are_finite():
    device, _ = _require_target_v100()
    torch.cuda.reset_peak_memory_stats(device)
    attention = _attention(device)
    optimizer = torch.optim.AdamW(attention.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    x = torch.randn(2, 16, 64, device=device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = attention(x)
        loss = output.float().square().mean()
    result = backward_and_step(
        loss,
        optimizer,
        attention.parameters(),
        scaler=scaler,
        max_grad_norm=1.0,
    )

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert result.finite is True
    assert result.skipped is False
    assert result.grad_norm >= 0.0
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    print(f"peak_memory_bytes={peak_memory_bytes}")
    assert peak_memory_bytes > 0

    restored_scaler = torch.amp.GradScaler("cuda", enabled=True)
    restored_scaler.load_state_dict(scaler.state_dict())
    assert restored_scaler.state_dict() == scaler.state_dict()


def test_v100_fp16_sdpa_matches_fp32_reference():
    device, _ = _require_target_v100()
    attention = _attention(device).eval()
    x = torch.randn(2, 16, 64, device=device)

    with torch.no_grad():
        reference = attention(x.float())
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            actual = attention(x)

    absolute_error = (actual.float() - reference).abs()
    max_absolute_error = float(absolute_error.max())
    max_relative_error = float(
        (absolute_error / reference.abs().clamp_min(1e-4)).max()
    )
    print(
        f"max_absolute_error={max_absolute_error:.8f} "
        f"max_relative_error={max_relative_error:.8f}"
    )
    torch.testing.assert_close(
        actual.float(), reference, rtol=5e-3, atol=5e-3
    )
