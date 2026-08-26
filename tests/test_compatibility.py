import pytest
import torch

from miniworld.compatibility import (
    flash_attention_available,
    resolve_attention_backend,
    resolve_sample_precision,
    resolve_training_dtype,
)


def test_flash_attention_availability_uses_import_spec(monkeypatch):
    monkeypatch.setattr(
        "miniworld.compatibility.importlib.util.find_spec",
        lambda name: object() if name == "flash_attn" else None,
    )
    assert flash_attention_available() is True


def test_v100_auto_attention_uses_sdpa():
    assert resolve_attention_backend(
        "auto",
        cuda_available=True,
        capability=(7, 0),
        flash_available=False,
    ) == "sdpa"


def test_supported_gpu_auto_attention_uses_available_flash():
    assert resolve_attention_backend(
        "auto",
        cuda_available=True,
        capability=(8, 0),
        flash_available=True,
    ) == "flash"


def test_explicit_sdpa_does_not_require_cuda():
    assert resolve_attention_backend(
        "sdpa",
        cuda_available=False,
        capability=None,
        flash_available=False,
    ) == "sdpa"


@pytest.mark.parametrize(
    ("cuda_available", "capability", "flash_available", "detected"),
    [
        (False, None, False, "CUDA unavailable"),
        (True, (7, 0), True, "7.0"),
        (True, (8, 0), False, "package unavailable"),
    ],
)
def test_explicit_unavailable_flash_has_actionable_error(
    cuda_available, capability, flash_available, detected
):
    with pytest.raises(RuntimeError) as exc_info:
        resolve_attention_backend(
            "flash",
            cuda_available=cuda_available,
            capability=capability,
            flash_available=flash_available,
        )

    message = str(exc_info.value)
    assert "FlashAttention" in message
    assert detected in message
    assert "sdpa" in message


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("no", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ],
)
def test_training_precision_maps_to_dtype(precision, expected):
    assert resolve_training_dtype(precision) is expected


@pytest.mark.parametrize(
    ("cuda_available", "capability", "expected"),
    [
        (False, None, (torch.float32, False)),
        (True, (7, 0), (torch.float16, True)),
        (True, (8, 0), (torch.bfloat16, True)),
    ],
)
def test_sample_auto_precision_uses_device_capability(
    cuda_available, capability, expected
):
    assert resolve_sample_precision(
        "auto",
        cuda_available=cuda_available,
        capability=capability,
    ) == expected


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_half_precision_sampling_rejects_cpu(precision):
    with pytest.raises(RuntimeError, match="CPU.*fp32"):
        resolve_sample_precision(
            precision,
            cuda_available=False,
            capability=None,
        )


@pytest.mark.parametrize("requested", ["invalid", ""])
def test_invalid_attention_backend_is_rejected(requested):
    with pytest.raises(ValueError, match="attention backend"):
        resolve_attention_backend(
            requested,
            cuda_available=False,
            capability=None,
            flash_available=False,
        )


@pytest.mark.parametrize("precision", ["invalid", ""])
def test_invalid_training_precision_is_rejected(precision):
    with pytest.raises(ValueError, match="training precision"):
        resolve_training_dtype(precision)


@pytest.mark.parametrize("precision", ["invalid", ""])
def test_invalid_sample_precision_is_rejected(precision):
    with pytest.raises(ValueError, match="sample precision"):
        resolve_sample_precision(
            precision,
            cuda_available=False,
            capability=None,
        )
