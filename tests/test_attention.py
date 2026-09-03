import torch
import torch.nn as nn

from miniworld.denoiser import Denoiser, DenoiserConfig
from miniworld.miniworld import (
    Attention,
    MiniWorldModel,
    _build_cached_block_causal_mask,
    _build_temporal_chunkwise_attn_mask,
)


def _make_attention() -> Attention:
    torch.manual_seed(7)
    return Attention(
        dim=8,
        num_heads=2,
        qkv_bias=True,
        qk_norm=False,
        proj_drop=0.0,
        attention_backend="sdpa",
    )


def test_sdpa_unmasked_forward_and_backward_are_finite():
    attention = _make_attention()
    x = torch.randn(2, 5, 8, requires_grad=True)

    output = attention(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attention.parameters()
    )


def test_sdpa_unmasked_matches_explicit_fp32_reference():
    attention = _make_attention().eval()
    x = torch.randn(2, 5, 8)

    actual = attention(x)

    batch, tokens, channels = x.shape
    qkv = attention.qkv(x).reshape(
        batch, tokens, 3, attention.num_heads, attention.head_dim
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    weights = torch.softmax(
        torch.matmul(query, key.transpose(-2, -1)) * attention.scale,
        dim=-1,
    )
    reference = torch.matmul(weights, value)
    reference = reference.transpose(1, 2).reshape(batch, tokens, channels)
    reference = attention.proj_drop(attention.proj(reference))

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def _make_model(attention_backend: str) -> MiniWorldModel:
    return MiniWorldModel(
        in_channels=2,
        hidden_size=8,
        cond_dim=3,
        depth=1,
        num_heads=2,
        patch_size=1,
        input_size=(2, 2),
        num_frames=2,
        mlp_ratio=2.0,
        use_qknorm=False,
        cond_per_token=False,
        adaln_lora_dim=2,
        attention_backend=attention_backend,
    )


def test_model_propagates_resolved_attention_backend_without_state_changes():
    sdpa_model = _make_model("sdpa")
    flash_model = _make_model("flash")

    assert sdpa_model.blocks[0].attn.attention_backend == "sdpa"
    assert flash_model.blocks[0].attn.attention_backend == "flash"
    assert sdpa_model.state_dict().keys() == flash_model.state_dict().keys()
    assert {
        name: tensor.shape for name, tensor in sdpa_model.state_dict().items()
    } == {
        name: tensor.shape for name, tensor in flash_model.state_dict().items()
    }


def test_temporal_mask_is_bidirectional_within_chunk_and_causal_across_chunks():
    mask = _build_temporal_chunkwise_attn_mask(
        seq_len=8,
        tokens_per_frame=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=2,
    )[0, 0]

    visible = torch.isfinite(mask)
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(visible, expected)
    assert torch.equal(mask[visible], torch.zeros_like(mask[visible]))
    assert torch.isneginf(mask[~visible]).all()


def test_cached_mask_exposes_all_past_and_keeps_current_block_causal():
    mask = _build_cached_block_causal_mask(
        n_past=3,
        n_cur=8,
        tokens_per_frame=2,
        chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[0, 0]

    assert mask.shape == (8, 11)
    assert torch.isfinite(mask[:, :3]).all()
    current_visible = torch.isfinite(mask[:, 3:])
    assert current_visible[:4, :4].all()
    assert not current_visible[:4, 4:].any()
    assert current_visible[4:].all()


def test_block_causal_mask_prevents_future_chunk_perturbation():
    attention = _make_attention().eval()
    x = torch.randn(1, 8, 8)
    changed = x.clone()
    changed[:, 4:] += 100.0
    mask = _build_temporal_chunkwise_attn_mask(
        seq_len=8,
        tokens_per_frame=2,
        device=x.device,
        dtype=x.dtype,
        chunk_size=2,
    )

    original_output = attention(x, attn_mask=mask)
    changed_output = attention(changed, attn_mask=mask)

    torch.testing.assert_close(original_output[:, :4], changed_output[:, :4])
    assert not torch.allclose(original_output[:, 4:], changed_output[:, 4:])


def test_cached_attention_responds_to_past_values():
    attention = _make_attention().eval()
    current = torch.randn(1, 4, 8)
    past_source = torch.randn(1, 3, 8)
    _, (past_key, past_value) = attention(past_source, return_kv=True)
    mask = _build_cached_block_causal_mask(
        n_past=3,
        n_cur=4,
        tokens_per_frame=2,
        chunk_size=2,
        device=current.device,
        dtype=current.dtype,
    )

    original = attention(current, attn_mask=mask, past_kv=(past_key, past_value))
    changed = attention(
        current,
        attn_mask=mask,
        past_kv=(past_key, past_value + 10.0),
    )

    assert not torch.allclose(original, changed)


def test_denoiser_propagates_resolved_attention_backend(monkeypatch):
    captured = {}

    class DummyModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setitem(
        __import__("miniworld.miniworld", fromlist=["MiniWorldModels"]).MiniWorldModels,
        "test",
        DummyModel,
    )
    denoiser = Denoiser(
        DenoiserConfig(wm_model="test", attention_backend="sdpa")
    )

    assert isinstance(denoiser.net, DummyModel)
    assert captured["attention_backend"] == "sdpa"
