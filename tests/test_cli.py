import pytest
import torch

from miniworld.train import build_grad_scaler, build_parser, validate_training_args
from miniworld.sample import apply_action_variant, build_parser as build_sample_parser


def _required_train_args():
    return [
        "--dataset",
        "droid",
        "--data_root",
        "/data",
        "--vae_checkpoint",
        "/vae.pt",
    ]


@pytest.mark.parametrize("precision", ["no", "fp16", "bf16"])
def test_training_parser_accepts_supported_precision(precision):
    args = build_parser().parse_args(
        _required_train_args() + ["--mixed_precision", precision]
    )
    assert args.mixed_precision == precision


@pytest.mark.parametrize("backend", ["auto", "sdpa", "flash"])
def test_training_parser_accepts_attention_backend(backend):
    args = build_parser().parse_args(
        _required_train_args() + ["--attention_backend", backend]
    )
    assert args.attention_backend == backend


def test_training_parser_defaults_gradient_clip_to_one():
    args = build_parser().parse_args(_required_train_args())
    assert args.max_grad_norm == 1.0


def test_training_parser_rejects_fp16_muon_combination():
    args = build_parser().parse_args(
        _required_train_args()
        + ["--mixed_precision", "fp16", "--use_muon"]
    )
    with pytest.raises(ValueError, match="FP16.*Muon"):
        validate_training_args(args)


@pytest.mark.parametrize(
    ("precision", "expected_enabled"),
    [("fp16", True), ("bf16", False), ("no", False)],
)
def test_grad_scaler_is_enabled_only_for_cuda_fp16(
    monkeypatch, precision, expected_enabled
):
    captured = {}

    def fake_grad_scaler(device, *, enabled):
        captured.update(device=device, enabled=enabled)
        return object()

    monkeypatch.setattr("miniworld.train.torch.amp.GradScaler", fake_grad_scaler)

    build_grad_scaler(precision, cuda_available=True)

    assert captured == {"device": "cuda", "enabled": expected_enabled}


def _required_sample_args():
    return [
        "--dataset",
        "droid",
        "--checkpoint",
        "/model.pt",
        "--vae_checkpoint",
        "/vae.pt",
    ]


@pytest.mark.parametrize("precision", ["auto", "fp16", "bf16", "fp32"])
def test_sampling_parser_accepts_supported_precision(precision):
    args = build_sample_parser().parse_args(
        _required_sample_args() + ["--precision", precision]
    )
    assert args.precision == precision


def test_sampling_parser_accepts_reproducible_action_controls():
    args = build_sample_parser().parse_args(
        _required_sample_args() + ["--seed", "123", "--action_variant", "shuffle"]
    )

    assert args.seed == 123
    assert args.action_variant == "shuffle"


def test_zero_action_variant_removes_action_conditioning():
    actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    transformed = apply_action_variant(actions, "zero")

    torch.testing.assert_close(transformed, torch.zeros_like(actions))


def test_shuffle_action_variant_reverses_time_without_mutating_input():
    actions = torch.tensor([[[1.0], [2.0], [3.0]]])

    transformed = apply_action_variant(actions, "shuffle")

    torch.testing.assert_close(transformed, torch.tensor([[[3.0], [2.0], [1.0]]]))
    torch.testing.assert_close(actions, torch.tensor([[[1.0], [2.0], [3.0]]]))


@pytest.mark.parametrize("backend", ["auto", "sdpa", "flash"])
def test_sampling_parser_accepts_attention_backend(backend):
    args = build_sample_parser().parse_args(
        _required_sample_args() + ["--attention_backend", backend]
    )
    assert args.attention_backend == backend
