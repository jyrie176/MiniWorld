import pytest

from miniworld.train import build_grad_scaler, build_parser, validate_training_args
from miniworld.sample import build_parser as build_sample_parser


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


@pytest.mark.parametrize("backend", ["auto", "sdpa", "flash"])
def test_sampling_parser_accepts_attention_backend(backend):
    args = build_sample_parser().parse_args(
        _required_sample_args() + ["--attention_backend", backend]
    )
    assert args.attention_backend == backend
