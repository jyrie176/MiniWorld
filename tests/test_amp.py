import torch

from miniworld.amp import AmpStepResult, backward_and_step


class _ScaledLoss:
    def __init__(self, calls):
        self.calls = calls

    def backward(self):
        self.calls.append("backward")


class _FakeScaler:
    def __init__(self, calls, before=1024.0, after=1024.0):
        self.calls = calls
        self.before = before
        self.after = after
        self.updated = False

    def scale(self, loss):
        self.calls.append("scale")
        return _ScaledLoss(self.calls)

    def unscale_(self, optimizer):
        self.calls.append("unscale")

    def step(self, optimizer):
        self.calls.append("step")

    def update(self):
        self.calls.append("update")
        self.updated = True

    def get_scale(self):
        return self.after if self.updated else self.before


class _FakeOptimizer:
    def __init__(self, calls):
        self.calls = calls

    def step(self):
        self.calls.append("optimizer.step")


def test_scaled_step_unscales_before_clipping(monkeypatch):
    calls = []
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    scaler = _FakeScaler(calls)
    optimizer = _FakeOptimizer(calls)

    def fake_clip(parameters, max_norm):
        assert list(parameters) == [parameter]
        assert max_norm == 1.0
        calls.append("clip")
        return torch.tensor(0.25)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)

    result = backward_and_step(
        torch.tensor(2.0),
        optimizer,
        [parameter],
        scaler=scaler,
        max_grad_norm=1.0,
    )

    assert calls == ["scale", "backward", "unscale", "clip", "step", "update"]
    assert result == AmpStepResult(
        grad_norm=0.25,
        loss_scale=1024.0,
        skipped=False,
        finite=True,
    )


def test_unscaled_step_backpropagates_clips_and_steps():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    loss = parameter.square().sum()

    result = backward_and_step(
        loss,
        optimizer,
        [parameter],
        scaler=None,
        max_grad_norm=1.0,
    )

    assert parameter.item() < 2.0
    assert result.loss_scale == 1.0
    assert result.skipped is False
    assert result.finite is True
    assert result.grad_norm == 4.0


def test_scaled_step_reports_overflow_when_scale_decreases(monkeypatch):
    calls = []
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    scaler = _FakeScaler(calls, before=1024.0, after=512.0)
    optimizer = _FakeOptimizer(calls)
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda parameters, max_norm: torch.tensor(float("inf")),
    )

    result = backward_and_step(
        torch.tensor(2.0),
        optimizer,
        [parameter],
        scaler=scaler,
        max_grad_norm=1.0,
    )

    assert result.loss_scale == 512.0
    assert result.skipped is True
    assert result.finite is False


def test_nonfinite_loss_does_not_backward_or_step():
    calls = []
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    scaler = _FakeScaler(calls)

    result = backward_and_step(
        torch.tensor(float("nan")),
        _FakeOptimizer(calls),
        [parameter],
        scaler=scaler,
        max_grad_norm=1.0,
    )

    assert calls == []
    assert result.skipped is True
    assert result.finite is False
    assert result.loss_scale == 1024.0


def test_unscaled_nonfinite_gradient_does_not_step(monkeypatch):
    calls = []
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = _FakeOptimizer(calls)
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda parameters, max_norm: torch.tensor(float("inf")),
    )

    result = backward_and_step(
        parameter.square().sum(),
        optimizer,
        [parameter],
        scaler=None,
        max_grad_norm=1.0,
    )

    assert "optimizer.step" not in calls
    assert result.skipped is True
    assert result.finite is False
