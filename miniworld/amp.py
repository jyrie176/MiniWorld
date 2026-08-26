"""Mixed-precision optimizer-step helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class AmpStepResult:
    grad_norm: float
    loss_scale: float
    skipped: bool
    finite: bool


def _current_scale(scaler) -> float:
    return float(scaler.get_scale()) if scaler is not None else 1.0


def _compute_grad_norm(parameters) -> torch.Tensor:
    gradients = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)
    return torch.stack([gradient.float() for gradient in gradients]).norm(2)


def backward_and_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    *,
    scaler=None,
    max_grad_norm: float = 1.0,
) -> AmpStepResult:
    """Backpropagate and step once, returning finite/scale telemetry."""
    parameter_list = list(parameters)
    if not bool(torch.isfinite(loss.detach()).all().item()):
        return AmpStepResult(
            grad_norm=float("nan"),
            loss_scale=_current_scale(scaler),
            skipped=True,
            finite=False,
        )

    scale_before = _current_scale(scaler)
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

    if max_grad_norm > 0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(parameter_list, max_grad_norm)
    else:
        grad_norm_tensor = _compute_grad_norm(parameter_list)
    grad_norm = float(grad_norm_tensor.detach().item())
    finite = bool(torch.isfinite(grad_norm_tensor).all().item())

    if scaler is None:
        optimizer.step()
        scale_after = 1.0
        skipped = False
    else:
        scaler.step(optimizer)
        scaler.update()
        scale_after = _current_scale(scaler)
        skipped = scale_after < scale_before

    return AmpStepResult(
        grad_norm=grad_norm,
        loss_scale=scale_after,
        skipped=skipped,
        finite=finite and not skipped,
    )
