import types

import torch
import torch.nn as nn

from miniworld.denoiser import Denoiser


class _PerfectVelocity(nn.Module):
    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, *args, **kwargs):
        return self.velocity


def test_returned_clean_prediction_inverts_forward_noising(monkeypatch):
    clean = torch.tensor([[[[[0.25]], [[-0.5]]]]])
    noise = torch.tensor([[[[[-0.75]], [[0.5]]]]])
    timesteps = torch.tensor([[0.9, 0.9]])
    clean_mask = torch.zeros_like(timesteps)

    denoiser = Denoiser.__new__(Denoiser)
    nn.Module.__init__(denoiser)
    denoiser.df_chunk_size = 2
    denoiser.net = _PerfectVelocity(clean - noise)
    denoiser._build_diffusion_forcing_timesteps = types.MethodType(
        lambda self, **kwargs: (timesteps, [slice(0, 2)], timesteps[:, :1], clean_mask),
        denoiser,
    )
    monkeypatch.setattr(torch, "randn_like", lambda tensor: noise.clone())

    _, predicted_clean, _ = denoiser.forward_diffusion_forcing(
        clean,
        torch.zeros(1, 2, 1),
        return_pred=True,
    )

    torch.testing.assert_close(predicted_clean, clean)
