"""Conditioning helpers for robot-action and camera-pose inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from miniworld.conditioning.poses import compute_ray_encoding, downsample_poses_to_latent


@dataclass
class ConditioningConfig:
    """Configuration for converting raw batch conditions into model conditions."""

    use_pose_cond: bool = False
    use_action_cond: bool = False
    pose_enc_freq: int = 15


def build_cond_seq_from_actions(actions: torch.Tensor) -> torch.Tensor:
    """Map raw per-frame actions to WAN-VAE latent-frame conditioning.

    WAN-VAE compresses time as 4x+1. Latent frame 0 is the seed frame and uses a
    zero action slot; each following latent frame receives the four raw actions
    that drive its decoded frame group.
    """
    batch, num_actions, action_dim = actions.shape
    if num_actions % 4 != 0:
        raise ValueError(f"Expected action length 4n for real actions, got {num_actions}")
    num_generated_latents = num_actions // 4
    cond = actions.new_zeros(batch, num_generated_latents + 1, 4 * action_dim)
    if num_generated_latents > 0:
        cond[:, 1:, :] = actions.reshape(batch, num_generated_latents, 4 * action_dim)
    return cond


def build_cond_seq_for_batch(
    *,
    cfg: ConditioningConfig,
    poses: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    t_latent: int,
    h_lat: int,
    w_lat: int,
) -> torch.Tensor:
    """Build the MiniWorld conditioning tensor for one batch."""
    if cfg.use_pose_cond:
        if poses is None:
            raise ValueError("Pose conditioning requested but batch has no poses")
        latent_poses = downsample_poses_to_latent(poses, t_latent)
        return compute_ray_encoding(latent_poses, h_lat, w_lat, freq=cfg.pose_enc_freq, normalize_trans=False)
    if cfg.use_action_cond:
        if actions is None:
            raise ValueError("Action conditioning requested but batch has no actions")
        return build_cond_seq_from_actions(actions)
    raise ValueError("Either pose or action conditioning must be enabled")
