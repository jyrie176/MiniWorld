"""Camera pose -> ray-encoding utilities for pose-conditioned world model.

Adapted from `GeometryForcing/utils/geometry_utils.py` with the following
simplifications:
  * Only the bits needed for ``ray_encoding`` conditioning are kept (the
    variant the user picked as the best-performing one).
  * ``rays`` accepts independent ``(h_res, w_res)`` so non-square latents
    (e.g. 15x20) are supported without distorting the intrinsics.

All functions follow this convention:
  * Raw camera pose layout: ``(B, T, 16)`` = ``[K(4), R(9 + T(3))]`` where the
    first 4 columns are normalised intrinsics ``(fx, fy, px, py)`` (pixel-coords
    divided by image size) and the last 12 columns are a flattened ``3x4``
    world-to-camera extrinsics matrix in row-major.
  * Ray encoding output: ``(B, T, 180, H_lat, W_lat)`` (6 ray dims * 2 trig fns
    * 15 NeRF frequencies = 180). This matches what `DiT3DPose` consumes when
    ``conditioning_type=ray_encoding``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from einops import einsum, rearrange, repeat


def _split_pose16(raw_poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(B, T, 16)`` -> ``R (B, T, 3, 3)``, ``T (B, T, 3)``, ``K (B, T, 4)``."""
    assert raw_poses.shape[-1] == 16, f"expected 16-dim pose, got {raw_poses.shape[-1]}"
    K, RT = raw_poses.split([4, 12], dim=-1)
    RT = rearrange(RT, "b t (i j) -> b t i j", i=3, j=4)
    R = RT[..., :3, :3]
    T = RT[..., :3, 3]
    return R, T, K


def _normalize_by_first(R: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-express all poses so the first frame is the world origin."""
    R_ref = R[:, 0]  # (B, 3, 3)
    T_ref = T[:, 0]  # (B, 3)
    R_inv = rearrange(R_ref, "b i j -> b j i")
    R_new = einsum(R, R_inv, "b t i j1, b j1 j2 -> b t i j2")
    T_new = T - einsum(R_new, T_ref, "b t i j, b j -> b t i")
    return R_new, T_new


def _compute_rays(
    R: torch.Tensor,
    T: torch.Tensor,
    K: torch.Tensor,
    h_res: int,
    w_res: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel ray origin / direction in world coords.

    Args:
        R: ``(B, T, 3, 3)`` world->cam rotation.
        T: ``(B, T, 3)`` world->cam translation.
        K: ``(B, T, 4)`` normalised intrinsics ``(fx, fy, px, py)``.
        h_res, w_res: target ray grid resolution (independent so non-square
            latents are handled correctly).

    Returns:
        origin: ``(B, T, H, W, 3)``
        direction: ``(B, T, H, W, 3)`` (unnormalised; norm encodes depth scale)
    """
    device, dtype = K.device, K.dtype

    coord_w, coord_h = torch.meshgrid(
        torch.linspace(0, w_res - 1, w_res, device=device, dtype=dtype),
        torch.linspace(0, h_res - 1, h_res, device=device, dtype=dtype),
        indexing="xy",
    )  # (H, W) each
    coord_w = rearrange(coord_w, "h w -> 1 1 h w") + 0.5
    coord_h = rearrange(coord_h, "h w -> 1 1 h w") + 0.5

    # Normalised K -> pixel-space K (separate W / H scaling for non-square grids).
    fx = (K[..., 0] * w_res).view(*K.shape[:-1], 1, 1)  # (B, T, 1, 1)
    fy = (K[..., 1] * h_res).view(*K.shape[:-1], 1, 1)
    px = (K[..., 2] * w_res).view(*K.shape[:-1], 1, 1)
    py = (K[..., 3] * h_res).view(*K.shape[:-1], 1, 1)

    x = (coord_w - px) / fx
    y = (coord_h - py) / fy
    z = torch.ones_like(x)
    direction = torch.stack([x, y, z], dim=-1)  # (B, T, H, W, 3)

    R_inv = rearrange(R, "b t i j -> b t j i")
    direction = einsum(R_inv, direction, "b t i j, b t h w j -> b t h w i")

    origin = -einsum(R_inv, T, "b t i j, b t j -> b t i")
    origin = repeat(origin, "b t i -> b t h w i", h=h_res, w=w_res).clone()
    return origin, direction


def _normalize_translation_scale(T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-clip translation-scale normalisation (CameraCtrl / lingbot style).

    Monocular SfM poses (e.g. RealEstate10K) have an arbitrary, per-clip metric
    scale, so raw camera translations vary wildly in magnitude across clips.
    Since the ray origin feeds NeRF frequency encoding ``sin(2^k pi x)`` -- which
    is very sensitive to the absolute magnitude of ``x`` -- this inconsistency
    hurts learning.  We rescale each clip so its largest camera displacement is
    ~1, making camera motion scale-invariant across clips.

    Args:
        T: ``(B, S, 3)`` camera translations, already expressed relative to the
            first frame (so the first frame sits at the origin).
        eps: guard so static / near-static clips (max norm ~ 0) are left
            unchanged ("only normalize when moving").

    Returns:
        ``(B, S, 3)`` translations divided by the per-clip max translation norm.
    """
    max_norm = torch.norm(T, dim=-1).amax(dim=1, keepdim=True)  # (B, 1)
    scale = torch.where(max_norm > eps, max_norm, torch.ones_like(max_norm))
    return T / scale.unsqueeze(-1)


def _nerf_pos_encoding(x: torch.Tensor, freq: int) -> torch.Tensor:
    """NeRF-style sin/cos positional encoding along the last dim."""
    scale = (
        2 ** torch.linspace(0, freq - 1, freq, device=x.device, dtype=x.dtype)
        * math.pi
    )
    encoding = rearrange(x[..., None] * scale, "b t h w i s -> b t h w (i s)")
    return torch.sin(torch.cat([encoding, encoding + 0.5 * math.pi], dim=-1))


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)  # always fp32 for geometry
def compute_ray_encoding(
    raw_poses: torch.Tensor,
    h_lat: int,
    w_lat: int,
    freq: int = 15,
    normalize_trans: bool = False,
) -> torch.Tensor:
    """End-to-end raw poses -> ray-encoding feature volume.

    Args:
        raw_poses: either ``(B, T, 16)`` or ``(B, T, K, 16)``. MiniWorld's RE10K
            pipeline uses ``K=4`` poses inside each WAN-VAE latent chunk.
        h_lat, w_lat: latent spatial size (= model input H, W after VAE).
        freq: NeRF frequency count. ``freq=15`` gives ``6 * 2 * 15 = 180``
            channels per pose.
        normalize_trans: if True, rescale each clip's camera
            translations so the largest displacement is ~1 (see
            ``_normalize_translation_scale``). Disabled by default to match
            DFoT's RealEstate10K preprocessing; static clips are untouched.

    Returns:
        ``(B, T, K * 6 * 2 * freq, H_lat, W_lat)`` float32. The ``K`` poses are
        ray-encoded independently then concatenated along the channel axis
        (so the spatial ``y_embedder`` sees ``K * 180`` channels). For the
        common ``(B, T, 16)`` input ``K=1`` and the output channel count is
        ``180``.
    """
    assert raw_poses.dim() in (3, 4), (
        f"raw_poses must be (B, T, 16) or (B, T, K, 16); got {raw_poses.shape}"
    )
    raw_poses = raw_poses.float()
    if raw_poses.dim() == 3:
        b, t_lat, _ = raw_poses.shape
        k_per_lat = 1
        flat = raw_poses  # (B, T, 16)
    else:
        b, t_lat, k_per_lat, _ = raw_poses.shape
        # Flatten K into the time axis so we can reuse the single-pose pipeline
        # (one shared normalisation anchor = first pose in the sequence).
        flat = raw_poses.reshape(b, t_lat * k_per_lat, 16)

    R, T, K = _split_pose16(flat)
    R, T = _normalize_by_first(R, T)
    if normalize_trans:
        T = _normalize_translation_scale(T)
    origin, direction = _compute_rays(R, T, K, h_res=h_lat, w_res=w_lat)
    enc = torch.cat(
        [
            _nerf_pos_encoding(origin, freq),
            _nerf_pos_encoding(direction, freq),
        ],
        dim=-1,
    )  # (B, T*K, H, W, 6 * 2 * freq)

    if k_per_lat == 1:
        return rearrange(enc, "b t h w c -> b t c h w").contiguous()
    return rearrange(
        enc, "b (t k) h w c -> b t (k c) h w", t=t_lat, k=k_per_lat,
    ).contiguous()


def downsample_poses_to_latent(
    raw_poses: torch.Tensor,
    t_latent: int,
) -> torch.Tensor:
    """Map per-raw-frame poses to four poses per WAN-style latent frame.

    The causal WAN VAE encodes ``T_raw = 4*(T_lat-1)+1`` raw frames into
    ``T_lat`` latents with the temporal grouping:
        * latent 0      -> raw [0]
        * latent j (>0) -> raw [4j-3, 4j-2, 4j-1, 4j]

    Latent 0 has only raw[0], so it is duplicated four times to keep the output
    shape consistent with action conditioning: ``(B, T_lat, 4, 16)``.
    """
    idx_per_latent = [[0, 0, 0, 0]]
    for j in range(1, t_latent):
        idx_per_latent.append([4 * j - 3, 4 * j - 2, 4 * j - 1, 4 * j])
    idx_flat = [i for chunk in idx_per_latent for i in chunk]
    assert raw_poses.shape[1] > max(idx_flat), (
        f"raw_poses has only {raw_poses.shape[1]} frames; need at least "
        f"{max(idx_flat) + 1} to build {t_latent} latent poses."
    )
    b = raw_poses.shape[0]
    return raw_poses[:, idx_flat].view(b, t_latent, 4, 16).contiguous()
