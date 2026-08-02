"""Procedural camera-trajectory utilities for pose-conditioned WM inference.

Builds ``(T, 16)`` pose tensors compatible with
``model.pose_utils.compute_ray_encoding`` so we can drive the world model
with **any** camera path (no GT video / poses required).

Pose layout (per frame, matches ``model.pose_utils._split_pose16``):
    ``[fx, fy, px, py, R(9, row-major), T(3)]`` = ``[K(4), RT(12)]``
where K is normalised (intrinsics divided by image W / H) and
(R, T) is world->camera (OpenCV convention: x=right, y=down, z=forward).

All trajectories here **start at the identity pose** at frame 0
(R=I, T=0). ``compute_ray_encoding`` will re-anchor the first frame as
the world origin anyway, so only relative camera motion w.r.t. frame 0
ever reaches the model.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch


###############################################################################
#                       Low-level rotation / look-at                          #
###############################################################################


def _yaw(a: float) -> np.ndarray:
    """Rotation around world-down axis (=+y). ``a > 0`` -> camera pans right
    (the world's +x moves toward camera-forward)."""
    c, s = math.cos(a), math.sin(a)
    # camera basis in world: right=(c,0,-s), down=(0,1,0), forward=(s,0,c)
    # R_w2c rows = [right; down; forward]
    return np.array(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _pitch(a: float) -> np.ndarray:
    """Rotation around world-right axis (=+x). ``a > 0`` -> camera tilts up."""
    c, s = math.cos(a), math.sin(a)
    # camera basis: right=(1,0,0), down=(0,c,s), forward=(0,-s,c)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=np.float64,
    )


def _look_at(
    eye: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray = np.array([0.0, -1.0, 0.0]),
) -> Tuple[np.ndarray, np.ndarray]:
    """world->camera (R, T) for a camera at ``eye`` looking at ``target``.

    OpenCV convention: camera frame is (right, down, forward) = (+x, +y, +z).
    ``world_up`` points along the world's "visual up" direction; in OpenCV
    image-y is down, so the canonical world-up is ``(0, -1, 0)``.
    """
    fwd = target - eye
    n = float(np.linalg.norm(fwd))
    if n < 1e-8:
        # Degenerate: fall back to identity orientation.
        R = np.eye(3, dtype=np.float64)
        T = -R @ eye
        return R, T
    fwd = fwd / n

    down_world = -world_up
    right = np.cross(down_world, fwd)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        # forward parallel to up -> pick any perpendicular right
        right = np.array([1.0, 0.0, 0.0])
        if abs(float(fwd @ right)) > 0.99:
            right = np.array([0.0, 0.0, 1.0])
    else:
        right = right / rn
    down = np.cross(fwd, right)

    R = np.stack([right, down, fwd], axis=0).astype(np.float64)  # world->cam rows
    T = -R @ eye
    return R, T


###############################################################################
#                       Trajectory primitives                                 #
###############################################################################


def _build_RT(traj_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]], num_frames: int) -> np.ndarray:
    """Sample ``traj_fn(s)`` at ``num_frames`` evenly-spaced ``s`` in [0, 1]
    and return a ``(num_frames, 12)`` row-major flattened RT.

    ``traj_fn(0.0)`` is expected to return the identity pose (R=I, T=0) so
    frame 0 anchors the world origin cleanly.
    """
    out = np.zeros((num_frames, 12), dtype=np.float32)
    for i in range(num_frames):
        s = i / max(num_frames - 1, 1)
        R, T = traj_fn(s)
        out[i, :9] = R.reshape(-1)
        out[i, 9:] = T.reshape(-1)
    return out


SUPPORTED_TRAJECTORIES = (
    "static",
    "forward",
    "backward",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "orbit_right",
    "orbit_left",
    "spiral",
    "zoom_in",
    "zoom_out",
)


def build_custom_trajectory(
    traj_type: str,
    num_frames: int,
    focal_norm: float = 0.7,
    magnitude: float = 1.0,
) -> torch.Tensor:
    """Build a ``(num_frames, 16)`` pose sequence for a named procedural path.

    Args:
        traj_type: one of :data:`SUPPORTED_TRAJECTORIES`.
        num_frames: number of *raw* frames the pose sequence must cover
            (i.e. ``eval_t_dataset = 4*(total_len-1)+2`` -- e.g. 126 for
            ``total_len=32``). ``compute_ray_encoding`` indexes into this.
        focal_norm: normalised focal length (``fx = fy = focal_norm``). RE10K
            videos typically sit near 0.5-1.0; smaller = wider FOV.
        magnitude: global scaling. At ``magnitude=1.0`` the defaults are:
              * translate ~0.5 units.
              * rotate up to 30 deg (pan / tilt).
              * orbit / spiral: 60 deg arc on a radius-``magnitude`` circle.
              * zoom: focal scales linearly to 1.5x (in) / 0.67x (out).
            For **rawscale** RE10K checkpoints (``normalize_trans=False``),
            ``magnitude=1.0`` looks nearly static.

    Returns:
        ``(num_frames, 16)`` float32 tensor on cpu.
    """
    if traj_type not in SUPPORTED_TRAJECTORIES:
        raise ValueError(
            f"Unknown trajectory '{traj_type}'. "
            f"Supported: {SUPPORTED_TRAJECTORIES}"
        )

    I = np.eye(3, dtype=np.float64)
    Z = np.zeros(3, dtype=np.float64)
    PI = math.pi

    # ----- translation / rotation only paths (K is constant) -----
    def f_static(s):
        return I, Z

    def f_forward(s):
        # camera center moves to (0, 0, +d) in world; T = -R @ c = (0,0,-d)
        d = 0.5 * magnitude * s
        return I, np.array([0.0, 0.0, -d])

    def f_backward(s):
        d = 0.5 * magnitude * s
        return I, np.array([0.0, 0.0, d])

    def f_pan_right(s):
        return _yaw(+(PI / 6) * magnitude * s), Z

    def f_pan_left(s):
        return _yaw(-(PI / 6) * magnitude * s), Z

    def f_tilt_up(s):
        return _pitch(+(PI / 6) * magnitude * s), Z

    def f_tilt_down(s):
        return _pitch(-(PI / 6) * magnitude * s), Z

    # Orbit / spiral are anchored so that frame 0 is exactly (R=I, T=0):
    # the camera starts at the world origin looking at a target one unit
    # away along +z (= (0, 0, r)), and pivots around that target while
    # keeping it in view.

    def _orbit(sign: float):
        def fn(s):
            # Radius scales with magnitude so rawscale mag>>1 also translates
            # farther (angle alone on r=1 caps |T| at ~2).
            a = sign * (PI / 3) * s
            r = 1.0 * magnitude
            target = np.array([0.0, 0.0, r])
            eye = np.array([r * math.sin(a), 0.0, r * (1.0 - math.cos(a))])
            return _look_at(eye, target)
        return fn

    def f_spiral(s):
        a = (PI / 3) * s
        r = 1.0 * magnitude
        target = np.array([0.0, 0.0, r])
        eye = np.array(
            [r * math.sin(a), -0.2 * magnitude * s, r * (1.0 - math.cos(a))]
        )
        return _look_at(eye, target)

    rt_dispatch = {
        "static": f_static,
        "forward": f_forward,
        "backward": f_backward,
        "pan_left": f_pan_left,
        "pan_right": f_pan_right,
        "tilt_up": f_tilt_up,
        "tilt_down": f_tilt_down,
        "orbit_left": _orbit(-1.0),
        "orbit_right": _orbit(+1.0),
        "spiral": f_spiral,
        # zoom paths keep RT = identity, vary K instead
        "zoom_in": f_static,
        "zoom_out": f_static,
    }
    RT = _build_RT(rt_dispatch[traj_type], num_frames)  # (T, 12)

    # ----- intrinsics K (T, 4) -----
    K = np.zeros((num_frames, 4), dtype=np.float32)
    for i in range(num_frames):
        s = i / max(num_frames - 1, 1)
        if traj_type == "zoom_in":
            scale = 1.0 + 0.5 * magnitude * s  # up to 1.5x at magnitude=1
        elif traj_type == "zoom_out":
            scale = 1.0 / (1.0 + 0.5 * magnitude * s)  # down to ~0.67x
        else:
            scale = 1.0
        K[i, 0] = focal_norm * scale  # fx
        K[i, 1] = focal_norm * scale  # fy
        K[i, 2] = 0.5  # px at image center
        K[i, 3] = 0.5  # py at image center

    pose16 = np.concatenate([K, RT], axis=1)  # (T, 16)
    return torch.from_numpy(pose16).to(torch.float32)


###############################################################################
#                       Init-image loading                                    #
###############################################################################


def load_init_image(path: str, resize_h: int, resize_w: int) -> torch.Tensor:
    """Load a single image (or first frame of a video) and return it as a
    ``(H, W, C)`` float32 tensor in ``[-1, 1]`` -- the same format that
    ``SimpleVideoDataset`` produces for a single frame.

    Supported inputs:
      * PIL-readable still images (.jpg / .png / .webp / ...).
      * Video files (.mp4 / .mov / ...). First frame is taken.
    """
    p = Path(path)
    assert p.exists(), f"--init_image not found: {path}"
    suffix = p.suffix.lower()

    if suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        import torchvision.io
        frames, _, _ = torchvision.io.read_video(
            os.fspath(p), pts_unit="sec", output_format="TCHW",
        )
        if frames.shape[0] == 0:
            raise RuntimeError(f"--init_image video decoded 0 frames: {path}")
        img = frames[0:1].float() / 255.0  # (1, C, H, W)
    else:
        from PIL import Image
        with Image.open(p) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (H, W, C)
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    if tuple(img.shape[-2:]) != (resize_h, resize_w):
        img = torch.nn.functional.interpolate(
            img, size=(resize_h, resize_w), mode="bilinear", align_corners=False,
        )
    img = img.squeeze(0).permute(1, 2, 0).contiguous()  # (H, W, C)
    img = img * 2.0 - 1.0
    return img
