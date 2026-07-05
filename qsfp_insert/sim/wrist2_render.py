"""Stable wrist_camera2 rendering — avoids TINY/HARDWARE black flicker in GUI/OpenCV.

Pattern from sim/scene.py refresh_camera_views + vision/teach_target.save_target:
  - GUI: always ER_BULLET_HARDWARE_OPENGL (for_opencv=False)
  - Reuse last good RGBA when a frame comes back blank
"""
from __future__ import annotations

import cv2
import numpy as np

from sim.wrist_camera2 import WristCamera2

_last_rgbd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def reset_wrist2_render_cache() -> None:
    global _last_rgbd
    _last_rgbd = None


def rgba_is_blank(rgba: np.ndarray) -> bool:
    rgb = rgba[..., :3]
    return float(rgb.max()) < 1.0


def render_rgbd(
    cam: WristCamera2,
    *,
    gui: bool = False,
    with_depth_seg: bool = True,
    use_cache: bool = True,
    warmup: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Render RGBA (+ optional depth/seg). GUI sessions use HARDWARE only."""
    global _last_rgbd
    rgba: np.ndarray | None = None
    depth: np.ndarray | None = None
    seg: np.ndarray | None = None

    for _ in range(max(1, warmup)):
        out = cam.render(with_depth_seg, for_opencv=not gui)
        if with_depth_seg:
            rgba, depth, seg = out
        else:
            rgba = out
            depth = seg = None

    if rgba is None:
        raise RuntimeError("wrist_camera2 render returned no image")

    if use_cache and rgba_is_blank(rgba) and _last_rgbd is not None:
        return _last_rgbd

    if not rgba_is_blank(rgba):
        if with_depth_seg and depth is not None and seg is not None:
            _last_rgbd = (rgba, depth, seg)
        elif not with_depth_seg:
            _last_rgbd = (rgba, np.zeros((cam.height, cam.width), dtype=np.float32), np.zeros((cam.height, cam.width), dtype=np.int32))

    if with_depth_seg and depth is not None and seg is not None:
        return rgba, depth, seg
    return rgba, None, None


def get_cached_rgbd() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    return _last_rgbd


def capture_servo_rgbd_gray(
    cam: WristCamera2,
    *,
    gui: bool = False,
    warmup: int = 1,
    use_cache: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One render per control step; use_cache=False for ViSP servo (avoid stale I*)."""
    rgba, depth, seg = render_rgbd(
        cam, gui=gui, with_depth_seg=True, use_cache=use_cache, warmup=warmup
    )
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray, (rgba, depth, seg)


def render_grayscale(
    cam: WristCamera2,
    *,
    gui: bool = False,
    warmup: int = 2,
    use_cache: bool = True,
) -> np.ndarray:
    """Aligned target I* capture — same policy as save_dvs_target_image."""
    rgba, _, _ = render_rgbd(cam, gui=gui, with_depth_seg=False, use_cache=use_cache, warmup=warmup)
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
