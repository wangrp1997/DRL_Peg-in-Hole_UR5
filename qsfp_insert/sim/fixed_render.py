"""Fixed camera render — same cache policy as sim/wrist2_render.py (HARDWARE + blank fallback)."""
from __future__ import annotations

import cv2
import numpy as np

from sim.fixed_camera import FixedCamera
from sim.wrist2_render import rgba_is_blank

_last_rgbd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def reset_fixed_render_cache() -> None:
    global _last_rgbd
    _last_rgbd = None


def get_cached_rgbd() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    return _last_rgbd


def render_rgbd(
    cam: FixedCamera,
    *,
    gui: bool = False,
    with_depth_seg: bool = True,
    use_cache: bool = True,
    warmup: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
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
        raise RuntimeError("fixed camera render returned no image")

    if use_cache and rgba_is_blank(rgba) and _last_rgbd is not None:
        return _last_rgbd

    if not rgba_is_blank(rgba):
        if with_depth_seg and depth is not None and seg is not None:
            _last_rgbd = (rgba, depth, seg)
        elif not with_depth_seg:
            _last_rgbd = (
                rgba,
                np.zeros((cam.height, cam.width), dtype=np.float32),
                np.zeros((cam.height, cam.width), dtype=np.int32),
            )

    if with_depth_seg and depth is not None and seg is not None:
        return rgba, depth, seg
    return rgba, None, None


def render_grayscale(
    cam: FixedCamera,
    *,
    gui: bool = False,
    warmup: int = 1,
    use_cache: bool = True,
) -> np.ndarray:
    rgba, _, _ = render_rgbd(cam, gui=gui, with_depth_seg=False, use_cache=use_cache, warmup=warmup)
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
