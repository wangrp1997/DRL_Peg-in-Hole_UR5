"""Headless RGB capture for IL dual-camera layout."""
from __future__ import annotations

import cv2
import numpy as np

from il_flow.scene import il_uses_hardware_render
from sim.fixed_camera import FixedCamera
from sim.fixed_render import render_rgbd as fixed_render_rgbd
from sim.wrist2_render import render_rgbd as wrist2_render_rgbd
from sim.wrist_camera2 import WristCamera2

CAM_WRIST2 = "wrist_camera2"
CAM_FIXED2 = "fixed_camera2"
IL_CAMERAS = (CAM_WRIST2, CAM_FIXED2)


def rgba_to_rgb(rgba: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _hardware_render(*, gui: bool) -> bool:
    return gui or il_uses_hardware_render()


def capture_camera_rgb(cam: WristCamera2 | FixedCamera, *, gui: bool = False, warmup: int | None = None) -> np.ndarray:
    hw = _hardware_render(gui=gui)
    w = 2 if hw and warmup is None else (warmup if warmup is not None else 1)
    if isinstance(cam, WristCamera2):
        rgba = wrist2_render_rgbd(cam, gui=hw, with_depth_seg=False, use_cache=False, warmup=w)[0]
    else:
        rgba = fixed_render_rgbd(cam, gui=hw, with_depth_seg=False, use_cache=False, warmup=w)[0]
    return rgba_to_rgb(rgba)


def capture_rgb_pair(
    wrist_cam: WristCamera2,
    fixed_cam: FixedCamera,
    *,
    gui: bool = False,
) -> dict[str, np.ndarray]:
    return {
        CAM_WRIST2: capture_camera_rgb(wrist_cam, gui=gui),
        CAM_FIXED2: capture_camera_rgb(fixed_cam, gui=gui),
    }


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
