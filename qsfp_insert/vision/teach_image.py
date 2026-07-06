"""Unified teach I* grayscale capture + PNG save (visp_flow / template_flow / demo)."""
from __future__ import annotations

import os

import cv2
import numpy as np

from sim.fixed_camera import FixedCamera
from sim.fixed_render import render_rgbd as fixed_render_rgbd
from sim.wrist2_render import capture_servo_rgbd_gray
from sim.wrist_camera2 import WristCamera2
from vision.corners import ProjectorCam


def _rgba_to_gray(rgba: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def capture_teach_gray(
    cam: ProjectorCam,
    *,
    gui: bool = False,
    warmup: int = 2,
) -> np.ndarray:
    """RGBD→gray at teach pose; with_depth_seg=True, fresh frame (no HARDWARE black edge)."""
    if isinstance(cam, WristCamera2):
        gray, _ = capture_servo_rgbd_gray(cam, gui=gui, warmup=warmup, use_cache=False)
        return gray
    if isinstance(cam, FixedCamera):
        rgba, _, _ = fixed_render_rgbd(
            cam, gui=gui, with_depth_seg=True, use_cache=False, warmup=warmup,
        )
        return _rgba_to_gray(rgba)
    raise TypeError(f"unsupported camera: {type(cam)}")


def save_teach_gray_png(gray: np.ndarray, png_path: str) -> str:
    parent = os.path.dirname(png_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cv2.imwrite(png_path, gray)
    return png_path
