"""Shared GUI preview — same idle/exit as visp_flow / corner_servo."""
from __future__ import annotations

import os
import time
from collections.abc import Callable

import cv2
import numpy as np
import pybullet as p

from sim.fixed_camera import FixedCamera
from sim.fixed_render import get_cached_rgbd as get_fixed_cached_rgbd
from sim.scene import (
    close_camera_windows,
    get_fixed_camera,
    get_wrist_camera2,
    refresh_camera_views,
)
from sim.wrist2_render import get_cached_rgbd as get_wrist2_cached_rgbd
from sim.wrist_camera2 import WristCamera2
from vision.corners import gt_image_keypoints

GuiLive = tuple[int, int, int, bool]  # robot_id, peg, hole_id, wrist2


def is_pybullet_connected() -> bool:
    try:
        return bool(p.getConnectionInfo()["isConnected"])
    except Exception:
        return False


def rgba_to_gray(rgba: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def gray_from_refresh_cache(cam: FixedCamera | WristCamera2) -> np.ndarray | None:
    if isinstance(cam, WristCamera2):
        cached = get_wrist2_cached_rgbd()
    elif isinstance(cam, FixedCamera):
        cached = get_fixed_cached_rgbd()
    else:
        return None
    if cached is None:
        return None
    return rgba_to_gray(cached[0])


def pause_gui(
    seconds: float,
    message: str,
    on_frame: Callable[[], None] | None = None,
) -> None:
    if seconds <= 0:
        if message:
            print(message)
        return
    print(message)
    deadline = time.monotonic() + seconds
    while is_pybullet_connected() and time.monotonic() < deadline:
        p.stepSimulation()
        if on_frame is not None:
            on_frame()
        time.sleep(1.0 / 240.0)


def gt_keypoint_gui_idle(
    robot_id: int,
    peg: int,
    hole_id: int,
    *,
    wrist2: bool = False,
    before_refresh: Callable[[], None] | None = None,
    detach_wrist2: bool = False,
    force_exit: int | None = None,
    overlay_provider: Callable[[], list | None] | None = None,
) -> None:
    """Keep GUI + keypoint overlay until PyBullet window is closed."""

    def _provider():
        if overlay_provider is not None:
            return overlay_provider()
        cam = get_wrist_camera2() if wrist2 else get_fixed_camera()
        if cam is None:
            return None
        return gt_image_keypoints(cam, robot_id, peg, hole_id)

    print("Close PyBullet window to exit.")
    while p.getConnectionInfo()["isConnected"]:
        p.stepSimulation()
        if before_refresh is not None:
            before_refresh()
        refresh_camera_views(_provider)
        time.sleep(1.0 / 240.0)
    close_camera_windows()
    if detach_wrist2:
        cam = get_wrist_camera2()
        if cam is not None:
            cam.detach()
    try:
        if p.getConnectionInfo()["isConnected"]:
            p.disconnect()
    except p.error:
        pass
    if force_exit is not None:
        os._exit(force_exit)


def refresh_keypoints(get_keypoint_sets: Callable[[], object | None]) -> None:
    if not is_pybullet_connected():
        return
    refresh_camera_views(get_keypoint_sets)
