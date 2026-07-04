"""Save DVS target image I* after Cartesian alignment (wrist_camera2, eye-in-hand)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np
import pybullet as p

from sim.wrist_camera2 import WristCamera2, attach_wrist_camera2

DEFAULT_DVS_TARGET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "teach",
    "dvs_targets",
)


def render_grayscale(cam: WristCamera2, warmup: int = 2, use_hardware: bool = False) -> np.ndarray:
    """Grab gray frame; GUI sessions use HARDWARE like refresh_camera_views."""
    rgba = None
    for _ in range(max(1, warmup)):
        rgba = cam.render(with_depth_seg=False, for_opencv=not use_hardware)
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def save_dvs_target_image(
    robot_id: int,
    ee_link: int,
    hole_xy: tuple[float, float],
    metrics: dict,
    out_dir: str | None = None,
    cam: WristCamera2 | None = None,
    gui: bool = False,
) -> tuple[str, str]:
    """Grab aligned gray frame from wrist_camera2; attach temporarily if cam is None."""
    out_dir = out_dir or DEFAULT_DVS_TARGET_DIR
    os.makedirs(out_dir, exist_ok=True)

    owned = cam is None
    if owned:
        cam = attach_wrist_camera2(robot_id, ee_link, hole_xy)
        for _ in range(10):
            p.stepSimulation()
    gray = render_grayscale(cam, use_hardware=gui)
    if owned:
        cam.detach()

    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"target_{tag}"
    png_path = os.path.join(out_dir, f"{stem}.png")
    json_path = os.path.join(out_dir, f"{stem}.json")

    cv2.imwrite(png_path, gray)
    meta = {
        "format": "gray8_png",
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "hole_xy": [float(hole_xy[0]), float(hole_xy[1])],
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return png_path, json_path
