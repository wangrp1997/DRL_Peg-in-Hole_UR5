"""Save DVS target image I* after Cartesian alignment (wrist_camera2, eye-in-hand)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np
import pybullet as p

from sim.wrist2_render import render_grayscale
from sim.wrist_camera2 import WristCamera2, attach_wrist_camera2

DEFAULT_DVS_TARGET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "teach",
    "dvs_targets",
)


def capture_dvs_target_gray(
    cam: WristCamera2,
    *,
    gui: bool = False,
    settle_steps: int = 10,
) -> np.ndarray:
    """Grab gray I* from wrist_camera2 after alignment (HARDWARE + cache if gui)."""
    for _ in range(settle_steps):
        p.stepSimulation()
    return render_grayscale(cam, gui=gui, warmup=2)


def save_dvs_target_image(
    robot_id: int,
    ee_link: int,
    hole_xy: tuple[float, float],
    metrics: dict,
    out_dir: str | None = None,
    cam: WristCamera2 | None = None,
    gui: bool = False,
) -> tuple[str, str]:
    """After align ok: wrist_camera2 gray PNG + JSON (servo_align --save_target)."""
    out_dir = out_dir or DEFAULT_DVS_TARGET_DIR
    os.makedirs(out_dir, exist_ok=True)

    owned = cam is None
    if owned:
        cam = attach_wrist_camera2(robot_id, ee_link, hole_xy)
        for _ in range(10):
            p.stepSimulation()
    gray = capture_dvs_target_gray(cam, gui=gui)
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
