"""Save IBVS desired features + DVS I* after Cartesian alignment."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from sim.wrist2_render import render_grayscale
from sim.wrist_camera2 import WristCamera2
from visp_flow._paths import TEACH_DIR
from visp_flow.ibvs_coarse import IbvsDesiredFeatures, teach_ibvs_desired
from vision.corners import ImageKeypoints

TEACH_STEM = "visp_teach"


def save_teach_bundle(
    ibvs: IbvsDesiredFeatures,
    dvs_gray: np.ndarray,
    hole_xy: tuple[float, float],
    metrics: dict,
    out_dir: str | None = None,
) -> tuple[str, str]:
    out_dir = out_dir or TEACH_DIR
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{TEACH_STEM}_dvs.png")
    json_path = os.path.join(out_dir, f"{TEACH_STEM}.json")

    cv2.imwrite(png_path, dvs_gray)
    meta = {
        "ibvs_pd": [{"x": x, "y": y, "Z": z} for x, y, z in ibvs.pd],
        "dvs_png": os.path.basename(png_path),
        "width": int(dvs_gray.shape[1]),
        "height": int(dvs_gray.shape[0]),
        "hole_xy": [float(hole_xy[0]), float(hole_xy[1])],
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "captured": "after_cartesian_align",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  DVS target I* (aligned): {png_path}")
    return png_path, json_path


def capture_aligned_teach(
    wrist_cam: WristCamera2,
    hole_kp: ImageKeypoints,
    robot_id: int,
    peg: int,
    *,
    gui: bool = False,
) -> tuple[IbvsDesiredFeatures, np.ndarray]:
    """I* + IBVS pd at aligned pose — same timing as servo_align --save_target."""
    ibvs = teach_ibvs_desired(wrist_cam, hole_kp, robot_id, peg)
    gray = render_grayscale(wrist_cam, gui=gui, warmup=2)
    return ibvs, gray
