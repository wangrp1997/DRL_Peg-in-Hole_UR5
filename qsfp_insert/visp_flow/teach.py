"""Save IBVS desired features + DVS I* after Cartesian alignment."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pybullet as p

from constants import PLATE_TOP_Z
from sim.wrist2_render import render_grayscale
from sim.wrist_camera2 import WristCamera2
from visp_flow._paths import TEACH_DIR
from visp_flow.ibvs_coarse import IbvsDesiredFeatures, teach_ibvs_desired
from vision.corners import ImageKeypoints

TEACH_STEM = "visp_teach"


def dvs_plane_depth_z(cam: WristCamera2, hole_xy: tuple[float, float]) -> float:
    """vpFeatureLuminance plane Z at teach — hole mouth centre in camera frame."""
    world_pt = (float(hole_xy[0]), float(hole_xy[1]), PLATE_TOP_Z)
    pos, orn = cam._pose()
    inv_pos, inv_orn = p.invertTransform(pos, orn)
    local, _ = p.multiplyTransforms(inv_pos, inv_orn, world_pt, [0.0, 0.0, 0.0, 1.0])
    z = float(-local[2])
    return z if z > 1e-6 else 0.10


def save_teach_bundle(
    ibvs: IbvsDesiredFeatures,
    dvs_gray: np.ndarray,
    hole_xy: tuple[float, float],
    metrics: dict,
    *,
    dvs_plane_z: float,
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
        "dvs_plane_z": float(dvs_plane_z),
        "width": int(dvs_gray.shape[1]),
        "height": int(dvs_gray.shape[0]),
        "hole_xy": [float(hole_xy[0]), float(hole_xy[1])],
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "captured": "after_cartesian_align",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  DVS target I* (aligned): {png_path}")
    print(f"  DVS plane Z (teach): {dvs_plane_z:.4f} m")
    return png_path, json_path


def capture_aligned_teach(
    wrist_cam: WristCamera2,
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
    robot_id: int,
    peg: int,
    hole_xy: tuple[float, float],
    *,
    gui: bool = False,
) -> tuple[IbvsDesiredFeatures | None, np.ndarray, float]:
    """I* + IBVS pd + DVS plane Z at aligned pose."""
    ibvs = teach_ibvs_desired(wrist_cam, hole_kp, peg_kp, robot_id, peg)
    gray = render_grayscale(wrist_cam, gui=gui, warmup=2)
    plane_z = dvs_plane_depth_z(wrist_cam, hole_xy)
    return ibvs, gray, plane_z
