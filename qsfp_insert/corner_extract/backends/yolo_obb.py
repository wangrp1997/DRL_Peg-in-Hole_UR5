"""YOLO pose corner detection for wrist2."""
from __future__ import annotations

import os
from collections.abc import Callable

import cv2
import numpy as np

from constants import PEG_H, PEG_W
from corner_extract.yolo_export import HOLE_CLASS, PEG_CLASS
from sim.wrist2_render import capture_servo_rgbd_gray
from sim.wrist_camera2 import WristCamera2
from vision.corners import (
    HOLE_HALF_X,
    HOLE_HALF_Y,
    ImageKeypoints,
    ProjectorCam,
    apply_infer_corner0_if_missing,
)

DEFAULT_YOLO_WEIGHTS = os.environ.get(
    "QSFP_YOLO_WEIGHTS",
    "/home/rw/Projects/ultralytics/runs/pose/qsfp_insert/corner_extract/runs/sim_wrist2_pose_n/weights/best.pt",
)

KPT_CONF_THRESH = 0.25


def _best_keypoints(result, class_id: int) -> np.ndarray | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best_i: int | None = None
    best_conf = -1.0
    for i in range(len(result.boxes)):
        if int(result.boxes.cls[i]) != class_id:
            continue
        conf = float(result.boxes.conf[i])
        if conf > best_conf:
            best_conf = conf
            best_i = i
    if best_i is None:
        return None
    return result.keypoints.data[best_i].cpu().numpy()


def _kpts_to_image_keypoints(
    name: str,
    prefix: str,
    color: tuple[int, int, int],
    kpts: np.ndarray,
    cam: ProjectorCam,
) -> ImageKeypoints:
    uv: list[tuple[float, float]] = []
    visible: list[bool] = []
    for x, y, cf in kpts:
        u, v = float(x), float(y)
        ok = (
            float(cf) >= KPT_CONF_THRESH
            and 0.0 <= u < cam.width
            and 0.0 <= v < cam.height
        )
        uv.append((u, v))
        visible.append(ok)
    return ImageKeypoints(name, prefix, color, uv, visible)


def _sets_usable(sets: list[ImageKeypoints]) -> bool:
    hole = next(s for s in sets if s.name == "hole")
    peg = next(s for s in sets if s.name == "peg")
    # Kabsch/IPPE needs four visible corners per object.
    return sum(hole.visible) >= 4 and sum(peg.visible) >= 4


def make_yolo_provider(
    cam: ProjectorCam,
    robot_id: int,
    peg: int,
    hole_id: int,
    *,
    infer_corner0: bool = False,
    gui: bool = False,
    yolo_weights: str | None = None,
) -> Callable[[], list[ImageKeypoints] | None]:
    del robot_id, peg, hole_id
    if not isinstance(cam, WristCamera2):
        raise TypeError("YOLO corner provider requires WristCamera2")

    weights = yolo_weights or DEFAULT_YOLO_WEIGHTS
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    from ultralytics import YOLO

    model = YOLO(weights)

    def provider() -> list[ImageKeypoints] | None:
        gray, _ = capture_servo_rgbd_gray(cam, gui=gui, warmup=1, use_cache=False)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        results = model.predict(bgr, verbose=False)
        if not results:
            return None
        result = results[0]
        hole_kpts = _best_keypoints(result, HOLE_CLASS)
        peg_kpts = _best_keypoints(result, PEG_CLASS)
        if hole_kpts is None or peg_kpts is None:
            return None

        sets = [
            _kpts_to_image_keypoints("hole", "H", (0, 220, 0), hole_kpts, cam),
            _kpts_to_image_keypoints("peg", "P", (0, 80, 255), peg_kpts, cam),
        ]
        # Peg P0 is never labeled (v=0); hole H0 only when YOLO marks it invisible.
        sets = [
            apply_infer_corner0_if_missing(sets[0], cam, HOLE_HALF_X, HOLE_HALF_Y),
            apply_infer_corner0_if_missing(sets[1], cam, PEG_W / 2.0, PEG_H / 2.0),
        ]
        if not _sets_usable(sets):
            return None
        return sets

    return provider
