"""Fixed eye-to-hand camera on opposite hole flank (IL layout)."""
from __future__ import annotations

import math
import os

import pybullet as p

from constants import (
    FIXED_CAM_FAR,
    FIXED_CAM_FOV,
    FIXED_CAM_HEIGHT,
    FIXED_CAM_NEAR,
    FIXED_CAM_UP,
    FIXED_CAM_WIDTH,
    PLATE_TOP_Z,
)
from il_flow._paths import URDF
from il_flow.il_constants import FIXED_CAM2_EYE_OFFSET, FIXED_CAM2_ROLL_DEG
from sim.fixed_camera import FixedCamera, camera_orientation, quat_look_at, target_for_hole


def _urdf(name: str) -> str:
    return os.path.join(URDF, name)


def eye_for_hole(hole_xy: tuple[float, float]) -> tuple[float, float, float]:
    tx, ty, tz = target_for_hole(hole_xy)
    ox, oy, oz = FIXED_CAM2_EYE_OFFSET
    return (tx + ox, ty + oy, tz + oz)


def camera_orientation2(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    orn = quat_look_at(eye, target, up=FIXED_CAM_UP)
    if abs(FIXED_CAM2_ROLL_DEG) < 1e-9:
        return orn
    roll_quat = p.getQuaternionFromEuler([0.0, 0.0, math.radians(FIXED_CAM2_ROLL_DEG)])
    return p.multiplyTransforms([0.0, 0.0, 0.0], orn, [0.0, 0.0, 0.0], roll_quat)[1]


def load_fixed_camera2(hole_xy: tuple[float, float]) -> FixedCamera:
    eye = eye_for_hole(hole_xy)
    target = target_for_hole(hole_xy)
    orn = camera_orientation2(eye, target)
    body_id = p.loadURDF(_urdf("fixed_camera2.urdf"), eye, orn, useFixedBase=True)
    return FixedCamera(body_id=body_id, hole_xy=hole_xy)
