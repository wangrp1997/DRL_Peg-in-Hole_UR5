"""PyBullet robot adapter: apply ViSP CAMERA_FRAME velocity (UR IBVS example pattern)."""
from __future__ import annotations

import numpy as np
import pybullet as p
from visp.core import ColVector, HomogeneousMatrix, RotationMatrix, TranslationVector, VelocityTwistMatrix

from constants import WRIST_CAM2_EE_LOCAL_ORN, WRIST_CAM2_EE_LOCAL_POS
from sim.cartesian_control import apply_cartesian_velocity
from visp_flow.require_visp import require_visp_python

require_visp_python()

_EMC: HomogeneousMatrix | None = None


def _eMc_homogeneous() -> HomogeneousMatrix:
    global _EMC
    if _EMC is not None:
        return _EMC
    rot = np.array(p.getMatrixFromQuaternion(WRIST_CAM2_EE_LOCAL_ORN), dtype=np.float64).reshape(3, 3)
    r = RotationMatrix()
    for i in range(3):
        for j in range(3):
            r[i, j] = float(rot[i, j])
    t = TranslationVector(
        float(WRIST_CAM2_EE_LOCAL_POS[0]),
        float(WRIST_CAM2_EE_LOCAL_POS[1]),
        float(WRIST_CAM2_EE_LOCAL_POS[2]),
    )
    eMc = HomogeneousMatrix()
    eMc.insert(r)
    eMc.insert(t)
    _EMC = eMc
    return eMc


def camera_velocity_to_ee_twist(v_cam) -> np.ndarray:
    """Map ViSP camera twist to EE twist via vpVelocityTwistMatrix (official ViSP)."""
    eMc = _eMc_homogeneous()
    eVc = VelocityTwistMatrix()
    eVc.buildFrom(eMc, True)
    v_ee = eVc * ColVector(v_cam)
    return np.array([v_ee[i] for i in range(6)], dtype=np.float64)


def apply_visp_camera_velocity(
    robot_id: int,
    peg_link: int,
    arm: list[int],
    v_cam,
) -> None:
    """servoUniversalRobotsIBVS.cpp: robot.setVelocity(vpRobot::CAMERA_FRAME, v_c)."""
    twist = camera_velocity_to_ee_twist(v_cam)
    apply_cartesian_velocity(robot_id, peg_link, arm, twist)
