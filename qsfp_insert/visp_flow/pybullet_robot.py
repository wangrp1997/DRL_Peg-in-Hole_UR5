"""PyBullet adapter for ViSP setVelocity(CAMERA_FRAME) — servoUniversalRobotsIBVS.cpp."""
from __future__ import annotations

import numpy as np
import pybullet as p
from visp.core import ColVector, HomogeneousMatrix, RotationMatrix, TranslationVector, VelocityTwistMatrix

from constants import CART_MAX_ANG, CART_MAX_LIN, WRIST_CAM2_EE_LOCAL_ORN, WRIST_CAM2_EE_LOCAL_POS
from sim.cartesian_control import apply_cartesian_velocity
from visp_flow.require_visp import require_visp_python

require_visp_python()

_EMC: HomogeneousMatrix | None = None


def _eMc_homogeneous() -> HomogeneousMatrix:
    """^eM_c — robot.set_eMc(eMc) in UR IBVS example."""
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


def _homogeneous_from_pose(pos, orn) -> HomogeneousMatrix:
    rot = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
    r = RotationMatrix()
    for i in range(3):
        for j in range(3):
            r[i, j] = float(rot[i, j])
    t = TranslationVector(float(pos[0]), float(pos[1]), float(pos[2]))
    m = HomogeneousMatrix()
    m.insert(r)
    m.insert(t)
    return m


def _fMe(robot_id: int, ee_link: int) -> HomogeneousMatrix:
    """Base → end-effector transform (same frame as UR get_fMe())."""
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    ee_pos, ee_orn = p.getLinkState(robot_id, ee_link)[:2]
    inv_pos, inv_orn = p.invertTransform(base_pos, base_orn)
    rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, ee_pos, ee_orn)
    return _homogeneous_from_pose(rel_pos, rel_orn)


def camera_velocity_to_world_twist(robot_id: int, ee_link: int, v_cam) -> np.ndarray:
    """UR: w_v_e = fVe * eVc * v_c  then speedL(w_v_e)."""
    fMe = _fMe(robot_id, ee_link)
    eMc = _eMc_homogeneous()

    fVe = VelocityTwistMatrix()
    fVe.buildFrom(fMe, False)
    eVc = VelocityTwistMatrix()
    eVc.buildFrom(eMc, True)

    v_world = fVe * eVc * ColVector(v_cam)
    twist = np.array([v_world[i] for i in range(6)], dtype=np.float64)

    lin = twist[:3]
    ang = twist[3:]
    ln = np.linalg.norm(lin)
    if ln > CART_MAX_LIN:
        lin = lin * (CART_MAX_LIN / ln)
    an = np.linalg.norm(ang)
    if an > CART_MAX_ANG:
        ang = ang * (CART_MAX_ANG / an)
    return np.concatenate([lin, ang])


def apply_visp_camera_velocity(
    robot_id: int,
    ee_link: int,
    arm: list[int],
    v_cam,
    *,
    peg_link: int | None = None,
    cam_body_id: int | None = None,
) -> None:
    """Apply v_c: UR fVe·eVc chain → world twist → peg-tip Jacobian (matches PnP preflight)."""
    del cam_body_id
    v_world = camera_velocity_to_world_twist(robot_id, ee_link, v_cam)
    if peg_link is None:
        raise ValueError("peg_link required for Cartesian IBVS execution")
    apply_cartesian_velocity(robot_id, peg_link, arm, v_world)
