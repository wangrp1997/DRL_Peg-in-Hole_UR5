"""PyBullet adapter for ViSP setVelocity(CAMERA_FRAME) — servoUniversalRobotsIBVS.cpp."""
from __future__ import annotations

import numpy as np
import pybullet as p
from visp.core import ColVector, HomogeneousMatrix, RotationMatrix, TranslationVector, VelocityTwistMatrix

from constants import CART_MAX_ANG, CART_MAX_LIN, CART_MAX_QDOT, CART_LAMBDA, WRIST_CAM2_EE_LOCAL_ORN, WRIST_CAM2_EE_LOCAL_POS
from sim.cartesian_control import damped_pinv
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


def _jacobian_ee(robot_id: int, ee_link: int, arm: list[int]) -> np.ndarray:
    q = [p.getJointState(robot_id, j)[0] for j in arm]
    z = [0.0] * len(arm)
    jt, jr = p.calculateJacobian(robot_id, ee_link, [0.0, 0.0, 0.0], q, z, z)
    return np.vstack([jt, jr])


def apply_visp_camera_velocity(
    robot_id: int,
    ee_link: int,
    arm: list[int],
    v_cam,
    *,
    cam_body_id: int | None = None,
) -> None:
    """Apply v_c like UR robot.setVelocity(vpRobot::CAMERA_FRAME, v_c)."""
    del cam_body_id  # chain uses eMc + fMe, not live camera pose
    v_world = camera_velocity_to_world_twist(robot_id, ee_link, v_cam)
    j = _jacobian_ee(robot_id, ee_link, arm)
    qdot = damped_pinv(j, CART_LAMBDA) @ v_world
    n = np.linalg.norm(qdot)
    if n > CART_MAX_QDOT:
        qdot *= CART_MAX_QDOT / n
    for j_idx, v in zip(arm, qdot):
        p.setJointMotorControl2(robot_id, j_idx, p.VELOCITY_CONTROL, targetVelocity=float(v), force=500.0)
