"""6D Cartesian velocity control via Jacobian (prep for visual servo)."""
from __future__ import annotations

import numpy as np
import pybullet as p

from constants import (
    ALIGN_Z_NOMINAL,
    ALIGN_Z_STANDOFF_MAX,
    ALIGN_Z_STANDOFF_MIN,
    CART_LAMBDA,
    CART_MAX_ANG,
    CART_MAX_LIN,
    CART_MAX_QDOT,
    CART_ROT_GAIN,
    CART_XY_GAIN,
    CART_Z_GAIN,
    PEG_L,
)


def damped_pinv(J: np.ndarray, lam: float = CART_LAMBDA) -> np.ndarray:
    n = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(n))


def jacobian_tip(robot_id: int, peg_link: int, arm: list[int]) -> np.ndarray:
    local = [0.0, 0.0, -PEG_L / 2.0]
    q = [p.getJointState(robot_id, j)[0] for j in arm]
    z = [0.0] * len(arm)
    jt, jr = p.calculateJacobian(robot_id, peg_link, local, q, z, z)
    return np.vstack([jt, jr])


def _clip_xy(vx: float, vy: float) -> tuple[float, float]:
    vxy = np.array([vx, vy])
    n = np.linalg.norm(vxy)
    if n > CART_MAX_LIN:
        vxy *= CART_MAX_LIN / n
    return float(vxy[0]), float(vxy[1])


def _clip_z(vz: float) -> float:
    return float(max(-CART_MAX_LIN, min(CART_MAX_LIN, vz)))


def _clip_angular(wx: float, wy: float, wz: float) -> tuple[float, float, float]:
    w = np.array([wx, wy, wz])
    n = np.linalg.norm(w)
    if n > CART_MAX_ANG:
        w *= CART_MAX_ANG / n
    return float(w[0]), float(w[1]), float(w[2])


def alignment_twist(dx: float, dy: float, standoff: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Proportional 6D twist; XY / Z / angular clipped separately so descent is not starved."""
    vx, vy = _clip_xy(-CART_XY_GAIN * dx, -CART_XY_GAIN * dy)
    if standoff > ALIGN_Z_STANDOFF_MAX:
        vz = -CART_MAX_LIN
    else:
        vz = _clip_z(CART_Z_GAIN * (ALIGN_Z_NOMINAL - standoff))
    wx, wy, wz = _clip_angular(-CART_ROT_GAIN * roll, -CART_ROT_GAIN * pitch, -CART_ROT_GAIN * yaw)
    return np.array([vx, vy, vz, wx, wy, wz])


def apply_joint_velocity(
    robot_id: int,
    arm: list[int],
    qdot: np.ndarray,
    force: float = 500.0,
) -> None:
    n = np.linalg.norm(qdot)
    if n > CART_MAX_QDOT:
        qdot = qdot * (CART_MAX_QDOT / n)
    for j, v in zip(arm, qdot):
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, targetVelocity=float(v), force=force)


def apply_cartesian_velocity(
    robot_id: int,
    peg_link: int,
    arm: list[int],
    twist: np.ndarray,
    force: float = 500.0,
) -> None:
    qdot = damped_pinv(jacobian_tip(robot_id, peg_link, arm)) @ twist
    n = np.linalg.norm(qdot)
    if n > CART_MAX_QDOT:
        qdot *= CART_MAX_QDOT / n
    for j, v in zip(arm, qdot):
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, targetVelocity=float(v), force=force)


def stop_arm(robot_id: int, arm: list[int], force: float = 500.0) -> None:
    for j in arm:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, targetVelocity=0.0, force=force)
