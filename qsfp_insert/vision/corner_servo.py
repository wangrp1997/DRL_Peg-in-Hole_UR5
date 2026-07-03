"""Closed-loop corner keypoint visual servo (fixed camera)."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pybullet as p

from constants import (
    ALIGN_Z_NOMINAL,
    ALIGN_Z_STANDOFF_MAX,
    CART_MAX_ANG,
    CART_MAX_LIN,
    CART_Z_GAIN,
    GUI_SERVO_REFRESH_EVERY,
    IBVS_BLEND_PX,
    IBVS_STALL_STEPS,
    PLATE_TOP_Z,
    SERVO_MAX_STEPS,
    SERVO_STALL_STEPS,
)
from geometry import AlignmentMetrics, metrics_converged, peg_tip_world
from sim.cartesian_control import alignment_twist, apply_cartesian_velocity, stop_arm
from sim.fixed_camera import FixedCamera
from vision.align import (
    AlignMethod,
    ibvs_pixel_error,
    ibvs_twist_tip,
    metrics_from_keypoints,
)
from vision.corners import ImageKeypoints, peg_tip_corners_world


KeypointProvider = Callable[[], list[ImageKeypoints] | None]


def _clip_twist(twist: np.ndarray, max_ang: float = CART_MAX_ANG) -> np.ndarray:
    lin = twist[:3]
    ang = twist[3:]
    ln = np.linalg.norm(lin)
    if ln > CART_MAX_LIN:
        lin = lin * (CART_MAX_LIN / ln)
    an = np.linalg.norm(ang)
    if an > max_ang:
        ang = ang * (max_ang / an)
    return np.concatenate([lin, ang])


def _ibvs_control_twist(
    keypoints: list[ImageKeypoints],
    cam: FixedCamera,
    robot_id: int,
    peg: int,
    standoff: float,
    metrics: AlignmentMetrics,
) -> np.ndarray | None:
    """Far: PnP coarse (valid 6D); near: IBVS fine (ViSP eye-to-hand J_img)."""
    hole_kp = next(s for s in keypoints if s.name == "hole")
    peg_kp = next(s for s in keypoints if s.name == "peg")
    tip = peg_tip_world(robot_id, peg)
    corners = peg_tip_corners_world(robot_id, peg)

    tw_pnp = alignment_twist(
        metrics["dx"], metrics["dy"], metrics["standoff"],
        metrics["roll"], metrics["pitch"], metrics["yaw"],
    )
    px = ibvs_pixel_error(hole_kp, peg_kp)
    ibvs_w = 0.0 if px is None else max(0.0, min(1.0, 1.0 - px / IBVS_BLEND_PX))

    tw_ibvs = ibvs_twist_tip(cam, hole_kp, peg_kp, corners, tip)
    if tw_ibvs is None:
        twist = tw_pnp
    else:
        if standoff > ALIGN_Z_STANDOFF_MAX:
            tw_ibvs[2] = -CART_MAX_LIN
        else:
            tw_ibvs[2] = max(-CART_MAX_LIN, min(CART_MAX_LIN, CART_Z_GAIN * (ALIGN_Z_NOMINAL - standoff)))
        twist = ibvs_w * tw_ibvs + (1.0 - ibvs_w) * tw_pnp
    return _clip_twist(twist, CART_MAX_ANG)


def run_corner_servo(
    robot_id: int,
    arm: list[int],
    peg: int,
    cam: FixedCamera,
    hole_xy: tuple[float, float],
    hole_orn,
    keypoint_provider: KeypointProvider,
    gui: bool = False,
    on_step: Callable[[], None] | None = None,
    align_method: AlignMethod | None = None,
) -> tuple[bool, AlignmentMetrics | None]:
    """Servo until corner metrics converge or stall/timeout."""
    stall = 0
    prev_px = float("inf")
    last_m: AlignmentMetrics | None = None
    use_ibvs = align_method == "ibvs"
    stall_limit = IBVS_STALL_STEPS if use_ibvs else SERVO_STALL_STEPS

    for step in range(SERVO_MAX_STEPS):
        keypoints = keypoint_provider()
        if keypoints is None:
            break
        standoff_hint = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
        m = metrics_from_keypoints(
            keypoints,
            cam,
            hole_xy,
            hole_orn,
            standoff_hint=standoff_hint,
            method=align_method,
        )
        if m is None:
            break
        last_m = m
        if metrics_converged(m):
            stop_arm(robot_id, arm)
            if on_step is not None:
                on_step()
            return True, m

        if use_ibvs:
            hole_kp = next(s for s in keypoints if s.name == "hole")
            peg_kp = next(s for s in keypoints if s.name == "peg")
            px = ibvs_pixel_error(hole_kp, peg_kp)
            if px is not None and abs(px - prev_px) < 0.1:
                stall += 1
            else:
                stall = 0
            prev_px = px if px is not None else prev_px
        else:
            stall = 0

        if stall >= stall_limit:
            break

        if use_ibvs:
            twist = _ibvs_control_twist(keypoints, cam, robot_id, peg, standoff_hint, m)
            if twist is None:
                break
        else:
            twist = alignment_twist(m["dx"], m["dy"], m["standoff"], m["roll"], m["pitch"], m["yaw"])
        apply_cartesian_velocity(robot_id, peg, arm, twist)
        p.stepSimulation()
        if on_step is not None and step % GUI_SERVO_REFRESH_EVERY == 0:
            on_step()
        elif gui:
            time.sleep(1.0 / 240.0)

    stop_arm(robot_id, arm)
    for _ in range(20):
        p.stepSimulation()
        if on_step is not None:
            on_step()

    if last_m is not None:
        keypoints = keypoint_provider()
        if keypoints is not None:
            standoff_hint = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
            m = metrics_from_keypoints(
                keypoints,
                cam,
                hole_xy,
                hole_orn,
                standoff_hint=standoff_hint,
                method=align_method,
            )
            if m is not None:
                last_m = m
        return metrics_converged(last_m), last_m
    return False, None
