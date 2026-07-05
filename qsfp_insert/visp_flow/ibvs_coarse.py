"""Eye-in-hand IBVS coarse — servoUniversalRobotsIBVS.cpp + PnP blend when far (demo hybrid)."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pybullet as p
from visp.core import PixelMeterConversion
from visp.visual_features import FeaturePoint
from visp.vs import Servo

from constants import (
    GUI_SERVO_REFRESH_EVERY,
    IBVS_STALL_STEPS,
    MIN_CORNERS_VISIBLE,
    PLATE_TOP_Z,
)
from geometry import metrics_converged, peg_tip_world
from sim.cartesian_control import alignment_twist, apply_cartesian_velocity, damped_pinv, jacobian_tip, stop_arm
from sim.wrist_camera2 import WristCamera2
from visp_flow.camera_params import camera_parameters_from_K
from visp_flow.visp_constants import (
    VISP_IBVS_BLEND_PX,
    VISP_IBVS_ERROR_TOL,
    VISP_IBVS_LAMBDA,
    VISP_IBVS_MAX_STEPS,
)
from visp_flow.pybullet_robot import camera_velocity_to_world_twist
from visp_flow.require_visp import require_visp_python
from vision.align import _corner_assignment, ibvs_pixel_error, metrics_from_keypoints
from vision.corners import ImageKeypoints, peg_tip_corners_world

require_visp_python()


@dataclass(frozen=True)
class IbvsDesiredFeatures:
    """Desired metric features pd[i] indexed by hole corner id; fixed peg↔hole pairing from teach."""

    pd: tuple[tuple[float, float, float], ...]
    pairs: tuple[tuple[int, int], ...]


def _world_to_cam_z(cam: WristCamera2, world_pt: tuple[float, float, float]) -> float:
    pos, orn = cam._pose()
    inv_pos, inv_orn = p.invertTransform(pos, orn)
    local, _ = p.multiplyTransforms(inv_pos, inv_orn, world_pt, [0.0, 0.0, 0.0, 1.0])
    return float(-local[2])


def _feature_xyz(
    vp_cam,
    cam: WristCamera2,
    u: float,
    v: float,
    world_pt: tuple[float, float, float],
    z_fallback: float,
) -> tuple[float, float, float]:
    """UR: FeatureBuilder x,y + changeFrame Z (cP[2])."""
    xm, ym = PixelMeterConversion.convertPoint(vp_cam, float(u), float(v))
    z = _world_to_cam_z(cam, world_pt)
    if z <= 1e-6:
        z = z_fallback
    return float(xm), float(ym), float(z)


def teach_ibvs_desired(
    cam: WristCamera2,
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
    robot_id: int,
    peg: int,
) -> IbvsDesiredFeatures | None:
    """Teach pd at aligned pose — desired hole-corner image coords + peg-corner depth."""
    pairs = _corner_assignment(hole_kp, peg_kp)
    if pairs is None or len(pairs) < MIN_CORNERS_VISIBLE:
        return None
    vp_cam = camera_parameters_from_K(cam.K)
    corners = peg_tip_corners_world(robot_id, peg)
    pd: list[tuple[float, float, float]] = [(0.0, 0.0, 0.12)] * 4
    for h, p_idx in pairs:
        hu, hv = hole_kp.uv[h]
        pd[h] = _feature_xyz(vp_cam, cam, hu, hv, corners[p_idx], 0.12)
    return IbvsDesiredFeatures(pd=tuple(pd), pairs=tuple(pairs))


def _apply_hybrid_control(
    robot_id: int,
    ee_link: int,
    peg: int,
    arm: list[int],
    v_cam,
    twist_pnp: np.ndarray,
    ibvs_weight: float,
) -> None:
    """Far: PnP (kabsch metrics); near: ViSP v_c via UR fVe·eVc chain."""
    w = float(max(0.0, min(1.0, ibvs_weight)))
    if w <= 0.0:
        apply_cartesian_velocity(robot_id, peg, arm, twist_pnp)
        return
    twist_visp = camera_velocity_to_world_twist(robot_id, ee_link, v_cam)
    if w >= 1.0:
        j = jacobian_tip(robot_id, peg, arm)
        qdot = damped_pinv(j) @ twist_visp
    else:
        j = jacobian_tip(robot_id, peg, arm)
        qdot = (1.0 - w) * (damped_pinv(j) @ twist_pnp) + w * (damped_pinv(j) @ twist_visp)
    n = np.linalg.norm(qdot)
    from constants import CART_MAX_QDOT

    if n > CART_MAX_QDOT:
        qdot *= CART_MAX_QDOT / n
    for j_idx, v in zip(arm, qdot):
        p.setJointMotorControl2(robot_id, j_idx, p.VELOCITY_CONTROL, targetVelocity=float(v), force=500.0)


def run_visp_ibvs_coarse(
    robot_id: int,
    ee_link: int,
    peg: int,
    arm: list[int],
    cam: WristCamera2,
    desired: IbvsDesiredFeatures,
    keypoint_provider: Callable[[], list[ImageKeypoints] | None],
    hole_xy: tuple[float, float],
    hole_orn,
    *,
    gui: bool = False,
    on_step: Callable[[], None] | None = None,
) -> tuple[bool, float]:
    vp_cam = camera_parameters_from_K(cam.K)

    p_feat = [FeaturePoint() for _ in range(4)]
    pd_feat = [FeaturePoint() for _ in range(4)]
    for i, (xm, ym, z) in enumerate(desired.pd):
        pd_feat[i].set_x(xm)
        pd_feat[i].set_y(ym)
        pd_feat[i].set_Z(z)

    task = Servo()
    for i in range(4):
        p_feat[i].set_x(desired.pd[i][0])
        p_feat[i].set_y(desired.pd[i][1])
        p_feat[i].set_Z(desired.pd[i][2])
        task.addFeature(p_feat[i], pd_feat[i])
    task.setServo(Servo.EYEINHAND_CAMERA)
    task.setInteractionMatrixType(Servo.CURRENT)
    task.setLambda(VISP_IBVS_LAMBDA)

    err_sq = float("inf")
    stall = 0
    prev_px = float("inf")
    try:
        for step in range(VISP_IBVS_MAX_STEPS):
            kps = keypoint_provider()
            if kps is None:
                stop_arm(robot_id, arm)
                return False, err_sq

            hole_kp = next(k for k in kps if k.name == "hole")
            peg_kp = next(k for k in kps if k.name == "peg")
            if len(desired.pairs) < MIN_CORNERS_VISIBLE:
                stop_arm(robot_id, arm)
                return False, err_sq

            corners_world = peg_tip_corners_world(robot_id, peg)
            for h, p_idx in desired.pairs:
                pu, pv = peg_kp.uv[p_idx]
                xm, ym, z = _feature_xyz(
                    vp_cam, cam, pu, pv, corners_world[p_idx], desired.pd[h][2]
                )
                p_feat[h].set_x(xm)
                p_feat[h].set_y(ym)
                p_feat[h].set_Z(z)

            v_c = task.computeControlLaw()
            err_sq = float(task.getError().sumSquare())

            standoff = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
            m = metrics_from_keypoints(kps, cam, hole_xy, hole_orn, standoff_hint=standoff)
            if m is None:
                stop_arm(robot_id, arm)
                return False, err_sq

            px = ibvs_pixel_error(hole_kp, peg_kp)
            ibvs_w = 0.0 if px is None else max(0.0, min(1.0, 1.0 - px / VISP_IBVS_BLEND_PX))
            twist_pnp = alignment_twist(
                m["dx"], m["dy"], m["standoff"], m["roll"], m["pitch"], m["yaw"]
            )
            _apply_hybrid_control(robot_id, ee_link, peg, arm, v_c, twist_pnp, ibvs_w)

            if ibvs_w > 0.5 and px is not None and abs(px - prev_px) < 0.1:
                stall += 1
            else:
                stall = 0
            prev_px = px if px is not None else prev_px

            p.stepSimulation()
            if on_step is not None and (not gui or step % GUI_SERVO_REFRESH_EVERY == 0):
                on_step()
            elif gui:
                time.sleep(1.0 / 240.0)

            if metrics_converged(m):
                stop_arm(robot_id, arm)
                return True, err_sq
            if err_sq < VISP_IBVS_ERROR_TOL and metrics_converged(m):
                stop_arm(robot_id, arm)
                return True, err_sq
            if stall >= IBVS_STALL_STEPS:
                break
    finally:
        task.kill()

    stop_arm(robot_id, arm)
    kps = keypoint_provider()
    if kps is not None:
        standoff = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
        m = metrics_from_keypoints(kps, cam, hole_xy, hole_orn, standoff_hint=standoff)
        if m is not None and metrics_converged(m):
            return True, err_sq
    return False, err_sq
