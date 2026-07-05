"""Eye-in-hand IBVS — servoUniversalRobotsIBVS.cpp (pure vpServo, no in-loop PnP)."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import pybullet as p
from visp.core import PixelMeterConversion
from visp.visual_features import FeaturePoint
from visp.vs import Servo

from constants import GUI_SERVO_REFRESH_EVERY, IBVS_STALL_STEPS, MIN_CORNERS_VISIBLE, PLATE_TOP_Z
from geometry import metrics_converged, peg_tip_world
from sim.cartesian_control import alignment_twist, apply_cartesian_velocity, stop_arm
from sim.wrist_camera2 import WristCamera2
from visp_flow.camera_params import camera_parameters_from_K
from visp_flow.visp_constants import (
    VISP_IBVS_ERROR_TOL,
    VISP_IBVS_LAMBDA,
    VISP_IBVS_MAX_STEPS,
    VISP_IBVS_START_PX,
    VISP_PNP_PREFLIGHT_MAX_STEPS,
)
from visp_flow.pybullet_robot import apply_visp_camera_velocity
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


def run_pnp_preflight_for_ibvs(
    robot_id: int,
    peg: int,
    arm: list[int],
    cam: WristCamera2,
    hole_xy: tuple[float, float],
    hole_orn,
    keypoint_provider: Callable[[], list[ImageKeypoints] | None],
    *,
    gui: bool = False,
    on_step: Callable[[], None] | None = None,
) -> tuple[bool, float, bool]:
    """Separate kabsch PnP stage: stop at VISP_IBVS_START_PX while GT still not converged.

    Returns (ready_for_ibvs, pixel_rms, gt_already_converged).
    """
    last_px = float("inf")
    for step in range(VISP_PNP_PREFLIGHT_MAX_STEPS):
        kps = keypoint_provider()
        if kps is None:
            stop_arm(robot_id, arm)
            return False, last_px, False

        hole_kp = next(k for k in kps if k.name == "hole")
        peg_kp = next(k for k in kps if k.name == "peg")
        px = ibvs_pixel_error(hole_kp, peg_kp)
        standoff = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
        m = metrics_from_keypoints(kps, cam, hole_xy, hole_orn, standoff_hint=standoff)
        if m is None:
            stop_arm(robot_id, arm)
            return False, last_px, False

        if px is not None and px <= VISP_IBVS_START_PX:
            stop_arm(robot_id, arm)
            if on_step is not None:
                on_step()
            return True, px, metrics_converged(m)

        if metrics_converged(m):
            stop_arm(robot_id, arm)
            return False, px if px is not None else last_px, True

        twist = alignment_twist(m["dx"], m["dy"], m["standoff"], m["roll"], m["pitch"], m["yaw"])
        apply_cartesian_velocity(robot_id, peg, arm, twist)
        p.stepSimulation()
        if on_step is not None and (not gui or step % GUI_SERVO_REFRESH_EVERY == 0):
            on_step()
        elif gui:
            time.sleep(1.0 / 240.0)
        last_px = px if px is not None else last_px

    stop_arm(robot_id, arm)
    return False, last_px, False


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
    """Pure ViSP IBVS — servoUniversalRobotsIBVS.cpp (CAMERA_FRAME, no in-loop PnP)."""
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
    prev_err = float("inf")
    try:
        for step in range(VISP_IBVS_MAX_STEPS):
            kps = keypoint_provider()
            if kps is None:
                stop_arm(robot_id, arm)
                return False, err_sq

            if len(desired.pairs) < MIN_CORNERS_VISIBLE:
                stop_arm(robot_id, arm)
                return False, err_sq

            corners_world = peg_tip_corners_world(robot_id, peg)
            peg_kp = next(k for k in kps if k.name == "peg")
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
            apply_visp_camera_velocity(robot_id, ee_link, arm, v_c)

            if abs(err_sq - prev_err) < 1e-8:
                stall += 1
            else:
                stall = 0
            prev_err = err_sq

            p.stepSimulation()
            if on_step is not None and (not gui or step % GUI_SERVO_REFRESH_EVERY == 0):
                on_step()
            elif gui:
                time.sleep(1.0 / 240.0)

            standoff = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
            m = metrics_from_keypoints(kps, cam, hole_xy, hole_orn, standoff_hint=standoff)
            if m is not None and metrics_converged(m):
                stop_arm(robot_id, arm)
                return True, err_sq
            if err_sq < VISP_IBVS_ERROR_TOL:
                stop_arm(robot_id, arm)
                return bool(m and metrics_converged(m)), err_sq
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
