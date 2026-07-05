"""Eye-in-hand IBVS coarse stage — structure from servoUniversalRobotsIBVS.cpp."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import pybullet as p
from visp.core import PixelMeterConversion
from visp.visual_features import FeaturePoint
from visp.vs import Servo

from constants import GUI_SERVO_REFRESH_EVERY, PLATE_TOP_Z
from geometry import metrics_converged, peg_tip_world
from sim.cartesian_control import stop_arm
from sim.wrist_camera2 import WristCamera2
from visp_flow.camera_params import camera_parameters_from_K
from visp_flow.visp_constants import VISP_IBVS_ERROR_TOL, VISP_IBVS_LAMBDA, VISP_IBVS_MAX_STEPS
from visp_flow.pybullet_robot import apply_visp_camera_velocity
from visp_flow.require_visp import require_visp_python
from vision.align import metrics_from_keypoints
from vision.corners import ImageKeypoints, peg_tip_corners_world

require_visp_python()


@dataclass(frozen=True)
class IbvsDesiredFeatures:
    """Taught desired feature coordinates (metric x,y,Z) — pd[i] in UR IBVS example."""

    pd: list[tuple[float, float, float]]


def _world_to_cam_z(cam: WristCamera2, world_pt: tuple[float, float, float]) -> float:
    pos, orn = cam._pose()
    inv_pos, inv_orn = p.invertTransform(pos, orn)
    local, _ = p.multiplyTransforms(inv_pos, inv_orn, world_pt, [0.0, 0.0, 0.0, 1.0])
    return float(-local[2])


def teach_ibvs_desired(
    cam: WristCamera2,
    hole_kp: ImageKeypoints,
    robot_id: int,
    peg: int,
) -> IbvsDesiredFeatures:
    vp_cam = camera_parameters_from_K(cam.K)
    corners = peg_tip_corners_world(robot_id, peg)
    pd: list[tuple[float, float, float]] = []
    for i in range(4):
        hu, hv = hole_kp.uv[i]
        xm, ym = PixelMeterConversion.convertPoint(vp_cam, float(hu), float(hv))
        z = _world_to_cam_z(cam, corners[i])
        if z <= 1e-6:
            z = 0.12
        pd.append((float(xm), float(ym), float(z)))
    return IbvsDesiredFeatures(pd=pd)


def run_visp_ibvs_coarse(
    robot_id: int,
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
    """ViSP IBVS coarse stage.

    Note: vpServo must be constructed inline here — creating Servo inside a nested
    helper after PyBullet load_scene() triggers a ViSP Python segfault.
    """
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
    try:
        for step in range(VISP_IBVS_MAX_STEPS):
            kps = keypoint_provider()
            if kps is None:
                stop_arm(robot_id, arm)
                return False, err_sq

            corners_world = peg_tip_corners_world(robot_id, peg)
            peg_kp = next(k for k in kps if k.name == "peg")
            for i in range(4):
                pu, pv = peg_kp.uv[i]
                xm, ym = PixelMeterConversion.convertPoint(vp_cam, float(pu), float(pv))
                z = _world_to_cam_z(cam, corners_world[i])
                if z <= 1e-6:
                    z = desired.pd[i][2]
                p_feat[i].set_x(float(xm))
                p_feat[i].set_y(float(ym))
                p_feat[i].set_Z(float(z))

            v_c = task.computeControlLaw()
            err_sq = float(task.getError().sumSquare())
            apply_visp_camera_velocity(robot_id, peg, arm, v_c)

            p.stepSimulation()
            if on_step is not None and (not gui or step % GUI_SERVO_REFRESH_EVERY == 0):
                on_step()
            elif gui:
                time.sleep(1.0 / 240.0)

            if err_sq < VISP_IBVS_ERROR_TOL:
                stop_arm(robot_id, arm)
                standoff = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
                m = metrics_from_keypoints(kps, cam, hole_xy, hole_orn, standoff_hint=standoff)
                return bool(m and metrics_converged(m)), err_sq
    finally:
        task.kill()

    stop_arm(robot_id, arm)
    return err_sq < VISP_IBVS_ERROR_TOL * 10, err_sq
