"""Eye-in-hand photometric DVS — photometricVisualServoing.cpp via native PhotometricServoTask."""
from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np
import pybullet as p

from constants import GUI_SERVO_REFRESH_EVERY
from sim.cartesian_control import stop_arm
from sim.wrist_camera2 import WristCamera2
from visp_flow.camera_params import camera_parameters_from_K
from visp_flow.visp_constants import (
    VISP_DVS_ABORT_ERR,
    VISP_DVS_ERROR_TOL,
    VISP_DVS_LAMBDA,
    VISP_DVS_MAX_STEPS,
    VISP_DVS_START_ERR,
)
from visp_flow.pybullet_robot import apply_visp_camera_velocity
from sim.wrist2_render import capture_servo_rgbd_gray


def _photometric_task():
    try:
        import photometric_servo as ps
    except ImportError as exc:
        raise ImportError(
            "编译 ViSP 光度扩展: cd qsfp_insert/visp_flow/native/build && cmake .. && make\n"
            "export PYTHONPATH=$PWD/qsfp_insert/visp_flow/native/build:$PWD:$PYTHONPATH"
        ) from exc
    return ps.PhotometricServoTask()


def run_visp_dvs_fine(
    robot_id: int,
    ee_link: int,
    peg: int,
    arm: list[int],
    wrist_cam: WristCamera2,
    target_gray: np.ndarray,
    *,
    plane_z: float,
    gui: bool = False,
    on_step: Callable[[], None] | None = None,
) -> tuple[bool, float, bool]:
    """Pure ViSP photometric servo — official task + UR CAMERA_FRAME execution.

    Returns (ok, err_sq, gated_out). gated_out=True if ||e||²_start > VISP_DVS_START_ERR.
    """
    vp_cam = camera_parameters_from_K(wrist_cam.K)
    Id = target_gray
    if Id.shape != (wrist_cam.height, wrist_cam.width):
        Id = cv2.resize(Id, (wrist_cam.width, wrist_cam.height), interpolation=cv2.INTER_AREA)

    I0, _buf = capture_servo_rgbd_gray(wrist_cam, gui=gui, warmup=2, use_cache=False)
    task = _photometric_task()
    task.init(
        np.ascontiguousarray(Id, dtype=np.uint8),
        float(vp_cam.get_px()),
        float(vp_cam.get_py()),
        float(vp_cam.get_u0()),
        float(vp_cam.get_v0()),
        float(plane_z),
        float(VISP_DVS_LAMBDA),
    )

    err_sq = float(task.error_at(np.ascontiguousarray(I0, dtype=np.uint8)))
    if err_sq > VISP_DVS_START_ERR:
        stop_arm(robot_id, arm)
        return False, err_sq, True

    err0 = err_sq
    for step in range(VISP_DVS_MAX_STEPS):
        I, _buf = capture_servo_rgbd_gray(wrist_cam, gui=gui, warmup=1, use_cache=False)
        v_list, err_sq = task.step(np.ascontiguousarray(I, dtype=np.uint8))
        apply_visp_camera_velocity(robot_id, ee_link, arm, v_list, peg_link=peg)

        p.stepSimulation()
        if on_step is not None and (not gui or step % GUI_SERVO_REFRESH_EVERY == 0):
            on_step()
        elif gui:
            time.sleep(1.0 / 240.0)

        if err_sq > max(VISP_DVS_ABORT_ERR, err0 * 10.0):
            stop_arm(robot_id, arm)
            return False, err_sq, False
        if err_sq < VISP_DVS_ERROR_TOL:
            stop_arm(robot_id, arm)
            return True, err_sq, False

    stop_arm(robot_id, arm)
    return err_sq < VISP_DVS_ERROR_TOL, err_sq, False
