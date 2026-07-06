"""One corner-servo episode: IK standoff → 6D perturb → keypoint align."""
from __future__ import annotations

import math
import random
import time
from typing import Any

import pybullet as p

from constants import (
    CORNER_SERVO_STANDOFF,
    HOLE_X_RANGE,
    HOLE_Y_RANGE,
    PERTURB_SERVO_PAUSE_S,
    SETTLE_IK_STEPS,
    SETTLE_IK_STEPS_GUI,
)
from geometry import alignment_metrics, peg_tip_world
from sim.perturbation import (
    Perturbation6,
    apply_tip_perturbation,
    format_perturbation_log,
    sample_perturbation6,
)
from vision.align import AlignMethod
from sim.scene import (
    close_camera_windows,
    connect,
    get_fixed_camera,
    load_scene,
    move_tip_to_standoff,
    refresh_camera_views,
    run_insert_after_align,
    set_hole_opaque,
)
from vision.corner_servo import run_corner_servo
from vision.corners import gt_image_keypoints
from vision.debug_markers import clear_gt_corner_markers, sync_gt_corner_markers


def sample_hole_xy(rng: random.Random) -> tuple[float, float]:
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def _pause_before_servo(gui: bool, seconds: float, on_frame=None) -> None:
    if seconds <= 0.0:
        return
    if not gui:
        return
    print(f"servo starts in {seconds:.0f}s …")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not p.getConnectionInfo()["isConnected"]:
            break
        p.stepSimulation()
        if on_frame is not None:
            on_frame()
        time.sleep(1.0 / 240.0)


def corner_servo_episode(
    gui: bool = False,
    hole_xy: tuple[float, float] | None = None,
    opencv_render: bool = False,
    perturb: Perturbation6 | None = None,
    rng: random.Random | None = None,
    gui_idle: bool = False,
    align_method: AlignMethod | None = None,
    insert: bool = False,
    infer_corner0: bool = False,
) -> tuple[bool, dict[str, Any] | None, tuple[float, float], tuple | None]:
    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(
        gui, hole_xy=hole_xy, opencv_render=opencv_render, fixed_cam=True
    )
    if gui:
        set_hole_opaque(hole_id)
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]

    settle_gui = SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS

    def _provider():
        cam = get_fixed_camera()
        if cam is None:
            return None
        return gt_image_keypoints(cam, robot_id, peg, hole_id, infer_corner0=infer_corner0)

    def _show_keypoints() -> None:
        if not gui:
            return
        cam = get_fixed_camera()
        kps = _provider()
        sync_gt_corner_markers(robot_id, peg, hole_id, kps, cam)
        refresh_camera_views(_provider)

    move_tip_to_standoff(
        robot_id, eef, arm, peg, hole_xy, CORNER_SERVO_STANDOFF, gui=gui, settle_steps=settle_gui
    )
    _show_keypoints()

    method = align_method or "kabsch"

    if perturb is None and rng is not None:
        perturb = sample_perturbation6(rng)
    if perturb is not None:
        apply_tip_perturbation(
            robot_id, eef, arm, peg, hole_xy, hole_orn, perturb, gui=gui, settle_steps=settle_gui
        )
        _show_keypoints()
        print(format_perturbation_log(perturb))
        _pause_before_servo(gui, PERTURB_SERVO_PAUSE_S, _show_keypoints if gui else None)

    cam = get_fixed_camera()
    if cam is None:
        p.disconnect()
        return False, None, hole_xy, None

    def _on_step():
        if gui:
            kps = _provider()
            sync_gt_corner_markers(robot_id, peg, hole_id, kps, cam)
            refresh_camera_views(_provider)

    aligned, m = run_corner_servo(
        robot_id,
        arm,
        peg,
        cam,
        hole_xy,
        hole_orn,
        _provider,
        gui=gui,
        on_step=_on_step if gui else None,
        align_method=align_method,
    )

    inserted: bool | None = None
    if insert and aligned:
        inserted = run_insert_after_align(robot_id, arm, eef, peg, hole_xy, gui=gui)
        if gui:
            _show_keypoints()

    info: dict[str, Any] | None = None
    if m is not None:
        info = {
            "align_method": method,
            "infer_corner0": infer_corner0,
            "insert": insert,
            "dx_mm": round(m["dx"] * 1e3, 3),
            "dy_mm": round(m["dy"] * 1e3, 3),
            "standoff_mm": round(m["standoff"] * 1e3, 3),
            "roll_deg": round(math.degrees(m["roll"]), 3),
            "pitch_deg": round(math.degrees(m["pitch"]), 3),
            "yaw_deg": round(math.degrees(m["yaw"]), 3),
            "perturb": perturb,
        }
        tip = peg_tip_world(robot_id, peg)
        peg_orn = p.getLinkState(robot_id, peg)[1]
        gt = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
        info["gt_aligned"] = gt["aligned"]
        info["gt_dx_mm"] = round(gt["dx"] * 1e3, 3)
        info["gt_dy_mm"] = round(gt["dy"] * 1e3, 3)
        info["gt_standoff_mm"] = round(gt["standoff"] * 1e3, 3)
        info["gt_roll_deg"] = round(math.degrees(gt["roll"]), 3)
        info["gt_pitch_deg"] = round(math.degrees(gt["pitch"]), 3)
        info["gt_yaw_deg"] = round(math.degrees(gt["yaw"]), 3)
        if insert:
            info["inserted"] = inserted

    live = None
    if gui and gui_idle:
        live = (robot_id, peg, hole_id, infer_corner0)

    if live is None:
        if gui:
            clear_gt_corner_markers()
            close_camera_windows()
        p.disconnect()

    return aligned, info, hole_xy, live


def corner_servo_gui_idle(robot_id: int, peg: int, hole_id: int, infer_corner0: bool = False) -> None:
    """Keep GUI + keypoint overlay until PyBullet window is closed."""
    from sim.gui_preview import gt_keypoint_gui_idle

    cam = get_fixed_camera()

    def _provider():
        if cam is None:
            return None
        return gt_image_keypoints(cam, robot_id, peg, hole_id, infer_corner0=infer_corner0)

    def _before():
        sync_gt_corner_markers(robot_id, peg, hole_id, _provider(), cam)

    gt_keypoint_gui_idle(robot_id, peg, hole_id, wrist2=False, before_refresh=_before)
    clear_gt_corner_markers()
