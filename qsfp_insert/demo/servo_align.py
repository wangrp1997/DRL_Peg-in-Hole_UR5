#!/usr/bin/env python3
"""UR5: Cartesian servo to hole mouth; optional insert."""
from __future__ import annotations

import argparse
import math
import sys
import time

import pybullet as p

from _paths import ROOT  # noqa: F401

from constants import (
    EE_LINEAR_STEP,
    HOLE_DEPTH,
    PLATE_TOP_Z,
    SERVO_GUI_SUBSTEPS,
    SERVO_MAX_STEPS,
    SERVO_STALL_STEPS,
    UR5_MIN_INSERT_DEPTH,
)
from geometry import alignment_metrics, is_aligned, is_inserted, peg_tip_world
from sim.cartesian_control import alignment_twist, apply_cartesian_velocity, stop_arm
from sim.scene import (
    add_gui_camera_args,
    close_camera_windows,
    connect,
    idle_gui,
    load_scene,
    refresh_camera_views,
    register_wrist_camera2,
    setup_wrist_camera2_views,
    step_tip_z,
    validate_gui_camera_args,
)
from sim.wrist_camera2 import WristCamera2, attach_wrist_camera2


def _insert_phase(robot_id, arm, eef, peg, hole_xy, gui: bool) -> bool:
    ee_orn0 = p.getLinkState(robot_id, eef)[1]
    for _ in range(800):
        step_tip_z(robot_id, eef, arm, peg, hole_xy, ee_orn0, -EE_LINEAR_STEP, gui)
        tip = peg_tip_world(robot_id, peg)
        if is_inserted(tip, hole_xy, min_insert_depth=UR5_MIN_INSERT_DEPTH):
            return True
        if tip[2] < PLATE_TOP_Z - HOLE_DEPTH:
            break
    tip = peg_tip_world(robot_id, peg)
    return is_inserted(tip, hole_xy, min_insert_depth=UR5_MIN_INSERT_DEPTH)


def _setup_wrist_camera2(
    robot_id: int,
    eef: int,
    hole_xy: tuple[float, float],
    gui: bool,
    opencv_render: bool,
    wrist_cam2: bool,
) -> WristCamera2 | None:
    if not (wrist_cam2 and gui):
        return None
    cam2 = attach_wrist_camera2(robot_id, eef, hole_xy)
    register_wrist_camera2(cam2)
    setup_wrist_camera2_views(True, opencv_render)
    refresh_camera_views()
    return cam2


def servo_align_episode(
    gui: bool = False,
    hole_xy: tuple[float, float] | None = None,
    opaque_hole: bool = False,
    insert: bool = False,
    opencv_render: bool = False,
    wrist_cam: bool = False,
    wrist_cam2: bool = False,
    fixed_cam: bool = False,
    save_target: bool = False,
    save_target_dir: str | None = None,
) -> tuple[bool, bool | None, dict, tuple[float, float], tuple[str, str] | None, WristCamera2 | None]:
    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(
        gui, hole_xy=hole_xy, opencv_render=opencv_render, wrist_cam=wrist_cam, fixed_cam=fixed_cam
    )
    if gui and opaque_hole:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 1.0])
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]

    cam2 = _setup_wrist_camera2(robot_id, eef, hole_xy, gui, opencv_render, wrist_cam2)

    stall = 0
    prev_standoff = float("inf")
    aligned = False
    for _ in range(SERVO_MAX_STEPS):
        tip = peg_tip_world(robot_id, peg)
        peg_orn = p.getLinkState(robot_id, peg)[1]
        if is_aligned(tip, peg_orn, hole_xy, hole_orn):
            aligned = True
            break
        m = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
        if abs(m["standoff"] - prev_standoff) < 5e-6:
            stall += 1
            if stall >= SERVO_STALL_STEPS:
                break
        else:
            stall = 0
        prev_standoff = m["standoff"]
        twist = alignment_twist(m["dx"], m["dy"], m["standoff"], m["roll"], m["pitch"], m["yaw"])
        apply_cartesian_velocity(robot_id, peg, arm, twist)
        substeps = SERVO_GUI_SUBSTEPS if gui else 1
        for _ in range(substeps):
            p.stepSimulation()
        if gui:
            refresh_camera_views()
            time.sleep(1.0 / 240.0)

    stop_arm(robot_id, arm)
    for _ in range(20):
        p.stepSimulation()

    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    m = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    aligned = is_aligned(tip, peg_orn, hole_xy, hole_orn)

    inserted: bool | None = None
    if insert:
        inserted = _insert_phase(robot_id, arm, eef, peg, hole_xy, gui) if aligned else False

    saved: tuple[str, str] | None = None
    if save_target and aligned:
        from vision.teach_target import save_dvs_target_image

        saved = save_dvs_target_image(
            robot_id, eef, hole_xy, m, out_dir=save_target_dir, cam=cam2, gui=gui
        )

    return aligned, inserted, m, hole_xy, saved, cam2


def run(
    gui: bool,
    insert: bool,
    opencv_render: bool,
    wrist_cam: bool,
    wrist_cam2: bool,
    fixed_cam: bool,
    save_target: bool,
    save_target_dir: str | None,
) -> bool:
    aligned, inserted, m, _, saved, cam2 = servo_align_episode(
        gui=gui,
        opaque_hole=gui,
        insert=insert,
        opencv_render=opencv_render,
        wrist_cam=wrist_cam,
        wrist_cam2=wrist_cam2,
        fixed_cam=fixed_cam,
        save_target=save_target,
        save_target_dir=save_target_dir,
    )
    print(
        f"dx={m['dx']*1e3:+.2f}mm dy={m['dy']*1e3:+.2f}mm "
        f"standoff={m['standoff']*1e3:.2f}mm "
        f"rpy=({math.degrees(m['roll']):+.2f}°, {math.degrees(m['pitch']):+.2f}°, {math.degrees(m['yaw']):+.2f}°)"
    )
    print("align ok" if aligned else "align fail")
    if saved is not None:
        png_path, json_path = saved
        print(f"saved DVS target (wrist_camera2): {png_path}")
        print(f"saved metadata:                 {json_path}")
    elif save_target and not aligned:
        print("save_target skipped (align fail)")
    ok = aligned
    if insert:
        print("insert ok" if inserted else "insert fail")
        ok = aligned and bool(inserted)
    if gui:
        idle_gui()
    if cam2 is not None:
        cam2.detach()
        register_wrist_camera2(None)
    close_camera_windows()
    p.disconnect()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_gui_camera_args(ap)
    ap.add_argument("--insert", action="store_true", help="After alignment, continue to insert")
    ap.add_argument(
        "--save_target",
        action="store_true",
        help="After align ok: wrist_camera2 gray PNG + JSON → teach/dvs_targets/",
    )
    ap.add_argument(
        "--save_target_dir",
        default=None,
        metavar="DIR",
        help="Override output dir for --save_target (default: qsfp_insert/teach/dvs_targets)",
    )
    args = ap.parse_args()
    validate_gui_camera_args(ap, args)
    sys.exit(
        0
        if run(
            args.gui,
            args.insert,
            args.opencv_render,
            args.wrist_cam,
            args.wrist_cam2,
            args.fixed_cam,
            args.save_target,
            args.save_target_dir,
        )
        else 1
    )
