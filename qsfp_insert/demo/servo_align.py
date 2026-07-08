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
    UR5_MIN_INSERT_DEPTH,
)
from geometry import alignment_metrics, is_inserted, is_xy_rpy_aligned, peg_tip_world
from sim.cartesian_align import run_cartesian_align
from sim.cartesian_align_policy import cartesian_align_target
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

DEFAULT_STANDOFF_MM = 4.0
STANDOFF_Z_BAND_M = 0.001


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
    standoff_mm: float = DEFAULT_STANDOFF_MM,
) -> tuple[bool, bool | None, dict, tuple[float, float], tuple[str, str] | None, WristCamera2 | None]:
    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(
        gui, hole_xy=hole_xy, opencv_render=opencv_render, wrist_cam=wrist_cam, fixed_cam=fixed_cam
    )
    if gui and opaque_hole:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 1.0])
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]

    cam2 = _setup_wrist_camera2(robot_id, eef, hole_xy, gui, opencv_render, wrist_cam2)

    target_z = standoff_mm * 1e-3

    def on_gui_step() -> None:
        refresh_camera_views()
        time.sleep(1.0 / 240.0)

    with cartesian_align_target(target_z, z_band_m=STANDOFF_Z_BAND_M):
        aligned, m = run_cartesian_align(
            robot_id,
            arm,
            peg,
            hole_xy,
            hole_orn,
            gui=gui,
            on_step=on_gui_step if gui else None,
        )

    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    gt = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    m = dict(m)
    m["gt_aligned"] = is_xy_rpy_aligned(gt["dx"], gt["dy"], gt["roll"], gt["pitch"], gt["yaw"])
    m["gt_dx"] = gt["dx"]
    m["gt_dy"] = gt["dy"]
    m["gt_standoff"] = gt["standoff"]
    m["gt_roll"] = gt["roll"]
    m["gt_pitch"] = gt["pitch"]
    m["gt_yaw"] = gt["yaw"]

    inserted: bool | None = None
    if insert:
        inserted = _insert_phase(robot_id, arm, eef, peg, hole_xy, gui) if aligned else False

    saved: tuple[str, str] | None = None
    if save_target and aligned:
        from vision.teach_target import save_dvs_target_image

        m = dict(m)
        m["target_standoff_mm"] = standoff_mm
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
    standoff_mm: float,
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
        standoff_mm=standoff_mm,
    )
    print(
        f"pose   dx={m['dx']*1e3:+.2f}mm dy={m['dy']*1e3:+.2f}mm "
        f"standoff={m['standoff']*1e3:.2f}mm (target {standoff_mm:.1f}mm) "
        f"rpy=({math.degrees(m['roll']):+.2f}°, {math.degrees(m['pitch']):+.2f}°, {math.degrees(m['yaw']):+.2f}°)"
    )
    print(
        f"GT     dx={m['gt_dx']*1e3:+.2f}mm dy={m['gt_dy']*1e3:+.2f}mm "
        f"standoff={m['gt_standoff']*1e3:.2f}mm "
        f"rpy=({math.degrees(m['gt_roll']):+.2f}°, {math.degrees(m['gt_pitch']):+.2f}°, {math.degrees(m['gt_yaw']):+.2f}°) "
        f"gt_aligned={m['gt_aligned']}"
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
        help="After align ok: teach/dvs_targets/dvs_target.png + .json",
    )
    ap.add_argument(
        "--save_target_dir",
        default=None,
        metavar="DIR",
        help="Override output dir for --save_target (default: qsfp_insert/teach/dvs_targets)",
    )
    ap.add_argument(
        "--standoff-mm",
        type=float,
        default=DEFAULT_STANDOFF_MM,
        metavar="MM",
        help=f"Target tip standoff above hole mouth in mm (default: {DEFAULT_STANDOFF_MM})",
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
            args.standoff_mm,
        )
        else 1
    )
