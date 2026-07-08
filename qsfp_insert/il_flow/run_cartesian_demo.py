#!/usr/bin/env python3
"""IL preview: Cartesian align at rest hole, dual cameras wrist2 + fixed2."""
from __future__ import annotations

import argparse
import math
import sys
import time

import pybullet as p

from il_flow._paths import ROOT  # noqa: F401

from constants import FIXED_CAM_DEMO_STANDOFF, HOLE_XY
from geometry import alignment_metrics, is_xy_rpy_aligned, peg_tip_world
from il_flow.scene import (
    il_gui_idle,
    load_il_scene,
    refresh_il_camera_views,
    teardown_il_scene,
)
from sim.cartesian_align import run_cartesian_align
from sim.cartesian_align_policy import cartesian_align_target
from sim.scene import connect, idle_gui, move_tip_to_standoff

DEFAULT_STANDOFF_MM = 4.0
STANDOFF_Z_BAND_M = 0.001


def run(gui: bool, standoff_mm: float, align: bool, opencv_render: bool) -> bool:
    connect(gui)
    robot_id, arm, _eef, peg, hole_id, hole_xy = load_il_scene(
        gui, hole_xy=HOLE_XY, opencv_render=opencv_render,
    )
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]

    standoff_m = FIXED_CAM_DEMO_STANDOFF
    move_tip_to_standoff(robot_id, _eef, arm, peg, hole_xy, standoff_m, gui=gui)
    if gui and opencv_render:
        refresh_il_camera_views()
    print(f"hole_xy={hole_xy} IK standoff={standoff_m * 1e3:.1f} mm (demo pose)")

    aligned = True
    m: dict = {}
    if align:
        target_z = standoff_mm * 1e-3

        def on_gui_step() -> None:
            if opencv_render:
                refresh_il_camera_views()
            time.sleep(1.0 / 240.0)

        with cartesian_align_target(target_z, z_band_m=STANDOFF_Z_BAND_M):
            aligned, m = run_cartesian_align(
                robot_id, arm, peg, hole_xy, hole_orn, gui=gui, on_step=on_gui_step if gui else None,
            )
        tip = peg_tip_world(robot_id, peg)
        peg_orn = p.getLinkState(robot_id, peg)[1]
        gt = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
        gt_ok = is_xy_rpy_aligned(gt["dx"], gt["dy"], gt["roll"], gt["pitch"], gt["yaw"])
        print(
            f"pose   dx={m['dx']*1e3:+.2f}mm dy={m['dy']*1e3:+.2f}mm "
            f"standoff={m['standoff']*1e3:.2f}mm (target {standoff_mm:.1f}mm) "
            f"rpy=({math.degrees(m['roll']):+.2f}°, {math.degrees(m['pitch']):+.2f}°, {math.degrees(m['yaw']):+.2f}°)"
        )
        print(
            f"GT     dx={gt['dx']*1e3:+.2f}mm dy={gt['dy']*1e3:+.2f}mm "
            f"standoff={gt['standoff']*1e3:.2f}mm "
            f"rpy=({math.degrees(gt['roll']):+.2f}°, {math.degrees(gt['pitch']):+.2f}°, {math.degrees(gt['yaw']):+.2f}°) "
            f"gt_aligned={gt_ok}"
        )
        print("align ok" if aligned else "align fail")
        if gui and opencv_render:
            refresh_il_camera_views()

    if gui:
        if opencv_render:
            print("OpenCV: wrist2 + fixed2. Ctrl+C to quit.")
        else:
            print("仅 Bullet 3D 场景（无相机预览窗）。关窗退出。")
        try:
            il_gui_idle(opencv_render=opencv_render)
        except KeyboardInterrupt:
            pass
        idle_gui()

    teardown_il_scene()
    p.disconnect()
    return aligned


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IL dual-cam Cartesian align preview")
    ap.add_argument("--gui", action="store_true", help="PyBullet 3D + wrist2 预览")
    ap.add_argument(
        "--opencv",
        action="store_true",
        help="额外开 OpenCV 双窗（卡；默认只用 Bullet 右下角预览）",
    )
    ap.add_argument(
        "--standoff-mm",
        type=float,
        default=DEFAULT_STANDOFF_MM,
        help=f"Cartesian align target standoff (default {DEFAULT_STANDOFF_MM})",
    )
    ap.add_argument(
        "--no-align",
        action="store_true",
        help="Skip Cartesian servo; only show IK demo standoff + cameras",
    )
    args = ap.parse_args()
    if not args.gui:
        ap.error("preview requires --gui")
    sys.exit(
        0
        if run(args.gui, args.standoff_mm, align=not args.no_align, opencv_render=args.opencv)
        else 1
    )
