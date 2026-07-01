#!/usr/bin/env python3
"""UR5 + rect peg + hole, IK insert."""
from __future__ import annotations

import argparse
import sys

import pybullet as p

from _paths import ROOT  # noqa: F401

from constants import EE_LINEAR_STEP, HOLE_DEPTH, PLATE_TOP_Z
from geometry import is_inserted, peg_tip_world
from sim.scene import (
    add_gui_camera_args,
    close_camera_windows,
    connect,
    idle_gui,
    load_scene,
    step_tip_z,
    validate_gui_camera_args,
)

_UR5_INSERT_DEPTH = 0.022


def run(gui: bool, opencv_render: bool, wrist_cam: bool, fixed_cam: bool) -> bool:
    connect(gui)
    robot_id, arm, eef, peg, _hole_id, hole_xy = load_scene(
        gui, opencv_render=opencv_render, wrist_cam=wrist_cam, fixed_cam=fixed_cam
    )
    ee_orn0 = p.getLinkState(robot_id, eef)[1]

    ok = False
    for _ in range(800):
        step_tip_z(robot_id, eef, arm, peg, hole_xy, ee_orn0, -EE_LINEAR_STEP, gui)
        tip = peg_tip_world(robot_id, peg)
        if is_inserted(tip, hole_xy, min_insert_depth=_UR5_INSERT_DEPTH):
            ok = True
            break
        if tip[2] < PLATE_TOP_Z - HOLE_DEPTH:
            break

    tip = peg_tip_world(robot_id, peg)
    ok = is_inserted(tip, hole_xy, min_insert_depth=_UR5_INSERT_DEPTH)
    print(f"tip_z={tip[2]:.4f} inserted={ok}")
    if gui:
        idle_gui()
    close_camera_windows()
    p.disconnect()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_gui_camera_args(ap)
    args = ap.parse_args()
    validate_gui_camera_args(ap, args)
    sys.exit(0 if run(args.gui, args.opencv_render, args.wrist_cam, args.fixed_cam) else 1)
