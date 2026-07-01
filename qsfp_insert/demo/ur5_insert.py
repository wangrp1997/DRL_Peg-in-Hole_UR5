#!/usr/bin/env python3
"""UR5 + rect peg + hole, IK insert. Usage: python qsfp_insert/demo/ur5_insert.py [--gui]"""
from __future__ import annotations

import argparse
import sys

import pybullet as p

from _paths import ROOT  # noqa: F401

from constants import EE_LINEAR_STEP, HOLE_DEPTH, PLATE_TOP_Z  # noqa: E402
from geometry import is_inserted, peg_tip_world  # noqa: E402
from ur5_common import connect, idle_gui, load_scene, settle, step_tip_z  # noqa: E402

_UR5_INSERT_DEPTH = 0.022


def run(gui: bool) -> bool:
    connect(gui)
    robot_id, arm, eef, peg, _hole_id, hole_xy = load_scene(gui)
    ee0, ee_orn0 = p.getLinkState(robot_id, eef)[:2]

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
    p.disconnect()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    sys.exit(0 if run(ap.parse_args().gui) else 1)
