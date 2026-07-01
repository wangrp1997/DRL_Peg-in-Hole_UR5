#!/usr/bin/env python3
"""UR5: Cartesian servo to hole mouth; optional insert. Usage: python qsfp_insert/demo/servo_align.py [--gui] [--insert]"""
from __future__ import annotations

import argparse
import math
import sys
import time

import pybullet as p

from _paths import ROOT  # noqa: F401

from cartesian_control import alignment_twist, apply_cartesian_velocity, stop_arm  # noqa: E402
from constants import EE_LINEAR_STEP, HOLE_DEPTH, PLATE_TOP_Z, SERVO_MAX_STEPS, SERVO_STALL_STEPS, UR5_MIN_INSERT_DEPTH  # noqa: E402
from geometry import alignment_metrics, is_aligned, is_inserted, peg_tip_world  # noqa: E402
from ur5_common import connect, idle_gui, load_scene, step_tip_z  # noqa: E402


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


def servo_align_episode(
    gui: bool = False,
    hole_xy: tuple[float, float] | None = None,
    opaque_hole: bool = False,
    insert: bool = False,
) -> tuple[bool, bool | None, dict, tuple[float, float]]:
    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(gui, hole_xy=hole_xy)
    if gui and opaque_hole:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 1.0])
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]

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
        p.stepSimulation()
        if gui:
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

    return aligned, inserted, m, hole_xy


def run(gui: bool, insert: bool) -> bool:
    aligned, inserted, m, _ = servo_align_episode(gui=gui, opaque_hole=gui, insert=insert)
    print(
        f"dx={m['dx']*1e3:+.2f}mm dy={m['dy']*1e3:+.2f}mm "
        f"standoff={m['standoff']*1e3:.2f}mm "
        f"rpy=({math.degrees(m['roll']):+.2f}°, {math.degrees(m['pitch']):+.2f}°, {math.degrees(m['yaw']):+.2f}°)"
    )
    print("align ok" if aligned else "align fail")
    ok = aligned
    if insert:
        print("insert ok" if inserted else "insert fail")
        ok = aligned and bool(inserted)
    if gui:
        idle_gui()
    p.disconnect()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--insert", action="store_true", help="After alignment, continue to insert")
    args = ap.parse_args()
    sys.exit(0 if run(args.gui, args.insert) else 1)
