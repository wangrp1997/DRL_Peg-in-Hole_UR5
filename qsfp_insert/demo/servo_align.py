#!/usr/bin/env python3
"""UR5: 6D Cartesian velocity servo to hole-mouth standoff. Usage: python qsfp_insert/demo/servo_align.py [--gui]"""
from __future__ import annotations

import argparse
import math
import sys
import time

import pybullet as p

from _paths import ROOT  # noqa: F401

from cartesian_control import alignment_twist, apply_cartesian_velocity, stop_arm  # noqa: E402
from geometry import alignment_metrics, is_aligned, peg_tip_world  # noqa: E402
from ur5_common import connect, idle_gui, load_scene  # noqa: E402


def run(gui: bool) -> bool:
    connect(gui)
    robot_id, arm, _eef, peg, hole_id, hole_xy = load_scene(gui)
    if gui:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 1.0])
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]

    ok = False
    for _ in range(4000):
        tip = peg_tip_world(robot_id, peg)
        peg_orn = p.getLinkState(robot_id, peg)[1]
        if is_aligned(tip, peg_orn, hole_xy, hole_orn):
            ok = True
            break
        m = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
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
    ok = is_aligned(tip, peg_orn, hole_xy, hole_orn)
    print(
        f"dx={m['dx']*1e3:+.2f}mm dy={m['dy']*1e3:+.2f}mm "
        f"standoff={m['standoff']*1e3:.2f}mm "
        f"rpy=({math.degrees(m['roll']):+.2f}°, {math.degrees(m['pitch']):+.2f}°, {math.degrees(m['yaw']):+.2f}°)"
    )
    print("align ok" if ok else "align fail")
    if gui:
        idle_gui()
    p.disconnect()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    sys.exit(0 if run(ap.parse_args().gui) else 1)
