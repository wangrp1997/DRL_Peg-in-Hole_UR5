#!/usr/bin/env python3
"""Preview fixed eye-to-hand camera."""
from __future__ import annotations

import argparse
import random
import sys

import pybullet as p

from _paths import ROOT  # noqa: F401

from constants import FIXED_CAM_DEMO_STANDOFF, HOLE_X_RANGE, HOLE_Y_RANGE
from geometry import alignment_metrics, peg_tip_world
from sim.fixed_camera import eye_for_hole, target_for_hole
from sim.scene import (
    add_gui_camera_args,
    close_camera_windows,
    connect,
    idle_gui,
    load_scene,
    move_tip_to_standoff,
    refresh_camera_views,
    validate_gui_camera_args,
)


def sample_hole_xy(seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def run(gui: bool, opencv_render: bool, hole_xy: tuple[float, float] | None) -> None:
    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(
        gui, hole_xy=hole_xy, opencv_render=opencv_render, fixed_cam=True
    )
    if gui:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 1.0])
    move_tip_to_standoff(robot_id, eef, arm, peg, hole_xy, FIXED_CAM_DEMO_STANDOFF, gui=gui)
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]
    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    m = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    print(f"hole_xy={hole_xy}")
    print(f"fixed eye={eye_for_hole(hole_xy)} target={target_for_hole(hole_xy)}")
    print(f"demo standoff={m['standoff']*1e3:.1f}mm (aligned band 3–12mm)")
    if gui:
        refresh_camera_views()
        idle_gui()
    close_camera_windows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_gui_camera_args(ap)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random hole XY in X{HOLE_X_RANGE} Y{HOLE_Y_RANGE}; omit = hole under rest peg tip",
    )
    args = ap.parse_args()
    if args.gui and not args.fixed_cam:
        args.fixed_cam = True
    validate_gui_camera_args(ap, args)
    hole_xy = sample_hole_xy(args.seed) if args.seed is not None else None
    if args.seed is not None:
        print(f"seed={args.seed} hole_xy={hole_xy}")
    run(args.gui, args.opencv_render, hole_xy)
    sys.exit(0)
