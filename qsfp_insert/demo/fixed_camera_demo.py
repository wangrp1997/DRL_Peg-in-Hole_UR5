#!/usr/bin/env python3
"""Preview fixed eye-to-hand camera."""
from __future__ import annotations

import argparse
import math
import random
import sys
import time

import pybullet as p

from _paths import ROOT  # noqa: F401

from constants import FIXED_CAM_DEMO_STANDOFF, HOLE_X_RANGE, HOLE_Y_RANGE, SERVO_MAX_STEPS, SERVO_STALL_STEPS
from geometry import alignment_metrics, is_aligned, peg_tip_world
from sim.cartesian_control import alignment_twist, apply_cartesian_velocity, stop_arm
from sim.fixed_camera import eye_for_hole, target_for_hole
from sim.scene import (
    add_gui_camera_args,
    close_camera_windows,
    connect,
    get_fixed_camera,
    idle_gui,
    load_scene,
    move_tip_to_standoff,
    refresh_camera_views,
    validate_gui_camera_args,
)
from vision.corners import gt_image_keypoints
from vision.debug_markers import clear_gt_corner_markers, sync_gt_corner_markers


def sample_hole_xy(seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def _keypoint_provider(robot_id: int, peg: int, hole_id: int):
    def _fn():
        cam = get_fixed_camera()
        if cam is None:
            return None
        return gt_image_keypoints(cam, robot_id, peg, hole_id)

    return _fn


def _align_phase(
    robot_id: int,
    arm: list[int],
    peg: int,
    hole_id: int,
    hole_xy: tuple[float, float],
    hole_orn,
    gui: bool,
    keypoint_provider=None,
) -> tuple[bool, dict]:
    """Cartesian servo from current pose (e.g. demo standoff) to hole mouth."""
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
            if keypoint_provider is not None:
                sync_gt_corner_markers(robot_id, peg, hole_id)
            refresh_camera_views(keypoint_provider)
            time.sleep(1.0 / 240.0)

    stop_arm(robot_id, arm)
    for _ in range(20):
        p.stepSimulation()
        if gui and keypoint_provider is not None:
            sync_gt_corner_markers(robot_id, peg, hole_id)
            refresh_camera_views(keypoint_provider)

    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    m = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    aligned = is_aligned(tip, peg_orn, hole_xy, hole_orn)
    return aligned, m


def _idle_with_keypoints(robot_id: int, peg: int, hole_id: int) -> None:
    print("Close PyBullet window to exit.")
    provider = _keypoint_provider(robot_id, peg, hole_id)
    while p.getConnectionInfo()["isConnected"]:
        p.stepSimulation()
        sync_gt_corner_markers(robot_id, peg, hole_id)
        refresh_camera_views(provider)
        time.sleep(1.0 / 240.0)


def run(
    gui: bool,
    opencv_render: bool,
    hole_xy: tuple[float, float] | None,
    draw_keypoints: bool,
    align: bool,
) -> None:
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
    m0 = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    print(f"hole_xy={hole_xy}")
    print(f"fixed eye={eye_for_hole(hole_xy)} target={target_for_hole(hole_xy)}")
    print(f"demo standoff={m0['standoff']*1e3:.1f}mm (aligned band 3–12mm)")

    provider = _keypoint_provider(robot_id, peg, hole_id) if draw_keypoints else None

    if align:
        print("aligning from demo standoff…")
        aligned, m = _align_phase(robot_id, arm, peg, hole_id, hole_xy, hole_orn, gui, provider)
        print(
            f"dx={m['dx']*1e3:+.2f}mm dy={m['dy']*1e3:+.2f}mm standoff={m['standoff']*1e3:.2f}mm "
            f"rpy=({math.degrees(m['roll']):+.2f}°, {math.degrees(m['pitch']):+.2f}°, {math.degrees(m['yaw']):+.2f}°)"
        )
        print("align ok" if aligned else "align fail")

    if gui:
        if draw_keypoints:
            cam = get_fixed_camera()
            if cam is not None and not align:
                sets = gt_image_keypoints(cam, robot_id, peg, hole_id)
                for s in sets:
                    n = sum(s.visible)
                    print(f"GT {s.name}: {n}/{len(s.uv)} corners visible in image")
            _idle_with_keypoints(robot_id, peg, hole_id)
        else:
            refresh_camera_views()
            idle_gui()
    clear_gt_corner_markers()
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
    ap.add_argument(
        "--draw_keypoints",
        action="store_true",
        help="Overlay GT hole (H0–H3) and peg tip (P0–P3) corners on fixed-cam OpenCV panel",
    )
    ap.add_argument(
        "--align",
        action="store_true",
        help="Cartesian servo from demo standoff to hole mouth (fixed cam stays on hole; peg moves in view)",
    )
    args = ap.parse_args()
    if args.gui and not args.fixed_cam:
        args.fixed_cam = True
    if (args.draw_keypoints or args.align) and args.gui:
        args.opencv_render = True
        args.fixed_cam = True
    validate_gui_camera_args(ap, args)
    hole_xy = sample_hole_xy(args.seed) if args.seed is not None else None
    if args.seed is not None:
        print(f"seed={args.seed} hole_xy={hole_xy}")
    run(args.gui, args.opencv_render, hole_xy, args.draw_keypoints, args.align)
    sys.exit(0)
