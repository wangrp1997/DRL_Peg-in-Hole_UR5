#!/usr/bin/env python3
"""Preview wrist_camera2 at IK standoff — same pattern as fixed_camera_demo."""
from __future__ import annotations

import argparse
import math
import random
import sys
import time

import pybullet as p

from _paths import ROOT  # noqa: F401

from constants import FIXED_CAM_DEMO_STANDOFF, HOLE_X_RANGE, HOLE_Y_RANGE, PLATE_TOP_Z
from geometry import alignment_metrics, peg_tip_world
from sim.fixed_camera import eye_for_hole, target_for_hole
from sim.scene import (
    close_camera_windows,
    connect,
    get_wrist_camera2,
    idle_gui,
    load_scene,
    move_ee,
    move_tip_to_standoff,
    refresh_camera_views,
    register_wrist_camera2,
    setup_wrist_camera2_views,
)
from sim.wrist_camera2 import attach_wrist_camera2
from vision.corners import gt_image_keypoints
from vision.debug_markers import clear_gt_corner_markers, sync_gt_corner_markers


def sample_hole_xy(seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


# Peg tip orbits hole centre at standoff (eye-in-hand: hole should shift in image).
_CIRCLE_RADIUS_M = 0.006
_CIRCLE_PERIOD_S = 10.0


def _move_tip_xy_offset(
    robot_id: int,
    eef: int,
    arm: list[int],
    peg: int,
    hole_xy: tuple[float, float],
    standoff: float,
    dx: float,
    dy: float,
    ee_orn,
) -> None:
    tip_z = PLATE_TOP_Z + standoff
    tx = hole_xy[0] + dx
    ty = hole_xy[1] + dy
    ee_pos, _ = p.getLinkState(robot_id, eef)[:2]
    tip = peg_tip_world(robot_id, peg)
    move_ee(
        robot_id,
        eef,
        arm,
        [
            ee_pos[0] - (tip[0] - tx),
            ee_pos[1] - (tip[1] - ty),
            ee_pos[2] + (tip_z - tip[2]),
        ],
        ee_orn,
    )


def _idle_circle(
    robot_id: int,
    arm: list[int],
    eef: int,
    peg: int,
    hole_id: int,
    hole_xy: tuple[float, float],
    ee_orn,
    draw_keypoints: bool,
) -> None:
    """Orbit peg tip on XY circle; refresh wrist_camera2 each step."""
    provider = _keypoint_provider(robot_id, peg, hole_id) if draw_keypoints else None
    print(
        f"circle move: r={_CIRCLE_RADIUS_M * 1e3:.1f}mm period={_CIRCLE_PERIOD_S:.0f}s "
        "(hole fixed in world; eye-in-hand → hole shifts in image). Close PyBullet to exit."
    )
    t0 = time.monotonic()
    while p.getConnectionInfo()["isConnected"]:
        t = time.monotonic() - t0
        theta = 2.0 * math.pi * (t / _CIRCLE_PERIOD_S)
        dx = _CIRCLE_RADIUS_M * math.cos(theta)
        dy = _CIRCLE_RADIUS_M * math.sin(theta)
        _move_tip_xy_offset(
            robot_id, eef, arm, peg, hole_xy, FIXED_CAM_DEMO_STANDOFF, dx, dy, ee_orn
        )
        p.stepSimulation()
        if draw_keypoints:
            sync_gt_corner_markers(robot_id, peg, hole_id)
        refresh_camera_views(provider)
        time.sleep(1.0 / 240.0)


def _keypoint_provider(robot_id: int, peg: int, hole_id: int):
    def _fn():
        cam = get_wrist_camera2()
        if cam is None:
            return None
        return gt_image_keypoints(cam, robot_id, peg, hole_id)

    return _fn


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
    move: bool,
) -> None:
    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(gui, hole_xy=hole_xy)
    if gui:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 1.0])

    move_tip_to_standoff(robot_id, eef, arm, peg, hole_xy, FIXED_CAM_DEMO_STANDOFF, gui=gui)
    cam2 = attach_wrist_camera2(robot_id, eef, hole_xy)
    register_wrist_camera2(cam2)
    setup_wrist_camera2_views(gui, opencv_render)

    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]
    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    m0 = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    cam_pos, _ = cam2._pose()

    print(f"hole_xy={hole_xy}")
    print(f"fixed_cam reference eye={eye_for_hole(hole_xy)} target={target_for_hole(hole_xy)}")
    print(f"wrist_camera2 world pos={tuple(round(x, 4) for x in cam_pos)}")
    print(f"intrinsics {cam2.width}x{cam2.height} FOV={cam2.fov}°")
    print(f"demo standoff={m0['standoff']*1e3:.1f}mm")

    if draw_keypoints:
        sets = gt_image_keypoints(cam2, robot_id, peg, hole_id)
        for s in sets:
            n = sum(s.visible)
            print(f"GT {s.name}: {n}/{len(s.uv)} corners visible in wrist_camera2 image")

    if gui:
        ee_orn = p.getLinkState(robot_id, eef)[1]
        if move:
            _idle_circle(robot_id, arm, eef, peg, hole_id, hole_xy, ee_orn, draw_keypoints)
        elif draw_keypoints:
            _idle_with_keypoints(robot_id, peg, hole_id)
        else:
            refresh_camera_views()
            idle_gui()

    cam2.detach()
    register_wrist_camera2(None)
    clear_gt_corner_markers()
    close_camera_windows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="wrist_camera2 preview at IK standoff (like fixed_camera_demo)")
    ap.add_argument("--gui", action="store_true", help="PyBullet 3D + wrist_camera2 corner RGB/Depth/Seg previews")
    ap.add_argument(
        "--opencv_render",
        action="store_true",
        help="OpenCV RGB|Depth|Seg panel (requires --gui; auto on with --draw_keypoints)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random hole XY in X{HOLE_X_RANGE} Y{HOLE_Y_RANGE}; omit = hole under rest peg tip",
    )
    ap.add_argument(
        "--draw_keypoints",
        action="store_true",
        help="Overlay GT hole (H0–H3) and peg (P0–P3) corners on OpenCV panel",
    )
    ap.add_argument(
        "--move",
        action="store_true",
        help="Orbit peg tip on XY circle (6 mm r, 10 s); watch hole shift in wrist_camera2 view",
    )
    args = ap.parse_args()
    if args.opencv_render and not args.gui:
        ap.error("--opencv_render requires --gui")
    if args.move and not args.gui:
        ap.error("--move requires --gui")
    if (args.draw_keypoints or args.move) and args.gui:
        args.opencv_render = True
    hole_xy = sample_hole_xy(args.seed) if args.seed is not None else None
    if args.seed is not None:
        print(f"seed={args.seed} hole_xy={hole_xy}")
    run(args.gui, args.opencv_render, hole_xy, args.draw_keypoints, args.move)
    sys.exit(0)
