#!/usr/bin/env python3
"""Corner keypoint visual servo: IK standoff → 6D perturb → align (GT corners for now)."""
from __future__ import annotations

import argparse
import random
import sys

from _paths import ROOT  # noqa: F401

from constants import CORNER_ALIGN_METHOD, HOLE_X_RANGE, HOLE_Y_RANGE
from vision.corner_servo_episode import corner_servo_episode, corner_servo_gui_idle, sample_hole_xy


def main() -> int:
    ap = argparse.ArgumentParser(description="Corner keypoint servo align (fixed camera)")
    ap.add_argument("--gui", action="store_true", help="PyBullet + OpenCV keypoint overlay")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help=f"Random hole XY {HOLE_X_RANGE}/{HOLE_Y_RANGE} and 6D tip perturbation",
    )
    ap.add_argument(
        "--align-method",
        choices=("kabsch", "ibvs"),
        default=CORNER_ALIGN_METHOD,
        help="6D error: kabsch (planar PnP) or ibvs",
    )
    ap.add_argument(
        "--insert",
        action="store_true",
        help="After alignment, continue descending to insert",
    )
    ap.add_argument(
        "--infer-corner0",
        action="store_true",
        help="Only corners 1–3 visible; infer corner 0 from known rectangle (magenta overlay)",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)
    hole_xy = sample_hole_xy(rng)
    opencv = args.gui

    print(
        f"seed={args.seed} hole_xy={hole_xy} align_method={args.align_method} "
        f"insert={args.insert} infer_corner0={args.infer_corner0}"
    )
    print("nominal pose: IK to standoff, then random 6D perturb, then corner servo")
    aligned, info, hole_xy, live = corner_servo_episode(
        gui=args.gui,
        hole_xy=hole_xy,
        opencv_render=opencv,
        rng=rng,
        gui_idle=args.gui,
        align_method=args.align_method,
        insert=args.insert,
        infer_corner0=args.infer_corner0,
    )

    if info is None:
        print("align fail (no valid corner metrics)")
        if live is not None:
            corner_servo_gui_idle(*live)
        return 1
    print(
        f"vision dx={info['dx_mm']:+.2f}mm dy={info['dy_mm']:+.2f}mm standoff={info['standoff_mm']:.2f}mm "
        f"rpy=({info['roll_deg']:+.2f}°, {info['pitch_deg']:+.2f}°, {info['yaw_deg']:+.2f}°)"
    )
    if "gt_dx_mm" in info:
        print(
            f"GT     dx={info['gt_dx_mm']:+.2f}mm dy={info['gt_dy_mm']:+.2f}mm "
            f"standoff={info['gt_standoff_mm']:.2f}mm "
            f"rpy=({info['gt_roll_deg']:+.2f}°, {info['gt_pitch_deg']:+.2f}°, {info['gt_yaw_deg']:+.2f}°) "
            f"gt_aligned={info['gt_aligned']}"
        )
    print("align ok" if aligned else "align fail")
    if info.get("gt_aligned") is False and aligned:
        print("warning: vision converged but GT still misaligned")

    if args.insert:
        inserted = info.get("inserted")
        print("insert ok" if inserted else "insert fail")
        ok = bool(aligned and inserted)
    else:
        ok = aligned

    if live is not None:
        corner_servo_gui_idle(*live)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
