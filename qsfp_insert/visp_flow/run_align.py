#!/usr/bin/env python3
"""Single ViSP baseline alignment episode."""
from __future__ import annotations

import argparse
import random
import sys

from visp_flow._paths import ROOT  # noqa: F401
from visp_flow.episode import visp_flow_episode
from sim.gui_preview import gt_keypoint_gui_idle


def main() -> int:
    ap = argparse.ArgumentParser(description="ViSP baseline: wrist2, kabsch|ibvs coarse + ViSP DVS fine")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--opencv_render", "--opencv", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coarse-method", choices=("kabsch", "ibvs"), default="kabsch")
    ap.add_argument(
        "--insert",
        action="store_true",
        help="After coarse alignment, continue descending to insert",
    )
    ap.add_argument(
        "--corners",
        choices=("gt", "deeplsd", "yolo"),
        default="gt",
        help="Corner source: gt (sim projection), deeplsd, yolo",
    )
    ap.add_argument(
        "--infer-corner0",
        action="store_true",
        help="Only corners 1–3 detected; infer corner 0 via parallelogram (sim GT test)",
    )
    ap.add_argument(
        "--lock-hole-corners",
        action="store_true",
        help="After perturb, freeze hole world corners from first frame; peg stays live YOLO",
    )
    ap.add_argument(
        "--skip-hole-h0",
        action="store_true",
        help="With --lock-hole-corners: lock from H1–H3 + parallelogram H0 (ignore YOLO H0)",
    )
    args = ap.parse_args()

    aligned, report, hole_xy, live = visp_flow_episode(
        gui=args.gui,
        opencv_render=args.opencv_render,
        rng=random.Random(args.seed),
        coarse_method=args.coarse_method,
        gui_idle=args.gui,
        insert=args.insert,
        corners=args.corners,  # type: ignore[arg-type]
        infer_corner0=args.infer_corner0,
        lock_hole_corners=args.lock_hole_corners,
        skip_hole_h0=args.skip_hole_h0,
    )
    if args.insert:
        ok = bool(report.get("coarse_ok") and report.get("inserted"))
    else:
        ok = aligned
    code = 0 if ok else 1
    if live is not None:
        rid, peg_i, hid, w2, provider = live
        gt_keypoint_gui_idle(
            rid, peg_i, hid,
            wrist2=w2,
            detach_wrist2=True,
            overlay_provider=provider,
            force_exit=code if args.opencv_render else None,
        )
    print(f"hole_xy={hole_xy} aligned={aligned}")
    if args.insert:
        print(f"coarse_ok={report.get('coarse_ok')} inserted={report.get('inserted')}")
    print(report)
    sys.exit(code)


if __name__ == "__main__":
    main()
