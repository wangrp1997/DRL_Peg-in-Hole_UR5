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
    args = ap.parse_args()

    aligned, report, hole_xy, live = visp_flow_episode(
        gui=args.gui,
        opencv_render=args.opencv_render,
        rng=random.Random(args.seed),
        coarse_method=args.coarse_method,
        gui_idle=args.gui,
        insert=args.insert,
    )
    if args.insert:
        ok = bool(report.get("coarse_ok") and report.get("inserted"))
    else:
        ok = aligned
    code = 0 if ok else 1
    if live is not None:
        rid, peg_i, hid, w2 = live
        gt_keypoint_gui_idle(
            rid, peg_i, hid,
            wrist2=w2,
            detach_wrist2=True,
            force_exit=code if args.opencv_render else None,
        )
    print(f"hole_xy={hole_xy} aligned={aligned}")
    if args.insert:
        print(f"coarse_ok={report.get('coarse_ok')} inserted={report.get('inserted')}")
    print(report)
    sys.exit(code)


if __name__ == "__main__":
    main()
