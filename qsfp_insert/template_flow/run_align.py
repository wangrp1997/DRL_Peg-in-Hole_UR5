#!/usr/bin/env python3
"""Single template coarse episode (kabsch | ibvs)."""
from __future__ import annotations

import argparse
import random
import sys

from template_flow._paths import ROOT  # noqa: F401
from template_flow.episode import template_flow_episode
from sim.gui_preview import gt_keypoint_gui_idle


def main() -> int:
    ap = argparse.ArgumentParser(description="Template coarse: teach + XFeat corners + kabsch|ibvs")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--opencv_render", "--opencv", action="store_true",
                    help="OpenCV RGB|Depth|Seg panel (requires --gui)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coarse-method", choices=("kabsch", "ibvs"), default="kabsch")
    ap.add_argument("--camera", choices=("fixed", "wrist2"), default="fixed")
    args = ap.parse_args()
    coarse_ok, report, hole_xy, live = template_flow_episode(
        gui=args.gui,
        opencv_render=args.opencv_render,
        rng=random.Random(args.seed),
        coarse_method=args.coarse_method,
        camera=args.camera,
        gui_idle=args.gui,
    )
    code = 0 if coarse_ok else 1
    if live is not None:
        rid, peg_i, hid, w2, tracker = live
        gt_keypoint_gui_idle(
            rid, peg_i, hid,
            wrist2=w2,
            detach_wrist2=True,
            force_exit=code if args.opencv_render else None,
            overlay_provider=tracker.display_keypoints,
        )
    print(f"hole_xy={hole_xy} coarse_ok={coarse_ok}")
    print(report)
    sys.exit(code)


if __name__ == "__main__":
    main()
