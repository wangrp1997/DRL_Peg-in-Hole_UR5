#!/usr/bin/env python3
"""Single ViSP baseline alignment episode."""
from __future__ import annotations

import argparse
import random

from visp_flow._paths import ROOT  # noqa: F401
from visp_flow.episode import visp_flow_episode


def main() -> None:
    ap = argparse.ArgumentParser(description="ViSP baseline: wrist2, kabsch|ibvs coarse + ViSP DVS fine")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--opencv_render", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coarse-method", choices=("kabsch", "ibvs"), default="kabsch")
    args = ap.parse_args()

    aligned, report, hole_xy = visp_flow_episode(
        gui=args.gui,
        opencv_render=args.opencv_render,
        rng=random.Random(args.seed),
        coarse_method=args.coarse_method,
    )
    print(f"hole_xy={hole_xy} aligned={aligned}")
    print(report)


if __name__ == "__main__":
    main()
