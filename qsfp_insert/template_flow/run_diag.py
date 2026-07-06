#!/usr/bin/env python3
"""Debug template corners vs GT (sim only)."""
from __future__ import annotations

import argparse
import random
import sys

from template_flow._paths import ROOT  # noqa: F401
from template_flow.diag import run_corner_diag


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare template corners to GT after perturb")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--camera", choices=("fixed", "wrist2"), default="fixed")
    args = ap.parse_args()
    ok = run_corner_diag(
        seed=args.seed,
        gui=args.gui,
        rng=random.Random(args.seed),
        camera=args.camera,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
