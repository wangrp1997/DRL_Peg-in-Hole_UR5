#!/usr/bin/env python3
"""10-episode template coarse eval — success = coarse_ok."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

from template_flow._paths import REPO_ROOT, ROOT  # noqa: F401
from template_flow.episode import template_flow_episode
from constants import HOLE_X_RANGE, HOLE_Y_RANGE


def sample_hole_xy(rng: random.Random) -> tuple[float, float]:
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def run(episodes: int, seed: int, output_dir: str, coarse_method: str, camera: str) -> dict:
    rng = random.Random(seed)
    rows = []
    ok = 0
    for ep in range(1, episodes + 1):
        hole_xy = sample_hole_xy(rng)
        coarse_ok, report, _, _ = template_flow_episode(
            gui=False,
            hole_xy=hole_xy,
            rng=rng,
            coarse_method=coarse_method,  # type: ignore[arg-type]
            camera=camera,  # type: ignore[arg-type]
        )
        success = bool(coarse_ok)
        ok += int(success)
        m = report.get("metrics") or {}
        rows.append({
            "episode": ep,
            "hole_x": round(hole_xy[0], 4),
            "hole_y": round(hole_xy[1], 4),
            "coarse_ok": coarse_ok,
            "success": success,
            "gt_aligned": report.get("gt_aligned"),
            "coarse_method": coarse_method,
            "dx_mm": round(m.get("dx", 0) * 1e3, 3),
            "dy_mm": round(m.get("dy", 0) * 1e3, 3),
        })
        print(f"[{ep}/{episodes}] hole=({hole_xy[0]:.3f},{hole_xy[1]:.3f}) coarse={coarse_ok}")
    rate = ok / episodes * 100
    summary = {
        "mode": f"template_coarse_{camera}",
        "camera": camera,
        "episodes": episodes,
        "successes": ok,
        "failures": episodes - ok,
        "success_rate_pct": round(rate, 1),
        "seed": seed,
        "coarse_method": coarse_method,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": rows,
    }
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_dir, f"template_flow_{camera}_{coarse_method}_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{ok}/{episodes} = {rate:.1f}% -> {out}")
    summary["_result_file"] = out
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coarse-method", choices=("kabsch", "ibvs"), default="kabsch")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--camera", choices=("fixed", "wrist2"), default="fixed",
                    help="fixed: hole=teach peg=XFeat; wrist2: peg=teach hole=XFeat")
    ap.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs"))
    args = ap.parse_args()
    if args.both:
        s_k = run(args.episodes, args.seed, args.output_dir, "kabsch", args.camera)
        s_i = run(args.episodes, args.seed, args.output_dir, "ibvs", args.camera)
        sys.exit(0 if s_k["failures"] + s_i["failures"] == 0 else 1)
    summary = run(args.episodes, args.seed, args.output_dir, args.coarse_method, args.camera)
    sys.exit(0 if summary["failures"] == 0 else 1)
