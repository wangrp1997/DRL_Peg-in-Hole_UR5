#!/usr/bin/env python3
"""Batch eval for corner keypoint visual servo."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

import pybullet as p

from _paths import ROOT, REPO_ROOT  # noqa: F401

from constants import CORNER_ALIGN_METHOD, HOLE_X_RANGE, HOLE_Y_RANGE
from vision.corner_servo_episode import corner_servo_episode, sample_hole_xy


def run(
    episodes: int,
    seed: int,
    output_dir: str,
    align_method: str,
    insert: bool,
    infer_corner0: bool,
) -> dict:
    rng = random.Random(seed)
    rows = []
    ok = 0

    for ep in range(1, episodes + 1):
        hole_xy = sample_hole_xy(rng)
        aligned, info, _, _ = corner_servo_episode(
            gui=False,
            hole_xy=hole_xy,
            rng=rng,
            align_method=align_method,
            insert=insert,
            infer_corner0=infer_corner0,
        )
        if insert:
            inserted = bool(info and info.get("inserted"))
            success = bool(aligned and inserted)
        else:
            success = bool(aligned and info and info.get("gt_aligned"))
        ok += int(success)
        row = {
            "episode": ep,
            "hole_x": round(hole_xy[0], 4),
            "hole_y": round(hole_xy[1], 4),
            "aligned": aligned,
            "success": success,
        }
        if info:
            row.update({k: v for k, v in info.items() if k != "perturb"})
            if info.get("perturb"):
                row["perturb"] = info["perturb"]
        rows.append(row)
        if insert:
            inserted = bool(info and info.get("inserted"))
            status = "insert ok" if success else f"align={'ok' if aligned else 'fail'} insert={'ok' if inserted else 'fail'}"
        else:
            status = "align ok" if success else "align fail"
        if info:
            print(
                f"[{ep}/{episodes}] hole=({hole_xy[0]:.3f},{hole_xy[1]:.3f}) {status} "
                f"dx={info['dx_mm']:+.2f}mm dy={info['dy_mm']:+.2f}mm standoff={info['standoff_mm']:.2f}mm "
                f"gt_aligned={info.get('gt_aligned')}"
                + (f" inserted={info.get('inserted')}" if insert else "")
            )
        else:
            print(f"[{ep}/{episodes}] hole=({hole_xy[0]:.3f},{hole_xy[1]:.3f}) {status} (no metrics)")

    rate = ok / episodes * 100
    label = "insert" if insert else "align"
    summary = {
        "mode": label,
        "episodes": episodes,
        "successes": ok,
        "failures": episodes - ok,
        "success_rate_pct": round(rate, 1),
        "seed": seed,
        "align_method": align_method,
        "insert": insert,
        "infer_corner0": infer_corner0,
        "hole_x_range": list(HOLE_X_RANGE),
        "hole_y_range": list(HOLE_Y_RANGE),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": rows,
    }
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_dir, f"corner_servo_eval_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{ok}/{episodes} = {rate:.1f}% -> {out}")
    summary["_result_file"] = out
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--align-method",
        choices=("kabsch", "ibvs", "dvs"),
        default=CORNER_ALIGN_METHOD,
    )
    ap.add_argument(
        "--insert",
        action="store_true",
        help="After alignment, continue descending to insert; count align+insert success",
    )
    ap.add_argument(
        "--infer-corner0",
        action="store_true",
        help="Only corners 1–3 visible; infer corner 0 from parallelogram",
    )
    ap.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs"))
    args = ap.parse_args()
    summary = run(
        args.episodes,
        args.seed,
        args.output_dir,
        args.align_method,
        args.insert,
        args.infer_corner0,
    )
    sys.exit(0 if summary["failures"] == 0 else 1)
