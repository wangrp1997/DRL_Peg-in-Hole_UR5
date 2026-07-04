#!/usr/bin/env python3
"""Random hole XY eval for servo_align. Usage: python qsfp_insert/demo/servo_align_eval.py [--episodes 10] [--insert]"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import pybullet as p

from _paths import ROOT  # noqa: F401

from constants import HOLE_X_RANGE, HOLE_Y_RANGE

_PROJECT_ROOT = os.path.dirname(ROOT)

from servo_align import servo_align_episode  # noqa: E402


def _sample_hole_xy(rng: random.Random) -> tuple[float, float]:
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def run(episodes: int, seed: int | None, output_dir: str, insert: bool) -> dict:
    rng = random.Random(seed)
    rows = []
    ok_count = 0

    for ep in range(1, episodes + 1):
        hole_xy = _sample_hole_xy(rng)
        aligned, inserted, m, _, _ = servo_align_episode(gui=False, hole_xy=hole_xy, insert=insert)
        p.disconnect()
        success = bool(inserted) if insert else aligned
        ok_count += int(success)
        row = {
            "episode": ep,
            "hole_x": round(hole_xy[0], 4),
            "hole_y": round(hole_xy[1], 4),
            "aligned": aligned,
            "inserted": inserted,
            "success": success,
            "dx_mm": round(m["dx"] * 1e3, 3),
            "dy_mm": round(m["dy"] * 1e3, 3),
            "standoff_mm": round(m["standoff"] * 1e3, 3),
            "roll_deg": round(math.degrees(m["roll"]), 3),
            "pitch_deg": round(math.degrees(m["pitch"]), 3),
            "yaw_deg": round(math.degrees(m["yaw"]), 3),
        }
        rows.append(row)
        if insert:
            status = "insert ok" if success else f"align={'ok' if aligned else 'fail'} insert fail"
        else:
            status = "align ok" if success else "align fail"
        print(
            f"[{ep}/{episodes}] hole=({hole_xy[0]:.3f},{hole_xy[1]:.3f}) {status} "
            f"dx={row['dx_mm']:+.2f}mm dy={row['dy_mm']:+.2f}mm standoff={row['standoff_mm']:.2f}mm"
        )

    rate = ok_count / episodes * 100
    label = "insert" if insert else "align"
    summary = {
        "mode": label,
        "episodes": episodes,
        "successes": ok_count,
        "failures": episodes - ok_count,
        "success_rate_pct": round(rate, 1),
        "hole_x_range": list(HOLE_X_RANGE),
        "hole_y_range": list(HOLE_Y_RANGE),
        "seed": seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": rows,
    }

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_dir, f"servo_align_eval_{label}_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["_result_file"] = out
    print(f"\n{ok_count}/{episodes} = {rate:.1f}% ({label}) -> {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--insert", action="store_true", help="Align then insert; count insert success")
    ap.add_argument("--output-dir", default=os.path.join(_PROJECT_ROOT, "outputs"))
    args = ap.parse_args()
    summary = run(args.episodes, args.seed, args.output_dir, args.insert)
    sys.exit(0 if summary["failures"] == 0 else 1)
