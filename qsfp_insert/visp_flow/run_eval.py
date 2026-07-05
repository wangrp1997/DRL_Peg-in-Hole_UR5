#!/usr/bin/env python3
"""10-episode ViSP baseline eval (wrist_camera2, kabsch|ibvs coarse + ViSP DVS fine)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

from visp_flow._paths import REPO_ROOT, ROOT  # noqa: F401
from visp_flow.episode import visp_flow_episode
from constants import HOLE_X_RANGE, HOLE_Y_RANGE


def sample_hole_xy(rng: random.Random) -> tuple[float, float]:
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def run(episodes: int, seed: int, output_dir: str, coarse_method: str) -> dict:
    rng = random.Random(seed)
    rows = []
    ok = 0

    for ep in range(1, episodes + 1):
        hole_xy = sample_hole_xy(rng)
        aligned, report, _ = visp_flow_episode(
            gui=False,
            hole_xy=hole_xy,
            rng=rng,
            coarse_method=coarse_method,  # type: ignore[arg-type]
        )
        success = bool(aligned and report.get("gt_aligned"))
        ok += int(success)
        m = report.get("metrics") or {}
        row = {
            "episode": ep,
            "hole_x": round(hole_xy[0], 4),
            "hole_y": round(hole_xy[1], 4),
            "aligned": aligned,
            "success": success,
            "coarse_ok": report.get("coarse_ok"),
            "dvs_ok": report.get("dvs_ok"),
            "dvs_gated": report.get("dvs_gated"),
            "dvs_err": report.get("dvs_err"),
            "coarse_method": coarse_method,
        }
        if m:
            row.update({
                "dx_mm": round(m.get("dx", 0) * 1e3, 3),
                "dy_mm": round(m.get("dy", 0) * 1e3, 3),
                "standoff_mm": round(m.get("standoff", 0) * 1e3, 3),
            })
        rows.append(row)
        print(
            f"[{ep}/{episodes}] hole=({hole_xy[0]:.3f},{hole_xy[1]:.3f}) "
            f"coarse={report.get('coarse_ok')} dvs={report.get('dvs_ok')} success={success}"
        )

    rate = ok / episodes * 100
    summary = {
        "mode": "visp_baseline",
        "episodes": episodes,
        "successes": ok,
        "failures": episodes - ok,
        "success_rate_pct": round(rate, 1),
        "seed": seed,
        "coarse_method": coarse_method,
        "hole_x_range": list(HOLE_X_RANGE),
        "hole_y_range": list(HOLE_Y_RANGE),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": rows,
    }
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_dir, f"visp_flow_eval_{ts}.json")
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
    ap.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs"))
    args = ap.parse_args()
    summary = run(args.episodes, args.seed, args.output_dir, args.coarse_method)
    sys.exit(0 if summary["failures"] == 0 else 1)
