#!/usr/bin/env python3
"""10-episode ViSP baseline eval (wrist_camera2, kabsch|ibvs coarse + ViSP DVS fine)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

from visp_flow._paths import REPO_ROOT, ROOT  # noqa: F401
from visp_flow.episode import visp_flow_episode
from constants import HOLE_X_RANGE, HOLE_Y_RANGE


def sample_hole_xy(rng: random.Random) -> tuple[float, float]:
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def run(
    episodes: int,
    seed: int,
    output_dir: str,
    coarse_method: str,
    insert: bool,
    *,
    corners: str = "gt",
    infer_corner0: bool = False,
) -> dict:
    rng = random.Random(seed)
    rows = []
    ok = 0

    for ep in range(1, episodes + 1):
        hole_xy = sample_hole_xy(rng)
        aligned, report, _, _ = visp_flow_episode(
            gui=False,
            hole_xy=hole_xy,
            rng=rng,
            coarse_method=coarse_method,  # type: ignore[arg-type]
            insert=insert,
            corners=corners,  # type: ignore[arg-type]
            infer_corner0=infer_corner0,
        )
        if insert:
            success = bool(report.get("coarse_ok") and report.get("inserted"))
        else:
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
            "inserted": report.get("inserted"),
            "dvs_ok": report.get("dvs_ok"),
            "dvs_gated": report.get("dvs_gated"),
            "dvs_err": report.get("dvs_err"),
            "coarse_method": coarse_method,
            "insert": insert,
            "corners": corners,
            "infer_corner0": infer_corner0,
        }
        if m:
            row.update({
                "dx_mm": round(m.get("dx", 0) * 1e3, 3),
                "dy_mm": round(m.get("dy", 0) * 1e3, 3),
                "standoff_mm": round(m.get("standoff", 0) * 1e3, 3),
                "roll_deg": round(math.degrees(m.get("roll", 0)), 3),
                "pitch_deg": round(math.degrees(m.get("pitch", 0)), 3),
                "yaw_deg": round(math.degrees(m.get("yaw", 0)), 3),
            })
        rows.append(row)
        if insert:
            status = (
                "insert ok" if success
                else f"coarse={'ok' if report.get('coarse_ok') else 'fail'} "
                     f"insert={'ok' if report.get('inserted') else 'fail'}"
            )
        else:
            status = "success" if success else "fail"
        m = report.get("metrics") or {}
        gt_flag = "ok" if report.get("gt_aligned") else "fail"
        rpy = ""
        if m:
            rpy = (
                f" GT_rpy=({math.degrees(m.get('roll', 0)):+.2f}°, "
                f"{math.degrees(m.get('pitch', 0)):+.2f}°, "
                f"{math.degrees(m.get('yaw', 0)):+.2f}°)"
            )
        print(
            f"[{ep}/{episodes}] hole=({hole_xy[0]:.3f},{hole_xy[1]:.3f}) "
            f"视觉粗对准={report.get('coarse_ok')} GT={gt_flag} dvs={report.get('dvs_ok')} {status}"
            f"{rpy}"
        )

    rate = ok / episodes * 100
    label = "visp_insert" if insert else "visp_baseline"
    summary = {
        "mode": label,
        "episodes": episodes,
        "successes": ok,
        "failures": episodes - ok,
        "success_rate_pct": round(rate, 1),
        "seed": seed,
        "coarse_method": coarse_method,
        "insert": insert,
        "corners": corners,
        "infer_corner0": infer_corner0,
        "hole_x_range": list(HOLE_X_RANGE),
        "hole_y_range": list(HOLE_Y_RANGE),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": rows,
    }
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_insert" if insert else ""
    out = os.path.join(output_dir, f"visp_flow_eval{suffix}_{ts}.json")
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
    ap.add_argument(
        "--insert",
        action="store_true",
        help="After coarse alignment, descend to insert; count coarse+insert success",
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
        help="Only corners 1–3 visible; infer corner 0 via parallelogram (sim test)",
    )
    ap.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs"))
    args = ap.parse_args()
    summary = run(
        args.episodes,
        args.seed,
        args.output_dir,
        args.coarse_method,
        args.insert,
        corners=args.corners,
        infer_corner0=args.infer_corner0,
    )
    sys.exit(0 if summary["failures"] == 0 else 1)
