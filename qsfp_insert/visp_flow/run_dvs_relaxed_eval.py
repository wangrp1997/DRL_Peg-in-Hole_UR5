#!/usr/bin/env python3
"""Eval: coarse (kabsch|ibvs) + DVS with relaxed start gate — 10 episodes each."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

from visp_flow._paths import REPO_ROOT
from visp_flow.episode import visp_flow_episode
from visp_flow.visp_constants import VISP_DVS_ERROR_TOL
from constants import HOLE_X_RANGE, HOLE_Y_RANGE


def run_one(
    episodes: int,
    seed: int,
    coarse_method: str,
    dvs_start_err: float,
    dvs_abort_err: float,
) -> dict:
    rng = random.Random(seed)
    rows = []
    for ep in range(1, episodes + 1):
        hole_xy = (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))
        _, report, _ = visp_flow_episode(
            gui=False,
            hole_xy=hole_xy,
            rng=rng,
            coarse_method=coarse_method,  # type: ignore[arg-type]
            always_run_dvs=True,
            dvs_start_err=dvs_start_err,
            dvs_abort_err=dvs_abort_err,
        )
        m = report.get("metrics") or {}
        row = {
            "episode": ep,
            "coarse_ok": report.get("coarse_ok"),
            "dvs_gated": report.get("dvs_gated"),
            "dvs_ok": report.get("dvs_ok"),
            "dvs_err": report.get("dvs_err"),
            "gt_aligned": report.get("gt_aligned"),
            "photometric_converged": bool(report.get("dvs_ok")),
        }
        if m:
            row["dx_mm"] = round(m.get("dx", 0) * 1e3, 3)
            row["dy_mm"] = round(m.get("dy", 0) * 1e3, 3)
        rows.append(row)
        print(
            f"[{coarse_method} {ep}/{episodes}] coarse={row['coarse_ok']} "
            f"gated={row['dvs_gated']} dvs_ok={row['dvs_ok']} "
            f"||e||²={row['dvs_err']:.4g} gt={row['gt_aligned']}"
        )

    dvs_ran = sum(1 for r in rows if not r["dvs_gated"])
    dvs_conv = sum(1 for r in rows if r["photometric_converged"])
    gt_ok = sum(1 for r in rows if r["gt_aligned"])
    return {
        "coarse_method": coarse_method,
        "episodes": episodes,
        "seed": seed,
        "dvs_start_err": dvs_start_err,
        "dvs_abort_err": dvs_abort_err,
        "dvs_convergence_tol": VISP_DVS_ERROR_TOL,
        "dvs_entered": dvs_ran,
        "dvs_photometric_ok": dvs_conv,
        "gt_aligned_after": gt_ok,
        "details": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dvs-start-err",
        type=float,
        default=1e10,
        help="Relaxed entry gate (default 1e10; official 10000)",
    )
    ap.add_argument("--dvs-abort-err", type=float, default=1e12)
    ap.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs"))
    args = ap.parse_args()

    summaries = []
    for method in ("kabsch", "ibvs"):
        print(f"\n=== {method} + relaxed DVS ===")
        summaries.append(
            run_one(args.episodes, args.seed, method, args.dvs_start_err, args.dvs_abort_err)
        )

    out = {
        "mode": "relaxed_dvs_eval",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": summaries,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(
        args.output_dir, f"visp_relaxed_dvs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
    for s in summaries:
        print(
            f"{s['coarse_method']}: entered={s['dvs_entered']}/{s['episodes']} "
            f"photometric_ok={s['dvs_photometric_ok']}/{s['episodes']} "
            f"gt={s['gt_aligned_after']}/{s['episodes']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
