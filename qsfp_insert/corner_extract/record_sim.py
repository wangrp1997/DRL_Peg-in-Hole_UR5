#!/usr/bin/env python3
"""Record wrist2 gray video: random perturb @ standoff → kabsch/ibvs align (真机式连续轨迹)."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np
import pybullet as p

from corner_extract._paths import DEFAULT_RAW_DIR, ROOT  # noqa: F401
from constants import (
    ALIGN_Z_STANDOFF_MIN,
    HOLE_X_RANGE,
    HOLE_Y_RANGE,
    PLATE_TOP_Z,
    SETTLE_IK_STEPS_GUI,
)
from geometry import peg_tip_world
from sim.cartesian_align_policy import cartesian_align_target
from sim.perturbation import (
    apply_tip_perturbation,
    format_perturbation_log,
    sample_collect_perturbation6,
)
from sim.scene import connect, load_scene, move_tip_to_standoff, set_hole_opaque
from sim.wrist_camera2 import attach_wrist_camera2
from vision.corners import gt_image_keypoints
from vision.corner_servo import run_corner_servo
from vision.teach_image import capture_teach_gray
from visp_flow.visp_constants import COARSE_STANDOFF

# Success height: lower tip → peg occludes more hole corners in wrist2 view.
TARGET_Z_MIN_M = ALIGN_Z_STANDOFF_MIN  # 3.0 mm
TARGET_Z_MAX_M = 0.0038  # 3.8 mm
TARGET_Z_BAND_M = 0.0005  # ±0.5 mm convergence


def sample_hole_xy(rng: random.Random) -> tuple[float, float]:
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def sample_target_standoff_z(rng: random.Random) -> float:
    return rng.uniform(TARGET_Z_MIN_M, TARGET_Z_MAX_M)


@contextmanager
def _record_align_policy(target_z: float, *, z_band_m: float = TARGET_Z_BAND_M):
    with cartesian_align_target(target_z, z_band_m=z_band_m):
        yield


@dataclass
class GrayVideoWriter:
    path: str
    fps: float = 20.0
    _writer: cv2.VideoWriter | None = field(default=None, init=False)
    frame_count: int = field(default=0, init=False)

    def write_gray(self, gray: np.ndarray) -> None:
        if self._writer is None:
            h, w = gray.shape[:2]
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self._writer.write(bgr)
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _clear_dataset_dir(out_dir: str) -> None:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def record_align_video(
    *,
    seed: int,
    out_dir: str,
    align_method: str = "kabsch",
    gui: bool = True,
    fps: float = 20.0,
    stem: str | None = None,
) -> dict:
    rng = random.Random(seed)
    hole_xy = sample_hole_xy(rng)
    perturb = sample_collect_perturbation6(rng)
    target_z = sample_target_standoff_z(rng)
    stem = stem or f"align_s{seed:04d}"
    video_path = os.path.join(out_dir, "videos", f"{stem}.mp4")
    meta_path = os.path.join(out_dir, "videos", f"{stem}.json")

    connect(gui=gui)
    wrist_cam = None
    recorder = GrayVideoWriter(video_path, fps=fps)
    recording = False
    aligned = False
    metrics: dict | None = None
    tip_standoff_mm: float | None = None

    def _capture() -> None:
        if not recording:
            return
        gray = capture_teach_gray(wrist_cam, gui=gui, warmup=1)
        recorder.write_gray(gray)

    try:
        robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(gui=gui, hole_xy=hole_xy)
        set_hole_opaque(hole_id)
        hole_orn = p.getBasePositionAndOrientation(hole_id)[1]
        wrist_cam = attach_wrist_camera2(robot_id, eef, hole_xy)

        move_tip_to_standoff(
            robot_id, eef, arm, peg, hole_xy, COARSE_STANDOFF,
            gui=gui, settle_steps=SETTLE_IK_STEPS_GUI,
        )

        apply_tip_perturbation(
            robot_id, eef, arm, peg, hole_xy, hole_orn, perturb,
            gui=gui, settle_steps=SETTLE_IK_STEPS_GUI,
        )
        print(format_perturbation_log(perturb))
        print(f"target standoff z={target_z * 1e3:.2f} mm (random, may occlude hole corners)")

        # 录制从扰动后 standoff 位姿开始（不录 teach / 扰动前）。
        recording = True
        for _ in range(SETTLE_IK_STEPS_GUI):
            p.stepSimulation()
        _capture()
        tip0 = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
        print(f"record start: tip standoff={tip0 * 1e3:.1f} mm (post-perturb @ COARSE_STANDOFF)")

        def _provider():
            return gt_image_keypoints(wrist_cam, robot_id, peg, hole_id, infer_corner0=False)

        with _record_align_policy(target_z):
            aligned, m = run_corner_servo(
                robot_id, arm, peg, wrist_cam, hole_xy, hole_orn, _provider,
                gui=gui, on_step=_capture, align_method=align_method, refresh_every=1,
            )

        if m is not None:
            tip_standoff_mm = round((peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z) * 1e3, 3)
            metrics = {
                "dx_mm": round(m["dx"] * 1e3, 3),
                "dy_mm": round(m["dy"] * 1e3, 3),
                "standoff_mm": round(m["standoff"] * 1e3, 3),
                "tip_standoff_mm": tip_standoff_mm,
                "roll_deg": round(float(np.degrees(m["roll"])), 3),
                "pitch_deg": round(float(np.degrees(m["pitch"])), 3),
                "yaw_deg": round(float(np.degrees(m["yaw"])), 3),
            }

    finally:
        recorder.close()
        if wrist_cam is not None:
            wrist_cam.detach()
        if p.getConnectionInfo()["isConnected"]:
            p.disconnect()

    meta = {
        "seed": seed,
        "hole_xy": list(hole_xy),
        "perturb": perturb,
        "target_standoff_z_mm": round(target_z * 1e3, 3),
        "align_method": align_method,
        "aligned": aligned,
        "metrics": metrics,
        "video": os.path.relpath(video_path, out_dir),
        "frames": recorder.frame_count,
        "fps": fps,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"video → {video_path} ({recorder.frame_count} frames, {recorder.frame_count / fps:.1f}s)")
    print(f"meta  → {meta_path}")
    print(f"aligned={aligned} metrics={metrics}")
    return meta


def _sample_video_seeds(master_seed: int, count: int) -> list[int]:
    rng = random.Random(master_seed)
    return rng.sample(range(1_000_000, 9_000_000), count)


def record_align_videos(
    *,
    count: int,
    master_seed: int,
    out_dir: str,
    align_method: str = "kabsch",
    gui: bool = True,
    fps: float = 20.0,
    skip_existing: bool = True,
) -> dict:
    if count < 1:
        raise ValueError("count must be >= 1")
    seeds = [master_seed] if count == 1 else _sample_video_seeds(master_seed, count)
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)

    rows: list[dict] = []
    n_ok = 0
    n_skip = 0
    for i, seed in enumerate(seeds, start=1):
        stem = f"align_s{seed:04d}"
        video_path = os.path.join(out_dir, "videos", f"{stem}.mp4")
        if skip_existing and os.path.isfile(video_path):
            print(f"[{i}/{count}] skip existing {stem}")
            n_skip += 1
            continue
        print(f"[{i}/{count}] seed={seed} ({stem})")
        meta = record_align_video(
            seed=seed,
            out_dir=out_dir,
            align_method=align_method,
            gui=gui,
            fps=fps,
        )
        rows.append(meta)
        if meta["aligned"]:
            n_ok += 1

    summary = {
        "master_seed": master_seed,
        "requested": count,
        "recorded": len(rows),
        "skipped": n_skip,
        "aligned": n_ok,
        "failed": len(rows) - n_ok,
        "seeds": seeds,
        "videos": rows,
    }
    meta_path = os.path.join(out_dir, "record_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(
        f"batch done: recorded={len(rows)} skipped={n_skip} "
        f"aligned={n_ok} failed={len(rows) - n_ok} → {meta_path}"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Record sim wrist2 align trajectory video(s)")
    ap.add_argument(
        "--count",
        type=int,
        default=1,
        help="number of videos; count>1 draws random seeds from --seed master RNG",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="video seed when count=1; master RNG seed when count>1",
    )
    ap.add_argument("--out", default=DEFAULT_RAW_DIR, help="raw videos dir (default: datasets/sim_wrist2_raw)")
    ap.add_argument("--align-method", choices=("kabsch", "ibvs"), default="kabsch")
    ap.add_argument("--headless", action="store_true", help="p.DIRECT (shadow artifacts)")
    ap.add_argument("--overwrite", action="store_true", help="clear raw videos dir first")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-record even if align_sXXXX.mp4 already exists",
    )
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()
    if args.overwrite:
        _clear_dataset_dir(args.out)
        print(f"cleared → {args.out}")

    summary = record_align_videos(
        count=args.count,
        master_seed=args.seed,
        out_dir=args.out,
        align_method=args.align_method,
        gui=not args.headless,
        fps=args.fps,
        skip_existing=not args.force,
    )
    if summary["recorded"] == 0:
        return 0
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
