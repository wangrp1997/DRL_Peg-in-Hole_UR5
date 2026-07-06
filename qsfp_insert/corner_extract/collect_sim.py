#!/usr/bin/env python3
"""Sim wrist2 YOLO pose dataset — same GUI/HARDWARE render path as visp_flow teach."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from typing import Literal

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from corner_extract._paths import DEFAULT_DATASET_DIR, ROOT  # noqa: F401
from corner_extract.visibility import apply_label_visibility
from corner_extract.yolo_dataset import write_data_yaml
from corner_extract.yolo_export import keypoint_sets_to_yolo_lines
from constants import (
    ALIGN_Z_NOMINAL,
    COLLECT_PERTURB_Z_M,
    HOLE_X_RANGE,
    HOLE_Y_RANGE,
    MIN_CORNERS_VISIBLE,
    SETTLE_IK_STEPS,
    SETTLE_IK_STEPS_GUI,
)
from sim.perturbation import (
    Perturbation6,
    apply_tip_perturbation,
    sample_collect_perturbation6,
)
from sim.scene import connect, load_scene, move_tip_to_standoff, set_hole_opaque
from sim.wrist_camera2 import attach_wrist_camera2
from vision.corners import ImageKeypoints, gt_image_keypoints
from vision.overlay import draw_keypoints_on_bgr
from vision.teach_image import capture_teach_gray_and_seg, save_teach_gray_png
from visp_flow.visp_constants import COARSE_STANDOFF

NEG_STANDOFF_MIN = 0.042
NEG_STANDOFF_MAX = 0.058
NEG_XY_M = 0.040
MAX_POSE_TRIES = 60

SampleKind = Literal["pos", "neg"]


def sample_hole_xy(rng: random.Random) -> tuple[float, float]:
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def sample_collect_perturb6(rng: random.Random) -> Perturbation6:
    return sample_collect_perturbation6(rng)


def sample_neg_perturb6(rng: random.Random) -> Perturbation6:
    p = sample_collect_perturb6(rng)
    p["dz"] = rng.uniform(0.0, COLLECT_PERTURB_Z_M)
    sign_x = -1.0 if rng.random() < 0.5 else 1.0
    sign_y = -1.0 if rng.random() < 0.5 else 1.0
    p["dx"] = sign_x * rng.uniform(NEG_XY_M * 0.85, NEG_XY_M)
    p["dy"] = sign_y * rng.uniform(NEG_XY_M * 0.85, NEG_XY_M)
    return p


def _visible_count(kp: ImageKeypoints) -> int:
    return sum(1 for ok in kp.visible if ok)


def _pick_pose(rng: random.Random, kind: SampleKind) -> tuple[float, Perturbation6]:
    if kind == "pos":
        standoff = rng.uniform(ALIGN_Z_NOMINAL, COARSE_STANDOFF)
        return standoff, sample_collect_perturb6(rng)
    standoff = rng.uniform(NEG_STANDOFF_MIN, NEG_STANDOFF_MAX)
    return standoff, sample_neg_perturb6(rng)


def _valid_sample(kps: list[ImageKeypoints], kind: SampleKind) -> bool:
    hole = next(s for s in kps if s.name == "hole")
    peg = next(s for s in kps if s.name == "peg")
    if _visible_count(peg) < MIN_CORNERS_VISIBLE:
        return False
    hole_vis = _visible_count(hole)
    if kind == "neg":
        return hole_vis == 0
    return hole_vis >= 1


def _reset_bullet_scene(gui: bool) -> None:
    """New scene in an existing PyBullet client (avoid reconnecting GUI each frame)."""
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)


def _settle_steps(gui: bool) -> int:
    return SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS


def capture_frame(
    rng: random.Random,
    kind: SampleKind,
    hole_xy: tuple[float, float],
    *,
    gui: bool,
    connected: bool,
) -> tuple[np.ndarray, list[ImageKeypoints], Perturbation6, float] | None:
    settle = _settle_steps(gui)
    for _ in range(MAX_POSE_TRIES):
        standoff, perturb = _pick_pose(rng, kind)
        wrist_cam = None
        own_connection = not connected
        try:
            if own_connection:
                connect(gui=gui)
            else:
                _reset_bullet_scene(gui)
            robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(gui=gui, hole_xy=hole_xy)
            if gui:
                set_hole_opaque(hole_id)
            hole_orn = p.getBasePositionAndOrientation(hole_id)[1]
            wrist_cam = attach_wrist_camera2(robot_id, eef, hole_xy)
            move_tip_to_standoff(
                robot_id, eef, arm, peg, hole_xy, standoff, gui=gui, settle_steps=settle,
            )
            apply_tip_perturbation(
                robot_id, eef, arm, peg, hole_xy, hole_orn, perturb,
                standoff=standoff, gui=gui, settle_steps=settle,
            )
            gray, seg = capture_teach_gray_and_seg(wrist_cam, gui=gui, warmup=2)
            kps = gt_image_keypoints(wrist_cam, robot_id, peg, hole_id)
            kps = apply_label_visibility(
                kps, wrist_cam, seg, hole_id=hole_id, robot_id=robot_id, peg_link=peg,
            )
            if _valid_sample(kps, kind):
                return gray, kps, perturb, standoff
        finally:
            if wrist_cam is not None:
                wrist_cam.detach()
            if own_connection and p.getConnectionInfo()["isConnected"]:
                p.disconnect()
    return None


def save_preview(path: str, gray: np.ndarray, kps: list[ImageKeypoints]) -> None:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    draw_keypoints_on_bgr(bgr, kps)
    cv2.imwrite(path, bgr)


def _clear_dataset_dir(out_dir: str) -> None:
    for sub in ("images", "labels", "previews"):
        path = os.path.join(out_dir, sub)
        if os.path.isdir(path):
            shutil.rmtree(path)
    for name in ("collect_meta.json", "data.yaml"):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def collect(
    count: int,
    seed: int,
    out_dir: str,
    pos_ratio: float,
    *,
    overwrite: bool = False,
    gui: bool = True,
) -> dict:
    if overwrite and os.path.isdir(out_dir):
        _clear_dataset_dir(out_dir)
        print(f"cleared → {out_dir}")

    rng = random.Random(seed)
    img_dir = os.path.join(out_dir, "images", "train")
    lbl_dir = os.path.join(out_dir, "labels", "train")
    prev_dir = os.path.join(out_dir, "previews")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    n_pos = max(0, min(count, int(round(count * pos_ratio))))
    n_neg = count - n_pos
    plan: list[SampleKind] = ["pos"] * n_pos + ["neg"] * n_neg
    rng.shuffle(plan)

    rows = []
    saved = 0
    connect(gui=gui)
    try:
        for i, kind in enumerate(plan):
            hole_xy = sample_hole_xy(rng)
            out = capture_frame(rng, kind, hole_xy, gui=gui, connected=True)
            if out is None:
                print(f"  skip {i}: no valid {kind} pose after {MAX_POSE_TRIES} tries")
                continue
            gray, kps, perturb, standoff = out
            stem = f"{saved:06d}"
            img_path = os.path.join(img_dir, f"{stem}.png")
            lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
            prev_path = os.path.join(prev_dir, f"{stem}_{kind}.jpg")
            save_teach_gray_png(gray, img_path)
            h, w = gray.shape[:2]
            include_hole = kind == "pos"
            lines = keypoint_sets_to_yolo_lines(kps, w, h, include_hole=include_hole)
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            save_preview(prev_path, gray, kps)

            hole = next(s for s in kps if s.name == "hole")
            peg = next(s for s in kps if s.name == "peg")
            rows.append({
                "id": stem,
                "kind": kind,
                "hole_xy": list(hole_xy),
                "standoff_mm": round(standoff * 1e3, 2),
                "hole_visible": _visible_count(hole),
                "peg_visible": _visible_count(peg),
                "perturb_mm": {k: round(v * 1e3, 2) if k in ("dx", "dy", "dz") else round(v, 4)
                               for k, v in perturb.items()},
            })
            saved += 1
            print(
                f"[{saved}/{count}] {stem} {kind} hole_vis={_visible_count(hole)} "
                f"peg_vis={_visible_count(peg)} z={standoff*1e3:.1f}mm"
            )
    finally:
        if p.getConnectionInfo()["isConnected"]:
            p.disconnect()

    write_data_yaml(out_dir)
    meta = {
        "requested": count,
        "saved": saved,
        "pos_ratio": pos_ratio,
        "seed": seed,
        "samples": rows,
    }
    meta_path = os.path.join(out_dir, "collect_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"done: {saved} frames → {out_dir}")
    print(f"previews → {prev_dir}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect sim wrist2 YOLO pose dataset")
    ap.add_argument("--count", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-ratio", type=float, default=0.8)
    ap.add_argument("--out", default=DEFAULT_DATASET_DIR)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="remove existing images/labels/previews before collecting",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="p.DIRECT (fast, TINY renderer — bottom shadow artifacts; not for training PNGs)",
    )
    args = ap.parse_args()
    collect(
        args.count, args.seed, args.out, args.pos_ratio,
        overwrite=args.overwrite, gui=not args.headless,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
