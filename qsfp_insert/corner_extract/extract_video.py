#!/usr/bin/env python3
"""Extract YOLO pose dataset from record_sim videos (mp4 frames + sim GT replay)."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np
import pybullet as p

from corner_extract._paths import DEFAULT_DATASET_DIR, DEFAULT_RAW_DIR, ROOT  # noqa: F401
from corner_extract.collect_sim import save_preview
from corner_extract.record_sim import _record_align_policy
from corner_extract.visibility import apply_label_visibility
from corner_extract.yolo_dataset import (
    assign_video_splits,
    clear_yolo_dataset,
    ensure_split_dirs,
    split_paths,
    write_data_yaml,
)
from corner_extract.yolo_export import keypoint_sets_to_yolo_lines
from constants import SETTLE_IK_STEPS_GUI
from sim.perturbation import Perturbation6, apply_tip_perturbation, format_perturbation_log
from sim.scene import connect, load_scene, move_tip_to_standoff, set_hole_opaque
from sim.wrist_camera2 import attach_wrist_camera2
from vision.corners import ImageKeypoints, gt_image_keypoints
from vision.corner_servo import run_corner_servo
from vision.teach_image import capture_teach_gray_and_seg, save_teach_gray_png
from visp_flow.visp_constants import COARSE_STANDOFF


def _visible_count(kp: ImageKeypoints) -> int:
    return sum(1 for ok in kp.visible if ok)


def extract_mp4_frames(video_path: str, stride: int) -> tuple[list[np.ndarray], int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    total = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if total % stride == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        total += 1
    cap.release()
    return frames, total


def replay_labels(
    meta: dict,
    stride: int,
    *,
    gui: bool,
) -> tuple[list[list[ImageKeypoints]], int]:
    hole_xy = tuple(meta["hole_xy"])
    perturb: Perturbation6 = meta["perturb"]
    target_z = float(meta["target_standoff_z_mm"]) * 1e-3
    align_method = meta.get("align_method", "kabsch")

    connect(gui=gui)
    wrist_cam = None
    labels: list[list[ImageKeypoints]] = []
    frame_idx = 0

    def _maybe_label() -> None:
        nonlocal frame_idx
        if frame_idx % stride == 0:
            _, seg = capture_teach_gray_and_seg(wrist_cam, gui=gui, warmup=1)
            kps = gt_image_keypoints(wrist_cam, robot_id, peg, hole_id, infer_corner0=False)
            kps = apply_label_visibility(
                kps, wrist_cam, seg, hole_id=hole_id, robot_id=robot_id, peg_link=peg,
            )
            labels.append(kps)
        frame_idx += 1

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

        for _ in range(SETTLE_IK_STEPS_GUI):
            p.stepSimulation()
        _maybe_label()

        def _provider():
            return gt_image_keypoints(wrist_cam, robot_id, peg, hole_id, infer_corner0=False)

        with _record_align_policy(target_z):
            run_corner_servo(
                robot_id, arm, peg, wrist_cam, hole_xy, hole_orn, _provider,
                gui=gui, on_step=_maybe_label, align_method=align_method, refresh_every=1,
            )
    finally:
        if wrist_cam is not None:
            wrist_cam.detach()
        if p.getConnectionInfo()["isConnected"]:
            p.disconnect()

    return labels, frame_idx


def extract_one(
    video_path: str,
    meta_path: str,
    out_dir: str,
    split: str,
    stride: int,
    *,
    gui: bool,
    next_ids: dict[str, int],
) -> list[dict]:
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    img_dir, lbl_dir, prev_dir = split_paths(out_dir, split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[{split}] extract ← {video_stem} (stride={stride})")
    frames, video_total = extract_mp4_frames(video_path, stride)
    print(f"[{split}] replay  ← {meta_path}")
    labels, replay_total = replay_labels(meta, stride, gui=gui)

    if len(frames) != len(labels):
        n = min(len(frames), len(labels))
        print(
            f"warn: frame mismatch video={video_total}→{len(frames)} "
            f"replay={replay_total}→{len(labels)}; using first {n}"
        )
        frames = frames[:n]
        labels = labels[:n]

    rows: list[dict] = []
    for gray, kps in zip(frames, labels):
        stem = f"{next_ids[split]:06d}"
        next_ids[split] += 1
        img_path = os.path.join(img_dir, f"{stem}.png")
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        prev_path = os.path.join(prev_dir, f"{stem}.jpg")

        save_teach_gray_png(gray, img_path)
        h, w = gray.shape[:2]
        lines = keypoint_sets_to_yolo_lines(kps, w, h, include_hole=True)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        save_preview(prev_path, gray, kps)

        hole = next(s for s in kps if s.name == "hole")
        peg = next(s for s in kps if s.name == "peg")
        rows.append({
            "id": stem,
            "split": split,
            "source_video": video_stem,
            "frame_stride": stride,
            "hole_visible": _visible_count(hole),
            "peg_visible": _visible_count(peg),
        })

    print(f"  → {len(rows)} samples → {split}")
    return rows


def _list_video_pairs(raw_dir: str, stem: str | None) -> list[tuple[str, str, str]]:
    videos_dir = os.path.join(raw_dir, "videos")
    if stem:
        candidates = [(stem, os.path.join(videos_dir, f"{stem}.mp4"))]
    else:
        candidates = []
        for mp4 in sorted(glob.glob(os.path.join(videos_dir, "*.mp4"))):
            base = os.path.splitext(os.path.basename(mp4))[0]
            candidates.append((base, mp4))

    pairs: list[tuple[str, str, str]] = []
    for video_stem, mp4 in candidates:
        meta = os.path.join(videos_dir, f"{video_stem}.json")
        if os.path.isfile(mp4) and os.path.isfile(meta):
            pairs.append((video_stem, mp4, meta))
    return pairs


def extract_dataset(
    raw_dir: str,
    dataset_dir: str,
    stride: int,
    *,
    stem: str | None = None,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    gui: bool = True,
    clear: bool = False,
) -> dict:
    pairs = _list_video_pairs(raw_dir, stem)
    if not pairs:
        raise FileNotFoundError(f"no video+json pairs under {os.path.join(raw_dir, 'videos')}")

    if clear:
        clear_yolo_dataset(dataset_dir)
        print(f"cleared YOLO dataset → {dataset_dir}")

    ensure_split_dirs(dataset_dir)
    split_map = assign_video_splits(
        [p[0] for p in pairs], val_ratio=val_ratio, split_seed=split_seed,
    )

    next_ids = {"train": 0, "val": 0}
    all_rows: list[dict] = []
    for video_stem, mp4, meta_path in pairs:
        split = split_map[video_stem]
        rows = extract_one(
            mp4, meta_path, dataset_dir, split, stride, gui=gui, next_ids=next_ids,
        )
        all_rows.extend(rows)

    write_data_yaml(dataset_dir)

    train_videos = sorted(s for s, sp in split_map.items() if sp == "train")
    val_videos = sorted(s for s, sp in split_map.items() if sp == "val")
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]

    out_meta = {
        "stride": stride,
        "val_ratio": val_ratio,
        "split_seed": split_seed,
        "raw_dir": raw_dir,
        "videos_total": len(pairs),
        "videos_train": train_videos,
        "videos_val": val_videos,
        "samples_total": len(all_rows),
        "samples_train": len(train_rows),
        "samples_val": len(val_rows),
        "hole_visible_train": dict(sorted(Counter(r["hole_visible"] for r in train_rows).items())),
        "hole_visible_val": dict(sorted(Counter(r["hole_visible"] for r in val_rows).items())),
        "entries": all_rows,
    }
    meta_path = os.path.join(dataset_dir, "extract_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2, ensure_ascii=False)

    print(f"done: train={len(train_rows)} val={len(val_rows)} → {dataset_dir}")
    print(f"previews → {os.path.join(dataset_dir, 'previews')}")
    return out_meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract YOLO dataset from record_sim videos")
    ap.add_argument("--raw", default=DEFAULT_RAW_DIR, help="raw videos root (sim_wrist2_raw)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET_DIR, help="YOLO dataset root (sim_wrist2)")
    ap.add_argument("--stride", type=int, default=10, help="save every Nth video frame")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="fraction of videos for val")
    ap.add_argument("--split-seed", type=int, default=42, help="RNG for train/val video split")
    ap.add_argument("--stem", default=None, help="video stem, e.g. align_s0042 (default: all)")
    ap.add_argument(
        "--clear",
        action="store_true",
        help="remove existing images/labels/previews/data.yaml before extract",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="p.DIRECT for label replay (images still from mp4)",
    )
    args = ap.parse_args()
    extract_dataset(
        args.raw,
        args.dataset,
        args.stride,
        stem=args.stem,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        gui=not args.headless,
        clear=args.clear,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
