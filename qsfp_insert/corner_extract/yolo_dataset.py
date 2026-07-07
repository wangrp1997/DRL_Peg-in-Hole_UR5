"""YOLO dataset layout helpers (Ultralytics convention)."""
from __future__ import annotations

import os
import random
import shutil


def write_data_yaml(out_dir: str) -> None:
    yaml_path = os.path.join(out_dir, "data.yaml")
    root = os.path.abspath(out_dir)
    content = f"""# YOLO pose — hole/peg 4 corners (sim wrist2)
path: {root}
train: images/train
val: images/val

names:
  0: hole
  1: peg

kpt_shape: [4, 3]
flip_idx: [0, 1, 2, 3]
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


def split_paths(out_dir: str, split: str) -> tuple[str, str, str]:
    """Return (images_dir, labels_dir, previews_dir) for train or val."""
    return (
        os.path.join(out_dir, "images", split),
        os.path.join(out_dir, "labels", split),
        os.path.join(out_dir, "previews", split),
    )


def ensure_split_dirs(out_dir: str) -> None:
    for split in ("train", "val"):
        for sub in split_paths(out_dir, split):
            os.makedirs(sub, exist_ok=True)


def clear_yolo_dataset(out_dir: str) -> None:
    for sub in ("images", "labels", "previews"):
        path = os.path.join(out_dir, sub)
        if os.path.isdir(path):
            shutil.rmtree(path)
    for name in ("data.yaml", "extract_meta.json"):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def assign_video_splits(
    video_stems: list[str],
    *,
    val_ratio: float = 0.2,
    split_seed: int = 42,
) -> dict[str, str]:
    """Assign whole videos to train or val (avoid frame leakage)."""
    if not video_stems:
        return {}
    rng = random.Random(split_seed)
    stems = sorted(video_stems)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_ratio)))
    if n_val >= len(stems):
        n_val = max(1, len(stems) - 1)
    val_set = set(stems[:n_val])
    return {s: ("val" if s in val_set else "train") for s in stems}
