#!/usr/bin/env python3
"""Convert il_flow_v1 dataset (dual RGB + goal_png) to LeRobot format."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "dataset" / "raw"
DEFAULT_REPO_ID = "local/qsfp_il"
DEFAULT_TASK = "qsfp peg-in-hole align"
ROBOT_TYPE = "ur5_qsfp"

CAM_WRIST2 = "wrist_camera2"
CAM_FIXED2 = "fixed_camera2"

KEY_WRIST = f"observation.images.{CAM_WRIST2}"
KEY_FIXED = f"observation.images.{CAM_FIXED2}"
KEY_GOAL_WRIST = "observation.images.goal_wrist"
KEY_GOAL_FIXED = "observation.images.goal_fixed"
KEY_STATE = "observation.state"
KEY_ACTION = "action"

CAM_DIRS = {
    KEY_WRIST: CAM_WRIST2,
    KEY_FIXED: CAM_FIXED2,
}
GOAL_FILES = {
    KEY_GOAL_WRIST: (CAM_WRIST2, "goal_rgb.png"),
    KEY_GOAL_FIXED: (CAM_FIXED2, "goal_rgb.png"),
}


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {img.shape}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_goal_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"goal image missing: {path}")
    return _bgr_to_rgb(img)


def _read_video_rgb(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(_bgr_to_rgb(bgr))
    cap.release()
    if not frames:
        raise ValueError(f"video has no frames: {path}")
    return frames


def _load_timesteps(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty timesteps: {path}")
    return rows


def _episode_dirs(raw_dir: Path) -> list[Path]:
    ep_root = raw_dir / "episodes"
    if not ep_root.is_dir():
        raise FileNotFoundError(f"missing episodes dir: {ep_root}")
    dirs = sorted(p for p in ep_root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not dirs:
        raise FileNotFoundError(f"no episodes under {ep_root}")
    return dirs


def _probe_shape(raw_dir: Path) -> tuple[int, int]:
    ep0 = _episode_dirs(raw_dir)[0]
    frames = _read_video_rgb(ep0 / CAM_WRIST2 / "rgb.mp4")
    h, w = frames[0].shape[:2]
    return h, w


def _build_features(height: int, width: int) -> dict:
    img_shape = (height, width, 3)
    img_names = ["height", "width", "channels"]
    joint_names = ["j0", "j1", "j2", "j3", "j4", "j5"]
    twist_names = ["vx", "vy", "vz", "wx", "wy", "wz"]
    features = {
        KEY_STATE: {
            "dtype": "float32",
            "shape": (6,),
            "names": {"axes": joint_names},
        },
        KEY_ACTION: {
            "dtype": "float32",
            "shape": (6,),
            "names": {"axes": twist_names},
        },
    }
    for key in (KEY_WRIST, KEY_FIXED, KEY_GOAL_WRIST, KEY_GOAL_FIXED):
        features[key] = {
            "dtype": "video",
            "shape": img_shape,
            "names": img_names,
        }
    return features


def _validate_episode(ep_dir: Path, meta: dict) -> None:
    ts_path = ep_dir / "timesteps.jsonl"
    steps = _load_timesteps(ts_path)
    n_ts = len(steps)
    n_meta = int(meta.get("frames", n_ts))
    if n_ts != n_meta:
        raise ValueError(f"{ep_dir.name}: timesteps={n_ts} meta.frames={n_meta}")
    for cam in (CAM_WRIST2, CAM_FIXED2):
        vid = ep_dir / cam / "rgb.mp4"
        n_vid = len(_read_video_rgb(vid))
        if n_vid != n_ts:
            raise ValueError(f"{ep_dir.name}/{cam}: video={n_vid} timesteps={n_ts}")
        goal = ep_dir / cam / "goal_rgb.png"
        if not goal.is_file():
            raise FileNotFoundError(goal)


def convert(
    *,
    raw_dir: Path,
    repo_id: str,
    out_root: Path | None,
    fps: float,
    task: str,
    max_episodes: int | None,
    overwrite: bool,
    image_writer_threads: int,
) -> Path:
    info_path = raw_dir / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("format") != "il_flow_v1":
            logging.warning("unexpected format %s", info.get("format"))
        if fps <= 0:
            fps = float(info.get("fps", 20.0))

    height, width = _probe_shape(raw_dir)
    features = _build_features(height, width)

    root = out_root if out_root is not None else None
    if root is not None:
        root = Path(root)
        if root.exists():
            if not overwrite:
                raise FileExistsError(f"output exists: {root} (pass --overwrite)")
            shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(fps)),
        features=features,
        root=root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_threads=image_writer_threads,
    )

    episode_dirs = _episode_dirs(raw_dir)
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    for ep_dir in episode_dirs:
        meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
        _validate_episode(ep_dir, meta)

        steps = _load_timesteps(ep_dir / "timesteps.jsonl")
        wrist_frames = _read_video_rgb(ep_dir / CAM_WRIST2 / "rgb.mp4")
        fixed_frames = _read_video_rgb(ep_dir / CAM_FIXED2 / "rgb.mp4")
        goal_wrist = _load_goal_rgb(ep_dir / CAM_WRIST2 / "goal_rgb.png")
        goal_fixed = _load_goal_rgb(ep_dir / CAM_FIXED2 / "goal_rgb.png")

        for i, step in enumerate(steps):
            frame = {
                KEY_WRIST: wrist_frames[i],
                KEY_FIXED: fixed_frames[i],
                KEY_GOAL_WRIST: goal_wrist,
                KEY_GOAL_FIXED: goal_fixed,
                KEY_STATE: np.asarray(step["qpos"], dtype=np.float32),
                KEY_ACTION: np.asarray(step["action"], dtype=np.float32),
                "task": task,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        logging.info(
            "saved %s  frames=%d  aligned=%s  gt_aligned=%s",
            ep_dir.name,
            len(steps),
            meta.get("aligned"),
            meta.get("gt_aligned"),
        )

    dataset.finalize()
    out = Path(dataset.root)
    logging.info("LeRobot dataset → %s  repo_id=%s  episodes=%d", out, repo_id, len(episode_dirs))
    return out


def validate_output(root: Path, repo_id: str, expected_episodes: int) -> None:
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    if meta.total_episodes != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episodes, got {meta.total_episodes}")
    for ep_idx in range(meta.total_episodes):
        for vid_key in meta.video_keys:
            vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            if not vid_path.is_file():
                raise FileNotFoundError(vid_path)
    logging.info(
        "validate ok: episodes=%d frames=%d cameras=%s",
        meta.total_episodes,
        meta.total_frames,
        meta.camera_keys,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert il_flow_v1 → LeRobot dataset")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="output directory (default: $HF_LEROBOT_HOME/{repo_id})",
    )
    ap.add_argument("--fps", type=float, default=0.0, help="override fps (default: meta/info.json)")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fps = args.fps if args.fps > 0 else 20.0

    out = convert(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        out_root=args.out_root,
        fps=fps,
        task=args.task,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        image_writer_threads=args.image_writer_threads,
    )

    if not args.skip_validate:
        expected = len(_episode_dirs(args.raw_dir)) if args.max_episodes is None else args.max_episodes
        validate_output(out, args.repo_id, expected)

    print(f"done → {out}")
    print(f"train: lerobot-train --dataset.repo_id={args.repo_id} --dataset.root={out} --policy=diffusion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
