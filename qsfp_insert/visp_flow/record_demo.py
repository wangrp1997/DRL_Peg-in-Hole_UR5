#!/usr/bin/env python3
"""Record wrist_camera2 demo video for ViSP baseline (GT + Kabsch + insert)."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

_QSFP_INSERT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_QSFP_INSERT)
for _p in (_QSFP_INSERT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim.wrist2_render import render_rgbd
from sim.wrist_camera2 import WristCamera2
from vision.overlay import draw_keypoints_on_bgr
from visp_flow.episode import visp_flow_episode

PHASE_LABELS = {
    "standoff": "Standoff 待机位",
    "teach": "Teach 示教对准",
    "teach_capture": "采集目标图像 I*",
    "perturb": "Perturb 6D 扰动",
    "coarse": "Kabsch 粗对准",
    "dvs_skip": "DVS 跳过（GT 已收敛）",
    "dvs": "ViSP DVS 细对准",
    "insert": "Insert 插入",
    "done": "完成",
}


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _banner(bgr: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    bar_h = 52
    cv2.rectangle(out, (0, 0), (w, bar_h), (24, 24, 24), -1)
    cv2.putText(out, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "GT + ViSP | wrist_camera2", (w - 260, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return out


@dataclass
class DemoRecorder:
    path: str
    fps: float = 20.0
    cam: WristCamera2 | None = None
    provider=None
    phase: str = "init"
    subtitle: str = ""
    stride: int = 4
    _writer: cv2.VideoWriter | None = field(default=None, init=False)
    _tick: int = field(default=0, init=False)
    frame_count: int = field(default=0, init=False)

    def set_phase(self, name: str) -> None:
        self.phase = name
        self.subtitle = PHASE_LABELS.get(name, name)
        for _ in range(int(self.fps * 0.8)):
            self.capture(force=True)

    def capture(self, *, force: bool = False) -> None:
        if self.cam is None:
            return
        self._tick += 1
        if not force and self._tick % self.stride != 0:
            return
        rgba, _, _ = render_rgbd(self.cam, gui=False, with_depth_seg=False, use_cache=False, warmup=1)
        bgr = _rgb_to_bgr(rgba)
        if self.provider is not None:
            kps = self.provider()
            if kps is not None:
                draw_keypoints_on_bgr(bgr, kps)
        title = f"① GT + ViSP  |  {self.subtitle}"
        panel = _banner(bgr, title, "Teach I* -> Perturb -> Kabsch -> [DVS] -> Insert")
        if self._writer is None:
            h, w = panel.shape[:2]
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self._writer.write(panel)
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _maybe_reencode(src: str, dst: str) -> bool:
    if not shutil_which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def shutil_which(name: str) -> str | None:
    from shutil import which
    return which(name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Record GT+ViSP demo mp4 (scheme ①)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="Output .mp4 path")
    ap.add_argument("--h264", action="store_true", help="Also write H.264 copy via ffmpeg")
    ap.add_argument("--gui", action="store_true", help="PyBullet GUI (slower; default headless)")
    args = ap.parse_args()

    out_dir = os.path.join(_QSFP_INSERT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"gt_visp_demo_s{args.seed:04d}"
    mp4_path = args.out or os.path.join(out_dir, f"{stem}.mp4")
    h264_path = os.path.splitext(mp4_path)[0] + "_h264.mp4"

    recorder = DemoRecorder(path=mp4_path, fps=args.fps)

    def on_setup(wrist_cam, provider) -> None:
        recorder.cam = wrist_cam
        recorder.provider = provider

    def on_phase(name: str) -> None:
        recorder.set_phase(name)

    def on_frame() -> None:
        recorder.capture()

    aligned, report, hole_xy, _live = visp_flow_episode(
        gui=args.gui,
        opencv_render=False,
        rng=random.Random(args.seed),
        coarse_method="kabsch",
        corners="gt",
        insert=True,
        gui_idle=False,
        on_frame=on_frame,
        on_phase=on_phase,
        on_setup=on_setup,
    )
    recorder.close()

    meta = {
        "seed": args.seed,
        "hole_xy": list(hole_xy),
        "aligned": aligned,
        "report": report,
        "video": mp4_path,
        "frames": recorder.frame_count,
        "duration_s": recorder.frame_count / args.fps if args.fps > 0 else 0,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = os.path.splitext(mp4_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"video -> {mp4_path} ({recorder.frame_count} frames, {meta['duration_s']:.1f}s)")
    print(f"meta  -> {meta_path}")
    print(f"coarse_ok={report.get('coarse_ok')} inserted={report.get('inserted')} dvs_skipped={report.get('dvs_skipped')}")

    if args.h264 and _maybe_reencode(mp4_path, h264_path):
        print(f"h264  -> {h264_path}")

    ok = bool(report.get("coarse_ok") and report.get("inserted"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
