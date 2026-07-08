#!/usr/bin/env python3
"""IL dataset: random hole + 6D perturb → corner/GT expert → dual RGB video."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
import pybullet as p

from il_flow._paths import IL_ROOT, ROOT  # noqa: F401

from constants import (
    HOLE_X_RANGE,
    HOLE_Y_RANGE,
    PLATE_TOP_Z,
    SETTLE_IK_STEPS,
    SETTLE_IK_STEPS_GUI,
)
from geometry import alignment_metrics, is_xy_rpy_aligned, metrics_converged, peg_tip_world
from il_flow.capture import IL_CAMERAS, capture_rgb_pair, rgb_to_bgr
from il_flow.il_constants import (
    IL_CART_MAX_ANG,
    IL_CART_MAX_LIN,
    IL_CART_MAX_QDOT,
    IL_HEADLESS_HOLD_S,
    IL_PHASE_PAUSE_S,
    IL_RECORD_HOLD_STEPS,
    IL_SIM_HZ,
    TARGET_STANDOFF_MAX_MM,
    TARGET_STANDOFF_MIN_MM,
    TARGET_Z_BAND_M,
)
from il_flow.scene import (
    connect_il,
    disconnect_il,
    get_il_cameras,
    load_il_scene,
    reset_il_simulation,
    teardown_il_episode,
)
from il_flow.validate import check_goal_corners, check_physics_final, check_physics_runtime
from sim.cartesian_align import run_cartesian_align
from sim.cartesian_align_policy import cartesian_align_target
from sim.cartesian_control import alignment_twist
from sim.perturbation import apply_tip_perturbation, format_perturbation_log, sample_collect_perturbation6
from sim.scene import move_tip_to_standoff, settle
from vision.align import AlignMethod, metrics_from_keypoints
from vision.corner_servo import run_corner_servo
from vision.corners import gt_image_keypoints
from vision.debug_markers import clear_gt_corner_markers, sync_gt_corner_markers
from visp_flow.visp_constants import COARSE_STANDOFF

DEFAULT_OUT_DIR = os.path.join(IL_ROOT, "dataset", "raw")
DATASET_FORMAT = "il_flow_v1"
DEFAULT_ALIGN_METHOD: AlignMethod = "kabsch"
ExpertKind = Literal["corner", "gt"]


def sample_hole_xy(rng: random.Random) -> tuple[float, float]:
    return (rng.uniform(*HOLE_X_RANGE), rng.uniform(*HOLE_Y_RANGE))


def sample_target_standoff_mm(rng: random.Random) -> float:
    return rng.uniform(TARGET_STANDOFF_MIN_MM, TARGET_STANDOFF_MAX_MM)


@contextmanager
def _il_servo_limits():
    """Slower Cartesian caps during IL expert only (does not affect demo/record_sim)."""
    import constants as c
    import sim.cartesian_control as cc
    import vision.corner_servo as cs

    mods = (c, cc, cs)
    keys = ("CART_MAX_LIN", "CART_MAX_ANG", "CART_MAX_QDOT")
    saved = {mod: {k: getattr(mod, k) for k in keys if hasattr(mod, k)} for mod in mods}
    for mod in mods:
        mod.CART_MAX_LIN = IL_CART_MAX_LIN
        mod.CART_MAX_ANG = IL_CART_MAX_ANG
        mod.CART_MAX_QDOT = IL_CART_MAX_QDOT
    try:
        yield
    finally:
        for mod, vals in saved.items():
            for k, v in vals.items():
                setattr(mod, k, v)


@dataclass
class RgbVideoWriter:
    path: str
    fps: float = 20.0
    _writer: cv2.VideoWriter | None = field(default=None, init=False)
    frame_count: int = 0

    def write_rgb(self, rgb: np.ndarray) -> None:
        if self._writer is None:
            h, w = rgb.shape[:2]
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self._writer.write(rgb_to_bgr(rgb))
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _write_rgb_png(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, rgb_to_bgr(rgb))


def _round_vec(xs, ndigits: int = 6) -> list[float]:
    return [round(float(x), ndigits) for x in xs]


def _peg_in_hole_dict(metrics: dict) -> dict:
    return {
        "pos": _round_vec([metrics["dx"], metrics["dy"], metrics["standoff"]]),
        "rpy": _round_vec([metrics["roll"], metrics["pitch"], metrics["yaw"]]),
    }


def _record_phase_hold(
    *,
    record_frame,
    fps: float,
    record_stride: int,
    gui: bool,
    label: str,
    pause_s: float,
    terminal: bool,
    on_tick=None,
) -> None:
    """Settle then record pause_s seconds of static frames (action=0) at video fps."""
    settle(IL_RECORD_HOLD_STEPS, gui)
    n_frames = max(1, round(pause_s * fps))
    print(f"{label}: {n_frames} hold frames ({pause_s:.1f}s @ {fps:.0f}fps)")
    period = 1.0 / fps
    for i in range(n_frames):
        for _ in range(record_stride):
            p.stepSimulation()
        if on_tick is not None:
            on_tick()
        record_frame([0.0] * 6, terminal=terminal and i == n_frames - 1)
        if gui:
            time.sleep(period)


def _snapshot_proprio(robot_id: int, arm: list[int], peg: int) -> dict:
    qpos = [p.getJointState(robot_id, j)[0] for j in arm]
    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    return {
        "qpos": _round_vec(qpos),
        "peg_tip_pos": _round_vec(tip),
        "peg_tip_orn": _round_vec(peg_orn),
    }


def record_episode(
    *,
    episode_id: int,
    seed: int,
    out_dir: str,
    standoff_mm: float | None,
    fps: float,
    gui: bool,
    expert: ExpertKind,
    align_method: AlignMethod,
    connected: bool = False,
) -> dict | None:
    rng = random.Random(seed)
    hole_xy = sample_hole_xy(rng)
    perturb = sample_collect_perturbation6(rng)
    target_standoff_mm = standoff_mm if standoff_mm is not None else sample_target_standoff_mm(rng)
    target_z = target_standoff_mm * 1e-3

    ep_name = f"episode_{episode_id:06d}"
    ep_dir = os.path.join(out_dir, "episodes", ep_name)
    os.makedirs(ep_dir, exist_ok=True)

    writers = {
        cam: RgbVideoWriter(os.path.join(ep_dir, cam, "rgb.mp4"), fps=fps)
        for cam in IL_CAMERAS
    }
    timesteps: list[dict] = []
    frame_idx = 0
    recording = False
    record_stride = max(1, round(IL_SIM_HZ / fps))
    hole_pos: tuple[float, float, float] | None = None
    hole_orn: tuple[float, float, float, float] | None = None
    skip_reason: str | None = None

    def _fail(reason: str) -> None:
        nonlocal skip_reason
        if skip_reason is None:
            skip_reason = reason

    def _runtime_ok() -> bool:
        if hole_pos is None or hole_orn is None:
            return True
        ok, msg = check_physics_runtime(robot_id, arm, hole_id, peg, hole_pos, hole_orn)
        if not ok:
            _fail(msg)
        return ok

    own_connection = not connected
    aligned = False
    gt_ok = False
    metrics: dict | None = None
    settle_steps = SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS

    try:
        if own_connection:
            connect_il(gui=gui)
        else:
            reset_il_simulation(gui=gui)
        robot_id, arm, eef, peg, hole_id, hole_xy = load_il_scene(
            gui=gui, hole_xy=hole_xy, opencv_render=False,
        )
        hole_pos, hole_orn = p.getBasePositionAndOrientation(hole_id)

        move_tip_to_standoff(
            robot_id, eef, arm, peg, hole_xy, COARSE_STANDOFF, gui=gui, settle_steps=settle_steps,
        )
        apply_tip_perturbation(
            robot_id, eef, arm, peg, hole_xy, hole_orn, perturb, gui=gui, settle_steps=settle_steps,
        )
        settle(settle_steps, gui)
        print(format_perturbation_log(perturb))
        print(f"target standoff z={target_standoff_mm:.2f} mm")

        wrist_cam, fixed_cam = get_il_cameras()
        assert wrist_cam is not None and fixed_cam is not None

        def _gt_metrics() -> dict:
            tip = peg_tip_world(robot_id, peg)
            peg_orn = p.getLinkState(robot_id, peg)[1]
            return alignment_metrics(tip, peg_orn, hole_xy, hole_orn)

        def _record_frame(action: list[float], *, terminal: bool) -> None:
            nonlocal frame_idx
            if not recording or skip_reason is not None:
                return
            if not _runtime_ok():
                return
            wrist_cam, fixed_cam = get_il_cameras()
            if wrist_cam is None or fixed_cam is None:
                return
            rgbs = capture_rgb_pair(wrist_cam, fixed_cam, gui=gui)
            for cam, rgb in rgbs.items():
                writers[cam].write_rgb(rgb)
            proprio = _snapshot_proprio(robot_id, arm, peg)
            timesteps.append(
                {
                    "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 4),
                    "action": action,
                    "terminal": terminal,
                    "qpos": proprio["qpos"],
                    "peg_tip_pos": proprio["peg_tip_pos"],
                    "peg_tip_orn": proprio["peg_tip_orn"],
                    "peg_in_hole": _peg_in_hole_dict(_gt_metrics()),
                }
            )
            frame_idx += 1

        def _provider():
            return gt_image_keypoints(wrist_cam, robot_id, peg, hole_id, infer_corner0=False)

        def _corner_metrics() -> dict | None:
            tip = peg_tip_world(robot_id, peg)
            standoff_hint = tip[2] - PLATE_TOP_Z
            kps = _provider()
            if kps is None:
                return None
            return metrics_from_keypoints(
                kps, wrist_cam, hole_xy, hole_orn,
                standoff_hint=standoff_hint, method=align_method,
            )

        def _metrics_for_record() -> dict:
            if expert == "corner":
                m_step = _corner_metrics()
                return m_step if m_step is not None else _gt_metrics()
            return _gt_metrics()

        rec_cb_idx = 0
        servo_recording_done = False

        def _on_servo_step() -> None:
            nonlocal rec_cb_idx, servo_recording_done
            if servo_recording_done or skip_reason is not None:
                return
            if not _runtime_ok():
                servo_recording_done = True
                return
            rec_cb_idx += 1
            if expert == "gt" and (rec_cb_idx - 1) % record_stride != 0:
                return
            if expert == "corner" and gui:
                kps = _provider()
                sync_gt_corner_markers(robot_id, peg, hole_id, kps, wrist_cam)
            m_step = _metrics_for_record()
            if metrics_converged(m_step):
                servo_recording_done = True
                return
            twist = alignment_twist(
                m_step["dx"], m_step["dy"], m_step["standoff"],
                m_step["roll"], m_step["pitch"], m_step["yaw"],
            )
            _record_frame([float(x) for x in twist], terminal=False)

        def _gui_markers() -> None:
            if expert == "corner" and gui:
                kps = _provider()
                sync_gt_corner_markers(robot_id, peg, hole_id, kps, wrist_cam)

        hold_s = IL_PHASE_PAUSE_S if gui else IL_HEADLESS_HOLD_S
        recording = True
        _record_phase_hold(
            record_frame=_record_frame,
            fps=fps,
            record_stride=record_stride,
            gui=gui,
            label="post-perturb",
            pause_s=hold_s,
            terminal=False,
            on_tick=_gui_markers,
        )

        with cartesian_align_target(target_z, z_band_m=TARGET_Z_BAND_M), _il_servo_limits():
            if expert == "gt":
                aligned, metrics = run_cartesian_align(
                    robot_id, arm, peg, hole_xy, hole_orn,
                    gui=gui, on_step=_on_servo_step,
                )
            else:
                aligned, metrics = run_corner_servo(
                    robot_id, arm, peg, wrist_cam, hole_xy, hole_orn, _provider,
                    gui=gui, on_step=_on_servo_step, align_method=align_method,
                    refresh_every=record_stride,
                )

        if aligned:
            servo_recording_done = True
            _record_phase_hold(
                record_frame=_record_frame,
                fps=fps,
                record_stride=record_stride,
                gui=gui,
                label="aligned",
                pause_s=hold_s,
                terminal=True,
                on_tick=_gui_markers,
            )

        if aligned and skip_reason is None:
            ok, msg = check_goal_corners(wrist_cam, robot_id, peg, hole_id)
            if not ok:
                _fail(msg)
            else:
                rgbs = capture_rgb_pair(wrist_cam, fixed_cam, gui=gui)
                for cam, rgb in rgbs.items():
                    _write_rgb_png(os.path.join(ep_dir, cam, "goal_rgb.png"), rgb)

        gt_ok = False
        if metrics is not None:
            gt_ok = is_xy_rpy_aligned(
                metrics["dx"], metrics["dy"], metrics["roll"], metrics["pitch"], metrics["yaw"],
            )
        if (
            aligned and gt_ok and skip_reason is None and metrics is not None
            and hole_pos is not None and hole_orn is not None
        ):
            ok, msg = check_physics_final(
                robot_id, arm, peg, hole_id, hole_xy, hole_pos, hole_orn, metrics,
            )
            if not ok:
                _fail(msg)

    finally:
        recording = False
        for w in writers.values():
            w.close()
        if gui:
            clear_gt_corner_markers()
        teardown_il_episode()
        if own_connection and p.getConnectionInfo()["isConnected"]:
            disconnect_il()

    success = aligned and gt_ok and skip_reason is None
    if not success:
        shutil.rmtree(ep_dir, ignore_errors=True)
        reason = skip_reason or (
            f"aligned={aligned} gt_aligned={gt_ok}" if not (aligned and gt_ok) else "unknown"
        )
        print(
            f"SKIP seed={seed} hole_xy={hole_xy} target_z={target_standoff_mm:.2f}mm "
            f"reason={reason}"
        )
        return None

    timesteps_path = os.path.join(ep_dir, "timesteps.jsonl")
    with open(timesteps_path, "w", encoding="utf-8") as f:
        for row in timesteps:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ep_meta = {
        "episode_id": episode_id,
        "episode_name": ep_name,
        "seed": seed,
        "hole_xy": list(hole_xy),
        "hole_pos": _round_vec(hole_pos) if hole_pos is not None else None,
        "hole_orn": _round_vec(hole_orn) if hole_orn is not None else None,
        "perturb": perturb,
        "target_standoff_mm": round(target_standoff_mm, 3),
        "expert": expert,
        "align_method": align_method if expert == "corner" else None,
        "coarse_standoff_mm": round(COARSE_STANDOFF * 1e3, 3),
        "aligned": aligned,
        "gt_aligned": gt_ok,
        "metrics": {
            "dx_mm": round(metrics["dx"] * 1e3, 3),
            "dy_mm": round(metrics["dy"] * 1e3, 3),
            "standoff_mm": round(metrics["standoff"] * 1e3, 3),
            "roll_deg": round(math.degrees(metrics["roll"]), 3),
            "pitch_deg": round(math.degrees(metrics["pitch"]), 3),
            "yaw_deg": round(math.degrees(metrics["yaw"]), 3),
        },
        "frames": frame_idx,
        "fps": fps,
        "record_stride": record_stride,
        "videos": {cam: f"{ep_name}/{cam}/rgb.mp4" for cam in IL_CAMERAS},
        "goals": {cam: f"{ep_name}/{cam}/goal_rgb.png" for cam in IL_CAMERAS},
        "timesteps": f"{ep_name}/timesteps.jsonl",
        "render_gui": gui,
    }
    with open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(ep_meta, f, indent=2, ensure_ascii=False)

    print(
        f"{ep_name} seed={seed} hole_xy={hole_xy} frames={frame_idx} "
        f"target_z={target_standoff_mm:.2f}mm aligned={aligned} gt_aligned={gt_ok}"
    )
    return ep_meta


def _sample_seeds(master_seed: int, count: int) -> list[int]:
    if count == 1:
        return [master_seed]
    rng = random.Random(master_seed)
    return rng.sample(range(1_000_000, 9_000_000), count)


def _next_episode_id(out_dir: str) -> int:
    ep_root = os.path.join(out_dir, "episodes")
    if not os.path.isdir(ep_root):
        return 0
    ids: list[int] = []
    for name in os.listdir(ep_root):
        if not name.startswith("episode_"):
            continue
        try:
            ids.append(int(name.split("_", 1)[1]))
        except ValueError:
            continue
    return max(ids, default=-1) + 1


def collect_dataset(
    *,
    episodes: int,
    master_seed: int,
    out_dir: str,
    standoff_mm: float | None,
    fps: float,
    overwrite: bool,
    gui: bool,
    expert: ExpertKind,
    align_method: AlignMethod,
    max_attempts: int | None = None,
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if overwrite and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(os.path.join(out_dir, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

    record_stride = max(1, round(IL_SIM_HZ / fps))
    expert_label = "cartesian_gt" if expert == "gt" else f"corner_servo_{align_method}"
    info = {
        "format": DATASET_FORMAT,
        "description": "QSFP IL: random hole + 6D perturb, dual RGB expert demos",
        "cameras": list(IL_CAMERAS),
        "action_dim": 6,
        "action_space": "cartesian_twist",
        "fps": fps,
        "record_stride": record_stride,
        "pose_frame": "world",
        "quat_order": "xyzw",
        "peg_in_hole_frame": "hole (origin at hole_pos, axes from hole_orn)",
        "timestep_state": ["qpos", "peg_tip_pos", "peg_tip_orn", "peg_in_hole"],
        "episode_state": ["hole_pos", "hole_orn"],
        "qpos_dim": 6,
        "expert": expert_label,
        "coarse_standoff_mm": round(COARSE_STANDOFF * 1e3, 3),
        "target_standoff_mm_range": [TARGET_STANDOFF_MIN_MM, TARGET_STANDOFF_MAX_MM],
        "target_standoff_mm_fixed": standoff_mm,
        "render_gui": gui,
    }
    with open(os.path.join(out_dir, "meta", "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    seed_rng = random.Random(master_seed)
    attempt_limit = max_attempts if max_attempts is not None else max(episodes * 5, episodes)
    seeds = _sample_seeds(master_seed, episodes) if episodes > 1 else [master_seed]
    seed_iter = iter(seeds)
    n_ok = 0
    n_skip = 0
    episode_id = 0 if overwrite else _next_episode_id(out_dir)
    used_seeds: list[int] = []

    connect_il(gui=gui)
    try:
        for attempt in range(attempt_limit):
            if n_ok >= episodes:
                break
            try:
                seed = next(seed_iter)
            except StopIteration:
                seed = seed_rng.randint(1_000_000, 9_999_999)
            used_seeds.append(seed)

            meta = record_episode(
                episode_id=episode_id,
                seed=seed,
                out_dir=out_dir,
                standoff_mm=standoff_mm,
                fps=fps,
                gui=gui,
                expert=expert,
                align_method=align_method,
                connected=True,
            )
            if meta is None:
                n_skip += 1
                continue

            n_ok += 1
            episode_id += 1
            with open(os.path.join(out_dir, "meta", "episodes.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    finally:
        if p.getConnectionInfo()["isConnected"]:
            disconnect_il()

    summary = {
        "out_dir": out_dir,
        "episodes_requested": episodes,
        "episodes_saved": n_ok,
        "skipped": n_skip,
        "master_seed": master_seed,
        "seeds_used": used_seeds,
        "expert": expert_label,
    }
    with open(os.path.join(out_dir, "meta", "collect_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"dataset → {out_dir}  saved {n_ok}/{episodes}  skipped {n_skip}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Dual-RGB IL dataset (corner/GT expert)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--standoff-mm",
        type=float,
        default=None,
        help="fixed target standoff (mm); default random 3.0–3.8",
    )
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--expert", choices=("corner", "gt"), default="corner")
    ap.add_argument("--align-method", choices=("kabsch", "ibvs"), default=DEFAULT_ALIGN_METHOD)
    args = ap.parse_args()

    if args.headless:
        print(
            "WARNING: --headless (EGL) may produce black artifacts on fixture/base; "
            "use default GUI for training data, or xvfb-run -a without --headless on servers."
        )

    summary = collect_dataset(
        episodes=args.episodes,
        master_seed=args.seed,
        out_dir=args.out_dir,
        standoff_mm=args.standoff_mm,
        fps=args.fps,
        overwrite=args.overwrite,
        gui=not args.headless,
        expert=args.expert,
        align_method=args.align_method,
    )
    return 0 if summary["episodes_saved"] == summary["episodes_requested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
