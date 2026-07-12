#!/usr/bin/env python3
"""PyBullet rollout eval for LeRobot Diffusion policy (goal-conditioned dual RGB)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

# Allow: python qsfp_insert/il_flow/run_eval.py (from repo root)
_QSFP_INSERT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_QSFP_INSERT)
for _p in (_QSFP_INSERT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pybullet as p
import torch

from il_flow._paths import IL_ROOT, REPO_ROOT, ROOT  # noqa: F401

from constants import SETTLE_IK_STEPS, SETTLE_IK_STEPS_GUI
from geometry import alignment_metrics, is_xy_rpy_aligned, metrics_converged, peg_tip_world
from il_flow.capture import CAM_FIXED2, CAM_WRIST2, capture_rgb_pair
from il_flow.collect_il import sample_hole_xy, sample_target_standoff_mm
from il_flow.il_constants import IL_SIM_HZ, TARGET_Z_BAND_M
from il_flow.scene import (
    connect_il,
    disconnect_il,
    get_il_cameras,
    load_il_scene,
    reset_il_simulation,
    teardown_il_episode,
)
from sim.cartesian_align import run_cartesian_align
from sim.cartesian_align_policy import cartesian_align_target
from sim.cartesian_control import apply_cartesian_velocity, stop_arm
from sim.perturbation import apply_tip_perturbation, format_perturbation_log, sample_collect_perturbation6
from sim.scene import move_tip_to_standoff, settle
from visp_flow.visp_constants import COARSE_STANDOFF

DEFAULT_MODEL = "rpwang/qsfp_il_dp"
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs")
ACTION_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz")


def _round_vec(xs, ndigits: int = 6) -> list[float]:
    return [round(float(x), ndigits) for x in xs]


def _qpos(robot_id: int, arm: list[int]) -> np.ndarray:
    return np.asarray([p.getJointState(robot_id, j)[0] for j in arm], dtype=np.float32)


def _save_arm_qpos(robot_id: int, arm: list[int]) -> list[float]:
    return [p.getJointState(robot_id, j)[0] for j in arm]


def _restore_arm_qpos(robot_id: int, arm: list[int], qpos: list[float], *, gui: bool) -> None:
    for j, q in zip(arm, qpos):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, q, force=500)
    settle(SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS, gui)


def capture_oracle_goals(
    *,
    robot_id: int,
    arm: list[int],
    peg: int,
    hole_xy: tuple[float, float],
    hole_orn: tuple[float, float, float, float],
    target_standoff_mm: float,
    wrist_cam,
    fixed_cam,
    gui: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """GT align → capture goal RGB → restore perturbed pose (matches training goal images)."""
    saved = _save_arm_qpos(robot_id, arm)
    target_z = target_standoff_mm * 1e-3
    with cartesian_align_target(target_z, z_band_m=TARGET_Z_BAND_M):
        aligned, _ = run_cartesian_align(
            robot_id, arm, peg, hole_xy, hole_orn, gui=gui,
        )
    rgbs = capture_rgb_pair(wrist_cam, fixed_cam, gui=gui)
    _restore_arm_qpos(robot_id, arm, saved, gui=gui)
    return rgbs[CAM_WRIST2], rgbs[CAM_FIXED2], aligned


def build_obs_batch(
    *,
    wrist_rgb: np.ndarray,
    fixed_rgb: np.ndarray,
    goal_wrist: np.ndarray,
    goal_fixed: np.ndarray,
    qpos: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    from lerobot.policies.utils import prepare_observation_for_inference

    raw = {
        "observation.state": qpos,
        "observation.images.wrist_camera2": wrist_rgb,
        "observation.images.fixed_camera2": fixed_rgb,
        "observation.images.goal_wrist": goal_wrist,
        "observation.images.goal_fixed": goal_fixed,
    }
    return prepare_observation_for_inference(raw, device)


def action_tensor_to_twist(action: torch.Tensor) -> np.ndarray:
    action = action.squeeze(0).detach().cpu().numpy()
    return np.asarray(action, dtype=np.float64)


def rollout_episode(
    *,
    seed: int,
    policy,
    preprocess,
    postprocess,
    device: torch.device,
    fps: float,
    max_steps: int,
    gui: bool,
    target_standoff_mm: float | None,
) -> dict:
    rng = random.Random(seed)
    hole_xy = sample_hole_xy(rng)
    perturb = sample_collect_perturbation6(rng)
    standoff_mm = target_standoff_mm if target_standoff_mm is not None else sample_target_standoff_mm(rng)
    settle_steps = SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS
    record_stride = max(1, round(IL_SIM_HZ / fps))

    reset_il_simulation(gui=gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_il_scene(gui=gui, hole_xy=hole_xy, opencv_render=False)
    hole_pos, hole_orn = p.getBasePositionAndOrientation(hole_id)

    move_tip_to_standoff(
        robot_id, eef, arm, peg, hole_xy, COARSE_STANDOFF, gui=gui, settle_steps=settle_steps,
    )
    apply_tip_perturbation(
        robot_id, eef, arm, peg, hole_xy, hole_orn, perturb, gui=gui, settle_steps=settle_steps,
    )
    settle(settle_steps, gui)
    print(format_perturbation_log(perturb))
    print(f"target standoff z={standoff_mm:.2f} mm")

    wrist_cam, fixed_cam = get_il_cameras()
    assert wrist_cam is not None and fixed_cam is not None

    goal_wrist, goal_fixed, goal_ok = capture_oracle_goals(
        robot_id=robot_id,
        arm=arm,
        peg=peg,
        hole_xy=hole_xy,
        hole_orn=hole_orn,
        target_standoff_mm=standoff_mm,
        wrist_cam=wrist_cam,
        fixed_cam=fixed_cam,
        gui=gui,
    )
    if not goal_ok:
        teardown_il_episode()
        return {
            "seed": seed,
            "hole_xy": list(hole_xy),
            "target_standoff_mm": round(standoff_mm, 3),
            "success": False,
            "reason": "oracle_goal_align_failed",
            "steps": 0,
        }

    policy.reset()
    success = False
    reason = "max_steps"
    steps = 0
    metrics: dict | None = None

    try:
        for step in range(max_steps):
            rgbs = capture_rgb_pair(wrist_cam, fixed_cam, gui=gui)
            qpos = _qpos(robot_id, arm)
            obs = build_obs_batch(
                wrist_rgb=rgbs[CAM_WRIST2],
                fixed_rgb=rgbs[CAM_FIXED2],
                goal_wrist=goal_wrist,
                goal_fixed=goal_fixed,
                qpos=qpos,
                device=device,
            )
            obs = preprocess(obs)
            action = policy.select_action(obs)
            action = postprocess(action)
            twist = action_tensor_to_twist(action)

            for _ in range(record_stride):
                apply_cartesian_velocity(robot_id, peg, arm, twist)
                p.stepSimulation()
                if gui:
                    time.sleep(1.0 / IL_SIM_HZ)

            steps = step + 1
            tip = peg_tip_world(robot_id, peg)
            peg_orn = p.getLinkState(robot_id, peg)[1]
            metrics = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)

            if metrics_converged(metrics):
                success = True
                reason = "converged"
                break
    finally:
        stop_arm(robot_id, arm)
        teardown_il_episode()

    gt_ok = False
    if metrics is not None:
        gt_ok = is_xy_rpy_aligned(
            metrics["dx"], metrics["dy"], metrics["roll"], metrics["pitch"], metrics["yaw"],
        )

    row = {
        "seed": seed,
        "hole_xy": _round_vec(hole_xy, 4),
        "target_standoff_mm": round(standoff_mm, 3),
        "success": success,
        "gt_aligned": gt_ok,
        "reason": reason,
        "steps": steps,
        "goal_oracle_ok": goal_ok,
    }
    if metrics is not None:
        row["metrics"] = {
            "dx_mm": round(metrics["dx"] * 1e3, 3),
            "dy_mm": round(metrics["dy"] * 1e3, 3),
            "standoff_mm": round(metrics["standoff"] * 1e3, 3),
            "roll_deg": round(np.degrees(metrics["roll"]), 3),
            "pitch_deg": round(np.degrees(metrics["pitch"]), 3),
            "yaw_deg": round(np.degrees(metrics["yaw"]), 3),
        }
    return row


def load_policy(model_id: str, device: torch.device):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=model_id)
    return policy, preprocess, postprocess


def run_eval(
    *,
    model_id: str,
    episodes: int,
    seed: int,
    output_dir: str,
    fps: float,
    max_steps: int,
    gui: bool,
    device: str | None,
    standoff_mm: float | None,
) -> dict:
    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    print(f"loading {model_id} → {device_t} (first run downloads from Hugging Face Hub)")
    policy, preprocess, postprocess = load_policy(model_id, device_t)

    rng = random.Random(seed)
    rows: list[dict] = []
    ok = 0

    connect_il(gui=gui)
    try:
        for ep in range(1, episodes + 1):
            ep_seed = rng.randint(1, 9_000_000) if episodes > 1 else seed
            row = rollout_episode(
                seed=ep_seed,
                policy=policy,
                preprocess=preprocess,
                postprocess=postprocess,
                device=device_t,
                fps=fps,
                max_steps=max_steps,
                gui=gui,
                target_standoff_mm=standoff_mm,
            )
            rows.append(row)
            ok += int(row["success"])
            m = row.get("metrics") or {}
            print(
                f"[{ep}/{episodes}] seed={ep_seed} hole={row['hole_xy']} "
                f"success={row['success']} steps={row['steps']} "
                f"dx={m.get('dx_mm', '?')} dy={m.get('dy_mm', '?')} "
                f"standoff={m.get('standoff_mm', '?')} reason={row['reason']}"
            )
    finally:
        if p.getConnectionInfo()["isConnected"]:
            disconnect_il()

    rate = ok / episodes * 100 if episodes else 0.0
    summary = {
        "mode": "il_flow_diffusion_rollout",
        "model": model_id,
        "episodes": episodes,
        "successes": ok,
        "failures": episodes - ok,
        "success_rate_pct": round(rate, 1),
        "seed": seed,
        "fps": fps,
        "max_steps": max_steps,
        "goal_mode": "oracle_gt",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": rows,
    }
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"il_flow_dp_eval_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{ok}/{episodes} = {rate:.1f}% → {out_path}")
    summary["_result_file"] = out_path
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="IL Diffusion policy rollout eval in PyBullet")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local checkpoint dir")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=20.0, help="policy control rate (match training fps)")
    ap.add_argument("--max-steps", type=int, default=120, help="max policy steps per episode")
    ap.add_argument("--standoff-mm", type=float, default=None, help="fixed target standoff; default random")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    try:
        summary = run_eval(
            model_id=args.model,
            episodes=args.episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            fps=args.fps,
            max_steps=args.max_steps,
            gui=not args.headless,
            device=args.device,
            standoff_mm=args.standoff_mm,
        )
    except ImportError as exc:
        print(
            "LeRobot not installed. Use conda env with lerobot, e.g.\n"
            "  conda activate lerobot312\n"
            "  pip install -e \"/path/to/lerobot[training,diffusion]\"",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
