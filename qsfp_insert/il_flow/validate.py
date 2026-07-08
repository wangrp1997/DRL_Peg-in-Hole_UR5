"""Episode sanity checks before saving IL data."""
from __future__ import annotations

import math

import pybullet as p

from constants import MIN_CORNERS_VISIBLE
from geometry import peg_tip_world
from vision.corners import (
    ProjectorCam,
    gt_image_keypoints,
    hole_pose_corners_ok,
)

# Peg tip workspace (catch sim explosions); generous for perturbed start, tight at goal.
_TIP_X = (0.38, 0.78)
_TIP_Y = (-0.05, 0.25)
_TIP_Z = (0.55, 0.95)
_HOLE_POS_TOL_M = 0.001
_QPOS_ABS_MAX = 2.0 * math.pi
_GOAL_STANDOFF_M = (0.001, 0.012)
_GOAL_XY_M = 0.002


def _finite_vec(xs) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


def check_physics_runtime(
    robot_id: int,
    arm: list[int],
    hole_id: int,
    peg: int,
    hole_pos_ref: tuple[float, float, float],
    hole_orn_ref: tuple[float, float, float, float],
) -> tuple[bool, str]:
    """Light check each recorded frame: finite joints, hole plate fixed."""
    for j in arm:
        q = p.getJointState(robot_id, j)[0]
        if not math.isfinite(q) or abs(q) > _QPOS_ABS_MAX:
            return False, f"bad qpos j{j}={q}"

    tip = peg_tip_world(robot_id, peg)
    if not _finite_vec(tip):
        return False, "peg tip non-finite"

    hole_pos, hole_orn = p.getBasePositionAndOrientation(hole_id)
    if not _finite_vec(hole_pos) or not _finite_vec(hole_orn):
        return False, "hole pose non-finite"
    if any(abs(float(a) - float(b)) > _HOLE_POS_TOL_M for a, b in zip(hole_pos, hole_pos_ref)):
        return False, f"hole moved pos={hole_pos} ref={hole_pos_ref}"
    if sum(abs(float(a) - float(b)) for a, b in zip(hole_orn, hole_orn_ref)) > 0.01:
        return False, "hole orientation changed"

    return True, "ok"


def check_physics_final(
    robot_id: int,
    arm: list[int],
    peg: int,
    hole_id: int,
    hole_xy: tuple[float, float],
    hole_pos_ref: tuple[float, float, float],
    hole_orn_ref: tuple[float, float, float, float],
    metrics: dict,
) -> tuple[bool, str]:
    ok, msg = check_physics_runtime(robot_id, arm, hole_id, peg, hole_pos_ref, hole_orn_ref)
    if not ok:
        return ok, msg

    tip = peg_tip_world(robot_id, peg)
    if not (_TIP_X[0] <= tip[0] <= _TIP_X[1] and _TIP_Y[0] <= tip[1] <= _TIP_Y[1] and _TIP_Z[0] <= tip[2] <= _TIP_Z[1]):
        return False, f"peg tip out of workspace {tip}"

    if abs(metrics["dx"]) > _GOAL_XY_M or abs(metrics["dy"]) > _GOAL_XY_M:
        return False, f"goal xy error dx={metrics['dx']:.4f} dy={metrics['dy']:.4f}"
    if not (_GOAL_STANDOFF_M[0] <= metrics["standoff"] <= _GOAL_STANDOFF_M[1]):
        return False, f"goal standoff {metrics['standoff']:.4f} m out of range"

    return True, "ok"


def check_goal_corners(
    cam: ProjectorCam,
    robot_id: int,
    peg: int,
    hole_id: int,
) -> tuple[bool, str]:
    """Goal image sanity: hole + peg corners project into wrist view."""
    kps = gt_image_keypoints(cam, robot_id, peg, hole_id, infer_corner0=False)
    hole_kp = next(s for s in kps if s.name == "hole")
    peg_kp = next(s for s in kps if s.name == "peg")
    n_hole = sum(hole_kp.visible)
    n_peg = sum(peg_kp.visible)
    if n_hole < MIN_CORNERS_VISIBLE:
        return False, f"goal hole corners {n_hole}/4 visible (need >={MIN_CORNERS_VISIBLE})"
    if n_peg < MIN_CORNERS_VISIBLE:
        return False, f"goal peg corners {n_peg}/4 visible (need >={MIN_CORNERS_VISIBLE})"
    if not hole_pose_corners_ok(hole_kp):
        return False, "goal hole pose corners 1–3 not all visible"
    return True, "ok"
