"""Insertion and alignment geometry helpers."""

from __future__ import annotations

from typing import TypedDict

import pybullet as p

from constants import (
    ALIGN_ANG_TOL,
    ALIGN_XY_TOL,
    ALIGN_Z_STANDOFF_MAX,
    ALIGN_Z_STANDOFF_MIN,
    HOLE_DEPTH,
    PEG_L,
    PLATE_TOP_Z,
)


class AlignmentMetrics(TypedDict):
    dx: float
    dy: float
    standoff: float
    roll: float
    pitch: float
    yaw: float
    aligned: bool


def peg_tip_world(body_id: int, link_index: int = -1, peg_length: float = PEG_L) -> tuple[float, float, float]:
    if link_index == -1:
        pos, orn = p.getBasePositionAndOrientation(body_id)
    else:
        pos, orn = p.getLinkState(body_id, link_index)[:2]
    mat = p.getMatrixFromQuaternion(orn)
    half = peg_length / 2.0
    return (
        pos[0] - half * mat[2],
        pos[1] - half * mat[5],
        pos[2] - half * mat[8],
    )


def nominal_peg_orn(hole_orn) -> tuple[float, float, float, float]:
    """Peg link quat when aligned with hole plate: 18.4‖hole X, 8.5‖hole Y, insert −hole Z."""
    return hole_orn


def rpy_error(actual_orn, target_orn) -> tuple[float, float, float]:
    inv_pos, inv_orn = p.invertTransform([0.0, 0.0, 0.0], target_orn)
    _, rel_orn = p.multiplyTransforms([0.0, 0.0, 0.0], actual_orn, inv_pos, inv_orn)
    return p.getEulerFromQuaternion(rel_orn)


def alignment_metrics(
    tip_xyz: tuple[float, float, float],
    peg_orn,
    hole_xy: tuple[float, float],
    hole_orn,
) -> AlignmentMetrics:
    dx = tip_xyz[0] - hole_xy[0]
    dy = tip_xyz[1] - hole_xy[1]
    standoff = tip_xyz[2] - PLATE_TOP_Z
    roll, pitch, yaw = rpy_error(peg_orn, nominal_peg_orn(hole_orn))
    return {
        "dx": dx,
        "dy": dy,
        "standoff": standoff,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "aligned": _check_aligned(dx, dy, standoff, roll, pitch, yaw),
    }


def _check_aligned(dx, dy, standoff, roll, pitch, yaw) -> bool:
    xy_ok = abs(dx) <= ALIGN_XY_TOL and abs(dy) <= ALIGN_XY_TOL
    z_ok = ALIGN_Z_STANDOFF_MIN <= standoff <= ALIGN_Z_STANDOFF_MAX
    rpy_ok = abs(roll) <= ALIGN_ANG_TOL and abs(pitch) <= ALIGN_ANG_TOL and abs(yaw) <= ALIGN_ANG_TOL
    return xy_ok and z_ok and rpy_ok


def is_aligned(
    tip_xyz: tuple[float, float, float],
    peg_orn,
    hole_xy: tuple[float, float],
    hole_orn,
) -> bool:
    return alignment_metrics(tip_xyz, peg_orn, hole_xy, hole_orn)["aligned"]


def is_inserted(
    tip_xyz: tuple[float, float, float],
    hole_xy: tuple[float, float],
    min_insert_depth: float | None = None,
) -> bool:
    depth = HOLE_DEPTH - 0.003 if min_insert_depth is None else min_insert_depth
    tip_z_max = PLATE_TOP_Z - depth
    dx = abs(tip_xyz[0] - hole_xy[0])
    dy = abs(tip_xyz[1] - hole_xy[1])
    return dx < 0.002 and dy < 0.002 and tip_xyz[2] <= tip_z_max
