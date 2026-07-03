"""Random 6D peg-tip perturbation after IK to a nominal standoff pose."""
from __future__ import annotations

import math
import random
from typing import TypedDict

import numpy as np
import pybullet as p

from constants import (
    CORNER_SERVO_STANDOFF,
    PEG_L,
    PLATE_TOP_Z,
    PERTURB_PITCH_RAD,
    PERTURB_ROLL_RAD,
    PERTURB_XY_M,
    PERTURB_YAW_RAD,
    PERTURB_Z_M,
    SETTLE_IK_STEPS,
    SETTLE_IK_STEPS_GUI,
)
from sim.scene import move_ee, settle


class Perturbation6(TypedDict):
    dx: float
    dy: float
    dz: float
    roll: float
    pitch: float
    yaw: float


def format_perturbation_log(perturb: Perturbation6) -> str:
    return (
        f"perturb applied: dx={perturb['dx'] * 1e3:+.2f}mm dy={perturb['dy'] * 1e3:+.2f}mm "
        f"dz={perturb['dz'] * 1e3:+.2f}mm "
        f"rpy=({math.degrees(perturb['roll']):+.2f}°, "
        f"{math.degrees(perturb['pitch']):+.2f}°, "
        f"{math.degrees(perturb['yaw']):+.2f}°)"
    )


def sample_perturbation6(rng: random.Random) -> Perturbation6:
    return {
        "dx": rng.uniform(-PERTURB_XY_M, PERTURB_XY_M),
        "dy": rng.uniform(-PERTURB_XY_M, PERTURB_XY_M),
        "dz": rng.uniform(-PERTURB_Z_M, PERTURB_Z_M),
        "roll": rng.uniform(-PERTURB_ROLL_RAD, PERTURB_ROLL_RAD),
        "pitch": rng.uniform(-PERTURB_PITCH_RAD, PERTURB_PITCH_RAD),
        "yaw": rng.uniform(-PERTURB_YAW_RAD, PERTURB_YAW_RAD),
    }


def _peg_link_pose_from_tip(
    tip_xyz: tuple[float, float, float],
    peg_orn: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    mat = p.getMatrixFromQuaternion(peg_orn)
    half = PEG_L / 2.0
    pos = (
        tip_xyz[0] + half * mat[2],
        tip_xyz[1] + half * mat[5],
        tip_xyz[2] + half * mat[8],
    )
    return pos, peg_orn


def move_peg_tip_pose(
    robot_id: int,
    eef: int,
    arm: list[int],
    peg: int,
    tip_xyz: tuple[float, float, float],
    peg_orn: tuple[float, float, float, float],
    gui: bool = False,
    settle_steps: int | None = None,
) -> None:
    """IK so peg tip and peg link orientation match the target (ee chain preserved)."""
    peg_pos, peg_orn = _peg_link_pose_from_tip(tip_xyz, peg_orn)
    ee_pos, ee_orn = p.getLinkState(robot_id, eef)[:2]
    peg_pos_cur, peg_orn_cur = p.getLinkState(robot_id, peg)[:2]
    inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
    pe_pos, pe_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, peg_pos_cur, peg_orn_cur)
    inv_pe_pos, inv_pe_orn = p.invertTransform(pe_pos, pe_orn)
    target_ee_pos, target_ee_orn = p.multiplyTransforms(peg_pos, peg_orn, inv_pe_pos, inv_pe_orn)
    move_ee(robot_id, eef, arm, list(target_ee_pos), target_ee_orn)
    if settle_steps is None:
        settle_steps = SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS
    settle(settle_steps, gui)


def apply_tip_perturbation(
    robot_id: int,
    eef: int,
    arm: list[int],
    peg: int,
    hole_xy: tuple[float, float],
    hole_orn,
    perturb: Perturbation6,
    standoff: float = CORNER_SERVO_STANDOFF,
    gui: bool = False,
    settle_steps: int | None = None,
) -> None:
    """Apply a 6D offset in the hole frame on top of the nominal standoff pose."""
    nominal_tip = np.array([hole_xy[0], hole_xy[1], PLATE_TOP_Z + standoff], dtype=np.float64)
    rot = np.array(p.getMatrixFromQuaternion(hole_orn), dtype=np.float64).reshape(3, 3)
    delta_local = np.array([perturb["dx"], perturb["dy"], perturb["dz"]], dtype=np.float64)
    target_tip = tuple(nominal_tip + rot @ delta_local)
    delta_orn = p.getQuaternionFromEuler([perturb["roll"], perturb["pitch"], perturb["yaw"]])
    target_peg_orn = p.multiplyTransforms([0.0, 0.0, 0.0], hole_orn, [0.0, 0.0, 0.0], delta_orn)[1]
    move_peg_tip_pose(robot_id, eef, arm, peg, target_tip, target_peg_orn, gui=gui, settle_steps=settle_steps)
