"""Cartesian proportional align until is_aligned (servo_align.py loop)."""
from __future__ import annotations

from collections.abc import Callable

import pybullet as p

from constants import SERVO_GUI_SUBSTEPS, SERVO_MAX_STEPS, SERVO_STALL_STEPS
from geometry import alignment_metrics, is_aligned, peg_tip_world
from sim.cartesian_control import alignment_twist, apply_cartesian_velocity, stop_arm


def run_cartesian_align(
    robot_id: int,
    arm: list[int],
    peg: int,
    hole_xy: tuple[float, float],
    hole_orn,
    *,
    gui: bool = False,
    on_step: Callable[[], None] | None = None,
) -> tuple[bool, dict]:
    stall = 0
    prev_standoff = float("inf")
    aligned = False
    m: dict = {}

    for _ in range(SERVO_MAX_STEPS):
        tip = peg_tip_world(robot_id, peg)
        peg_orn = p.getLinkState(robot_id, peg)[1]
        if is_aligned(tip, peg_orn, hole_xy, hole_orn):
            aligned = True
            m = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
            break
        m = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
        if abs(m["standoff"] - prev_standoff) < 5e-6:
            stall += 1
            if stall >= SERVO_STALL_STEPS:
                break
        else:
            stall = 0
        prev_standoff = m["standoff"]
        twist = alignment_twist(m["dx"], m["dy"], m["standoff"], m["roll"], m["pitch"], m["yaw"])
        apply_cartesian_velocity(robot_id, peg, arm, twist)
        substeps = SERVO_GUI_SUBSTEPS if gui else 1
        for _ in range(substeps):
            p.stepSimulation()
        if on_step is not None:
            on_step()

    stop_arm(robot_id, arm)
    for _ in range(20):
        p.stepSimulation()
        if on_step is not None:
            on_step()

    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    m = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    aligned = is_aligned(tip, peg_orn, hole_xy, hole_orn)
    return aligned, m
