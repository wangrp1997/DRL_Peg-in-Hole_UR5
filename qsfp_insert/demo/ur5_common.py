"""Shared UR5 + table + hole scene for qsfp_insert demos."""
from __future__ import annotations

import os
import time

import pybullet as p
import pybullet_data

from _paths import ROOT, URDF  # noqa: F401

from constants import FIXTURE_CENTER_Z, PLATE_TOP_Z, REST_POSES, ROBOT_BASE_Z
from geometry import peg_tip_world


def urdf(name: str) -> str:
    return os.path.join(URDF, name)


def link_index(body_id: int, name: str) -> int:
    for i in range(p.getNumJoints(body_id)):
        if p.getJointInfo(body_id, i)[12].decode() == name:
            return i
    raise ValueError(name)


def arm_joints(robot_id: int) -> list[int]:
    return [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED][:6]


def connect(gui: bool) -> None:
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)


def _joint_positions(robot_id: int, arm: list[int]) -> list[float]:
    return [p.getJointState(robot_id, j)[0] for j in arm]


def move_ee(robot_id, eef, arm, pos, orn) -> None:
    """IK with restPoses=current joints to stay on same branch (approx. straight Cartesian)."""
    rest = _joint_positions(robot_id, arm)
    q = p.calculateInverseKinematics(
        robot_id,
        eef,
        pos,
        orn,
        maxNumIterations=100,
        residualThreshold=1e-5,
        restPoses=rest,
    )
    for j, v in zip(arm, q[:6]):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, v, force=500)


def step_ee_z(robot_id, eef, arm, ee_xy, ee_orn, dz: float, gui: bool = False) -> None:
    """Move EE along world Z with fixed XY and orientation (linear approach)."""
    ee_pos, _ = p.getLinkState(robot_id, eef)[:2]
    move_ee(robot_id, eef, arm, [ee_xy[0], ee_xy[1], ee_pos[2] + dz], ee_orn)
    settle(10, gui)


def step_tip_z(robot_id, eef, arm, peg, hole_xy, ee_orn, dz: float, gui: bool = False) -> None:
    """Move peg tip along world Z through hole centre (TCP linear / MoveL)."""
    ee_pos, _ = p.getLinkState(robot_id, eef)[:2]
    tip = peg_tip_world(robot_id, peg)
    move_ee(
        robot_id,
        eef,
        arm,
        [ee_pos[0] - (tip[0] - hole_xy[0]), ee_pos[1] - (tip[1] - hole_xy[1]), ee_pos[2] + dz],
        ee_orn,
    )
    settle(10, gui)


def load_fixture(hole_xy: tuple[float, float]) -> int:
    return p.loadURDF(
        urdf("qsfp_fixture_base.urdf"),
        [hole_xy[0], hole_xy[1], FIXTURE_CENTER_Z],
        useFixedBase=True,
    )


def load_scene(gui: bool):
    """Return robot_id, arm, eef_idx, peg_idx, hole_id, hole_xy."""
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.4, 0, 0], p.getQuaternionFromEuler([0, 0, 1.57079632679]))

    robot_id = p.loadURDF(urdf("ur5_robotiq_85_qsfp_peg.urdf"), [0, 0, ROBOT_BASE_Z], useFixedBase=True)
    arm = arm_joints(robot_id)
    eef = link_index(robot_id, "ee_link")
    peg = link_index(robot_id, "qsfp_peg_link")

    for j, q in zip(arm, REST_POSES):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, q, force=500)
    settle(120)

    tip0 = peg_tip_world(robot_id, peg)
    hole_xy = (tip0[0], tip0[1])
    load_fixture(hole_xy)
    hole_id = p.loadURDF(urdf("qsfp_dd_hole_plate.urdf"), [hole_xy[0], hole_xy[1], PLATE_TOP_Z], useFixedBase=True)
    if gui:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 0.25])
        p.resetDebugVisualizerCamera(1.2, 110, -40, [0.5, 0, 0.6])

    return robot_id, arm, eef, peg, hole_id, hole_xy


def settle(steps: int = 10, gui: bool = False) -> None:
    for _ in range(steps):
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / 240.0)


def idle_gui() -> None:
    print("Close PyBullet window to exit.")
    while p.getConnectionInfo()["isConnected"]:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
