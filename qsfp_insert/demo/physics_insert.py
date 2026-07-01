#!/usr/bin/env python3
"""Peg + hole plate, kinematic insert."""
from __future__ import annotations

import argparse
import os
import sys
import time

import pybullet as p
import pybullet_data

from _paths import ROOT, URDF  # noqa: F401

from constants import HOLE_XY, INSERTED_TIP_Z, PEG_L, PLATE_TOP_Z
from geometry import is_inserted, peg_tip_world
from sim.scene import load_fixture


def _urdf(name: str) -> str:
    return os.path.join(URDF, name)


def _idle_gui() -> None:
    while p.getConnectionInfo()["isConnected"]:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def run(gui: bool) -> bool:
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.loadURDF("plane.urdf")

    hole_xy = HOLE_XY
    load_fixture(hole_xy)
    hole_id = p.loadURDF(
        _urdf("qsfp_dd_hole_plate.urdf"),
        [hole_xy[0], hole_xy[1], PLATE_TOP_Z],
        useFixedBase=True,
    )
    if gui:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 0.25])
        p.resetDebugVisualizerCamera(1.2, 90, -35, [0.5, 0, 0.6])

    peg_id = p.loadURDF(
        _urdf("qsfp_dd_rect_peg.urdf"),
        [hole_xy[0], hole_xy[1], PLATE_TOP_Z + 0.002 + PEG_L / 2],
    )

    pos, orn = p.getBasePositionAndOrientation(peg_id)
    for _ in range(400):
        pos = [pos[0], pos[1], pos[2] - 0.0005]
        p.resetBasePositionAndOrientation(peg_id, pos, orn)
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / 240.0)
        if is_inserted(peg_tip_world(peg_id), hole_xy):
            break

    tip = peg_tip_world(peg_id)
    ok = is_inserted(tip, hole_xy)
    print(f"tip_z={tip[2]:.4f} target<={INSERTED_TIP_Z:.4f} inserted={ok}")
    if gui:
        print("Close PyBullet window to exit.")
        _idle_gui()
    p.disconnect()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    sys.exit(0 if run(ap.parse_args().gui) else 1)
