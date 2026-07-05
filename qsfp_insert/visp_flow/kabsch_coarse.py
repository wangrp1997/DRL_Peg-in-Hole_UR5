"""Geometric kabsch coarse alignment — same loop as fixed_cam corner_servo demo."""
from __future__ import annotations

from collections.abc import Callable

from sim.wrist_camera2 import WristCamera2
from vision.corner_servo import run_corner_servo
from vision.corners import ImageKeypoints


def run_kabsch_coarse(
    robot_id: int,
    peg: int,
    arm: list[int],
    cam: WristCamera2,
    hole_xy: tuple[float, float],
    hole_orn,
    keypoint_provider: Callable[[], list[ImageKeypoints] | None],
    *,
    gui: bool = False,
    on_step: Callable[[], None] | None = None,
) -> tuple[bool, dict | None]:
    return run_corner_servo(
        robot_id,
        arm,
        peg,
        cam,
        hole_xy,
        hole_orn,
        keypoint_provider,
        gui=gui,
        on_step=on_step,
        align_method="kabsch",
    )
