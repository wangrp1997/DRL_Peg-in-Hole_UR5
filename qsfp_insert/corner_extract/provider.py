"""Unified keypoint provider factory for --corners backend selection."""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from corner_extract.backends.deeplsd_geom import make_deeplsd_provider
from corner_extract.backends.gt import make_gt_provider
from corner_extract.backends.yolo_obb import make_yolo_provider
from corner_extract.postprocess import wrap_lock_hole
from vision.corners import ImageKeypoints, ProjectorCam

CornerMode = Literal["gt", "deeplsd", "yolo"]

_BACKENDS = {
    "gt": make_gt_provider,
    "deeplsd": make_deeplsd_provider,
    "yolo": make_yolo_provider,
}


def make_keypoint_provider(
    mode: CornerMode,
    cam: ProjectorCam,
    robot_id: int,
    peg: int,
    hole_id: int,
    *,
    infer_corner0: bool = False,
    lock_hole_corners: bool = False,
    skip_hole_h0: bool = False,
    gui: bool = False,
    yolo_weights: str | None = None,
) -> Callable[[], list[ImageKeypoints] | None]:
    factory = _BACKENDS.get(mode)
    if factory is None:
        raise ValueError(f"unknown corners mode: {mode!r}")
    if skip_hole_h0 and not lock_hole_corners:
        raise ValueError("--skip-hole-h0 requires --lock-hole-corners")
    provider = factory(
        cam,
        robot_id,
        peg,
        hole_id,
        infer_corner0=infer_corner0,
        gui=gui,
        yolo_weights=yolo_weights,
    )
    if lock_hole_corners and mode != "yolo":
        raise ValueError("--lock-hole-corners only applies to --corners yolo")
    if skip_hole_h0 and mode != "yolo":
        raise ValueError("--skip-hole-h0 only applies to --corners yolo")
    return wrap_lock_hole(
        provider, cam, enabled=lock_hole_corners, skip_hole_h0=skip_hole_h0,
    )
