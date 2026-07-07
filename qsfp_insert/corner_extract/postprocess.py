"""Corner ordering, visibility, and provider post-processing."""
from __future__ import annotations

from collections.abc import Callable

from constants import PLATE_TOP_Z
from vision.align import refine_hole_corners_h123
from vision.corners import (
    ImageKeypoints,
    ProjectorCam,
    apply_infer_corner0,
    apply_infer_corner0_if_missing,
    infer_corner0_uv,
    project_corners,
)

__all__ = [
    "ImageKeypoints",
    "ProjectorCam",
    "apply_infer_corner0",
    "apply_infer_corner0_if_missing",
    "infer_corner0_uv",
    "wrap_lock_hole",
]


def _hole_corners_world_from_kp(
    hole_kp: ImageKeypoints,
    cam: ProjectorCam,
    plane_z: float,
) -> list[tuple[float, float, float]] | None:
    pts: list[tuple[float, float, float]] = []
    for i in range(4):
        if not hole_kp.visible[i]:
            return None
        u, v = hole_kp.uv[i]
        pt = cam.world_point_on_plane(u, v, plane_z)
        if pt is None:
            return None
        pts.append(pt)
    return pts


def _hole_keypoints_from_world(
    world_pts: list[tuple[float, float, float]],
    cam: ProjectorCam,
) -> ImageKeypoints:
    uv, visible = project_corners(cam, world_pts)
    return ImageKeypoints(
        "hole",
        "H",
        (0, 220, 0),
        uv,
        visible,
        inferred=(False, False, False, False),
    )


def wrap_lock_hole(
    provider: Callable[[], list[ImageKeypoints] | None],
    cam: ProjectorCam,
    *,
    enabled: bool = False,
    skip_hole_h0: bool = False,
    plane_z: float = PLATE_TOP_Z,
) -> Callable[[], list[ImageKeypoints] | None]:
    """After perturb: lock hole 4 corners in world frame; peg stays live YOLO."""
    if not enabled:
        return provider

    locked_world: list[tuple[float, float, float]] | None = None

    def wrapped() -> list[ImageKeypoints] | None:
        nonlocal locked_world
        sets = provider()
        if sets is None:
            return None
        hole = next((s for s in sets if s.name == "hole"), None)
        peg = next((s for s in sets if s.name == "peg"), None)
        if hole is None or peg is None:
            return None

        if locked_world is None:
            if skip_hole_h0:
                hole_lock = refine_hole_corners_h123(hole, cam)
            else:
                hole_lock = hole
            if hole_lock is None:
                return None
            world_pts = _hole_corners_world_from_kp(hole_lock, cam, plane_z)
            if world_pts is None:
                return None
            locked_world = world_pts
            mode = "H1–H3 + 19×9mm IPPE" if skip_hole_h0 else "YOLO H0–H3"
            print(
                f"lock-hole: frozen 4 hole corners on plate Z={plane_z:.4f} m "
                f"({mode}; peg still live YOLO)"
            )

        hole_locked = _hole_keypoints_from_world(locked_world, cam)
        if sum(hole_locked.visible) < 4:
            return None
        return [hole_locked, peg]

    return wrapped
