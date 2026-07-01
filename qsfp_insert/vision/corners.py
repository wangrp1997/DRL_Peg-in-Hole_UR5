"""GT 3D corner points for hole mouth and peg tip; project to fixed camera."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pybullet as p

from constants import HOLE_DEPTH, PEG_H, PEG_L, PEG_W
from sim.fixed_camera import FixedCamera

# qsfp_dd_hole_plate.urdf opening 19 × 9 mm.
# PyBullet base origin is HOLE_DEPTH/2 below mouth (inertial at z=-HOLE_DEPTH/2); mouth at +HOLE_DEPTH/2.
HOLE_HALF_X = 0.0095
HOLE_HALF_Y = 0.0045
HOLE_MOUTH_Z_LOCAL = HOLE_DEPTH / 2.0


@dataclass(frozen=True)
class ImageKeypoints:
    name: str
    prefix: str
    color_bgr: tuple[int, int, int]
    uv: list[tuple[float, float]]
    visible: list[bool]


def _link_to_world(body_id: int, link_index: int, local: tuple[float, float, float]) -> tuple[float, float, float]:
    if link_index == -1:
        pos, orn = p.getBasePositionAndOrientation(body_id)
    else:
        pos, orn = p.getLinkState(body_id, link_index)[:2]
    world, _ = p.multiplyTransforms(pos, orn, local, [0.0, 0.0, 0.0, 1.0])
    return world


def hole_corners_local() -> list[tuple[float, float, float]]:
    hx, hy, z = HOLE_HALF_X, HOLE_HALF_Y, HOLE_MOUTH_Z_LOCAL
    return [
        (-hx, -hy, z),
        (hx, -hy, z),
        (hx, hy, z),
        (-hx, hy, z),
    ]


def peg_tip_corners_local() -> list[tuple[float, float, float]]:
    """Peg link frame: tip at −Z; rectangular face at z = −PEG_L/2."""
    hw, hh, z = PEG_W / 2.0, PEG_H / 2.0, -PEG_L / 2.0
    return [
        (-hw, -hh, z),
        (hw, -hh, z),
        (hw, hh, z),
        (-hw, hh, z),
    ]


def hole_corners_world(hole_id: int) -> list[tuple[float, float, float]]:
    return [_link_to_world(hole_id, -1, pt) for pt in hole_corners_local()]


def peg_tip_corners_world(robot_id: int, peg_link: int) -> list[tuple[float, float, float]]:
    return [_link_to_world(robot_id, peg_link, pt) for pt in peg_tip_corners_local()]


def project_corners(
    cam: FixedCamera,
    world_pts: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float]], list[bool]]:
    uv: list[tuple[float, float]] = []
    visible: list[bool] = []
    for pt in world_pts:
        u, v, z = cam.project_world(pt)
        ok = (
            math.isfinite(u)
            and math.isfinite(v)
            and -1.0 <= z <= 1.0
            and 0.0 <= u < cam.width
            and 0.0 <= v < cam.height
        )
        uv.append((u, v))
        visible.append(ok)
    return uv, visible


def gt_image_keypoints(
    cam: FixedCamera,
    robot_id: int,
    peg_link: int,
    hole_id: int,
) -> list[ImageKeypoints]:
    hole_uv, hole_vis = project_corners(cam, hole_corners_world(hole_id))
    peg_uv, peg_vis = project_corners(cam, peg_tip_corners_world(robot_id, peg_link))
    return [
        ImageKeypoints("hole", "H", (0, 220, 0), hole_uv, hole_vis),
        ImageKeypoints("peg", "P", (0, 80, 255), peg_uv, peg_vis),
    ]
