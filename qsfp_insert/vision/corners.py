"""GT 3D corner points for hole mouth and peg tip; project to fixed camera."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pybullet as p

from constants import HOLE_DEPTH, PEG_H, PEG_L, PEG_W, PLATE_TOP_Z
from sim.fixed_camera import FixedCamera

# qsfp_dd_hole_plate.urdf opening 19 × 9 mm.
# PyBullet base origin is HOLE_DEPTH/2 below mouth (inertial at z=-HOLE_DEPTH/2); mouth at +HOLE_DEPTH/2.
HOLE_HALF_X = 0.0095
HOLE_HALF_Y = 0.0045
HOLE_MOUTH_Z_LOCAL = HOLE_DEPTH / 2.0

# BGR for OpenCV overlay when corner 0 is inferred from 1–3 (not directly detected).
INFERRED_COLOR_BGR = (255, 0, 255)
INFERRED_MARKER_COLOR = (1.0, 0.0, 1.0)


@dataclass(frozen=True)
class ImageKeypoints:
    name: str
    prefix: str
    color_bgr: tuple[int, int, int]
    uv: list[tuple[float, float]]
    visible: list[bool]
    inferred: tuple[bool, bool, bool, bool] = (False, False, False, False)


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


def infer_corner0_uv(
    cam: FixedCamera,
    uv: list[tuple[float, float]],
    visible: list[bool],
    half_x: float,
    half_y: float,
) -> tuple[tuple[float, float], bool]:
    """Infer image of corner 0 from visible 1–3 via parallelogram (p0 = p1 + p3 − p2).

    Corners 0–3 are ordered around the rectangle; only 1–3 are detected. The fourth
    image point is the parallelogram completion used before 4-point planar PnP (IPPE).
    half_x/half_y are unused here but kept for call-site symmetry with object model.
    """
    del half_x, half_y
    if not (visible[1] and visible[2] and visible[3]):
        return uv[0], False
    u0 = uv[1][0] + uv[3][0] - uv[2][0]
    v0 = uv[1][1] + uv[3][1] - uv[2][1]
    if not (math.isfinite(u0) and math.isfinite(v0) and 0.0 <= u0 < cam.width and 0.0 <= v0 < cam.height):
        return uv[0], False
    return (u0, v0), True


def apply_infer_corner0(
    kp: ImageKeypoints,
    cam: FixedCamera,
    half_x: float,
    half_y: float,
) -> ImageKeypoints:
    """Simulate only corners 1–3 detected; corner 0 from parallelogram in image plane."""
    uv = list(kp.uv)
    visible = list(kp.visible)
    inferred = list(kp.inferred)
    visible[0] = False
    inferred[0] = False
    uv0, ok = infer_corner0_uv(cam, uv, visible, half_x, half_y)
    if ok:
        uv[0] = uv0
        visible[0] = True
        inferred[0] = True
    return replace(kp, uv=uv, visible=visible, inferred=tuple(inferred))


def marker_world_point(
    cam: FixedCamera,
    kp: ImageKeypoints,
    index: int,
    fallback_world: tuple[float, float, float],
    plane_z: float,
) -> tuple[float, float, float]:
    if not kp.inferred[index]:
        return fallback_world
    u, v = kp.uv[index]
    pt = cam.world_point_on_plane(u, v, plane_z)
    return pt if pt is not None else fallback_world


def gt_image_keypoints(
    cam: FixedCamera,
    robot_id: int,
    peg_link: int,
    hole_id: int,
    infer_corner0: bool = False,
) -> list[ImageKeypoints]:
    hole_uv, hole_vis = project_corners(cam, hole_corners_world(hole_id))
    peg_uv, peg_vis = project_corners(cam, peg_tip_corners_world(robot_id, peg_link))
    sets = [
        ImageKeypoints("hole", "H", (0, 220, 0), hole_uv, hole_vis),
        ImageKeypoints("peg", "P", (0, 80, 255), peg_uv, peg_vis),
    ]
    if not infer_corner0:
        return sets
    return [
        apply_infer_corner0(sets[0], cam, HOLE_HALF_X, HOLE_HALF_Y),
        apply_infer_corner0(sets[1], cam, PEG_W / 2.0, PEG_H / 2.0),
    ]
