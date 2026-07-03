"""Small PyBullet GUI markers at GT corner world positions."""
from __future__ import annotations

import pybullet as p

from constants import PLATE_TOP_Z
from geometry import peg_tip_world
from vision.corners import (
    INFERRED_MARKER_COLOR,
    ImageKeypoints,
    hole_corners_world,
    marker_world_point,
    peg_tip_corners_world,
)

_HOLE_COLOR = (0.0, 0.85, 0.0)
_PEG_COLOR = (1.0, 0.45, 0.0)
_POINT_SIZE = 10.0

_hole_marker_ids: list[int | None] = [None] * 4
_peg_marker_ids: list[int | None] = [None] * 4


def _clear_marker_ids(ids: list[int | None]) -> None:
    for mid in ids:
        if mid is not None and mid >= 0:
            p.removeUserDebugItem(mid)


def _set_corner_marker(
    point: tuple[float, float, float],
    color: tuple[float, float, float],
    marker_id: int | None,
) -> int | None:
    if marker_id is not None and marker_id >= 0:
        new_id = p.addUserDebugPoints(
            [point],
            [color],
            pointSize=_POINT_SIZE,
            lifeTime=0,
            replaceItemUniqueId=marker_id,
        )
    else:
        new_id = p.addUserDebugPoints([point], [color], pointSize=_POINT_SIZE, lifeTime=0)
    return new_id if new_id >= 0 else marker_id


def sync_gt_corner_markers(
    robot_id: int,
    peg_link: int,
    hole_id: int,
    keypoints: list[ImageKeypoints] | None = None,
    cam=None,
) -> None:
    """Update hole (green) and peg (orange) corner markers; inferred corners in magenta."""
    global _hole_marker_ids, _peg_marker_ids

    hole_pts = hole_corners_world(hole_id)
    peg_pts = peg_tip_corners_world(robot_id, peg_link)
    peg_plane_z = peg_tip_world(robot_id, peg_link)[2]

    hole_kp = next((s for s in keypoints if s.name == "hole"), None) if keypoints else None
    peg_kp = next((s for s in keypoints if s.name == "peg"), None) if keypoints else None

    for i in range(4):
        if hole_kp is not None and cam is not None and hole_kp.visible[i]:
            hpt = marker_world_point(cam, hole_kp, i, hole_pts[i], PLATE_TOP_Z)
            hcol = INFERRED_MARKER_COLOR if hole_kp.inferred[i] else _HOLE_COLOR
        else:
            hpt, hcol = hole_pts[i], _HOLE_COLOR
        _hole_marker_ids[i] = _set_corner_marker(hpt, hcol, _hole_marker_ids[i])

        if peg_kp is not None and cam is not None and peg_kp.visible[i]:
            ppt = marker_world_point(cam, peg_kp, i, peg_pts[i], peg_plane_z)
            pcol = INFERRED_MARKER_COLOR if peg_kp.inferred[i] else _PEG_COLOR
        else:
            ppt, pcol = peg_pts[i], _PEG_COLOR
        _peg_marker_ids[i] = _set_corner_marker(ppt, pcol, _peg_marker_ids[i])


def clear_gt_corner_markers() -> None:
    """Remove marker debug items (call on exit)."""
    global _hole_marker_ids, _peg_marker_ids
    _clear_marker_ids(_hole_marker_ids)
    _clear_marker_ids(_peg_marker_ids)
    _hole_marker_ids = [None] * 4
    _peg_marker_ids = [None] * 4
