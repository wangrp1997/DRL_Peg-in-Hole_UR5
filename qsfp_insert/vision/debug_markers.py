"""Small PyBullet GUI markers at GT corner world positions."""
from __future__ import annotations

import pybullet as p

from vision.corners import hole_corners_world, peg_tip_corners_world

_HOLE_COLOR = (0.0, 0.85, 0.0)
_PEG_COLOR = (1.0, 0.45, 0.0)
_POINT_SIZE = 10.0

_hole_marker_id: int | None = None
_peg_marker_id: int | None = None


def _update_point_batch(
    points: list,
    color: tuple[float, float, float],
    marker_id: int | None,
) -> int | None:
    if not points:
        if marker_id is not None and marker_id >= 0:
            p.removeUserDebugItem(marker_id)
        return None
    colors = [color] * len(points)
    if marker_id is not None and marker_id >= 0:
        new_id = p.addUserDebugPoints(
            points,
            colors,
            pointSize=_POINT_SIZE,
            lifeTime=0,
            replaceItemUniqueId=marker_id,
        )
    else:
        new_id = p.addUserDebugPoints(
            points,
            colors,
            pointSize=_POINT_SIZE,
            lifeTime=0,
        )
    return new_id if new_id >= 0 else marker_id


def sync_gt_corner_markers(robot_id: int, peg_link: int, hole_id: int) -> None:
    """In-place update of hole (green) and peg (orange) corner markers."""
    global _hole_marker_id, _peg_marker_id

    hole_pts = hole_corners_world(hole_id)
    peg_pts = peg_tip_corners_world(robot_id, peg_link)
    _hole_marker_id = _update_point_batch(hole_pts, _HOLE_COLOR, _hole_marker_id)
    _peg_marker_id = _update_point_batch(peg_pts, _PEG_COLOR, _peg_marker_id)


def clear_gt_corner_markers() -> None:
    """Remove marker debug items (call on exit)."""
    global _hole_marker_id, _peg_marker_id
    for mid in (_hole_marker_id, _peg_marker_id):
        if mid is not None and mid >= 0:
            p.removeUserDebugItem(mid)
    _hole_marker_id = None
    _peg_marker_id = None
