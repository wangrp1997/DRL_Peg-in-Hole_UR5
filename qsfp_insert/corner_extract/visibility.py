"""Corner visibility for labeling — in-frame + seg occlusion (not GT projection only)."""
from __future__ import annotations

import math

import numpy as np

from vision.corners import ImageKeypoints, ProjectorCam, replace


def _seg_object_link(seg_val: int) -> tuple[int, int]:
    obj = int(seg_val) & ((1 << 24) - 1)
    link = int(seg_val >> 24) - 1
    return obj, link


def _sample_seg(seg: np.ndarray, u: float, v: float, r: int = 2) -> list[tuple[int, int]]:
    h, w = seg.shape
    ui, vi = int(round(u)), int(round(v))
    out: list[tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x, y = ui + dx, vi + dy
            if 0 <= x < w and 0 <= y < h:
                out.append(_seg_object_link(int(seg[y, x])))
    return out


def _in_frame(cam: ProjectorCam, u: float, v: float) -> bool:
    return (
        math.isfinite(u)
        and math.isfinite(v)
        and 0.0 <= u < cam.width
        and 0.0 <= v < cam.height
    )


def _hole_pixel_visible(
    seg: np.ndarray,
    u: float,
    v: float,
    hole_id: int,
) -> bool:
    hits = _sample_seg(seg, u, v)
    return any(obj == hole_id for obj, _ in hits)


def _peg_pixel_visible(
    seg: np.ndarray,
    u: float,
    v: float,
    robot_id: int,
    peg_link: int,
) -> bool:
    hits = _sample_seg(seg, u, v)
    return any(obj == robot_id and link == peg_link for obj, link in hits)


def apply_label_visibility(
    kps: list[ImageKeypoints],
    cam: ProjectorCam,
    seg: np.ndarray,
    *,
    hole_id: int,
    robot_id: int,
    peg_link: int,
    peg_hide_index0: bool = True,
) -> list[ImageKeypoints]:
    """Realistic v for YOLO: out-of-frame or wrong seg → not visible."""
    out: list[ImageKeypoints] = []
    for kp in kps:
        uv = list(kp.uv)
        visible = list(kp.visible)
        for i, (u, v) in enumerate(kp.uv):
            if kp.name == "peg" and peg_hide_index0 and i == 0:
                visible[i] = False
                continue
            if not _in_frame(cam, u, v):
                visible[i] = False
                continue
            if kp.name == "hole":
                visible[i] = _hole_pixel_visible(seg, u, v, hole_id)
            elif kp.name == "peg":
                visible[i] = _peg_pixel_visible(seg, u, v, robot_id, peg_link)
            else:
                visible[i] = False
        out.append(replace(kp, uv=uv, visible=visible))
    return out
