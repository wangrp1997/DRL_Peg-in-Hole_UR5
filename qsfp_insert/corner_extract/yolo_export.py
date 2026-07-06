"""ImageKeypoints → Ultralytics YOLO pose label lines."""
from __future__ import annotations

from vision.corners import ImageKeypoints

HOLE_CLASS = 0
PEG_CLASS = 1


def _bbox_norm(kp: ImageKeypoints, width: int, height: int) -> tuple[float, float, float, float] | None:
    xs = [u for (u, v), ok in zip(kp.uv, kp.visible) if ok]
    ys = [v for (u, v), ok in zip(kp.uv, kp.visible) if ok]
    if not xs:
        xs = [u for u, v in kp.uv]
        ys = [v for u, v in kp.uv]
    if not xs:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 4.0
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = min(float(width - 1), x1 + pad)
    y1 = min(float(height - 1), y1 + pad)
    cx = (x0 + x1) / 2.0 / width
    cy = (y0 + y1) / 2.0 / height
    bw = max((x1 - x0) / width, 1e-4)
    bh = max((y1 - y0) / height, 1e-4)
    return cx, cy, bw, bh


def keypoints_to_yolo_line(
    class_id: int,
    kp: ImageKeypoints,
    width: int,
    height: int,
    *,
    require_visible: bool = False,
) -> str | None:
    if require_visible and not any(kp.visible):
        return None
    box = _bbox_norm(kp, width, height)
    if box is None:
        return None
    cx, cy, bw, bh = box
    parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for (u, v), ok in zip(kp.uv, kp.visible):
        if ok:
            parts.append(f"{u / width:.6f}")
            parts.append(f"{v / height:.6f}")
            parts.append("2")
        else:
            parts.append("0.0")
            parts.append("0.0")
            parts.append("0")
    return " ".join(parts)


def keypoint_sets_to_yolo_lines(
    sets: list[ImageKeypoints],
    width: int,
    height: int,
    *,
    include_hole: bool = True,
) -> list[str]:
    lines: list[str] = []
    for kp in sets:
        if kp.name == "hole":
            if not include_hole:
                continue
            line = keypoints_to_yolo_line(HOLE_CLASS, kp, width, height)
        elif kp.name == "peg":
            line = keypoints_to_yolo_line(PEG_CLASS, kp, width, height, require_visible=True)
        else:
            continue
        if line is not None:
            lines.append(line)
    return lines
