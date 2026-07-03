"""Draw GT keypoints on OpenCV BGR images."""
from __future__ import annotations

from vision.corners import INFERRED_COLOR_BGR, ImageKeypoints


def draw_keypoints_on_bgr(bgr, sets: list[ImageKeypoints]) -> None:
    """Draw circles/labels in-place on a BGR image (single RGB→BGR conversion upstream)."""
    import cv2

    for s in sets:
        for i, ((u, v), ok) in enumerate(zip(s.uv, s.visible)):
            if not ok:
                continue
            pt = (int(round(u)), int(round(v)))
            color = INFERRED_COLOR_BGR if s.inferred[i] else s.color_bgr
            if s.inferred[i]:
                cv2.drawMarker(bgr, pt, color, cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
            else:
                cv2.circle(bgr, pt, 6, color, 2, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"{s.prefix}{i}",
                (pt[0] + 7, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
