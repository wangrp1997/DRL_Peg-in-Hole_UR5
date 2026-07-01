"""Draw GT keypoints on OpenCV BGR images."""
from __future__ import annotations

import numpy as np

from vision.corners import ImageKeypoints


def draw_keypoints_on_bgr(bgr: np.ndarray, sets: list[ImageKeypoints]) -> None:
    """Draw circles/labels in-place on a BGR image (single RGB→BGR conversion upstream)."""
    import cv2

    for s in sets:
        for i, ((u, v), ok) in enumerate(zip(s.uv, s.visible)):
            if not ok:
                continue
            pt = (int(round(u)), int(round(v)))
            cv2.circle(bgr, pt, 6, s.color_bgr, 2, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"{s.prefix}{i}",
                (pt[0] + 7, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                s.color_bgr,
                1,
                cv2.LINE_AA,
            )
