"""Ground-truth corners from simulation 3D model projection."""
from __future__ import annotations

from collections.abc import Callable

from vision.corners import ImageKeypoints, ProjectorCam, gt_image_keypoints


def make_gt_provider(
    cam: ProjectorCam,
    robot_id: int,
    peg: int,
    hole_id: int,
    *,
    infer_corner0: bool = False,
    **_,
) -> Callable[[], list[ImageKeypoints] | None]:
    def provider() -> list[ImageKeypoints] | None:
        return gt_image_keypoints(
            cam, robot_id, peg, hole_id, infer_corner0=infer_corner0,
        )

    return provider
