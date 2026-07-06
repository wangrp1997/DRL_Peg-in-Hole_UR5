"""DeepLSD line segments + rectangle geometry (not implemented yet)."""
from __future__ import annotations

from collections.abc import Callable

from vision.corners import ImageKeypoints, ProjectorCam


def make_deeplsd_provider(
    cam: ProjectorCam,
    robot_id: int,
    peg: int,
    hole_id: int,
    **_,
) -> Callable[[], list[ImageKeypoints] | None]:
    del cam, robot_id, peg, hole_id
    raise NotImplementedError("--corners deeplsd: not implemented yet")
