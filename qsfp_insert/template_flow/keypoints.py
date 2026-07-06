"""Runtime corners — fixed: hole=teach, peg=XFeat; wrist2: peg=teach, hole=XFeat."""
from __future__ import annotations

from sim.fixed_camera import FixedCamera
from sim.fixed_render import render_grayscale as render_fixed_gray
from sim.wrist2_render import render_grayscale as render_wrist2_gray
from sim.wrist_camera2 import WristCamera2
from template_flow.match_teach import (
    estimate_homography_roi,
    gray_to_bgr,
    uv_in_image,
    warp_uv_points,
)
from template_flow.teach import CameraMode, TemplateBundle
from template_flow.tuning import (
    COARSE_MAX_PROVIDER_CALLS,
    MATCH_EVERY,
)
from vision.corners import (
    ImageKeypoints,
    ProjectorCam,
    apply_infer_corner0,
    hole_corners_world,
    peg_tip_corners_world,
    project_corners,
)
from constants import PEG_W, PEG_H

TemplateCam = ProjectorCam


def _max_corner_shift(
    teach_uv: tuple[tuple[float, float], ...],
    curr_uv: list[tuple[float, float]],
) -> float:
    return max(
        ((u - u0) ** 2 + (v - v0) ** 2) ** 0.5
        for (u0, v0), (u, v) in zip(teach_uv, curr_uv)
    )


def warp_shift_sane(bundle: TemplateBundle, kps: list[ImageKeypoints]) -> bool:
    """Reject wild homography jumps on the warped corner set only."""
    from template_flow.tuning import SANITY_MAX_CORNER_SHIFT_PX

    if bundle.camera == "wrist2":
        hole = next(s for s in kps if s.name == "hole")
        return _max_corner_shift(bundle.hole_uv, list(hole.uv)) <= SANITY_MAX_CORNER_SHIFT_PX
    peg = next(s for s in kps if s.name == "peg")
    return _max_corner_shift(bundle.peg_uv, list(peg.uv)) <= SANITY_MAX_CORNER_SHIFT_PX


def _render_gray(cam: TemplateCam, *, gui: bool, warmup: int = 1):
    if isinstance(cam, WristCamera2):
        return render_wrist2_gray(cam, gui=gui, warmup=warmup)
    return render_fixed_gray(cam, gui=gui, warmup=warmup)


def teach_keypoints_from_bundle(bundle: TemplateBundle) -> list[ImageKeypoints]:
    return [
        ImageKeypoints("hole", "H", (0, 220, 0), bundle.hole_uv, [True] * 4),
        ImageKeypoints("peg", "P", (0, 80, 255), bundle.peg_uv, [True] * 4),
    ]


def keypoints_from_template(
    bundle: TemplateBundle,
    current_gray,
    cam: TemplateCam,
    *,
    allow_slide: bool = True,
) -> list[ImageKeypoints] | None:
    cur_bgr = gray_to_bgr(current_gray)
    H = estimate_homography_roi(
        bundle.ref_bgr, cur_bgr, bundle.roi_crop_ref, allow_slide=allow_slide,
    )
    if H is None:
        return None

    if bundle.camera == "wrist2":
        hole_uv = warp_uv_points(H, bundle.hole_uv)
        if not uv_in_image(hole_uv, cam.width, cam.height):
            return None
        hole_kp = ImageKeypoints("hole", "H", (0, 220, 0), hole_uv, [True] * 4)
        peg_kp = ImageKeypoints("peg", "P", (0, 80, 255), list(bundle.peg_uv), [True] * 4)
        return [hole_kp, peg_kp]

    hole_kp = ImageKeypoints(
        "hole", "H", (0, 220, 0), list(bundle.hole_uv), [True] * 4,
    )
    peg_uv: list[tuple[float, float]] = [(0.0, 0.0)] * 4
    for i in (1, 2, 3):
        peg_uv[i] = warp_uv_points(H, [bundle.peg_uv[i]])[0]
    peg_raw = ImageKeypoints("peg", "P", (0, 80, 255), peg_uv, [False, True, True, True])
    peg_kp = apply_infer_corner0(peg_raw, cam, PEG_W / 2.0, PEG_H / 2.0)
    if not all(peg_kp.visible) or not uv_in_image(list(peg_kp.uv), cam.width, cam.height):
        return None
    return [hole_kp, peg_kp]


def project_tracked_keypoints(
    bundle: TemplateBundle,
    cam: TemplateCam,
    robot_id: int,
    peg: int,
    hole_id: int,
) -> list[ImageKeypoints] | None:
    """Kinematic fill-in between XFeat rematches — teach static corner + project moving set."""
    if bundle.camera == "wrist2":
        hole_uv, hole_vis = project_corners(cam, hole_corners_world(hole_id))
        if sum(hole_vis) < 4:
            return None
        return [
            ImageKeypoints("hole", "H", (0, 220, 0), hole_uv, hole_vis),
            ImageKeypoints("peg", "P", (0, 80, 255), list(bundle.peg_uv), [True] * 4),
        ]
    peg_uv, peg_vis = project_corners(cam, peg_tip_corners_world(robot_id, peg))
    peg_raw = ImageKeypoints("peg", "P", (0, 80, 255), peg_uv, peg_vis)
    peg_kp = apply_infer_corner0(peg_raw, cam, PEG_W / 2.0, PEG_H / 2.0)
    if not all(peg_kp.visible[i] for i in (1, 2, 3)):
        return None
    return [
        ImageKeypoints("hole", "H", (0, 220, 0), list(bundle.hole_uv), [True] * 4),
        peg_kp,
    ]


class TemplateKeypointTracker:
    def __init__(
        self,
        bundle: TemplateBundle,
        cam: TemplateCam,
        *,
        gui: bool = False,
        match_every: int = MATCH_EVERY,
        max_provider_calls: int = COARSE_MAX_PROVIDER_CALLS,
        robot_id: int | None = None,
        peg: int | None = None,
        hole_id: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.cam = cam
        self.gui = gui
        self.match_every = max(1, match_every)
        self.max_provider_calls = max_provider_calls
        self._robot_id = robot_id
        self._peg = peg
        self._hole_id = hole_id
        self._kps: list[ImageKeypoints] | None = None
        self._provider_calls = 0
        self._match_step = 0
        self._fresh_render = False

    def __call__(self) -> list[ImageKeypoints] | None:
        return self.track()

    def last(self) -> list[ImageKeypoints] | None:
        return self._kps

    def _kinematic_keypoints(self) -> list[ImageKeypoints] | None:
        if self._robot_id is None or self._peg is None or self._hole_id is None:
            return self._kps
        return project_tracked_keypoints(
            self.bundle, self.cam, self._robot_id, self._peg, self._hole_id,
        )

    def track(self, *, gray=None) -> list[ImageKeypoints] | None:
        self._provider_calls += 1
        if self._provider_calls > self.max_provider_calls:
            return None
        self._match_step += 1
        need_xfeat = self._match_step == 1 or self._match_step % self.match_every == 0

        if need_xfeat:
            if gray is None:
                if self.gui:
                    return self._kps
                gray = _render_gray(self.cam, gui=False, warmup=1)
            new_kps = keypoints_from_template(
                self.bundle,
                gray,
                self.cam,
                allow_slide=(self._match_step == 1),
            )
            if new_kps is not None and warp_shift_sane(self.bundle, new_kps):
                self._kps = new_kps
            elif self._kps is None:
                return None

        return self._kps

    def mark_fresh_render(self) -> None:
        self._fresh_render = True

    def consume_fresh_render(self) -> bool:
        """True if track() just rendered — GUI can reuse wrist2 cache."""
        fresh = self._fresh_render
        self._fresh_render = False
        return fresh

    def display_keypoints(self) -> list[ImageKeypoints] | None:
        """OpenCV overlay — last XFeat match, or teach corners before first match."""
        if self._kps is not None:
            return self._kps
        return teach_keypoints_from_bundle(self.bundle)

    def track_for_display(self) -> list[ImageKeypoints] | None:
        return self.display_keypoints()


def make_template_keypoint_tracker(
    bundle: TemplateBundle,
    cam: TemplateCam,
    *,
    gui: bool = False,
    match_every: int = MATCH_EVERY,
    robot_id: int | None = None,
    peg: int | None = None,
    hole_id: int | None = None,
) -> TemplateKeypointTracker:
    return TemplateKeypointTracker(
        bundle,
        cam,
        gui=gui,
        match_every=match_every,
        robot_id=robot_id,
        peg=peg,
        hole_id=hole_id,
    )
