"""Teach bundle — fixed_cam or wrist2, manual corners (simulated)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from constants import PEG_W, PEG_H
from template_flow._paths import TEACH_DIR
from vision.teach_image import capture_teach_gray, save_teach_gray_png
from visp_flow.ibvs_coarse import IbvsDesiredFeatures
from template_flow.match_teach import build_roi_crop_ref, gray_to_bgr
from template_flow.tuning import ZONE_MARGIN
from vision.corners import ImageKeypoints, ProjectorCam, apply_infer_corner0, gt_image_keypoints

TEACH_STEM = "template_teach"
CameraMode = Literal["fixed", "wrist2"]


@dataclass(frozen=True)
class TemplateBundle:
    gray: object
    ref_bgr: object
    hole_uv: tuple[tuple[float, float], ...]
    peg_uv: tuple[tuple[float, float], ...]
    ibvs_desired: IbvsDesiredFeatures | None
    width: int
    height: int
    hole_xy: tuple[float, float]
    roi_crop_ref: object
    camera: CameraMode = "fixed"


def simulate_manual_mark_at_teach(
    cam: ProjectorCam,
    robot_id: int,
    peg: int,
    hole_id: int,
) -> tuple[ImageKeypoints, ImageKeypoints] | None:
    gt = gt_image_keypoints(cam, robot_id, peg, hole_id, infer_corner0=False)
    hole_kp = next(s for s in gt if s.name == "hole")
    peg_gt = next(s for s in gt if s.name == "peg")
    peg_kp = apply_infer_corner0(peg_gt, cam, PEG_W / 2.0, PEG_H / 2.0)
    if not all(hole_kp.visible) or not all(peg_kp.visible[i] for i in (1, 2, 3)):
        return None
    return hole_kp, peg_kp


def capture_template_bundle(
    cam: ProjectorCam,
    robot_id: int,
    peg: int,
    hole_id: int,
    hole_xy: tuple[float, float],
    *,
    camera: CameraMode = "fixed",
    gui: bool = False,
    gray=None,
) -> TemplateBundle | None:
    marked = simulate_manual_mark_at_teach(cam, robot_id, peg, hole_id)
    if marked is None:
        return None
    hole_kp, peg_kp = marked
    if gray is None:
        gray = capture_teach_gray(cam, gui=gui, warmup=2)
    ref_bgr = gray_to_bgr(gray)
    if camera == "wrist2":
        peg_bottom = max(v for _, v in peg_kp.uv)
        roi_crop_ref = build_roi_crop_ref(
            ref_bgr,
            tuple(hole_kp.uv),
            mask_above_y=peg_bottom + ZONE_MARGIN,
        )
    else:
        roi_crop_ref = build_roi_crop_ref(ref_bgr, tuple(peg_kp.uv))
    return TemplateBundle(
        gray=gray,
        ref_bgr=ref_bgr,
        hole_uv=tuple(hole_kp.uv),
        peg_uv=tuple(peg_kp.uv),
        ibvs_desired=None,
        width=cam.width,
        height=cam.height,
        hole_xy=hole_xy,
        roi_crop_ref=roi_crop_ref,
        camera=camera,
    )


def save_template_bundle(
    bundle: TemplateBundle,
    metrics: dict[str, Any] | None = None,
    *,
    out_dir: str | None = None,
    tracker: str | None = None,
) -> tuple[str, str]:
    out_dir = out_dir or TEACH_DIR
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{TEACH_STEM}.png")
    json_path = os.path.join(out_dir, f"{TEACH_STEM}.json")
    save_teach_gray_png(bundle.gray, png_path)
    if tracker is None:
        if bundle.camera == "wrist2":
            tracker = "peg=teach_uv; hole=XFeat ROI (realtime_demo.py)"
        else:
            tracker = "hole=teach_uv; peg=XFeat ROI (realtime_demo.py)"
    meta: dict[str, Any] = {
        "template_png": os.path.basename(png_path),
        "camera": bundle.camera,
        "width": bundle.width,
        "height": bundle.height,
        "hole_xy": [float(bundle.hole_xy[0]), float(bundle.hole_xy[1])],
        "hole_uv": [{"u": u, "v": v} for u, v in bundle.hole_uv],
        "peg_uv": [{"u": u, "v": v} for u, v in bundle.peg_uv],
        "peg_uv_marked": [{"u": u, "v": v} for u, v in bundle.peg_uv[1:4]],
        "captured": "manual_mark_sim_at_aligned_pose",
        "tracker": tracker,
    }
    if metrics:
        meta["metrics"] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  template I* saved: {png_path}")
    return png_path, json_path
