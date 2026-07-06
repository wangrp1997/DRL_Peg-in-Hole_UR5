"""Template vs GT — XFeat ROI diagnostics (simulation only)."""
from __future__ import annotations

import random

import pybullet as p

from constants import CORNER_SERVO_STANDOFF
from sim.cartesian_align import run_cartesian_align
from sim.fixed_render import render_grayscale as render_fixed_gray
from sim.perturbation import apply_tip_perturbation, format_perturbation_log, sample_perturbation6
from sim.scene import (
    connect,
    get_fixed_camera,
    load_scene,
    move_tip_to_standoff,
    register_wrist_camera2,
    setup_wrist_camera2_views,
)
from sim.wrist2_render import render_grayscale as render_wrist2_gray
from sim.wrist_camera2 import attach_wrist_camera2
from template_flow.keypoints import keypoints_from_template
from template_flow.teach import CameraMode, capture_template_bundle
from visp_flow.visp_constants import COARSE_STANDOFF
from vision.corners import gt_image_keypoints


def _print_compare(name: str, est_kp, gt_kp) -> tuple[float, float]:
    errs = [
        ((e[0] - g[0]) ** 2 + (e[1] - g[1]) ** 2) ** 0.5
        for e, g in zip(est_kp.uv, gt_kp.uv)
    ]
    print(f"  {name}:")
    for i, e in enumerate(errs):
        inf = " (inferred)" if gt_kp.inferred[i] else ""
        print(
            f"    [{i}] err={e:5.1f}px  "
            f"est=({est_kp.uv[i][0]:6.1f},{est_kp.uv[i][1]:6.1f})  "
            f"gt=({gt_kp.uv[i][0]:6.1f},{gt_kp.uv[i][1]:6.1f}){inf}"
        )
    return max(errs), sum(errs) / 4.0


def run_corner_diag(
    *,
    seed: int = 42,
    gui: bool = False,
    rng: random.Random | None = None,
    camera: CameraMode = "fixed",
) -> bool:
    rng = rng or random.Random(seed)
    use_wrist2 = camera == "wrist2"
    standoff = COARSE_STANDOFF if use_wrist2 else CORNER_SERVO_STANDOFF

    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(
        gui, fixed_cam=not use_wrist2,
    )
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]

    cam = None
    if use_wrist2:
        cam = attach_wrist_camera2(robot_id, eef, hole_xy)
        if gui:
            register_wrist_camera2(cam)
            setup_wrist_camera2_views(True, False)
    else:
        cam = get_fixed_camera()

    if cam is None:
        print("camera missing")
        p.disconnect()
        return False

    move_tip_to_standoff(robot_id, eef, arm, peg, hole_xy, standoff, gui=gui)
    run_cartesian_align(robot_id, arm, peg, hole_xy, hole_orn, gui=gui)
    bundle = capture_template_bundle(
        cam, robot_id, peg, hole_id, hole_xy, camera=camera, gui=gui,
    )
    if bundle is None:
        print("teach bundle failed")
        p.disconnect()
        return False

    move_tip_to_standoff(robot_id, eef, arm, peg, hole_xy, standoff, gui=gui)
    perturb = sample_perturbation6(rng)
    apply_tip_perturbation(robot_id, eef, arm, peg, hole_xy, hole_orn, perturb, gui=gui)
    print(format_perturbation_log(perturb))

    gray = render_wrist2_gray(cam, gui=gui, warmup=2) if use_wrist2 else render_fixed_gray(cam, gui=gui, warmup=2)
    est = keypoints_from_template(bundle, gray, cam)
    gt = gt_image_keypoints(cam, robot_id, peg, hole_id)
    if est is None:
        print("XFeat ROI homography failed (provider None)")
        p.disconnect()
        return False

    est_hole = next(s for s in est if s.name == "hole")
    est_peg = next(s for s in est if s.name == "peg")
    gt_hole = next(s for s in gt if s.name == "hole")
    gt_peg = next(s for s in gt if s.name == "peg")

    print(f"\n=== template vs GT  (seed={seed}, {camera}) ===")
    if use_wrist2:
        mx_p, _ = _print_compare("peg (teach_uv)", est_peg, gt_peg)
        mx_h, _ = _print_compare("hole (XFeat ROI)", est_hole, gt_hole)
    else:
        mx_h, _ = _print_compare("hole (teach_uv)", est_hole, gt_hole)
        mx_p, _ = _print_compare("peg (XFeat ROI + infer0)", est_peg, gt_peg)
    print(f"\nmax err hole={mx_h:.1f}px peg={mx_p:.1f}px")
    p.disconnect()
    return True
