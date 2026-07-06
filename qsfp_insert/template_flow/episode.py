"""Template coarse episode — fixed_cam or wrist2 + XFeat template corners."""
from __future__ import annotations

import random
from typing import Any, Literal

import pybullet as p

from constants import CORNER_SERVO_STANDOFF, PERTURB_SERVO_PAUSE_S, SETTLE_IK_STEPS, SETTLE_IK_STEPS_GUI
from geometry import alignment_metrics, metrics_converged, peg_tip_world
from sim.cartesian_align import run_cartesian_align
from sim.gui_preview import gray_from_refresh_cache, is_pybullet_connected, pause_gui
from sim.perturbation import (
    Perturbation6,
    apply_tip_perturbation,
    format_perturbation_log,
    sample_perturbation6,
)
from sim.scene import (
    close_camera_windows,
    connect,
    get_fixed_camera,
    load_scene,
    move_tip_to_standoff,
    refresh_camera_views,
    register_wrist_camera2,
    set_hole_opaque,
    setup_wrist_camera2_views,
)
from sim.wrist_camera2 import attach_wrist_camera2
from template_flow.keypoints import make_template_keypoint_tracker
from template_flow.teach import CameraMode, capture_template_bundle, save_template_bundle
from template_flow.tuning import MATCH_EVERY, MATCH_EVERY_GUI
from visp_flow.kabsch_coarse import run_kabsch_coarse
from visp_flow.visp_constants import COARSE_STANDOFF
from vision.corners import gt_image_keypoints

CoarseMethod = Literal["kabsch", "ibvs"]
PHASE_PAUSE_S = PERTURB_SERVO_PAUSE_S


def template_flow_episode(
    gui: bool = False,
    hole_xy: tuple[float, float] | None = None,
    opencv_render: bool = False,
    perturb: Perturbation6 | None = None,
    rng: random.Random | None = None,
    coarse_method: CoarseMethod = "kabsch",
    teach_dir: str | None = None,
    camera: CameraMode = "fixed",
    gui_idle: bool = False,
) -> tuple[bool, dict[str, Any], tuple[float, float], tuple | None]:
    if coarse_method == "ibvs" and camera == "fixed":
        print("template_flow 固定相机仅支持 kabsch coarse（IBVS 需 wrist2）")
        coarse_method = "kabsch"

    use_wrist2 = camera == "wrist2"
    standoff = COARSE_STANDOFF if use_wrist2 else CORNER_SERVO_STANDOFF

    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(
        gui,
        hole_xy=hole_xy,
        opencv_render=opencv_render,
        fixed_cam=not use_wrist2,
    )
    if gui:
        set_hole_opaque(hole_id)
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]
    settle_gui = SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS

    wrist_cam = None
    if use_wrist2:
        wrist_cam = attach_wrist_camera2(robot_id, eef, hole_xy)
        if gui:
            register_wrist_camera2(wrist_cam)
            setup_wrist_camera2_views(True, opencv_render)

    provider_holder: list = []
    teach_path = ""

    def _active_cam():
        return wrist_cam if use_wrist2 else get_fixed_camera()

    def _overlay_kps():
        if provider_holder:
            return provider_holder[0].display_keypoints()
        cam = _active_cam()
        if cam is None:
            return None
        return gt_image_keypoints(cam, robot_id, peg, hole_id)

    def _template_provider():
        if not is_pybullet_connected() or not provider_holder:
            return None
        tracker = provider_holder[0]
        if not gui:
            return tracker.track()
        cam = _active_cam()
        will_match = (
            tracker._kps is None
            or (tracker._match_step + 1) % tracker.match_every == 0
        )
        if will_match and cam is not None:
            refresh_camera_views(_overlay_kps, render=True)
            gray = gray_from_refresh_cache(cam)
            if gray is not None:
                tracker.mark_fresh_render()
                return tracker.track(gray=gray)
        return tracker.track()

    def _on_frame_corners():
        if not gui or not is_pybullet_connected():
            return
        render = True
        if provider_holder:
            render = not provider_holder[0].consume_fresh_render()
        refresh_camera_views(_overlay_kps, render=render)

    move_tip_to_standoff(
        robot_id, eef, arm, peg, hole_xy, standoff, gui=gui, settle_steps=settle_gui,
    )
    _on_frame_corners()

    teach_ok, m_align = run_cartesian_align(
        robot_id, arm, peg, hole_xy, hole_orn, gui=gui, on_step=_on_frame_corners if gui else None,
    )
    print(f"teach visit (cartesian align): ok={teach_ok}")

    cam = _active_cam()
    if teach_ok and cam is not None:
        if gui:
            print("示教：构建 XFeat ROI 模板（accelerated_features）…")
        bundle = capture_template_bundle(
            cam, robot_id, peg, hole_id, hole_xy, camera=camera, gui=gui,
        )
        if bundle is not None:
            _, teach_path = save_template_bundle(bundle, m_align, out_dir=teach_dir)
            _on_frame_corners()
            label = "wrist2" if use_wrist2 else "固定相机"
            if gui:
                pause_gui(PHASE_PAUSE_S, f"示教完成（{label} + XFeat template）", _on_frame_corners)
            tracker = make_template_keypoint_tracker(
                bundle, cam, gui=gui, match_every=MATCH_EVERY_GUI if gui else MATCH_EVERY,
                robot_id=robot_id, peg=peg, hole_id=hole_id,
            )
            provider_holder.clear()
            provider_holder.append(tracker)
            print(f"template teach saved: {teach_path}")
        else:
            print("teach skipped: manual mark failed")
    else:
        print("teach skipped: align failed or no camera")

    if not is_pybullet_connected():
        close_camera_windows()
        return False, {"teach_ok": teach_ok, "coarse_ok": False}, hole_xy, None

    move_tip_to_standoff(
        robot_id, eef, arm, peg, hole_xy, standoff, gui=gui, settle_steps=settle_gui,
    )
    _on_frame_corners()

    if perturb is None and rng is not None:
        perturb = sample_perturbation6(rng)
    if perturb is not None:
        apply_tip_perturbation(
            robot_id, eef, arm, peg, hole_xy, hole_orn, perturb, gui=gui, settle_steps=settle_gui,
        )
        _on_frame_corners()
        print(format_perturbation_log(perturb))

    pause_gui(
        PHASE_PAUSE_S,
        f"准备开始第一阶段伺服（{camera} 模板 / {coarse_method}）…",
        _on_frame_corners if gui else None,
    )

    coarse_ok = False
    coarse_metrics = None
    cam = _active_cam()
    if cam is not None and provider_holder and is_pybullet_connected():
        if gui:
            print("XFeat 首帧匹配（绘制 OpenCV 角点）…")
            refresh_camera_views(_overlay_kps)
            gray = gray_from_refresh_cache(cam)
            if gray is not None:
                provider_holder[0].mark_fresh_render()
                provider_holder[0].track(gray=gray)
        coarse_ok, coarse_metrics = run_kabsch_coarse(
            robot_id,
            peg,
            arm,
            cam,
            hole_xy,
            hole_orn,
            _template_provider,
            gui=gui,
            on_step=_on_frame_corners if gui else None,
        )
    print(f"第一阶段结束: coarse (kabsch) ok={coarse_ok}")

    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    mf = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    gt_ok = metrics_converged(mf)

    report: dict[str, Any] = {
        "mode": f"template_coarse_{camera}",
        "camera": camera,
        "coarse_method": coarse_method,
        "teach_ok": teach_ok,
        "coarse_ok": coarse_ok,
        "teach_path": teach_path,
        "metrics": mf,
        "gt_aligned": gt_ok,
    }
    if coarse_metrics is not None:
        report["coarse_metrics"] = coarse_metrics

    if gui and gui_idle and coarse_ok:
        pause_gui(20.0, "粗对准完成，查看 XFeat 角点…", _on_frame_corners)

    live = None
    if gui and gui_idle and provider_holder:
        live = (robot_id, peg, hole_id, use_wrist2, provider_holder[0])
    if live is None:
        if gui:
            close_camera_windows()
        if use_wrist2 and wrist_cam is not None:
            wrist_cam.detach()
        if is_pybullet_connected():
            p.disconnect()
    return coarse_ok, report, hole_xy, live
