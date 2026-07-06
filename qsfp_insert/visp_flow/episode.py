"""ViSP baseline episode — same rhythm as corner_servo demo."""
from __future__ import annotations

import random
from typing import Any, Literal

import pybullet as p

from constants import PERTURB_SERVO_PAUSE_S, PLATE_TOP_Z, SETTLE_IK_STEPS, SETTLE_IK_STEPS_GUI
from geometry import alignment_metrics, metrics_converged, peg_tip_world
from sim.cartesian_align import run_cartesian_align
from sim.perturbation import (
    Perturbation6,
    apply_tip_perturbation,
    format_perturbation_log,
    sample_perturbation6,
)
from sim.gui_preview import is_pybullet_connected, pause_gui
from sim.scene import (
    close_camera_windows,
    connect,
    load_scene,
    move_tip_to_standoff,
    refresh_camera_views,
    register_wrist_camera2,
    run_insert_after_align,
    set_hole_opaque,
    setup_wrist_camera2_views,
)
from visp_flow.visp_constants import COARSE_STANDOFF, VISP_DVS_START_ERR
from visp_flow.dvs_fine import run_visp_dvs_fine
from visp_flow.ibvs_coarse import run_pnp_preflight_for_ibvs, run_visp_ibvs_coarse
from visp_flow.kabsch_coarse import run_kabsch_coarse
from visp_flow.require_visp import require_visp_python
from visp_flow.teach import capture_aligned_teach, save_teach_bundle
from vision.corners import gt_image_keypoints
from vision.align import metrics_from_keypoints
from sim.wrist_camera2 import attach_wrist_camera2

CoarseMethod = Literal["kabsch", "ibvs"]
PHASE_PAUSE_S = PERTURB_SERVO_PAUSE_S  # 3 s


def visp_flow_episode(
    gui: bool = False,
    hole_xy: tuple[float, float] | None = None,
    opencv_render: bool = False,
    perturb: Perturbation6 | None = None,
    rng: random.Random | None = None,
    coarse_method: CoarseMethod = "kabsch",
    teach_dir: str | None = None,
    always_run_dvs: bool = False,
    dvs_start_err: float | None = None,
    dvs_abort_err: float | None = None,
    gui_idle: bool = False,
    insert: bool = False,
) -> tuple[bool, dict[str, Any], tuple[float, float], tuple[int, int, int, bool] | None]:
    require_visp_python()

    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(
        gui, hole_xy=hole_xy, opencv_render=opencv_render,
    )
    if gui:
        set_hole_opaque(hole_id)
    hole_orn = p.getBasePositionAndOrientation(hole_id)[1]
    settle_gui = SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS

    wrist_cam = attach_wrist_camera2(robot_id, eef, hole_xy)
    if gui:
        register_wrist_camera2(wrist_cam)
        setup_wrist_camera2_views(True, opencv_render)

    def _provider():
        return gt_image_keypoints(wrist_cam, robot_id, peg, hole_id)

    def _on_frame_corners():
        if gui and is_pybullet_connected():
            refresh_camera_views(_provider)

    def _on_frame_cam_only():
        if gui and is_pybullet_connected():
            refresh_camera_views(None, render=False)

    # A) IK standoff（episode 初始位）
    move_tip_to_standoff(
        robot_id, eef, arm, peg, hole_xy, COARSE_STANDOFF, gui=gui, settle_steps=settle_gui
    )
    _on_frame_corners()

    # B) 仅去对准位采 I*（不启动伺服流程）
    teach_ok, m_align = run_cartesian_align(
        robot_id, arm, peg, hole_xy, hole_orn, gui=gui, on_step=_on_frame_corners if gui else None
    )
    print(f"teach visit (cartesian align): ok={teach_ok}")

    teach_path = ""
    dvs_target = None
    dvs_plane_z = 0.10
    ibvs_desired = None
    if teach_ok:
        kps0 = _provider()
        hole_kp = next(s for s in kps0 if s.name == "hole")
        peg_kp = next(s for s in kps0 if s.name == "peg")
        ibvs_desired, dvs_target, dvs_plane_z = capture_aligned_teach(
            wrist_cam, hole_kp, peg_kp, robot_id, peg, hole_xy, gui=gui
        )
        if ibvs_desired is not None:
            _, teach_path = save_teach_bundle(
                ibvs_desired, dvs_target, hole_xy, m_align, dvs_plane_z=dvs_plane_z, out_dir=teach_dir
            )
            print(f"visp teach saved: {teach_path}")
        else:
            print("teach skipped: IBVS corner assignment failed")
    else:
        print("teach skipped: align failed at teach pose")

    # C) 回到 IK standoff，再扰动（与 corner_servo demo 一致）
    move_tip_to_standoff(
        robot_id, eef, arm, peg, hole_xy, COARSE_STANDOFF, gui=gui, settle_steps=settle_gui
    )
    _on_frame_corners()

    if perturb is None and rng is not None:
        perturb = sample_perturbation6(rng)
    if perturb is not None:
        apply_tip_perturbation(
            robot_id, eef, arm, peg, hole_xy, hole_orn, perturb, gui=gui, settle_steps=settle_gui
        )
        _on_frame_corners()
        print(format_perturbation_log(perturb))

    # D) 第一阶段：粗对准（角点可视化 ON）
    pause_gui(
        PHASE_PAUSE_S,
        f"准备开始第一阶段伺服（粗对准 / {coarse_method}）…",
        _on_frame_corners if gui else None,
    )

    coarse_ok = False
    ibvs_err = 0.0
    coarse_metrics = None
    if ibvs_desired is not None:
        if coarse_method == "ibvs":
            pre_ok, pre_px, pre_gt = run_pnp_preflight_for_ibvs(
                robot_id,
                peg,
                arm,
                wrist_cam,
                hole_xy,
                hole_orn,
                _provider,
                gui=gui,
                on_step=_on_frame_corners if gui else None,
            )
            standoff = peg_tip_world(robot_id, peg)[2] - PLATE_TOP_Z
            kps_pre = _provider()
            m_pre = None
            if kps_pre is not None:
                m_pre = metrics_from_keypoints(
                    kps_pre, wrist_cam, hole_xy, hole_orn, standoff_hint=standoff
                )
            if pre_ok:
                print(
                    f"IBVS preflight: px={pre_px:.1f} → 纯 ViSP IBVS "
                    f"(GT aligned={pre_gt or (m_pre and metrics_converged(m_pre))})"
                )
                coarse_ok, ibvs_err = run_visp_ibvs_coarse(
                    robot_id,
                    eef,
                    peg,
                    arm,
                    wrist_cam,
                    ibvs_desired,
                    _provider,
                    hole_xy,
                    hole_orn,
                    gui=gui,
                    on_step=_on_frame_corners if gui else None,
                )
            elif pre_gt:
                print(f"IBVS preflight: PnP 在 px>{pre_px:.1f} 时 GT 已收敛，仍跑纯 IBVS")
                coarse_ok, ibvs_err = run_visp_ibvs_coarse(
                    robot_id,
                    eef,
                    peg,
                    arm,
                    wrist_cam,
                    ibvs_desired,
                    _provider,
                    hole_xy,
                    hole_orn,
                    gui=gui,
                    on_step=_on_frame_corners if gui else None,
                )
            else:
                print(f"IBVS preflight: 未进入启动包络 (px≈{pre_px:.1f})，仍尝试纯 IBVS")
                coarse_ok, ibvs_err = run_visp_ibvs_coarse(
                    robot_id,
                    eef,
                    peg,
                    arm,
                    wrist_cam,
                    ibvs_desired,
                    _provider,
                    hole_xy,
                    hole_orn,
                    gui=gui,
                    on_step=_on_frame_corners if gui else None,
                )
        else:
            coarse_ok, coarse_metrics = run_kabsch_coarse(
                robot_id,
                peg,
                arm,
                wrist_cam,
                hole_xy,
                hole_orn,
                _provider,
                gui=gui,
                on_step=_on_frame_corners if gui else None,
            )
    print(f"第一阶段结束: coarse ({coarse_method}) ok={coarse_ok}")

    dvs_ok = False
    dvs_err = 0.0
    dvs_gated = False
    dvs_skipped = coarse_ok and not always_run_dvs
    if dvs_skipped:
        print("第二阶段跳过: 第一阶段已对准")
        if gui and gui_idle and not insert:
            pause_gui(20.0, "粗对准完成（跳过 DVS），查看终态…", _on_frame_corners)
    elif dvs_target is not None:
        pause_gui(
            PHASE_PAUSE_S,
            "准备开始第二阶段伺服（ViSP 光度 DVS）…",
            _on_frame_cam_only if gui else None,
        )
        dvs_ok, dvs_err, dvs_gated = run_visp_dvs_fine(
            robot_id,
            wrist_cam.ee_link,
            peg,
            arm,
            wrist_cam,
            dvs_target,
            plane_z=dvs_plane_z,
            gui=gui,
            on_step=_on_frame_cam_only if gui else None,
            start_err=dvs_start_err,
            abort_err=dvs_abort_err,
        )
        gate = dvs_start_err if dvs_start_err is not None else VISP_DVS_START_ERR
        if dvs_gated:
            print(
                f"第二阶段跳过: DVS 光度误差 ||e||²={dvs_err:.6g} > {gate} "
                f"(未在示教 capture 区)"
            )
        else:
            print(f"第二阶段结束: dvs ok={dvs_ok} ||e||²={dvs_err:.6g}")
    else:
        print("第二阶段跳过: 无 I*")

    inserted: bool | None = None
    if insert and coarse_ok:
        inserted = run_insert_after_align(robot_id, arm, eef, peg, hole_xy, gui=gui)
        print(f"插入: {'ok' if inserted else 'fail'}")
        if gui:
            _on_frame_corners()
        if gui and gui_idle:
            pause_gui(20.0, "插入完成，查看终态…", _on_frame_corners)

    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    mf = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    gt_ok = metrics_converged(mf)
    aligned = teach_ok and gt_ok and (dvs_ok if always_run_dvs or not coarse_ok else (dvs_skipped or dvs_ok))

    report: dict[str, Any] = {
        "coarse_method": coarse_method,
        "teach_ok": teach_ok,
        "coarse_ok": coarse_ok,
        "insert": insert,
        "dvs_ok": dvs_ok,
        "dvs_skipped": dvs_skipped,
        "dvs_gated": dvs_gated,
        "ibvs_err": ibvs_err,
        "dvs_err": dvs_err,
        "teach_path": teach_path,
        "metrics": mf,
        "aligned": aligned,
        "gt_aligned": gt_ok,
    }
    if inserted is not None:
        report["inserted"] = inserted
    if coarse_metrics is not None:
        report["coarse_metrics"] = coarse_metrics

    live = (robot_id, peg, hole_id, True) if (gui and gui_idle) else None
    if live is None:
        if gui:
            close_camera_windows()
        wrist_cam.detach()
        if is_pybullet_connected():
            p.disconnect()
    return aligned, report, hole_xy, live
