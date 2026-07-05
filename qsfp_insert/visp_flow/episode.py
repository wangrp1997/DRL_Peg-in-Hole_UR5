"""ViSP baseline episode — same rhythm as corner_servo demo."""
from __future__ import annotations

import random
import time
from typing import Any, Literal

import pybullet as p

from constants import PERTURB_SERVO_PAUSE_S, SETTLE_IK_STEPS, SETTLE_IK_STEPS_GUI
from geometry import alignment_metrics, metrics_converged, peg_tip_world
from sim.cartesian_align import run_cartesian_align
from sim.perturbation import (
    Perturbation6,
    apply_tip_perturbation,
    format_perturbation_log,
    sample_perturbation6,
)
from sim.scene import (
    close_camera_windows,
    connect,
    load_scene,
    move_tip_to_standoff,
    refresh_camera_views,
    register_wrist_camera2,
    set_hole_opaque,
    setup_wrist_camera2_views,
)
from sim.wrist_camera2 import attach_wrist_camera2
from visp_flow.visp_constants import COARSE_STANDOFF
from visp_flow.dvs_fine import run_visp_dvs_fine
from visp_flow.ibvs_coarse import run_visp_ibvs_coarse
from visp_flow.kabsch_coarse import run_kabsch_coarse
from visp_flow.require_visp import require_visp_python
from visp_flow.teach import capture_aligned_teach, save_teach_bundle
from vision.corners import gt_image_keypoints

CoarseMethod = Literal["kabsch", "ibvs"]
PHASE_PAUSE_S = PERTURB_SERVO_PAUSE_S  # 3 s


def _pause_gui(gui: bool, seconds: float, message: str, on_frame=None) -> None:
    if not gui or seconds <= 0:
        if message:
            print(message)
        return
    print(message)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not p.getConnectionInfo()["isConnected"]:
            break
        p.stepSimulation()
        if on_frame is not None:
            on_frame()
        time.sleep(1.0 / 240.0)


def visp_flow_episode(
    gui: bool = False,
    hole_xy: tuple[float, float] | None = None,
    opencv_render: bool = False,
    perturb: Perturbation6 | None = None,
    rng: random.Random | None = None,
    coarse_method: CoarseMethod = "kabsch",
    teach_dir: str | None = None,
) -> tuple[bool, dict[str, Any], tuple[float, float]]:
    require_visp_python()

    connect(gui)
    robot_id, arm, eef, peg, hole_id, hole_xy = load_scene(gui, hole_xy=hole_xy, opencv_render=False)
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
        if gui:
            refresh_camera_views(_provider)

    def _on_frame_cam_only():
        if gui:
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
    ibvs_desired = None
    if teach_ok:
        kps0 = _provider()
        hole_kp = next(s for s in kps0 if s.name == "hole")
        peg_kp = next(s for s in kps0 if s.name == "peg")
        ibvs_desired, dvs_target = capture_aligned_teach(
            wrist_cam, hole_kp, peg_kp, robot_id, peg, gui=gui
        )
        if ibvs_desired is not None:
            _, teach_path = save_teach_bundle(ibvs_desired, dvs_target, hole_xy, m_align, out_dir=teach_dir)
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
    _pause_gui(
        gui,
        PHASE_PAUSE_S,
        f"准备开始第一阶段伺服（粗对准 / {coarse_method}）…",
        _on_frame_corners,
    )

    coarse_ok = False
    ibvs_err = 0.0
    coarse_metrics = None
    if ibvs_desired is not None:
        if coarse_method == "ibvs":
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
    dvs_skipped = coarse_ok
    if dvs_skipped:
        print("第二阶段跳过: 第一阶段已对准")
    elif dvs_target is not None:
        _pause_gui(
            gui,
            PHASE_PAUSE_S,
            "准备开始第二阶段伺服（ViSP 光度 DVS）…",
            _on_frame_cam_only,
        )
        dvs_ok, dvs_err = run_visp_dvs_fine(
            robot_id,
            wrist_cam.ee_link,
            arm,
            wrist_cam,
            dvs_target,
            gui=gui,
            on_step=_on_frame_cam_only if gui else None,
        )
        print(f"第二阶段结束: dvs ok={dvs_ok} ||e||²={dvs_err:.6g}")
    else:
        print("第二阶段跳过: 无 I*")

    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    mf = alignment_metrics(tip, peg_orn, hole_xy, hole_orn)
    gt_ok = metrics_converged(mf)
    aligned = teach_ok and coarse_ok and gt_ok and (dvs_skipped or dvs_ok)

    report: dict[str, Any] = {
        "coarse_method": coarse_method,
        "teach_ok": teach_ok,
        "coarse_ok": coarse_ok,
        "dvs_ok": dvs_ok,
        "dvs_skipped": dvs_skipped,
        "ibvs_err": ibvs_err,
        "dvs_err": dvs_err,
        "teach_path": teach_path,
        "metrics": mf,
        "aligned": aligned,
        "gt_aligned": gt_ok,
    }
    if coarse_metrics is not None:
        report["coarse_metrics"] = coarse_metrics

    if gui:
        close_camera_windows()
    wrist_cam.detach()
    p.disconnect()
    return aligned, report, hole_xy
