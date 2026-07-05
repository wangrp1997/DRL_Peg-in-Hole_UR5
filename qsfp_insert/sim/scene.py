"""Shared UR5 + table + hole scene and optional camera previews."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

from constants import EE_LINEAR_STEP, FIXTURE_CENTER_Z, HOLE_DEPTH, PLATE_TOP_Z, REST_POSES, ROBOT_BASE_Z, UR5_MIN_INSERT_DEPTH
from geometry import is_inserted, peg_tip_world
from rlenv import PegInHoleGymEnv
from sim._paths import URDF
from sim.fixed_camera import FixedCamera, load_fixed_camera
from sim.wrist_camera import WristCamera

_rlenv_cam = PegInHoleGymEnv.__new__(PegInHoleGymEnv)
_rlenv_cam.image_width = 100
_rlenv_cam.image_height = 100

_WRIST_WINDOW = "wrist_camera (RGB | Depth | Seg)"
_FIXED_WINDOW = "fixed_camera (RGB | Depth | Seg)"
_WRIST2_WINDOW = "wrist_camera2 (RGB | Depth | Seg)"

_wrist_cam: int | None = None
_wrist_camera: WristCamera | None = None
_wrist_camera2 = None  # WristCamera2 | None
_gui_robot_id: int | None = None
_fixed_cam: FixedCamera | None = None
_wrist_opencv = False
_fixed_opencv = False
_gui_wrist = False  # PyBullet corner previews (HARDWARE_OPENGL) — only one per frame
_gui_fixed = False
_gui_wrist2 = False
_wrist2_opencv = False
_wrist2_last_rgba = None


from sim.wrist2_render import rgba_is_blank, render_rgbd, reset_wrist2_render_cache, get_cached_rgbd


def urdf(name: str) -> str:
    return os.path.join(URDF, name)


def link_index(body_id: int, name: str) -> int:
    for i in range(p.getNumJoints(body_id)):
        if p.getJointInfo(body_id, i)[12].decode() == name:
            return i
    raise ValueError(name)


def arm_joints(robot_id: int) -> list[int]:
    return [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED][:6]


def add_gui_camera_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--gui", action="store_true", help="PyBullet 3D window only")
    ap.add_argument(
        "--opencv_render",
        action="store_true",
        help="OpenCV RGB|Depth|Seg panel(s); requires --gui and a camera flag",
    )
    ap.add_argument(
        "--wrist_cam",
        action="store_true",
        help="Legacy wrist camera on camera_link: GUI corner previews; + OpenCV if --opencv_render",
    )
    ap.add_argument(
        "--wrist_cam2",
        action="store_true",
        help="DVS wrist_camera2 on ee_link: GUI corner previews; + OpenCV if --opencv_render",
    )
    ap.add_argument("--fixed_cam", action="store_true", help="Fixed camera URDF: GUI corner previews; + OpenCV if --opencv_render")


def validate_gui_camera_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    save_target = bool(getattr(args, "save_target", False))
    if args.opencv_render and not args.gui:
        ap.error("--opencv_render requires --gui")
    if args.opencv_render and not args.wrist_cam and not args.fixed_cam and not args.wrist_cam2 and not save_target:
        ap.error("--opencv_render requires --wrist_cam, --wrist_cam2, --fixed_cam, or --save_target")
    if (args.wrist_cam or args.fixed_cam or args.wrist_cam2) and not args.gui:
        ap.error("--wrist_cam, --wrist_cam2 and --fixed_cam require --gui")
    if args.wrist_cam and args.wrist_cam2:
        ap.error("--wrist_cam and --wrist_cam2 are mutually exclusive")


def _set_gui_corner_previews(on: bool) -> None:
    v = 1 if on else 0
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, v)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, v)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, v)


def _cam_views_active() -> bool:
    return _gui_wrist or _gui_fixed or _gui_wrist2 or _wrist_opencv or _fixed_opencv or _wrist2_opencv


def register_wrist_camera2(cam) -> None:
    """Register DVS wrist_camera2 body (after attach_wrist_camera2)."""
    global _wrist_camera2
    _wrist_camera2 = cam


def get_wrist_camera2():
    return _wrist_camera2


def setup_wrist_camera2_views(gui: bool, opencv_render: bool) -> None:
    """OpenCV panel + PyBullet corner previews (same pattern as --wrist_cam)."""
    global _gui_wrist2, _wrist2_opencv
    _gui_wrist2 = gui and _wrist_camera2 is not None
    _wrist2_opencv = gui and opencv_render and _wrist_camera2 is not None
    if _gui_wrist2 or _wrist2_opencv:
        _set_gui_corner_previews(True)
    if _wrist2_opencv and _wrist_camera2 is not None:
        w, h = _wrist_camera2.width, _wrist_camera2.height
        _open_preview_window(_WRIST2_WINDOW, w * 3 + 24, h + 8, 820, 0)


def connect(gui: bool) -> None:
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        _set_gui_corner_previews(False)


def _joint_positions(robot_id: int, arm: list[int]) -> list[float]:
    return [p.getJointState(robot_id, j)[0] for j in arm]


def move_ee(robot_id, eef, arm, pos, orn) -> None:
    rest = _joint_positions(robot_id, arm)
    q = p.calculateInverseKinematics(
        robot_id,
        eef,
        pos,
        orn,
        maxNumIterations=100,
        residualThreshold=1e-5,
        restPoses=rest,
    )
    for j, v in zip(arm, q[:6]):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, v, force=500)


def step_ee_z(robot_id, eef, arm, ee_xy, ee_orn, dz: float, gui: bool = False) -> None:
    ee_pos, _ = p.getLinkState(robot_id, eef)[:2]
    move_ee(robot_id, eef, arm, [ee_xy[0], ee_xy[1], ee_pos[2] + dz], ee_orn)
    settle(10, gui)


def step_tip_z(robot_id, eef, arm, peg, hole_xy, ee_orn, dz: float, gui: bool = False) -> None:
    ee_pos, _ = p.getLinkState(robot_id, eef)[:2]
    tip = peg_tip_world(robot_id, peg)
    move_ee(
        robot_id,
        eef,
        arm,
        [ee_pos[0] - (tip[0] - hole_xy[0]), ee_pos[1] - (tip[1] - hole_xy[1]), ee_pos[2] + dz],
        ee_orn,
    )
    settle(10, gui)


def run_insert_after_align(
    robot_id: int,
    arm: list[int],
    eef: int,
    peg: int,
    hole_xy: tuple[float, float],
    gui: bool = False,
) -> bool:
    """Descend along −Z after alignment until inserted or depth limit."""
    ee_orn0 = p.getLinkState(robot_id, eef)[1]
    for _ in range(800):
        step_tip_z(robot_id, eef, arm, peg, hole_xy, ee_orn0, -EE_LINEAR_STEP, gui)
        tip = peg_tip_world(robot_id, peg)
        if is_inserted(tip, hole_xy, min_insert_depth=UR5_MIN_INSERT_DEPTH):
            return True
        if tip[2] < PLATE_TOP_Z - HOLE_DEPTH:
            break
    tip = peg_tip_world(robot_id, peg)
    return is_inserted(tip, hole_xy, min_insert_depth=UR5_MIN_INSERT_DEPTH)


def move_tip_to_standoff(
    robot_id,
    eef,
    arm,
    peg,
    hole_xy: tuple[float, float],
    standoff: float,
    ee_orn=None,
    gui: bool = False,
    settle_steps: int | None = None,
) -> None:
    """IK: peg tip at hole_xy, standoff metres above plate mouth (PLATE_TOP_Z)."""
    if ee_orn is None:
        ee_orn = p.getLinkState(robot_id, eef)[1]
    ee_pos, _ = p.getLinkState(robot_id, eef)[:2]
    tip = peg_tip_world(robot_id, peg)
    target_tip_z = PLATE_TOP_Z + standoff
    move_ee(
        robot_id,
        eef,
        arm,
        [
            ee_pos[0] - (tip[0] - hole_xy[0]),
            ee_pos[1] - (tip[1] - hole_xy[1]),
            ee_pos[2] + (target_tip_z - tip[2]),
        ],
        ee_orn,
    )
    if settle_steps is None:
        from constants import SETTLE_IK_STEPS, SETTLE_IK_STEPS_GUI

        settle_steps = SETTLE_IK_STEPS_GUI if gui else SETTLE_IK_STEPS
    settle(settle_steps, gui)


def set_hole_opaque(hole_id: int) -> None:
    p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 1.0])


def load_fixture(hole_xy: tuple[float, float]) -> int:
    return p.loadURDF(
        urdf("qsfp_fixture_base.urdf"),
        [hole_xy[0], hole_xy[1], FIXTURE_CENTER_Z],
        useFixedBase=True,
    )


def load_scene(
    gui: bool,
    hole_xy: tuple[float, float] | None = None,
    opencv_render: bool = False,
    wrist_cam: bool = False,
    fixed_cam: bool = False,
):
    """Return robot_id, arm, eef_idx, peg_idx, hole_id, hole_xy."""
    global _wrist_cam, _wrist_camera, _gui_robot_id, _fixed_cam, _wrist_opencv, _fixed_opencv, _gui_wrist, _gui_fixed
    _wrist_opencv = gui and opencv_render and wrist_cam
    _fixed_opencv = gui and opencv_render and fixed_cam
    _gui_wrist = gui and wrist_cam
    _gui_fixed = gui and fixed_cam and not wrist_cam

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.4, 0, 0], p.getQuaternionFromEuler([0, 0, 1.57079632679]))

    robot_id = p.loadURDF(urdf("ur5_robotiq_85_qsfp_peg.urdf"), [0, 0, ROBOT_BASE_Z], useFixedBase=True)
    arm = arm_joints(robot_id)
    eef = link_index(robot_id, "ee_link")
    peg = link_index(robot_id, "qsfp_peg_link")

    for j, q in zip(arm, REST_POSES):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, q, force=500)
    settle(120, gui)

    if hole_xy is None:
        tip0 = peg_tip_world(robot_id, peg)
        hole_xy = (tip0[0], tip0[1])
    load_fixture(hole_xy)
    hole_id = p.loadURDF(urdf("qsfp_dd_hole_plate.urdf"), [hole_xy[0], hole_xy[1], PLATE_TOP_Z], useFixedBase=True)

    if fixed_cam:
        _fixed_cam = load_fixed_camera(hole_xy)
    else:
        _fixed_cam = None

    if wrist_cam:
        cam_link = link_index(robot_id, "camera_link")
        _wrist_cam = cam_link
        _wrist_camera = WristCamera(robot_id=robot_id, link_index=cam_link)
    else:
        _wrist_cam = None
        _wrist_camera = None

    if gui:
        p.changeVisualShape(hole_id, -1, rgbaColor=[0.55, 0.55, 0.55, 0.25])
        p.resetDebugVisualizerCamera(1.2, 110, -40, [0.5, 0, 0.6])
        _set_gui_corner_previews(wrist_cam or fixed_cam)
        _gui_robot_id = robot_id
        if _wrist_opencv and _wrist_camera is not None:
            w, h = _wrist_camera.width, _wrist_camera.height
            _open_preview_window(_WRIST_WINDOW, w * 3 + 24, h + 8, 820, 0)
        if _fixed_opencv and _fixed_cam is not None:
            _open_preview_window(_FIXED_WINDOW, _fixed_cam.width * 3 + 24, _fixed_cam.height + 8, 820, 520)
        if _cam_views_active():
            refresh_camera_views()
    else:
        _gui_robot_id = None

    return robot_id, arm, eef, peg, hole_id, hole_xy


def _open_preview_window(name: str, w: int, h: int, x: int, y: int) -> None:
    import cv2

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.moveWindow(name, x, y)


def _depth_bgr(depth: np.ndarray) -> np.ndarray:
    import cv2

    d = depth.copy()
    valid = (d > 0) & np.isfinite(d)
    if not np.any(valid):
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(d[valid], (2, 98))
    if hi <= lo:
        hi = lo + 1e-3
    return cv2.applyColorMap((np.clip((d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _seg_bgr(seg: np.ndarray) -> np.ndarray:
    import cv2

    obj = seg & ((1 << 24) - 1)
    hue = ((obj.astype(np.uint32) * 47) % 180).astype(np.uint8)
    hsv = np.stack([hue, np.where(obj > 0, 200, 0).astype(np.uint8), np.where(obj > 0, 255, 30).astype(np.uint8)], -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _show_panel(window: str, rgba, depth, seg, keypoint_sets=None) -> None:
    import cv2

    # PyBullet RGBA → BGR; ignore alpha (TINY/HARDWARE mix caused black flicker if mishandled)
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if keypoint_sets:
        from vision.overlay import draw_keypoints_on_bgr

        draw_keypoints_on_bgr(bgr, keypoint_sets)
    panel = np.hstack(
        [
            _label(bgr, "RGB"),
            _label(_depth_bgr(depth), "Depth"),
            _label(_seg_bgr(seg), "Seg"),
        ]
    )
    cv2.imshow(window, panel)
    cv2.waitKey(1)


def _render_wrist_cam(with_depth_seg: bool, for_opencv: bool):
    """Wrist camera render via WristCamera (640×480, same as fixed_cam)."""
    if _wrist_camera is None:
        raise RuntimeError("wrist camera not initialized")
    return _wrist_camera.render(with_depth_seg, for_opencv=for_opencv)


def get_wrist_camera() -> WristCamera | None:
    return _wrist_camera


def get_fixed_camera() -> FixedCamera | None:
    return _fixed_cam


def show_fixed_camera_panel(get_keypoint_sets=None) -> None:
    """OpenCV fixed-cam panel; reuses refresh_camera_views (HARDWARE when GUI fixed active)."""
    refresh_camera_views(get_keypoint_sets)


def refresh_camera_views(get_keypoint_sets=None, *, render: bool = True) -> None:
    """One HARDWARE render per active cam; OpenCV reuses that buffer (fixed_cam pattern)."""
    global _wrist2_last_rgba
    wrist2_buf = None
    wrist_buf = None
    fixed_buf = None

    if _gui_wrist2 and _wrist_camera2 is not None:
        if render:
            wrist2_buf = render_rgbd(_wrist_camera2, gui=True, with_depth_seg=True, use_cache=True, warmup=1)
            wrist2_buf = (wrist2_buf[0], wrist2_buf[1], wrist2_buf[2])
        else:
            cached = get_cached_rgbd()
            if cached is not None:
                wrist2_buf = cached
    elif _gui_wrist and _wrist_cam is not None and _gui_robot_id is not None:
        wrist_buf = _render_wrist_cam(True, for_opencv=False)
    elif _gui_fixed and _fixed_cam is not None:
        fixed_buf = _fixed_cam.render(True, for_opencv=False)

    if _wrist2_opencv:
        sets = get_keypoint_sets() if get_keypoint_sets is not None else None
        if wrist2_buf is not None:
            rgba, depth, seg = wrist2_buf
        elif not render:
            cached = get_cached_rgbd()
            if cached is not None:
                rgba, depth, seg = cached
            else:
                rgba = depth = seg = None
        elif _wrist_camera2 is not None:
            rgba, depth, seg = render_rgbd(_wrist_camera2, gui=True, with_depth_seg=True, use_cache=True, warmup=1)
        else:
            rgba = depth = seg = None
        if rgba is not None and not rgba_is_blank(rgba):
            _wrist2_last_rgba = (rgba, depth, seg)
        elif _wrist2_last_rgba is not None:
            rgba, depth, seg = _wrist2_last_rgba
        if rgba is not None:
            _show_panel(_WRIST2_WINDOW, rgba, depth, seg, sets)

    if _wrist_opencv:
        if wrist_buf is not None:
            _show_panel(_WRIST_WINDOW, *wrist_buf)
        elif _wrist_cam is not None and _gui_robot_id is not None:
            _show_panel(_WRIST_WINDOW, *_render_wrist_cam(True, for_opencv=True))

    if _fixed_opencv:
        sets = get_keypoint_sets() if get_keypoint_sets is not None else None
        if fixed_buf is not None:
            _show_panel(_FIXED_WINDOW, *fixed_buf, sets)
        elif _fixed_cam is not None:
            _show_panel(_FIXED_WINDOW, *_fixed_cam.render(True, for_opencv=True), sets)


def close_camera_windows() -> None:
    global _wrist_cam, _wrist_camera, _wrist_camera2, _gui_robot_id, _fixed_cam
    global _wrist_opencv, _fixed_opencv, _gui_wrist, _gui_fixed, _gui_wrist2, _wrist2_opencv, _wrist2_last_rgba
    import cv2

    for name in (_WRIST_WINDOW, _FIXED_WINDOW, _WRIST2_WINDOW):
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass
    _wrist_cam = None
    _wrist_camera = None
    _wrist_camera2 = None
    _gui_robot_id = None
    _fixed_cam = None
    _wrist_opencv = False
    _fixed_opencv = False
    _gui_wrist = False
    _gui_fixed = False
    _gui_wrist2 = False
    _wrist2_opencv = False
    _wrist2_last_rgba = None
    reset_wrist2_render_cache()


def settle(steps: int = 10, gui: bool = False) -> None:
    for _ in range(steps):
        p.stepSimulation()
        if gui and _cam_views_active():
            refresh_camera_views()
        if gui:
            time.sleep(1.0 / 240.0)


def idle_gui() -> None:
    print("Close PyBullet window to exit.")
    while p.getConnectionInfo()["isConnected"]:
        p.stepSimulation()
        if _cam_views_active():
            refresh_camera_views()
        time.sleep(1.0 / 240.0)
