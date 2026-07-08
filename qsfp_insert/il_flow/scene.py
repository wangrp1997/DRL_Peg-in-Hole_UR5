"""IL scene: no wrist_cam1, wrist_camera2 + fixed_camera2 (opposite flank), dual OpenCV."""
from __future__ import annotations

import os
import time

import pybullet as p

from constants import (
    FIXED_CAM_HEIGHT,
    FIXED_CAM_WIDTH,
    HOLE_XY,
    PLATE_TOP_Z,
    REST_POSES,
    ROBOT_BASE_Z,
)
from il_flow._paths import URDF as IL_URDF
from il_flow.fixed_camera2 import load_fixed_camera2
from sim.fixed_camera import FixedCamera
from sim.fixed_render import render_rgbd as fixed_render_rgbd
from sim._paths import URDF as PKG_URDF
from sim.scene import (
    arm_joints,
    connect,
    link_index,
    load_fixture,
    move_tip_to_standoff,
    register_wrist_camera2,
    set_hole_opaque,
    settle,
)
from sim.wrist2_render import render_rgbd as wrist2_render_rgbd
from sim.wrist_camera2 import WristCamera2, attach_wrist_camera2

_WRIST2_WINDOW = "wrist_camera2 (RGB | Depth | Seg)"
_FIXED2_WINDOW = "fixed_camera2 (RGB | Depth | Seg)"

_wrist_camera2: WristCamera2 | None = None
_fixed_cam2: FixedCamera | None = None
_il_hardware_render = False


def il_uses_hardware_render() -> bool:
    """True when GUI or EGL headless (HARDWARE OpenGL, not TINY)."""
    return _il_hardware_render


def _load_egl_renderer() -> bool:
    import os
    import pkgutil

    os.environ.setdefault("PYBULLET_EGL", "1")
    loader = pkgutil.get_loader("eglRenderer")
    if loader is None:
        print("il_flow: eglRenderer not found — headless falls back to TINY (black edges)")
        return False
    plugin_id = p.loadPlugin(loader.get_filename(), "_eglRendererPlugin")
    if plugin_id < 0:
        print("il_flow: EGL plugin load failed — headless falls back to TINY (black edges)")
        return False
    return True


def connect_il(*, gui: bool) -> None:
    """GUI → HARDWARE window; headless → DIRECT + EGL plugin + HARDWARE (no TINY)."""
    global _il_hardware_render
    if gui:
        _il_hardware_render = True
        connect(gui=True)
        return
    connect(gui=False)
    _il_hardware_render = _load_egl_renderer()
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)


def reset_il_render_mode() -> None:
    global _il_hardware_render
    _il_hardware_render = False


def _il_urdf(name: str) -> str:
    return os.path.join(IL_URDF, name)


def _open_preview_window(name: str, w: int, h: int, x: int, y: int) -> None:
    import cv2

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.moveWindow(name, x, y)


def _show_panel(window: str, rgba, depth, seg) -> None:
    from sim.scene import _show_panel as base_show

    base_show(window, rgba, depth, seg, keypoint_sets=None)


def get_il_cameras() -> tuple[WristCamera2 | None, FixedCamera | None]:
    return _wrist_camera2, _fixed_cam2


def _disable_bullet_camera_previews() -> None:
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)


def load_il_scene(
    gui: bool,
    hole_xy: tuple[float, float] | None = None,
    *,
    opencv_render: bool = False,
) -> tuple[int, list[int], int, int, int, tuple[float, float]]:
    """Robot without camera_link; fixed2 opposite flank; wrist2 on ee_link."""
    global _wrist_camera2, _fixed_cam2

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.4, 0, 0], p.getQuaternionFromEuler([0, 0, 1.57079632679]))

    robot_id = p.loadURDF(_il_urdf("ur5_robotiq_85_il.urdf"), [0, 0, ROBOT_BASE_Z], useFixedBase=True)
    arm = arm_joints(robot_id)
    eef = link_index(robot_id, "ee_link")
    peg = link_index(robot_id, "qsfp_peg_link")

    for j, q in zip(arm, REST_POSES):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, q, force=500)
    settle(120, gui)

    if hole_xy is None:
        hole_xy = HOLE_XY
    load_fixture(hole_xy)
    hole_id = p.loadURDF(
        os.path.join(PKG_URDF, "qsfp_dd_hole_plate.urdf"),
        [hole_xy[0], hole_xy[1], PLATE_TOP_Z],
        useFixedBase=True,
    )

    _fixed_cam2 = load_fixed_camera2(hole_xy)
    _wrist_camera2 = attach_wrist_camera2(robot_id, eef, hole_xy)
    register_wrist_camera2(_wrist_camera2)

    if gui:
        set_hole_opaque(hole_id)
        p.resetDebugVisualizerCamera(1.2, 110, -40, [0.5, 0, 0.6])
        _disable_bullet_camera_previews()
        if opencv_render:
            w, h = FIXED_CAM_WIDTH, FIXED_CAM_HEIGHT
            _open_preview_window(_WRIST2_WINDOW, w * 3 + 24, h + 8, 40, 40)
            _open_preview_window(_FIXED2_WINDOW, w * 3 + 24, h + 8, 40, 560)
            refresh_il_camera_views()

    return robot_id, arm, eef, peg, hole_id, hole_xy


def refresh_il_camera_views() -> None:
    if _wrist_camera2 is not None:
        rgba, depth, seg = wrist2_render_rgbd(
            _wrist_camera2, gui=True, with_depth_seg=True, use_cache=True, warmup=1,
        )
        _show_panel(_WRIST2_WINDOW, rgba, depth, seg)
    if _fixed_cam2 is not None:
        rgba, depth, seg = fixed_render_rgbd(
            _fixed_cam2, gui=True, with_depth_seg=True, use_cache=True, warmup=1,
        )
        _show_panel(_FIXED2_WINDOW, rgba, depth, seg)


def reset_il_simulation(*, gui: bool) -> None:
    """Clear bodies/GPU mesh cache in-place — keep PyBullet client (GUI or EGL) open."""
    import pybullet_data

    from sim.fixed_render import reset_fixed_render_cache
    from sim.wrist2_render import reset_wrist2_render_cache

    teardown_il_episode()
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    reset_wrist2_render_cache()
    reset_fixed_render_cache()
    if gui:
        _disable_bullet_camera_previews()


def teardown_il_episode() -> None:
    """Detach IL cameras between episodes; does not disconnect PyBullet."""
    global _wrist_camera2, _fixed_cam2

    if _wrist_camera2 is not None:
        _wrist_camera2.detach()
        _wrist_camera2 = None
    register_wrist_camera2(None)
    _fixed_cam2 = None


def teardown_il_scene() -> None:
    """End of session: detach cameras and drop render-mode flag."""
    teardown_il_episode()
    reset_il_render_mode()

    try:
        import cv2

        for name in (_WRIST2_WINDOW, _FIXED2_WINDOW):
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass
        cv2.waitKey(1)
    except Exception:
        pass


def disconnect_il() -> None:
    teardown_il_scene()
    if p.getConnectionInfo()["isConnected"]:
        p.disconnect()


def il_gui_idle(refresh_hz: float = 30.0, *, opencv_render: bool = False) -> None:
    period = 1.0 / refresh_hz
    try:
        while p.getConnectionInfo()["isConnected"]:
            p.stepSimulation()
            if opencv_render:
                refresh_il_camera_views()
            time.sleep(period)
    except KeyboardInterrupt:
        pass
