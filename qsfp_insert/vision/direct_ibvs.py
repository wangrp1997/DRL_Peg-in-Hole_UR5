"""Direct / photometric IBVS: teach aligned ROI once, servo by matching target image.

Collewet–Marchand–Chaumette photometric visual servoing: pixel luminance + image gradient
interaction matrix, no feature matching. Eye-to-hand chain reuses align.py J_img helpers.

Refs:
  - Collewet, Marchand, Chaumette, "Photometric Visual Servoing", IEEE TRO 2011
  - ViSP vpFeatureLuminance / photometricVisualServoing.cpp: https://github.com/lagadic/visp
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
import pybullet as p

from constants import (
    CART_MAX_ANG,
    CART_MAX_LIN,
    DVS_LAMBDA,
    DVS_MAX_SAMPLES,
    DVS_MAX_STEPS,
    DVS_MIN_GRAD,
    DVS_RENDER_SCALE,
    DVS_ROI_MARGIN_PX,
    DVS_SAMPLE_STEP,
    DVS_SIM_SUBSTEPS,
    DVS_SSD_TOL,
    DVS_TWIST_GAIN,
    GUI_SERVO_REFRESH_EVERY,
    PLATE_TOP_Z,
    SERVO_STALL_STEPS,
)
from geometry import AlignmentMetrics, alignment_metrics, metrics_converged, peg_tip_world
from sim.cartesian_control import apply_cartesian_velocity, damped_pinv, stop_arm
from sim.fixed_camera import FixedCamera, depth_buffer_to_meters
from vision.align import (
    _cam_rot_world_to_cam,
    _corner_cam_z,
    _ee_to_cam_point_jacobian,
    _interaction_matrix_points,
)
from vision.corners import peg_tip_corners_world


@dataclass(frozen=True)
class LumSample:
    u: float
    v: float
    intensity: float
    ix: float
    iy: float
    world_pt: tuple[float, float, float]


@dataclass(frozen=True)
class TaughtPatch:
    """Grayscale ROI + luminance samples captured at aligned standoff (target I*)."""

    template: np.ndarray
    roi: tuple[int, int, int, int]
    samples: tuple[LumSample, ...]
    depth_m: float
    standoff_m: float


def _render_gray_depth(cam: FixedCamera) -> tuple[np.ndarray, np.ndarray]:
    view, proj = cam._view_projection()
    w = max(32, int(cam.width * DVS_RENDER_SCALE))
    h = max(32, int(cam.height * DVS_RENDER_SCALE))
    _, _, rgba, depth, _ = p.getCameraImage(
        w,
        h,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    rgba_img = np.reshape(rgba, (h, w, 4))
    depth_m = depth_buffer_to_meters(np.reshape(depth, (h, w)).astype(np.float32), cam.near, cam.far)
    gray = cv2.cvtColor(rgba_img[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    return gray, depth_m


def render_gray(cam: FixedCamera) -> np.ndarray:
    gray, _ = _render_gray_depth(cam)
    return gray


def _sample_depth_dvs(depth_map: np.ndarray, u: float, v: float) -> float | None:
    h, w = depth_map.shape
    ui = max(0, min(w - 1, int(round(u))))
    vi = max(0, min(h - 1, int(round(v))))
    depth_m = float(depth_map[vi, ui])
    if depth_m >= 2.9:
        return None
    return depth_m


def _world_from_dvs_pixel(
    cam: FixedCamera, depth_map: np.ndarray, u: float, v: float
) -> tuple[float, float, float] | None:
    depth_m = _sample_depth_dvs(depth_map, u, v)
    if depth_m is None:
        return None
    u_full = u / DVS_RENDER_SCALE
    v_full = v / DVS_RENDER_SCALE
    origin, direction = cam.ray_world(u_full, v_full)
    pt = origin + depth_m * direction
    return float(pt[0]), float(pt[1]), float(pt[2])


def _roi_from_scene(
    cam: FixedCamera,
    hole_xy: tuple[float, float],
    robot_id: int,
    peg: int,
) -> tuple[int, int, int, int]:
    """ROI covering hole mouth and peg tip region (union of projections)."""
    hx, hy = hole_xy
    world_pts = [
        (hx, hy, PLATE_TOP_Z),
        (hx - 0.025, hy - 0.015, PLATE_TOP_Z),
        (hx + 0.025, hy + 0.015, PLATE_TOP_Z),
    ]
    world_pts.extend(peg_tip_corners_world(robot_id, peg))
    world_pts.append(peg_tip_world(robot_id, peg))
    us, vs = [], []
    for pt in world_pts:
        u, v, _ = cam.project_world(pt)
        if math.isfinite(u) and math.isfinite(v):
            us.append(u * DVS_RENDER_SCALE)
            vs.append(v * DVS_RENDER_SCALE)
    if not us:
        cx = int(cam.width * DVS_RENDER_SCALE) // 2
        cy = int(cam.height * DVS_RENDER_SCALE) // 2
        return cx - 40, cy - 30, 80, 60
    x0 = max(0, int(min(us)) - DVS_ROI_MARGIN_PX)
    y0 = max(0, int(min(vs)) - DVS_ROI_MARGIN_PX)
    x1 = min(int(cam.width * DVS_RENDER_SCALE), int(max(us)) + DVS_ROI_MARGIN_PX)
    y1 = min(int(cam.height * DVS_RENDER_SCALE), int(max(vs)) + DVS_ROI_MARGIN_PX)
    w = max(32, x1 - x0)
    h = max(32, y1 - y0)
    return x0, y0, w, h


def _build_luminance_samples(
    cam: FixedCamera,
    gray: np.ndarray,
    depth: np.ndarray,
    roi: tuple[int, int, int, int],
    peg_center: tuple[float, float, float],
    peg_radius: float = 0.028,
) -> tuple[LumSample, ...]:
    x, y, w, h = roi
    patch = gray[y : y + h, x : x + w]
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    min_g2 = DVS_MIN_GRAD * DVS_MIN_GRAD
    peg_c = np.array(peg_center, dtype=np.float64)
    out: list[LumSample] = []
    for vi in range(3, h - 3, DVS_SAMPLE_STEP):
        for ui in range(3, w - 3, DVS_SAMPLE_STEP):
            ix, iy = float(gx[vi, ui]), float(gy[vi, ui])
            if ix * ix + iy * iy < min_g2:
                continue
            u, v = float(x + ui), float(y + vi)
            world_pt = _world_from_dvs_pixel(cam, depth, u, v)
            if world_pt is None:
                continue
            if float(np.linalg.norm(np.array(world_pt) - peg_c)) > peg_radius:
                continue
            out.append(
                LumSample(
                    u=u,
                    v=v,
                    intensity=float(patch[vi, ui]),
                    ix=ix,
                    iy=iy,
                    world_pt=world_pt,
                )
            )
            if len(out) >= DVS_MAX_SAMPLES:
                return tuple(out)
    return tuple(out)


def teach_aligned_patch(
    cam: FixedCamera,
    hole_xy: tuple[float, float],
    robot_id: int,
    peg: int,
) -> TaughtPatch:
    """Capture I* and luminance interaction samples at aligned standoff."""
    gray, depth = _render_gray_depth(cam)
    roi = _roi_from_scene(cam, hole_xy, robot_id, peg)
    x, y, w, h = roi
    template = gray[y : y + h, x : x + w].copy()
    tip = peg_tip_world(robot_id, peg)
    samples = _build_luminance_samples(cam, gray, depth, roi, tip)
    standoff_m = tip[2] - PLATE_TOP_Z
    depth_m = max(0.05, standoff_m + 0.01)
    return TaughtPatch(
        template=template, roi=roi, samples=samples, depth_m=depth_m, standoff_m=standoff_m
    )


def _extract_patch(gray: np.ndarray, taught: TaughtPatch) -> np.ndarray:
    x, y, w, h = taught.roi
    return gray[y : y + h, x : x + w]


def patch_ssd(gray: np.ndarray, taught: TaughtPatch) -> float:
    cur = _extract_patch(gray, taught)
    if cur.shape != taught.template.shape:
        return float("inf")
    diff = cur - taught.template
    return float(np.mean(diff * diff))


def _clip_twist(twist: np.ndarray) -> np.ndarray:
    lin = twist[:3]
    ang = twist[3:]
    ln = np.linalg.norm(lin)
    if ln > CART_MAX_LIN:
        lin = lin * (CART_MAX_LIN / ln)
    an = np.linalg.norm(ang)
    if an > CART_MAX_ANG:
        ang = ang * (CART_MAX_ANG / an)
    return np.concatenate([lin, ang])


def _photometric_twist(
    cam: FixedCamera,
    taught: TaughtPatch,
    gray: np.ndarray,
    robot_id: int,
    peg: int,
) -> np.ndarray | None:
    """v = −λ L⁺ e with luminance rows L_I = Ix·L_u + Iy·L_v (Collewet / ViSP)."""
    if not taught.samples:
        return None
    tip = np.array(peg_tip_world(robot_id, peg), dtype=np.float64)
    r_cw = _cam_rot_world_to_cam(cam)
    x, y, w, h = taught.roi
    patch = gray[y : y + h, x : x + w]
    if patch.shape != taught.template.shape:
        return None

    rows: list[np.ndarray] = []
    errs: list[float] = []
    for s in taught.samples:
        ui = int(round(s.u - x))
        vi = int(round(s.v - y))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        cur_i = float(patch[vi, ui])
        e = cur_i - s.intensity
        z = _corner_cam_z(cam, s.world_pt)
        if z is None or z <= 1e-6:
            continue
        u_full = s.u / DVS_RENDER_SCALE
        v_full = s.v / DVS_RENDER_SCALE
        l_pt = _interaction_matrix_points(cam, [(u_full, v_full)], [z])
        if l_pt is None:
            continue
        l_i = s.ix * l_pt[0] + s.iy * l_pt[1]
        r = np.array(s.world_pt, dtype=np.float64) - tip
        m_i = _ee_to_cam_point_jacobian(r_cw, r)
        w = math.sqrt(s.ix * s.ix + s.iy * s.iy)
        rows.append(w * (l_i @ m_i))
        errs.append(w * e)

    if len(rows) < 8:
        return None
    j = np.vstack(rows)
    e_vec = np.array(errs, dtype=np.float64)
    ssd = float(np.mean(e_vec * e_vec))
    gain = DVS_TWIST_GAIN * min(1.0, max(0.35, math.sqrt(ssd) / 40.0))
    try:
        twist = gain * damped_pinv(j, DVS_LAMBDA) @ (-e_vec)
    except np.linalg.LinAlgError:
        twist, _, _, _ = np.linalg.lstsq(j, -e_vec, rcond=DVS_LAMBDA)
        twist = gain * twist
    patch_ssd_val = patch_ssd(gray, taught)
    if patch_ssd_val > DVS_SSD_TOL * 3:
        twist[2] = min(twist[2], -CART_MAX_LIN * 0.15)
    return _clip_twist(twist)


def _gt_metrics(robot_id: int, peg: int, hole_xy: tuple[float, float], hole_orn) -> AlignmentMetrics:
    tip = peg_tip_world(robot_id, peg)
    peg_orn = p.getLinkState(robot_id, peg)[1]
    return alignment_metrics(tip, peg_orn, hole_xy, hole_orn)


def run_direct_image_servo(
    robot_id: int,
    arm: list[int],
    peg: int,
    cam: FixedCamera,
    hole_xy: tuple[float, float],
    hole_orn,
    taught: TaughtPatch,
    gui: bool = False,
    on_step: Callable[[], None] | None = None,
) -> tuple[bool, AlignmentMetrics | None, float | None]:
    """Servo until SSD(current, I*) ≤ tol using photometric IBVS (no keypoints)."""
    stall = 0
    prev_ssd = float("inf")
    last_ssd: float | None = None
    last_m: AlignmentMetrics | None = None

    for step in range(DVS_MAX_STEPS):
        gray = render_gray(cam)
        last_ssd = patch_ssd(gray, taught)
        last_m = _gt_metrics(robot_id, peg, hole_xy, hole_orn)

        if last_ssd <= DVS_SSD_TOL and metrics_converged(last_m):
            stop_arm(robot_id, arm)
            if on_step is not None:
                on_step()
            return True, last_m, last_ssd

        twist = _photometric_twist(cam, taught, gray, robot_id, peg)
        if twist is None:
            break

        if abs(last_ssd - prev_ssd) < 2.0:
            stall += 1
        else:
            stall = 0
        prev_ssd = last_ssd
        if stall >= SERVO_STALL_STEPS // max(1, DVS_SIM_SUBSTEPS):
            break

        for _ in range(DVS_SIM_SUBSTEPS):
            apply_cartesian_velocity(robot_id, peg, arm, twist)
            p.stepSimulation()
        if on_step is not None and step % GUI_SERVO_REFRESH_EVERY == 0:
            on_step()
        elif gui:
            time.sleep(DVS_SIM_SUBSTEPS / 240.0)

    stop_arm(robot_id, arm)
    for _ in range(20):
        p.stepSimulation()
        if on_step is not None:
            on_step()

    last_m = _gt_metrics(robot_id, peg, hole_xy, hole_orn)
    ok = last_ssd is not None and last_ssd <= DVS_SSD_TOL and metrics_converged(last_m)
    return ok, last_m, last_ssd
