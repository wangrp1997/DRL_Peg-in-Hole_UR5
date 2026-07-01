"""Fixed eye-to-hand camera: URDF mount, 640×480 RGB-D render, K and projection helpers."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pybullet as p

from constants import (
    FIXED_CAM_EYE_OFFSET,
    FIXED_CAM_FAR,
    FIXED_CAM_FOV,
    FIXED_CAM_HEIGHT,
    FIXED_CAM_NEAR,
    FIXED_CAM_ROLL_DEG,
    FIXED_CAM_UP,
    FIXED_CAM_WIDTH,
    PLATE_TOP_Z,
)
from sim._paths import URDF


def _urdf(name: str) -> str:
    return os.path.join(URDF, name)


@dataclass(frozen=True)
class FixedCamera:
    body_id: int
    hole_xy: tuple[float, float]
    width: int = FIXED_CAM_WIDTH
    height: int = FIXED_CAM_HEIGHT
    fov: float = FIXED_CAM_FOV
    near: float = FIXED_CAM_NEAR
    far: float = FIXED_CAM_FAR

    def _pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return p.getBasePositionAndOrientation(self.body_id)

    @property
    def K(self) -> np.ndarray:
        fy = (self.height / 2.0) / math.tan(math.radians(self.fov / 2.0))
        fx = fy
        cx, cy = self.width / 2.0, self.height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def render(self, with_depth_seg: bool = True, for_opencv: bool = True):
        """for_opencv=True uses TINY_RENDERER so GUI corner previews stay on wrist camera only."""
        link_pos, link_ori = self._pose()
        rot = p.getMatrixFromQuaternion(link_ori)
        up = [rot[0], rot[3], rot[6]]
        target = list(target_for_hole(self.hole_xy))
        view = p.computeViewMatrix(link_pos, target, up)
        proj = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far,
        )
        renderer = p.ER_TINY_RENDERER if for_opencv else p.ER_BULLET_HARDWARE_OPENGL
        kwargs = dict(
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=renderer,
        )
        if with_depth_seg:
            kwargs["flags"] = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        w, h, rgba, depth, seg = p.getCameraImage(self.width, self.height, **kwargs)
        rgba_img = np.reshape(rgba, (h, w, 4))
        if not with_depth_seg:
            return rgba_img
        depth_m = depth_buffer_to_meters(np.reshape(depth, (h, w)).astype(np.float32), self.near, self.far)
        seg_buf = np.reshape(seg, (h, w)).astype(np.int32)
        return rgba_img, depth_m, seg_buf

    def project_world(self, point_world: tuple[float, float, float]) -> tuple[float, float, float]:
        """World point → (u, v, Z_cam metres). OpenCV-style camera frame (+Z forward)."""
        link_pos, link_ori = self._pose()
        inv_pos, inv_orn = p.invertTransform(link_pos, link_ori)
        p_cam, _ = p.multiplyTransforms(inv_pos, inv_orn, point_world, [0.0, 0.0, 0.0, 1.0])
        x_cv, y_cv, z_cv = -p_cam[0], -p_cam[1], -p_cam[2]
        if z_cv <= 1e-6:
            return float("nan"), float("nan"), float("nan")
        u = self.K[0, 0] * x_cv / z_cv + self.K[0, 2]
        v = self.K[1, 1] * y_cv / z_cv + self.K[1, 2]
        return float(u), float(v), float(z_cv)

    def backproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        x = (u - self.K[0, 2]) * depth_m / self.K[0, 0]
        y = (v - self.K[1, 2]) * depth_m / self.K[1, 1]
        return np.array([x, y, depth_m], dtype=np.float64)


def depth_buffer_to_meters(depth_buf: np.ndarray, near: float, far: float) -> np.ndarray:
    return (far * near / (far - (far - near) * depth_buf)).astype(np.float32)


def target_for_hole(hole_xy: tuple[float, float]) -> tuple[float, float, float]:
    return (hole_xy[0], hole_xy[1], PLATE_TOP_Z)


def eye_for_hole(hole_xy: tuple[float, float]) -> tuple[float, float, float]:
    tx, ty, tz = target_for_hole(hole_xy)
    ox, oy, oz = FIXED_CAM_EYE_OFFSET
    return (tx + ox, ty + oy, tz + oz)


def _quat_from_axes(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> tuple[float, float, float, float]:
    r = np.column_stack([x_axis, y_axis, z_axis])
    tr = float(np.trace(r))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def quat_look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up=FIXED_CAM_UP,
) -> tuple[float, float, float, float]:
    """Orientation for fixed_camera_link: +X up, −Z forward (matches wrist camera_link)."""
    forward = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    up_v = np.array(up, dtype=np.float64)
    x_axis = np.cross(up_v, z_axis)
    n = np.linalg.norm(x_axis)
    if n < 1e-8:
        up_v = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up_v, z_axis)
        n = np.linalg.norm(x_axis)
    x_axis /= n
    y_axis = np.cross(z_axis, x_axis)
    return _quat_from_axes(x_axis, y_axis, z_axis)


def camera_orientation(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Look-at + optional roll around link +Z (optical axis)."""
    orn = quat_look_at(eye, target)
    if abs(FIXED_CAM_ROLL_DEG) < 1e-9:
        return orn
    roll_quat = p.getQuaternionFromEuler([0.0, 0.0, math.radians(FIXED_CAM_ROLL_DEG)])
    return p.multiplyTransforms([0.0, 0.0, 0.0], orn, [0.0, 0.0, 0.0], roll_quat)[1]


def load_fixed_camera(hole_xy: tuple[float, float]) -> FixedCamera:
    eye = eye_for_hole(hole_xy)
    target = target_for_hole(hole_xy)
    orn = camera_orientation(eye, target)
    body_id = p.loadURDF(_urdf("fixed_camera.urdf"), eye, orn, useFixedBase=True)
    return FixedCamera(body_id=body_id, hole_xy=hole_xy)


def sync_fixed_camera(cam: FixedCamera, hole_xy: tuple[float, float]) -> None:
    eye = eye_for_hole(hole_xy)
    target = target_for_hole(hole_xy)
    p.resetBasePositionAndOrientation(cam.body_id, eye, camera_orientation(eye, target))
