"""Eye-in-hand wrist camera on camera_link — same intrinsics / optical frame as fixed_camera."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p

from constants import (
    FIXED_CAM_FAR,
    FIXED_CAM_FOV,
    FIXED_CAM_HEIGHT,
    FIXED_CAM_NEAR,
    FIXED_CAM_WIDTH,
    PLATE_TOP_Z,
)
from sim.fixed_camera import depth_buffer_to_meters, project_world_gl


@dataclass(frozen=True)
class WristCamera:
    """640×480 camera rigidly attached to robot camera_link (+X up, −Z forward)."""

    robot_id: int
    link_index: int
    width: int = FIXED_CAM_WIDTH
    height: int = FIXED_CAM_HEIGHT
    fov: float = FIXED_CAM_FOV
    near: float = FIXED_CAM_NEAR
    far: float = FIXED_CAM_FAR

    @property
    def K(self) -> np.ndarray:
        fy = (self.height / 2.0) / math.tan(math.radians(self.fov / 2.0))
        fx = fy
        cx, cy = self.width / 2.0, self.height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def link_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return p.getLinkState(self.robot_id, self.link_index, computeForwardKinematics=True)[:2]

    def _view_projection(self) -> tuple[list[float], list[float], tuple[float, float, float]]:
        link_pos, link_ori = self.link_pose()
        rot = p.getMatrixFromQuaternion(link_ori)
        forward = [-rot[2], -rot[5], -rot[8]]
        up = [rot[0], rot[3], rot[6]]
        target = [
            link_pos[0] + forward[0] * 0.2,
            link_pos[1] + forward[1] * 0.2,
            link_pos[2] + forward[2] * 0.2,
        ]
        view = p.computeViewMatrix(link_pos, target, up)
        proj = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far,
        )
        return view, proj, link_pos

    def render(self, with_depth_seg: bool = True, for_opencv: bool = True):
        view, proj, _ = self._view_projection()
        renderer = p.ER_TINY_RENDERER if for_opencv else p.ER_BULLET_HARDWARE_OPENGL
        kwargs = dict(viewMatrix=view, projectionMatrix=proj, renderer=renderer)
        if with_depth_seg:
            kwargs["flags"] = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        _, _, rgba, depth, seg = p.getCameraImage(self.width, self.height, **kwargs)
        rgba_img = np.reshape(rgba, (self.height, self.width, 4))
        if not with_depth_seg:
            return rgba_img
        depth_m = depth_buffer_to_meters(
            np.reshape(depth, (self.height, self.width)).astype(np.float32), self.near, self.far
        )
        seg_buf = np.reshape(seg, (self.height, self.width)).astype(np.int32)
        return rgba_img, depth_m, seg_buf

    def project_world(self, point_world: tuple[float, float, float]) -> tuple[float, float, float]:
        view, proj, _ = self._view_projection()
        return project_world_gl(view, proj, point_world, self.width, self.height)

    def world_point_on_plane(self, u: float, v: float, plane_z: float = PLATE_TOP_Z) -> tuple[float, float, float] | None:
        origin, direction = self.ray_world(u, v)
        if abs(direction[2]) < 1e-9:
            return None
        t = (plane_z - origin[2]) / direction[2]
        if t < 0.0:
            return None
        pt = origin + t * direction
        return float(pt[0]), float(pt[1]), float(pt[2])

    def ray_world(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        view, proj, _ = self._view_projection()
        from sim.fixed_camera import _mat4_col_major

        inv_vp = np.linalg.inv(_mat4_col_major(proj) @ _mat4_col_major(view))
        ndc_x = 2.0 * u / self.width - 1.0
        ndc_y = 1.0 - 2.0 * v / self.height

        def _world(ndc_z: float) -> np.ndarray:
            h = inv_vp @ np.array([ndc_x, ndc_y, ndc_z, 1.0], dtype=np.float64)
            return h[:3] / h[3]

        p_near, p_far = _world(-1.0), _world(1.0)
        direction = p_far - p_near
        direction /= np.linalg.norm(direction)
        return p_near, direction
