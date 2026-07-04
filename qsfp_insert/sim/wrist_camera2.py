"""Eye-in-hand wrist_camera2: mounted on ee_link, standoff pose matches fixed_cam view."""
from __future__ import annotations

import math
import os
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
from sim._paths import URDF
from sim.fixed_camera import (
    _mat4_col_major,
    camera_orientation,
    depth_buffer_to_meters,
    eye_for_hole,
    project_world_gl,
    target_for_hole,
)


@dataclass(frozen=True)
class WristCamera2:
    """640×480 camera body fixed to ee_link; at attach time world pose = fixed_cam."""

    body_id: int
    robot_id: int
    ee_link: int
    hole_xy: tuple[float, float]
    constraint_id: int
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

    def _pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return p.getBasePositionAndOrientation(self.body_id)

    def _view_projection(self) -> tuple[list[float], list[float]]:
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
        return view, proj

    def render(self, with_depth_seg: bool = True, for_opencv: bool = True):
        view, proj = self._view_projection()
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
        view, proj = self._view_projection()
        return project_world_gl(view, proj, point_world, self.width, self.height)

    def ray_world(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        view, proj = self._view_projection()
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

    def detach(self) -> None:
        if self.constraint_id >= 0:
            p.removeConstraint(self.constraint_id)
        p.removeBody(self.body_id)


def attach_wrist_camera2(
    robot_id: int,
    ee_link: int,
    hole_xy: tuple[float, float],
) -> WristCamera2:
    """After IK standoff: fix wrist_camera2 on ee_link so world pose matches fixed_cam."""
    eye_w = eye_for_hole(hole_xy)
    orn_w = camera_orientation(eye_w, target_for_hole(hole_xy))
    ee_pos, ee_orn = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)[:2]
    inv_pos, inv_orn = p.invertTransform(ee_pos, ee_orn)
    local_pos, local_orn = p.multiplyTransforms(inv_pos, inv_orn, eye_w, orn_w)

    urdf_path = os.path.join(URDF, "wrist_camera2.urdf")
    body_id = p.loadURDF(urdf_path, eye_w, orn_w, useFixedBase=False)
    cid = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link,
        childBodyUniqueId=body_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0.0, 0.0, 0.0],
        parentFramePosition=local_pos,
        parentFrameOrientation=local_orn,
        childFramePosition=[0.0, 0.0, 0.0],
        childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    p.changeConstraint(cid, maxForce=1e6)
    for _ in range(10):
        p.stepSimulation()

    return WristCamera2(
        body_id=body_id,
        robot_id=robot_id,
        ee_link=ee_link,
        hole_xy=hole_xy,
        constraint_id=cid,
    )
