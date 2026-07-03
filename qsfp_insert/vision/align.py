"""6-DOF alignment error from matched peg/hole image corners (geometric corner servo)."""
from __future__ import annotations

import itertools
import math
from typing import Literal

import cv2
import numpy as np
import pybullet as p

from constants import (
    CART_LAMBDA,
    CORNER_ALIGN_METHOD,
    CORNER_SERVO_STANDOFF,
    IBVS_BLEND_PX,
    IBVS_LAMBDA,
    IBVS_PIXEL_TOL,
    IBVS_TWIST_GAIN,
    MIN_CORNERS_VISIBLE,
    PEG_H,
    PEG_W,
    PLATE_TOP_Z,
)
from geometry import AlignmentMetrics, metrics_converged, rpy_error
from sim.fixed_camera import FixedCamera
from vision.corners import HOLE_HALF_X, HOLE_HALF_Y, ImageKeypoints

AlignMethod = Literal["kabsch", "ibvs"]


def _set_by_name(sets: list[ImageKeypoints]) -> tuple[ImageKeypoints, ImageKeypoints]:
    hole = next(s for s in sets if s.name == "hole")
    peg = next(s for s in sets if s.name == "peg")
    return hole, peg


def _insert_up(hole_orn) -> np.ndarray:
    """World +Z offset from hole mouth toward peg standoff (opposite insert axis)."""
    mat = np.array(p.getMatrixFromQuaternion(hole_orn), dtype=np.float64).reshape(3, 3)
    return mat[:, 2]


def _sample_depth(depth_map: np.ndarray, cam: FixedCamera, u: float, v: float) -> float | None:
    ui = int(round(u))
    vi = int(round(v))
    ui = max(0, min(cam.width - 1, ui))
    vi = max(0, min(cam.height - 1, vi))
    depth_m = float(depth_map[vi, ui])
    if depth_m >= cam.far * 0.99:
        return None
    return depth_m


def _corner_world_on_plane(
    cam: FixedCamera,
    u: float,
    v: float,
    plane_z: float,
) -> np.ndarray | None:
    pt = cam.world_point_on_plane(u, v, plane_z)
    if pt is None:
        return None
    return np.array(pt, dtype=np.float64)


def _peg_corner_world(
    cam: FixedCamera,
    u: float,
    v: float,
    hole_pt: np.ndarray,
    eye: np.ndarray,
    standoff: float,
    up: np.ndarray,
    depth_map: np.ndarray | None,
) -> np.ndarray | None:
    if depth_map is not None:
        depth_m = _sample_depth(depth_map, cam, u, v)
        if depth_m is not None:
            origin, direction = cam.ray_world(u, v)
            return origin + depth_m * direction
    origin, direction = cam.ray_world(u, v)
    th = float(np.linalg.norm(hole_pt - eye))
    tp = th - standoff * float(np.dot(up, direction))
    if tp <= 0.0:
        return None
    return origin + tp * direction


def _quat_from_corner_pts(pts: np.ndarray) -> tuple[float, float, float, float]:
    p0 = pts[0]
    p1 = pts[1]
    p3 = pts[min(3, len(pts) - 1)]
    x = p1 - p0
    x /= np.linalg.norm(x)
    y = p3 - p0
    z = np.cross(x, y)
    zn = np.linalg.norm(z)
    if zn < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    z /= zn
    y = np.cross(z, x)
    return _mat3_to_quat(np.column_stack([x, y, z]))


def _mat3_to_quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    x = rot[:, 0]
    y = rot[:, 1]
    z = np.cross(x, y)
    zn = np.linalg.norm(z)
    if zn < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    z = z / zn
    y = np.cross(z, x)
    mat = np.column_stack([x, y, z])
    tr = float(np.trace(mat))
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0) * 2.0)
        return (
            (mat[2, 1] - mat[1, 2]) / s,
            (mat[0, 2] - mat[2, 0]) / s,
            (mat[1, 0] - mat[0, 1]) / s,
            0.25 * s,
        )
    if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = float(np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0)
        return (0.25 * s, (mat[0, 1] + mat[1, 0]) / s, (mat[0, 2] + mat[2, 0]) / s, (mat[2, 1] - mat[1, 2]) / s)
    if mat[1, 1] > mat[2, 2]:
        s = float(np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0)
        return ((mat[0, 1] + mat[1, 0]) / s, 0.25 * s, (mat[1, 2] + mat[2, 1]) / s, (mat[0, 2] - mat[2, 0]) / s)
    s = float(np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0)
    return ((mat[0, 2] + mat[2, 0]) / s, (mat[1, 2] + mat[2, 1]) / s, 0.25 * s, (mat[1, 0] - mat[0, 1]) / s)


def _world_to_cam(cam: FixedCamera, pt_world: np.ndarray) -> np.ndarray:
    cam_pos, cam_orn = cam._pose()
    inv_pos, inv_orn = p.invertTransform(cam_pos, cam_orn)
    pt_cam, _ = p.multiplyTransforms(inv_pos, inv_orn, pt_world.tolist(), [0.0, 0.0, 0.0, 1.0])
    return np.array(pt_cam, dtype=np.float64)


def _cam_rot_to_world(cam: FixedCamera) -> np.ndarray:
    _, cam_orn = cam._pose()
    return np.array(p.getMatrixFromQuaternion(cam_orn), dtype=np.float64).reshape(3, 3)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _cam_rot_world_to_cam(cam: FixedCamera) -> np.ndarray:
    return _cam_rot_to_world(cam).T


def _depth_cam_z(cam: FixedCamera, depth_map: np.ndarray, u: float, v: float) -> float | None:
    """Ray depth from render → camera-frame Z for interaction matrix."""
    ray_len = _sample_depth(depth_map, cam, u, v)
    if ray_len is None:
        return None
    origin, direction = cam.ray_world(u, v)
    pt_cam = _world_to_cam(cam, origin + ray_len * direction)
    z = _cam_depth(pt_cam)
    return z if z > 1e-6 else None


def _ee_to_cam_point_jacobian(r_cw: np.ndarray, r_world: np.ndarray) -> np.ndarray:
    """Map peg-tip twist (world) → 3D point twist (camera): v_p = v + ω×r."""
    s = _skew(r_world)
    return np.vstack([np.hstack([r_cw, -r_cw @ s]), np.hstack([np.zeros((3, 3)), r_cw])])


def _interaction_matrix_point(cam: FixedCamera, u: float, v: float, z: float) -> np.ndarray:
    mat = _interaction_matrix_points(cam, [(u, v)], [z])
    if mat is None:
        raise ValueError("invalid depth for interaction matrix")
    return mat


def _corner_assignment(hole_kp: ImageKeypoints, peg_kp: ImageKeypoints) -> list[tuple[int, int]] | None:
    """Best peg↔hole corner pairing (handles yaw ambiguity via 4! search)."""
    best_pairs: list[tuple[int, int]] | None = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(4)):
        cost = 0.0
        n = 0
        for h, p in enumerate(perm):
            if not (hole_kp.visible[h] and peg_kp.visible[p]):
                continue
            du = peg_kp.uv[p][0] - hole_kp.uv[h][0]
            dv = peg_kp.uv[p][1] - hole_kp.uv[h][1]
            cost += du * du + dv * dv
            n += 1
        if n < MIN_CORNERS_VISIBLE or cost >= best_cost:
            continue
        best_cost = cost
        best_pairs = [
            (h, p)
            for h, p in enumerate(perm)
            if hole_kp.visible[h] and peg_kp.visible[p]
        ]
    return best_pairs


def ibvs_pixel_error(hole_kp: ImageKeypoints, peg_kp: ImageKeypoints) -> float | None:
    """RMS pixel residual ||s_peg − s_hole|| over best-matched corners."""
    pairs = _corner_assignment(hole_kp, peg_kp)
    if pairs is None:
        return None
    sq = 0.0
    for h, p in pairs:
        du = peg_kp.uv[p][0] - hole_kp.uv[h][0]
        dv = peg_kp.uv[p][1] - hole_kp.uv[h][1]
        sq += du * du + dv * dv
    return float(math.sqrt(sq / len(pairs)))


def ibvs_pixels_converged(hole_kp: ImageKeypoints, peg_kp: ImageKeypoints) -> bool:
    err = ibvs_pixel_error(hole_kp, peg_kp)
    return err is not None and err <= IBVS_PIXEL_TOL


def _ibvs_feature_error(hole_kp: ImageKeypoints, peg_kp: ImageKeypoints) -> np.ndarray | None:
    pairs = _corner_assignment(hole_kp, peg_kp)
    if pairs is None:
        return None
    err: list[float] = []
    for h, p in pairs:
        err.extend([peg_kp.uv[p][0] - hole_kp.uv[h][0], peg_kp.uv[p][1] - hole_kp.uv[h][1]])
    return np.array(err, dtype=np.float64)


def _hole_corner_cam_z(cam: FixedCamera, u: float, v: float) -> float | None:
    pt = cam.world_point_on_plane(u, v, PLATE_TOP_Z)
    if pt is None:
        return None
    return _corner_cam_z(cam, pt)


def _damped_pinv(j: np.ndarray, lam: float = CART_LAMBDA) -> np.ndarray:
    n = j.shape[0]
    return j.T @ np.linalg.inv(j @ j.T + lam**2 * np.eye(n))


def _paired_corners_3d(
    cam: FixedCamera,
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
    z_hole: float,
    standoff: float,
    hole_orn,
    depth_map: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], list[np.ndarray]] | None:
    eye = np.array(cam._pose()[0], dtype=np.float64)
    up = _insert_up(hole_orn)
    hole_pts: list[np.ndarray] = []
    peg_pts: list[np.ndarray] = []
    peg_uv: list[tuple[float, float]] = []

    for i in range(4):
        if not (hole_kp.visible[i] and peg_kp.visible[i]):
            continue
        hu, hv = hole_kp.uv[i]
        pu, pv = peg_kp.uv[i]
        hw = _corner_world_on_plane(cam, hu, hv, z_hole)
        if hw is None:
            return None
        pw = _peg_corner_world(cam, pu, pv, hw, eye, standoff, up, depth_map)
        if pw is None:
            return None
        hole_pts.append(hw)
        peg_pts.append(pw)
        peg_uv.append((pu, pv))

    if len(hole_pts) < MIN_CORNERS_VISIBLE:
        return None

    peg_cam = [_world_to_cam(cam, pw) for pw in peg_pts]
    return np.stack(hole_pts), np.stack(peg_pts), peg_uv, peg_cam


def _interaction_matrix_points(
    cam: FixedCamera,
    uv: list[tuple[float, float]],
    z_cam: list[float],
) -> np.ndarray | None:
    fx = cam.K[0, 0]
    fy = cam.K[1, 1]
    cx = cam.K[0, 2]
    cy = cam.K[1, 2]
    rows: list[list[float]] = []
    for (u, v), z in zip(uv, z_cam):
        if z <= 1e-6:
            return None
        uc = u - cx
        vc = v - cy
        rows.append([-fx / z, 0.0, uc / z, uc * vc / fx, -(fx * fx + uc * uc) / fx, vc])
        rows.append([0.0, -fy / z, vc / z, (fy * fy + vc * vc) / fy, -uc * vc / fy, -uc])
    return np.array(rows, dtype=np.float64)


def _rect_object_points(half_x: float, half_y: float) -> np.ndarray:
    return np.array(
        [
            [-half_x, -half_y, 0.0],
            [half_x, -half_y, 0.0],
            [half_x, half_y, 0.0],
            [-half_x, half_y, 0.0],
        ],
        dtype=np.float64,
    )


def _solve_planar_pose(
    uv: list[tuple[float, float]],
    object_pts: np.ndarray,
    cam: FixedCamera,
    score_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """4 coplanar corners → 6D pose: IPPE init, 3-point disambiguation if needed, LM refine."""
    n = len(uv)
    if n < 4 or object_pts.shape[0] < 4:
        return None
    img = np.array(uv, dtype=np.float64)
    obj = np.asarray(object_pts[:4], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, cam.K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        return None
    if score_indices is None:
        score_indices = list(range(4))

    def _valid(rvec, tvec) -> bool:
        rot_c, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        z = rot_c @ obj.T + t.reshape(3, 1)
        return not np.any(z[2, :] <= 0.0)

    def _reproj_err(rvec, tvec, indices: list[int]) -> float:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, dist)
        err = 0.0
        for i in indices:
            du = float(proj[i, 0, 0]) - img[i, 0]
            dv = float(proj[i, 0, 1]) - img[i, 1]
            err += du * du + dv * dv
        return err

    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    for rvec, tvec in zip(rvecs, tvecs):
        if not _valid(rvec, tvec):
            continue
        candidates.append((rvec, tvec, _reproj_err(rvec, tvec, score_indices)))

    # When corner 0 is inferred (score on 1–3), pick IPPE branch via 3-point PnP + parallelogram p0.
    if score_indices == [1, 2, 3] and len(candidates) > 1:
        img3 = img[[1, 2, 3]]
        obj3 = obj[[1, 2, 3]]
        ok3, rvecs3, tvecs3, _ = cv2.solvePnPGeneric(obj3, img3, cam.K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if ok3 and len(rvecs3) > 0:
            para0 = img[0]
            best_pose: tuple[np.ndarray, np.ndarray] | None = None
            best_d = float("inf")
            for rvec, tvec in zip(rvecs3, tvecs3):
                if not _valid(rvec, tvec):
                    continue
                proj, _ = cv2.projectPoints(obj[[0]], rvec, tvec, cam.K, dist)
                d = float((proj[0, 0, 0] - para0[0]) ** 2 + (proj[0, 0, 1] - para0[1]) ** 2)
                if d < best_d:
                    best_d = d
                    best_pose = (rvec, tvec)
            if best_pose is not None:
                candidates = [(best_pose[0], best_pose[1], _reproj_err(best_pose[0], best_pose[1], score_indices))]

    if not candidates:
        return None
    rvec, tvec, _ = min(candidates, key=lambda c: c[2])
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        cam.K,
        dist,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rot_c, _ = cv2.Rodrigues(rvec)
    return rot_c, tvec.reshape(3)


def _planar_pose_indices(kp: ImageKeypoints) -> tuple[list[int], list[int]] | None:
    """All four corners visible; score IPPE on non-inferred corners only."""
    if sum(kp.visible) < 4:
        return None
    all_idx = list(range(4))
    score = [i for i in all_idx if not kp.inferred[i]]
    return all_idx, score if score else all_idx


def _cam_to_world(cam: FixedCamera, rot_c: np.ndarray, trans_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_wc = _cam_rot_to_world(cam)
    eye = np.array(cam._pose()[0], dtype=np.float64)
    return rot_wc @ rot_c, rot_wc @ trans_c + eye


def _matched_uv(
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[int]] | None:
    hole_uv: list[tuple[float, float]] = []
    peg_uv: list[tuple[float, float]] = []
    indices: list[int] = []
    for i in range(4):
        if hole_kp.visible[i] and peg_kp.visible[i]:
            hole_uv.append(hole_kp.uv[i])
            peg_uv.append(peg_kp.uv[i])
            indices.append(i)
    if len(indices) < MIN_CORNERS_VISIBLE:
        return None
    return hole_uv, peg_uv, indices


def _metrics_kabsch(
    cam: FixedCamera,
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
    standoff: float,
    hole_orn,
    depth_map: np.ndarray | None,
) -> AlignmentMetrics | None:
    """Parallelogram-inferred 4th corner (if any) + 4-point IPPE → peg 6D relative to hole."""
    del hole_orn, depth_map
    hole_idx = _planar_pose_indices(hole_kp)
    peg_idx = _planar_pose_indices(peg_kp)
    if hole_idx is None or peg_idx is None:
        return None
    hole_corners, hole_score = hole_idx
    peg_corners, peg_score = peg_idx
    hole_uv = [hole_kp.uv[i] for i in hole_corners]
    peg_uv = [peg_kp.uv[i] for i in peg_corners]
    hole_obj = _rect_object_points(HOLE_HALF_X, HOLE_HALF_Y)
    peg_obj = _rect_object_points(PEG_W / 2.0, PEG_H / 2.0)
    hole_pose = _solve_planar_pose(hole_uv, hole_obj, cam, hole_score)
    peg_pose = _solve_planar_pose(peg_uv, peg_obj, cam, peg_score)
    if hole_pose is None or peg_pose is None:
        return None
    rot_h, trans_h = _cam_to_world(cam, *hole_pose)
    rot_p, trans_p = _cam_to_world(cam, *peg_pose)
    rot_rel = rot_h.T @ rot_p
    trans_rel = rot_h.T @ (trans_p - trans_h)
    roll, pitch, yaw = rpy_error(_mat3_to_quat(rot_rel), (0.0, 0.0, 0.0, 1.0))
    return {
        "dx": float(trans_rel[0]),
        "dy": float(trans_rel[1]),
        "standoff": standoff,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "aligned": False,
    }


def _metrics_ibvs(
    cam: FixedCamera,
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
    standoff: float,
    hole_orn,
    depth_map: np.ndarray | None,
) -> AlignmentMetrics | None:
    """Pose error via planar PnP (same as kabsch); IBVS only drives control."""
    del depth_map
    return _metrics_kabsch(cam, hole_kp, peg_kp, standoff, hole_orn, None)


def _corner_cam_z(cam: FixedCamera, pt_world: tuple[float, float, float] | np.ndarray) -> float | None:
    pc = _world_to_cam(cam, np.array(pt_world, dtype=np.float64))
    z = _cam_depth(pc)
    return z if z > 1e-6 else None


def _hole_corner_cam_z(cam: FixedCamera, u: float, v: float) -> float | None:
    pt = cam.world_point_on_plane(u, v, PLATE_TOP_Z)
    if pt is None:
        return None
    return _corner_cam_z(cam, pt)


def _ibvs_image_jacobian(
    cam: FixedCamera,
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
    peg_corners_world: list[tuple[float, float, float]],
    tip_world: tuple[float, float, float],
    depth_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Eye-to-hand J_img (2k×6): L at peg corners, matched peg↔hole via _corner_assignment."""
    pairs = _corner_assignment(hole_kp, peg_kp)
    e = _ibvs_feature_error(hole_kp, peg_kp)
    if pairs is None or e is None:
        return None

    r_cw = _cam_rot_world_to_cam(cam)
    tip = np.array(tip_world, dtype=np.float64)
    j_rows: list[np.ndarray] = []

    for h, p in pairs:
        pu, pv = peg_kp.uv[p]
        z = _corner_cam_z(cam, peg_corners_world[p])
        if z is None and depth_map is not None:
            z = _depth_cam_z(cam, depth_map, pu, pv)
        if z is None or z <= 1e-6:
            return None

        r = np.array(peg_corners_world[p], dtype=np.float64) - tip
        l_i = _interaction_matrix_point(cam, pu, pv, z)
        m_i = _ee_to_cam_point_jacobian(r_cw, r)
        j_rows.append(l_i @ m_i)

    if len(j_rows) < MIN_CORNERS_VISIBLE:
        return None
    return np.vstack(j_rows), e


def ibvs_joint_velocities(
    robot_id: int,
    peg_link: int,
    arm: list[int],
    cam: FixedCamera,
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
    peg_corners_world: list[tuple[float, float, float]],
    tip_world: tuple[float, float, float],
    depth_map: np.ndarray | None = None,
) -> np.ndarray | None:
    """q̇ = λ (J_img J_robot⁺)⁺ (−e); eye-to-hand IBVS in joint space (ViSP-style chain)."""
    from sim.cartesian_control import damped_pinv, jacobian_tip

    built = _ibvs_image_jacobian(cam, hole_kp, peg_kp, peg_corners_world, tip_world, depth_map)
    if built is None:
        return None
    j_img, e = built

    px = ibvs_pixel_error(hole_kp, peg_kp)
    if px is None:
        return None
    gain = IBVS_TWIST_GAIN * min(1.0, IBVS_BLEND_PX / max(px, 1.0))

    j_robot = jacobian_tip(robot_id, peg_link, arm)
    j_full = j_img @ damped_pinv(j_robot, IBVS_LAMBDA)
    return gain * damped_pinv(j_full, IBVS_LAMBDA) @ (-e)


def ibvs_twist_tip(
    cam: FixedCamera,
    hole_kp: ImageKeypoints,
    peg_kp: ImageKeypoints,
    peg_corners_world: list[tuple[float, float, float]],
    tip_world: tuple[float, float, float],
    depth_map: np.ndarray | None = None,
) -> np.ndarray | None:
    """Legacy Cartesian IBVS (superseded by ibvs_joint_velocities in the servo loop)."""
    built = _ibvs_image_jacobian(cam, hole_kp, peg_kp, peg_corners_world, tip_world, depth_map)
    if built is None:
        return None
    j_img, e = built
    px = ibvs_pixel_error(hole_kp, peg_kp)
    if px is None:
        return None
    gain = IBVS_TWIST_GAIN
    return gain * _damped_pinv(j_img, IBVS_LAMBDA) @ (-e)


def _cam_depth(pt_cam: np.ndarray) -> float:
    return float(-pt_cam[2])


def metrics_from_keypoints(
    keypoints: list[ImageKeypoints],
    cam: FixedCamera,
    hole_xy: tuple[float, float],
    hole_orn=None,
        depth_map: np.ndarray | None = None,
        standoff_hint: float | None = None,
        method: AlignMethod | None = None,
) -> AlignmentMetrics | None:
    """P/H corner pixels → 6D alignment error (planar PnP or 8-feature IBVS)."""
    del hole_xy
    if hole_orn is None:
        return None
    hole_kp, peg_kp = _set_by_name(keypoints)
    if sum(hole_kp.visible) < MIN_CORNERS_VISIBLE or sum(peg_kp.visible) < MIN_CORNERS_VISIBLE:
        return None

    standoff = standoff_hint if standoff_hint is not None else CORNER_SERVO_STANDOFF
    align_method = method if method is not None else CORNER_ALIGN_METHOD

    if align_method == "ibvs":
        m = _metrics_ibvs(cam, hole_kp, peg_kp, standoff, hole_orn, depth_map)
    else:
        m = _metrics_kabsch(cam, hole_kp, peg_kp, standoff, hole_orn, depth_map)

    if m is None:
        return None
    m["aligned"] = metrics_converged(m)
    return m
