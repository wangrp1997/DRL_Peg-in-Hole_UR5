"""XFeat peg-ROI homography — accelerated_features/realtime_demo.py on peg crop only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from template_flow._paths import ensure_xfeat_path
from template_flow.tuning import (
    HOMOGRAPHY_RANSAC,
    MATCH_COSSIM,
    PEG_HOMOGRAPHY_MIN_INLIERS,
    PEG_SLIDE_STEP,
    XFEAT_TOP_K,
    ZONE_MARGIN,
    ZONE_SEARCH,
)

_XFeat = None


def _get_xfeat():
    global _XFeat
    if _XFeat is None:
        ensure_xfeat_path()
        from modules.xfeat import XFeat  # type: ignore[import-not-found]
        _XFeat = XFeat(top_k=XFEAT_TOP_K)
    return _XFeat


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(gray)


def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
    return torch.tensor(bgr).permute(2, 0, 1).float()[None]


def _bbox(
    points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    margin: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, int(min(xs) - margin))
    y0 = max(0, int(min(ys) - margin))
    x1 = min(width, int(max(xs) + margin))
    y1 = min(height, int(max(ys) + margin))
    return x0, y0, max(x0 + 8, x1), max(y0 + 8, y1)


def _expand_bbox(
    x0: int, y0: int, x1: int, y1: int,
    search: int, width: int, height: int,
    *,
    y_min: int | None = None,
) -> tuple[int, int, int, int]:
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw = (x1 - x0) / 2.0 + search
    hh = (y1 - y0) / 2.0 + search
    sx0 = max(0, int(cx - hw))
    sy0 = max(0, int(cy - hh))
    if y_min is not None:
        sy0 = max(sy0, y_min)
    sx1 = min(width, int(cx + hw))
    sy1 = min(height, int(cy + hh))
    return sx0, sy0, max(sx0 + 8, sx1), max(sy0 + 8, sy1)


@dataclass(frozen=True)
class RoiCropRef:
    teach_box: tuple[int, int, int, int]
    search_box: tuple[int, int, int, int]
    xfeat_ref: dict[str, Any]


# backward alias
PegCropRef = RoiCropRef


def build_roi_crop_ref(
    ref_bgr: np.ndarray,
    zone_uv: tuple[tuple[float, float], ...],
    *,
    mask_above_y: float | None = None,
) -> RoiCropRef:
    h, w = ref_bgr.shape[:2]
    tx0, ty0, tx1, ty1 = _bbox(zone_uv, ZONE_MARGIN, w, h)
    y_min = int(mask_above_y) if mask_above_y is not None else None
    sx0, sy0, sx1, sy1 = _expand_bbox(
        tx0, ty0, tx1, ty1, ZONE_SEARCH, w, h, y_min=y_min,
    )
    teach_crop = ref_bgr[ty0:ty1, tx0:tx1]
    return RoiCropRef(
        teach_box=(tx0, ty0, tx1, ty1),
        search_box=(sx0, sy0, sx1, sy1),
        xfeat_ref=_get_xfeat().detectAndCompute(_to_tensor(teach_crop), top_k=XFEAT_TOP_K)[0],
    )


def build_peg_crop_ref(ref_bgr: np.ndarray, peg_uv: tuple[tuple[float, float], ...]) -> RoiCropRef:
    return build_roi_crop_ref(ref_bgr, peg_uv)


def _homography_mnn(
    ref_feats: dict[str, Any],
    cur_bgr: np.ndarray,
    *,
    min_inliers: int,
) -> np.ndarray | None:
    if ref_feats["descriptors"].shape[0] < 6:
        return None
    xfeat = _get_xfeat()
    current = xfeat.detectAndCompute(_to_tensor(cur_bgr), top_k=XFEAT_TOP_K)[0]
    if current["descriptors"].shape[0] < 6:
        return None
    idx0, idx1 = xfeat.match(
        ref_feats["descriptors"], current["descriptors"], MATCH_COSSIM,
    )
    if len(idx0) <= 4:
        return None
    points1 = ref_feats["keypoints"][idx0].cpu().numpy()
    points2 = current["keypoints"][idx1].cpu().numpy()
    H, inliers = cv2.findHomography(
        points1, points2, cv2.USAC_MAGSAC, HOMOGRAPHY_RANSAC,
        maxIters=700, confidence=0.995,
    )
    if H is None or inliers is None:
        return None
    if int(inliers.flatten().sum()) < min_inliers:
        return None
    return H


def _crop_to_full(
    H_crop: np.ndarray,
    teach_off: tuple[int, int],
    cur_off: tuple[int, int],
) -> np.ndarray:
    tx0, ty0 = teach_off
    cx0, cy0 = cur_off
    t_off = np.array([[1, 0, -tx0], [0, 1, -ty0], [0, 0, 1]], dtype=np.float64)
    c_off = np.array([[1, 0, cx0], [0, 1, cy0], [0, 0, 1]], dtype=np.float64)
    return c_off @ H_crop @ t_off


def _slide_match_roi(
    teach_crop: np.ndarray,
    cur_bgr: np.ndarray,
    teach_box: tuple[int, int, int, int],
    search_box: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Equal-size patches in search window — XFeat match_xfeat (demo API)."""
    tx0, ty0, tx1, ty1 = teach_box
    sx0, sy0, sx1, sy1 = search_box
    tw, th = tx1 - tx0, ty1 - ty0
    if tw < 8 or th < 8 or sx1 - sx0 < tw or sy1 - sy0 < th:
        return None
    t0 = _to_tensor(teach_crop)
    xfeat = _get_xfeat()
    ref_out = xfeat.detectAndCompute(t0, top_k=XFEAT_TOP_K)[0]
    if ref_out["descriptors"].shape[0] < 6:
        return None
    best_inl = 0
    best: tuple[np.ndarray, int, int] | None = None
    step = max(8, PEG_SLIDE_STEP)
    for y in range(sy0, sy1 - th + 1, step):
        for x in range(sx0, sx1 - tw + 1, step):
            patch = cur_bgr[y : y + th, x : x + tw]
            out1 = xfeat.detectAndCompute(_to_tensor(patch), top_k=XFEAT_TOP_K)[0]
            if out1["descriptors"].shape[0] < 6:
                continue
            try:
                mk0, mk1 = xfeat.match_xfeat(t0, _to_tensor(patch), top_k=XFEAT_TOP_K)
            except IndexError:
                continue
            if len(mk0) < 6:
                continue
            H, inliers = cv2.findHomography(
                mk0, mk1, cv2.USAC_MAGSAC, HOMOGRAPHY_RANSAC,
                maxIters=700, confidence=0.995,
            )
            if H is None or inliers is None:
                continue
            n = int(inliers.flatten().sum())
            if n > best_inl:
                best_inl, best = n, (H, x, y)
    if best is None or best_inl < PEG_HOMOGRAPHY_MIN_INLIERS:
        return None
    H_crop, cx0, cy0 = best
    return _crop_to_full(H_crop, (tx0, ty0), (cx0, cy0))


def estimate_homography_roi(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    roi_ref: RoiCropRef,
    *,
    allow_slide: bool = True,
) -> np.ndarray | None:
    tx0, ty0, tx1, ty1 = roi_ref.teach_box
    sx0, sy0, sx1, sy1 = roi_ref.search_box
    if tx1 - tx0 < 8 or ty1 - ty0 < 8 or sx1 - sx0 < 8 or sy1 - sy0 < 8:
        return None
    teach_crop = ref_bgr[ty0:ty1, tx0:tx1]
    cur_crop = cur_bgr[sy0:sy1, sx0:sx1]
    H_crop = _homography_mnn(
        roi_ref.xfeat_ref, cur_crop, min_inliers=PEG_HOMOGRAPHY_MIN_INLIERS,
    )
    if H_crop is not None:
        return _crop_to_full(H_crop, (tx0, ty0), (sx0, sy0))
    if not allow_slide:
        return None
    H_slide = _slide_match_roi(teach_crop, cur_bgr, roi_ref.teach_box, roi_ref.search_box)
    if H_slide is not None:
        return H_slide
    if teach_crop.size == 0 or cur_crop.size == 0:
        return None
    xfeat = _get_xfeat()
    out_ref = xfeat.detectAndCompute(_to_tensor(teach_crop), top_k=XFEAT_TOP_K)[0]
    out_cur = xfeat.detectAndCompute(_to_tensor(cur_crop), top_k=XFEAT_TOP_K)[0]
    if out_ref["descriptors"].shape[0] < 6 or out_cur["descriptors"].shape[0] < 6:
        return None
    try:
        mk0, mk1 = xfeat.match_xfeat(
            _to_tensor(teach_crop), _to_tensor(cur_crop), top_k=XFEAT_TOP_K,
        )
    except IndexError:
        return None
    if len(mk0) <= 4:
        return None
    H_crop, inliers = cv2.findHomography(
        mk0, mk1, cv2.USAC_MAGSAC, HOMOGRAPHY_RANSAC,
        maxIters=700, confidence=0.995,
    )
    if H_crop is None or inliers is None:
        return None
    if int(inliers.flatten().sum()) < PEG_HOMOGRAPHY_MIN_INLIERS:
        return None
    return _crop_to_full(H_crop, (tx0, ty0), (sx0, sy0))


def estimate_homography_peg_roi(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    peg_ref: RoiCropRef,
) -> np.ndarray | None:
    return estimate_homography_roi(ref_bgr, cur_bgr, peg_ref)


def warp_uv_points(
    H: np.ndarray,
    uv: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> list[tuple[float, float]]:
    pts = np.array(uv, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return [(float(p[0][0]), float(p[0][1])) for p in out]


def uv_in_image(
    uv: list[tuple[float, float]],
    width: int,
    height: int,
) -> bool:
    for u, v in uv:
        if not (0.0 <= u < width and 0.0 <= v < height):
            return False
    return True
