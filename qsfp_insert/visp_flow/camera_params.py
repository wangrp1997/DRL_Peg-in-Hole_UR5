"""PyBullet camera K → ViSP CameraParameters."""
from __future__ import annotations

import numpy as np

from visp_flow.require_visp import require_visp_python


def camera_parameters_from_K(K: np.ndarray):
    require_visp_python()
    from visp.core import CameraParameters

    fx, fy = float(K[0, 0]), float(K[1, 1])
    u0, v0 = float(K[0, 2]), float(K[1, 2])
    return CameraParameters(fx, fy, u0, v0)


def uv_to_metric(cam, u: float, v: float) -> tuple[float, float]:
    from visp.core import PixelMeterConversion

    return PixelMeterConversion.convertPoint(cam, float(u), float(v))
