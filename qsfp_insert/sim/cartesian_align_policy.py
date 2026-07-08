"""Temporarily retarget Cartesian align Z to a chosen standoff (servo / record)."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from constants import (
    ALIGN_ANG_TOL,
    ALIGN_XY_TOL,
    ALIGN_Z_STANDOFF_MAX,
)


@contextmanager
def cartesian_align_target(
    target_z: float,
    *,
    z_band_m: float = 0.003,
) -> Iterator[None]:
    """Servo drives to target_z; aligned when |standoff − target_z| ≤ z_band_m."""
    import constants
    from sim import cartesian_control
    import vision.corner_servo as corner_servo_mod
    import geometry as geometry_mod

    orig_check = geometry_mod._check_aligned
    orig_c = constants.ALIGN_Z_NOMINAL
    orig_cc = cartesian_control.ALIGN_Z_NOMINAL
    orig_cs = corner_servo_mod.ALIGN_Z_NOMINAL
    orig_zmax_c = constants.ALIGN_Z_STANDOFF_MAX
    orig_zmax_cs = corner_servo_mod.ALIGN_Z_STANDOFF_MAX

    def _check(dx, dy, standoff, roll, pitch, yaw) -> bool:
        xy_ok = abs(dx) <= ALIGN_XY_TOL and abs(dy) <= ALIGN_XY_TOL
        z_ok = abs(standoff - target_z) <= z_band_m
        rpy_ok = (
            abs(roll) <= ALIGN_ANG_TOL
            and abs(pitch) <= ALIGN_ANG_TOL
            and abs(yaw) <= ALIGN_ANG_TOL
        )
        return xy_ok and z_ok and rpy_ok

    fine_z_max = target_z + 0.003

    geometry_mod._check_aligned = _check
    constants.ALIGN_Z_NOMINAL = target_z
    cartesian_control.ALIGN_Z_NOMINAL = target_z
    corner_servo_mod.ALIGN_Z_NOMINAL = target_z
    constants.ALIGN_Z_STANDOFF_MAX = fine_z_max
    corner_servo_mod.ALIGN_Z_STANDOFF_MAX = fine_z_max
    try:
        yield
    finally:
        geometry_mod._check_aligned = orig_check
        constants.ALIGN_Z_NOMINAL = orig_c
        cartesian_control.ALIGN_Z_NOMINAL = orig_cc
        corner_servo_mod.ALIGN_Z_NOMINAL = orig_cs
        constants.ALIGN_Z_STANDOFF_MAX = orig_zmax_c
        corner_servo_mod.ALIGN_Z_STANDOFF_MAX = orig_zmax_cs
