"""IL scene cameras — fixed2 on opposite side of hole axis from fixed_cam / wrist2."""

# Original fixed_cam: (0.10, +0.06, 0.05) — flip Y to opposite flank (avoid wrist2 clash).
FIXED_CAM2_EYE_OFFSET = (0.10, -0.06, 0.05)
FIXED_CAM2_ROLL_DEG = 90.0
IL_SIM_HZ = 240.0  # physics rate; rgb.mp4 sampled at collect --fps, not every sim step
IL_PHASE_PAUSE_S = 1.0  # hold before/after servo in video (recorded at --fps)
IL_HEADLESS_HOLD_S = 1.0
IL_RECORD_HOLD_STEPS = 40  # physics settle before hold segment
# Expert servo caps (default sim: lin 0.05, ang 1.2, qdot 3.0) — slightly slower for IL video
IL_CART_MAX_LIN = 0.03
IL_CART_MAX_ANG = 0.85
IL_CART_MAX_QDOT = 2.0
TARGET_STANDOFF_MIN_MM = 3.0
TARGET_STANDOFF_MAX_MM = 3.8
TARGET_Z_BAND_M = 0.0005  # ±0.5 mm, same as record_sim
