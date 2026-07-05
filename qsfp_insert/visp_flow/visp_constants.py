"""ViSP flow gains (mirrors third_party/visp examples)."""

# IBVS coarse — servoUniversalRobotsIBVS.cpp + PnP blend when far
VISP_IBVS_LAMBDA = 0.5
VISP_IBVS_ERROR_TOL = 1e-4
VISP_IBVS_MAX_STEPS = 28800  # match corner_servo SERVO_MAX_STEPS
VISP_IBVS_BLEND_PX = 15.0  # ViSP fine only below this RMS px (far → kabsch-style PnP)

# Photometric fine — photometricVisualServoing.cpp
VISP_DVS_LAMBDA = 30.0
VISP_DVS_DEPTH_Z = 0.10  # vpFeatureLuminance plane depth [m] at teach pose
VISP_DVS_ERROR_TOL = 10000.0  # photometricVisualServoing.cpp: while (normError > 10000)
VISP_DVS_MAX_STEPS = 200

# Scene
COARSE_STANDOFF = 0.035
