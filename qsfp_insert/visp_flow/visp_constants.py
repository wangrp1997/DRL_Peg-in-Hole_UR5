"""ViSP flow gains (mirrors third_party/visp examples)."""

# IBVS — servoUniversalRobotsIBVS.cpp
VISP_IBVS_LAMBDA = 0.5
VISP_IBVS_ERROR_TOL = 5e-5  # official convergence_threshold (sumSquare)
VISP_IBVS_MAX_STEPS = 28800
VISP_IBVS_START_PX = 35.0
VISP_IBVS_ABORT_XY_M = 0.012
VISP_IBVS_ABORT_STANDOFF_MIN = 0.001
VISP_IBVS_ABORT_STANDOFF_MAX = 0.022
VISP_IBVS_ABORT_ANG_RAD = 0.12
VISP_PNP_PREFLIGHT_MAX_STEPS = 14400

# Photometric fine — photometricVisualServoing.cpp
VISP_DVS_LAMBDA = 30.0
VISP_DVS_ERROR_TOL = 10000.0  # while (normError > 10000)
VISP_DVS_START_ERR = 10000.0  # gate: same scale as official loop (start near teach pose)
VISP_DVS_ABORT_ERR = 50000.0  # abort if ||e||² explodes during servo
VISP_DVS_MAX_STEPS = 200

# Scene
COARSE_STANDOFF = 0.035
