"""ViSP flow gains (mirrors third_party/visp examples)."""

# IBVS — servoUniversalRobotsIBVS.cpp
VISP_IBVS_LAMBDA = 0.5
VISP_IBVS_ERROR_TOL = 5e-5  # official convergence_threshold (sumSquare)
VISP_IBVS_MAX_STEPS = 28800
# Pixel RMS envelope to start pure IBVS (stop PnP before GT converges; official ~hand-click close)
VISP_IBVS_START_PX = 35.0
VISP_PNP_PREFLIGHT_MAX_STEPS = 14400  # separate kabsch stage; must stop before GT converge

# Photometric fine — photometricVisualServoing.cpp
VISP_DVS_LAMBDA = 30.0
VISP_DVS_DEPTH_Z = 0.10  # vpFeatureLuminance plane depth [m] at teach pose
VISP_DVS_ERROR_TOL = 10000.0  # photometricVisualServoing.cpp: while (normError > 10000)
VISP_DVS_MAX_STEPS = 200

# Scene
COARSE_STANDOFF = 0.035
