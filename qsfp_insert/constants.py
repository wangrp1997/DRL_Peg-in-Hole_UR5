"""QSFP-DD peg-in-hole nominal dimensions (metres)."""

PEG_W = 0.0184
PEG_H = 0.0085
PEG_L = 0.08
HOLE_DEPTH = 0.05

TABLE_TOP_Z = 0.62
FIXTURE_HEIGHT = 0.08
FIXTURE_TOP_Z = TABLE_TOP_Z + FIXTURE_HEIGHT
FIXTURE_CENTER_Z = TABLE_TOP_Z + FIXTURE_HEIGHT / 2
# Hole plate URDF origin = hole mouth (z=0); body extends down HOLE_DEPTH to z=-HOLE_DEPTH.
# Bottom of plate sits on fixture top → mouth is one plate thickness above fixture.
PLATE_TOP_Z = FIXTURE_TOP_Z + HOLE_DEPTH
HOLE_BOTTOM_Z = PLATE_TOP_Z - HOLE_DEPTH
INSERTED_TIP_Z = HOLE_BOTTOM_Z + 0.003

HOLE_XY = (0.55, 0.0)
ROBOT_BASE_Z = 0.62
REST_POSES = [0, -1.57, 1.57, -1.5, -1.57, 0.0]

EE_LINEAR_STEP = 0.002

# Cartesian velocity servo (servo_align / future 6D visual servo)
CART_XY_GAIN = 2.0
CART_Z_GAIN = 2.0
CART_ROT_GAIN = 3.0
CART_MAX_LIN = 0.02
CART_MAX_ANG = 0.8
CART_MAX_QDOT = 1.5
CART_LAMBDA = 0.08

ALIGN_XY_TOL = 0.0005
ALIGN_Z_STANDOFF_MIN = 0.003
ALIGN_Z_STANDOFF_MAX = 0.012
ALIGN_Z_NOMINAL = 0.006
ALIGN_ANG_TOL = 0.026
