import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, "urdf")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
