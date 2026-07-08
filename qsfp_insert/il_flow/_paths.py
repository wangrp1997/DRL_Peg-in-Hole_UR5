import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
IL_ROOT = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(IL_ROOT, "urdf")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
