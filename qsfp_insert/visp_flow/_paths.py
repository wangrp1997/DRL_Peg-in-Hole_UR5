import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
VISP_THIRD_PARTY = os.path.join(REPO_ROOT, "third_party", "visp")
TEACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teach")
NATIVE_BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "build")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.isdir(NATIVE_BUILD_DIR) and NATIVE_BUILD_DIR not in sys.path:
    sys.path.insert(0, NATIVE_BUILD_DIR)
