import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "sim_wrist2")
DEFAULT_RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "sim_wrist2_raw")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
