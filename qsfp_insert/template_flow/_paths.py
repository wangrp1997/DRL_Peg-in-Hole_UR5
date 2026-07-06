import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
TEACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teach")
XFEAT_ROOT = os.path.join(REPO_ROOT, "third_party", "accelerated_features")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def ensure_xfeat_path() -> str:
    if not os.path.isdir(XFEAT_ROOT):
        raise RuntimeError(
            f"XFeat not found at {XFEAT_ROOT}. "
            "git clone https://github.com/verlab/accelerated_features.git third_party/accelerated_features"
        )
    if XFEAT_ROOT not in sys.path:
        sys.path.insert(0, XFEAT_ROOT)
    return XFEAT_ROOT
