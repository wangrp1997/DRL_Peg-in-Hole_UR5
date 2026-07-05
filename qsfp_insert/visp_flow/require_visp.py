"""Ensure ViSP Python bindings are importable."""
from __future__ import annotations

from visp_flow._paths import VISP_THIRD_PARTY


def require_visp_python() -> None:
    try:
        import visp.core  # noqa: F401
        import visp.visual_features  # noqa: F401
        import visp.vs  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ViSP Python 模块未安装。请编译 third_party/visp 并启用 Python 绑定，"
            "例如:\n"
            f"  cd {VISP_THIRD_PARTY} && mkdir -p build && cd build\n"
            "  cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON3=ON\n"
            "  make -j$(nproc) && pip install ./modules/python/stubs\n"
            "详见 qsfp_insert/README.md「ViSP 完整流程」。"
        ) from exc
