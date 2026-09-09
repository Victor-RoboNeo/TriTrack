"""Install a namespace for ``whole_body_tracking.sirac`` without importing Isaac Lab.

The real ``whole_body_tracking/__init__.py`` imports gym task registration and
therefore ``omni.kit``. Unit tests and dummy smokes must not require Kit.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent / "whole_body_tracking"


def install() -> None:
    existing = sys.modules.get("whole_body_tracking")
    if existing is not None and getattr(existing, "__path__", None):
        # Already a real or stub package.
        if getattr(existing, "_sirac_isaacfree", False) or "omni" in sys.modules:
            return
        # A partial failed import: replace only if it has no sirac attr and no path use yet.
    m = types.ModuleType("whole_body_tracking")
    m.__path__ = [str(_PKG_DIR)]
    m.__file__ = str(_PKG_DIR / "__init__.py")
    m._sirac_isaacfree = True  # type: ignore[attr-defined]
    sys.modules["whole_body_tracking"] = m
