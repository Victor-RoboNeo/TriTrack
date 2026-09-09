"""Run SIRAC unit tests without requiring pytest to be installed as a package.

    python tests/sirac/run_all.py
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "source"))
import sirac_isaacfree  # noqa: E402

sirac_isaacfree.install()
sys.path.insert(0, str(ROOT / "source" / "whole_body_tracking"))
sys.path.insert(0, str(HERE))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    modules = [_load(p) for p in sorted(HERE.glob("test_*.py"))]
    failed = 0
    total = 0
    for mod in modules:
        tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
        for fn in tests:
            total += 1
            try:
                fn()
                print(f"PASS  {mod.__name__}.{fn.__name__}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {mod.__name__}.{fn.__name__}: {e}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
