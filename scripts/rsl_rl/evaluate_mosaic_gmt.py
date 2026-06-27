# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Backward-compatible entry point; delegates to :mod:`evaluate_policy`.

Prefer ``scripts/rsl_rl/evaluate_policy.py`` for new scripts and documentation.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "evaluate_policy.py"), run_name="__main__")
