#!/usr/bin/env python3
"""Top-level entry point for the synthetic-motion CLI.

Sets ``WHOLE_BODY_TRACKING_NO_TASKS=1`` before importing the package so that the
``whole_body_tracking/__init__.py`` doesn't pull in Isaac (the synth writer is
fully offline). The body-names dump tool — which DOES need Isaac — is invoked
separately via ``python -m whole_body_tracking.synth.dump_body_names``.

Usage::

    python scripts/synth_cli.py list
    python scripts/synth_cli.py generate-all \\
        --seed-dir /home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/test/loco/walk_forward \\
        --out-root /home/lsn/Datasets/SONIC_npzs/g1/npz_synth
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("WHOLE_BODY_TRACKING_NO_TASKS", "1")

    # Source-tree layout: ensure the editable package is importable when running from
    # the repo root without an install. Mirrors how scripts/rsl_rl/play.py is invoked.
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    src_dir = os.path.join(repo_root, "source", "whole_body_tracking")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from whole_body_tracking.synth.cli import main as cli_main

    return int(cli_main(sys.argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
