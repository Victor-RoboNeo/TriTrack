"""Offline motion-tools: SONIC clip resampling.

``resample.py`` temporally resamples motion-clip NPZs (speed factor) while keeping the
output format identical to the training clips, so resampled directories slot into the
existing pipeline (point ``MOTION_DIR`` at the output directory). The package is
import-isolated from runtime command/observation code in ``mdp/``.
"""
