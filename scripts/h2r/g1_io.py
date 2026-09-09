"""Load G1 npz torso + wrists in world (meters, Isaac)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import LW_I, RW_I, TORSO_I


def load_g1_tlr(path: str | Path) -> tuple[np.ndarray, float]:
    d = np.load(str(path))
    pos = np.asarray(d["body_pos_w"], dtype=np.float64)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    tlr = np.stack([pos[:, TORSO_I], pos[:, LW_I], pos[:, RW_I]], axis=1)
    return tlr, fps


def overlay_npz(src: str | Path, dst: str | Path, tlr_world: np.ndarray) -> None:
    """Copy G1/P1 npz and replace torso + wrist world positions. Other bodies untouched."""
    src = Path(src)
    dst = Path(dst)
    d = dict(np.load(str(src)))
    pos = np.array(d["body_pos_w"], dtype=np.float32, copy=True)
    t = min(pos.shape[0], tlr_world.shape[0])
    pos[:t, TORSO_I] = tlr_world[:t, 0]
    pos[:t, LW_I] = tlr_world[:t, 1]
    pos[:t, RW_I] = tlr_world[:t, 2]
    if tlr_world.shape[0] < pos.shape[0]:
        pos[t:, TORSO_I] = tlr_world[-1, 0]
        pos[t:, LW_I] = tlr_world[-1, 1]
        pos[t:, RW_I] = tlr_world[-1, 2]
    d["body_pos_w"] = pos
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(dst), **d)
