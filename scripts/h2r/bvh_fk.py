"""Minimal BVH FK for SOMA Uniform. Positions in meters, Isaac Z-up."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CM = 0.01


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    z = np.zeros_like(a)
    o = np.ones_like(a)
    r = np.zeros(a.shape + (3, 3), dtype=np.float64)
    r[..., 0, 0] = c
    r[..., 0, 1] = -s
    r[..., 1, 0] = s
    r[..., 1, 1] = c
    r[..., 2, 2] = o
    return r


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    o = np.ones_like(a)
    r = np.zeros(a.shape + (3, 3), dtype=np.float64)
    r[..., 0, 0] = c
    r[..., 0, 2] = s
    r[..., 1, 1] = o
    r[..., 2, 0] = -s
    r[..., 2, 2] = c
    return r


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    o = np.ones_like(a)
    r = np.zeros(a.shape + (3, 3), dtype=np.float64)
    r[..., 0, 0] = o
    r[..., 1, 1] = c
    r[..., 1, 2] = -s
    r[..., 2, 1] = s
    r[..., 2, 2] = c
    return r


def euler_zyx(zdeg, ydeg, xdeg):
    z, y, x = np.deg2rad(zdeg), np.deg2rad(ydeg), np.deg2rad(xdeg)
    return _rz(z) @ _ry(y) @ _rx(x)


@dataclass
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)
    ch_index: list[int] = field(default_factory=list)


def parse_bvh(path: str | Path) -> tuple[list[Joint], np.ndarray, float]:
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()
    joints: list[Joint] = []
    stack: list[int] = []
    ch_cursor = 0
    i = 0
    while i < len(lines):
        toks = lines[i].strip().split()
        if not toks:
            i += 1
            continue
        if toks[0] in ("ROOT", "JOINT"):
            name = toks[1]
            parent = stack[-1] if stack else -1
            joints.append(Joint(name, parent, np.zeros(3)))
            stack.append(len(joints) - 1)
        elif toks[0] == "End":
            name = f"{joints[stack[-1]].name}End"
            joints.append(Joint(name, stack[-1], np.zeros(3)))
            stack.append(len(joints) - 1)
        elif toks[0] == "OFFSET" and stack:
            joints[stack[-1]].offset = np.array(list(map(float, toks[1:4])), dtype=np.float64)
        elif toks[0] == "CHANNELS" and stack:
            n = int(toks[1])
            ch = toks[2 : 2 + n]
            j = joints[stack[-1]]
            j.channels = ch
            j.ch_index = list(range(ch_cursor, ch_cursor + n))
            ch_cursor += n
        elif toks[0] == "}":
            if stack:
                stack.pop()
        elif toks[0] == "MOTION":
            i += 1
            break
        i += 1
    nframes = int(lines[i].split()[1])
    i += 1
    dt = float(lines[i].split()[2])
    i += 1
    rows = []
    for k in range(nframes):
        rows.append(np.fromstring(lines[i + k], sep=" ", dtype=np.float64))
    motion = np.stack(rows, axis=0)
    return joints, motion, dt


def fk_points(joints: list[Joint], motion: np.ndarray, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    t = motion.shape[0]
    n = len(joints)
    pos = np.zeros((n, t, 3), dtype=np.float64)
    rot = np.zeros((n, t, 3, 3), dtype=np.float64)
    rot[...] = np.eye(3)
    want = set(names)
    idx = {j.name: i for i, j in enumerate(joints)}
    for missing in want - set(idx):
        raise KeyError(f"BVH missing joint {missing}")
    for ji, j in enumerate(joints):
        trans = np.broadcast_to(j.offset, (t, 3)).copy()
        z = y = x = np.zeros(t)
        for ch, ci in zip(j.channels, j.ch_index):
            col = motion[:, ci]
            cl = ch.lower()
            if "position" in cl:
                ax = 0 if cl.startswith("x") else 1 if cl.startswith("y") else 2
                trans[:, ax] = col
            elif cl.startswith("zrot"):
                z = col
            elif cl.startswith("yrot"):
                y = col
            elif cl.startswith("xrot"):
                x = col
        rloc = euler_zyx(z, y, x)
        if j.parent < 0:
            pos[ji] = trans
            rot[ji] = rloc
        else:
            rp = rot[j.parent]
            pos[ji] = pos[j.parent] + np.einsum("tij,tj->ti", rp, trans)
            rot[ji] = np.einsum("tij,tjk->tik", rp, rloc)
    out = {}
    for name in names:
        p_cm = pos[idx[name]]
        # BVH Y-up (cm) -> Isaac Z-up (m): (x, y, z)_bvh -> (x, z, y)_isaac / 100
        p = np.empty_like(p_cm)
        p[:, 0] = p_cm[:, 0] * CM
        p[:, 1] = p_cm[:, 2] * CM
        p[:, 2] = p_cm[:, 1] * CM
        out[name] = p
    return out


def extract_hlr(path: str | Path) -> tuple[np.ndarray, float]:
    joints, motion, dt = parse_bvh(path)
    pts = fk_points(joints, motion, ("Head", "LeftHand", "RightHand"))
    hlr = np.stack([pts["Head"], pts["LeftHand"], pts["RightHand"]], axis=1)  # [T,3,3]
    fps = 1.0 / max(dt, 1e-8)
    return hlr, float(fps)
