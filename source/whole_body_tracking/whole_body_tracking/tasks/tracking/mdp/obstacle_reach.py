"""Obstacle-reach scene primitives: scene-JSON loading, per-phase randomized sampling,
and oriented-bounding-box (OBB) keep-out math.

This is the data/geometry core shared by the ObstacleReach task (see
[[obstacle-reach-task-design]]). It is deliberately framework-free (torch + numpy + json
only) so it can be unit-tested without Isaac.

Conventions
-----------
- Frame: Isaac/npz world, z-up, meters. The robot start is the env origin, facing +x; the
  obstacle "station" sits ~D meters ahead (+x). All tensors here are in this env-relative
  frame; the env adds ``scene.env_origins`` when placing markers / reading body positions.
- An obstacle is an oriented box: center (3), HALF extents (3), quat wxyz (4).
- Phases: 0 free reach, 1 above box (lean), 2 into open container (bend+down), 4 under a
  low slab (squat). "height" = the target z, the headline knob per phase.
- A scene is padded to ``MAX_OBSTACLES`` boxes with a boolean ``valid`` mask (phase 2's
  open container is 5 walls: floor + 4 sides, open top).
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import torch

MAX_OBSTACLES = 5
_IDENT_QUAT = (1.0, 0.0, 0.0, 0.0)

# Per-phase config. The obstacle GEOMETRY (sizes + layout) and the target now come from the
# AUTHORED scene-editor JSON (``_PHASE_TEMPLATE``), so they're tweakable in the GUI and loaded
# automatically here — not hardcoded. Only POSE randomizes per reset: the station distance D
# ahead of the robot + a small lateral y (phase 0 / free reach also randomizes target height).
PHASE_SPECS: dict[int, dict] = {
    0: dict(kind="free", height=(0.20, 1.20), lateral=(-0.20, 0.20)),
    1: dict(kind="template", lateral=(-0.10, 0.10)),
    2: dict(kind="template", lateral=(-0.10, 0.10)),
    4: dict(kind="template", lateral=(-0.10, 0.10)),
}
APPROACH_RANGE = (0.80, 1.20)   # station distance D ahead of the robot start

# Canonical authored scene per phase (scene_editor/library/...). The obstacle SIZE + layout +
# target are read from here; the robot collides with FIXED-size rigid boxes (option 2) and the
# station is re-anchored to a randomized distance D each reset. Tweak these JSONs in the editor
# (e.g. make the phase-1 cube taller) and they're picked up on the next run. Override the
# library root with ``OBSTACLE_REACH_LIBRARY`` if needed.
_PHASE_TEMPLATE: dict[int, str] = {
    1: "phase_1_height_0p8/phase_1_height_0p8.json",
    2: "phase_2_height_0p4/phase_2_height_0p4.json",
    4: "phase_4_height_0p4/phase_4_height_0p4.json",
}


def _library_dir() -> Path:
    env = os.environ.get("OBSTACLE_REACH_LIBRARY", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[6] / "scene_editor" / "library"


@functools.lru_cache(maxsize=None)
def load_phase_template(phase: int) -> dict:
    """Authored obstacle geometry for a phase, re-anchored to the target's ground point.

    Reads the canonical scene-editor JSON (:data:`_PHASE_TEMPLATE`). Obstacle centers are
    stored RELATIVE to the authored target's (x, y) so :func:`sample_phase` can drop the whole
    station at a randomized distance D: ``obstacle_center = (D + dx, y + dy, z)``,
    ``target = (D, y, target_z)``. Returns ``{halves, rel_centers, quats, target_z}``.
    """
    j = load_scene_json(_library_dir() / _PHASE_TEMPLATE[phase])
    if j["target"] is None:
        raise ValueError(f"phase {phase} template {_PHASE_TEMPLATE[phase]!r} has no target")
    tx, ty, _tz = j["target"]
    halves, rel_centers, quats = [], [], []
    for o in j["obstacles"][:MAX_OBSTACLES]:
        px, py, pz = o["pos"]
        sx, sy, sz = o["size"]
        halves.append((sx / 2.0, sy / 2.0, sz / 2.0))
        rel_centers.append((px - tx, py - ty, pz))
        quats.append(tuple(o["quat"]))
    return {"halves": halves, "rel_centers": rel_centers, "quats": quats, "target_z": float(j["target"][2])}


def phase_box_half_sizes(phase: int) -> list[tuple[float, float, float]]:
    """FIXED per-box HALF extents (x, y, z) for a phase's rigid colliders, in slot order
    (matches the slots :func:`sample_phase` fills). Sourced from the authored template; empty
    for the free-reach phase. Same sizes spawn the kinematic colliders and fill the OBB obs."""
    if PHASE_SPECS[phase]["kind"] == "free":
        return []
    return [tuple(h) for h in load_phase_template(phase)["halves"]]


def num_boxes(phase: int) -> int:
    """Number of rigid box colliders for a phase (0 free, 1 box/slab, 5 container)."""
    return len(phase_box_half_sizes(phase))


# --------------------------------------------------------------------------- quat math
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the inverse of wxyz quaternion ``q`` (broadcasting leading dims)."""
    q_w = q[..., 0:1]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


# --------------------------------------------------------------------------- OBB keep-out
def obb_penetration(
    points: torch.Tensor,      # (N, B, 3) body positions to keep out
    centers: torch.Tensor,     # (N, K, 3)
    half: torch.Tensor,        # (N, K, 3)
    quat: torch.Tensor,        # (N, K, 4) wxyz
    valid: torch.Tensor | None = None,  # (N, K) bool
) -> torch.Tensor:
    """Per-body penetration depth (m) into the nearest obstacle. 0 when outside all boxes.

    Inside an OBB the depth is the distance to the closest face (``min_i(half_i - |local_i|)``);
    outside it is 0. Returns (N, B), the max over obstacles.
    """
    N, B = points.shape[0], points.shape[1]
    K = centers.shape[1]
    rel = points[:, None, :, :] - centers[:, :, None, :]            # (N, K, B, 3)
    q = quat[:, :, None, :].expand(N, K, B, 4)
    local = quat_rotate_inverse(q, rel)                             # (N, K, B, 3)
    d = half[:, :, None, :] - local.abs()                          # (N, K, B, 3)
    inside = (d > 0).all(dim=-1)                                    # (N, K, B)
    depth = torch.where(inside, d.clamp(min=0.0).amin(dim=-1), torch.zeros(N, K, B, device=points.device))
    if valid is not None:
        depth = depth * valid[:, :, None].to(depth.dtype)
    return depth.amax(dim=1)                                        # (N, B)


# --------------------------------------------------------------------------- scene container
class ObstacleScene:
    """Batched obstacle boxes + reach target for ``N`` envs.

    Attributes are all torch tensors on ``device``:
      centers (N,K,3), half (N,K,3), quat (N,K,4), valid (N,K) bool, target (N,3).
    """

    def __init__(self, centers, half, quat, valid, target):
        self.centers = centers
        self.half = half
        self.quat = quat
        self.valid = valid
        self.target = target

    @property
    def num_envs(self) -> int:
        return self.centers.shape[0]

    def penetration(self, points: torch.Tensor) -> torch.Tensor:
        return obb_penetration(points, self.centers, self.half, self.quat, self.valid)

    @staticmethod
    def empty(num_envs: int, device) -> "ObstacleScene":
        z = torch.zeros(num_envs, MAX_OBSTACLES, 3, device=device)
        q = torch.zeros(num_envs, MAX_OBSTACLES, 4, device=device)
        q[..., 0] = 1.0
        return ObstacleScene(
            centers=z.clone(), half=z.clone(), quat=q,
            valid=torch.zeros(num_envs, MAX_OBSTACLES, dtype=torch.bool, device=device),
            target=torch.zeros(num_envs, 3, device=device),
        )


# --------------------------------------------------------------------------- helpers
def _u(lo, hi, n, device, gen=None):
    return lo + (hi - lo) * torch.rand(n, device=device, generator=gen)


# --------------------------------------------------------------------------- sampling (train)
def sample_phase(phase: int, num_envs: int, device, gen=None) -> ObstacleScene:
    """Sample a batch of valid scenes for ``phase`` (per-reset POSE randomization).

    Obstacle SIZE + layout + target come from the authored template
    (:func:`load_phase_template`); only the station is randomized — placed ~D ahead (+x) of the
    robot start (origin) with a small lateral y, the authored layout dropped at that station.
    Free reach (phase 0) has no obstacle and randomizes the target height instead.
    """
    spec = PHASE_SPECS[phase]
    s = ObstacleScene.empty(num_envs, device)
    eid = torch.arange(num_envs, device=device)
    D = _u(*APPROACH_RANGE, num_envs, device, gen)                 # station distance ahead
    y = _u(*spec.get("lateral", (-0.10, 0.10)), num_envs, device, gen)

    if spec["kind"] == "free":
        z = _u(*spec["height"], num_envs, device, gen)
        s.target = torch.stack([D + 0.05, y, z], dim=-1)
        return s

    # Authored template re-anchored to the randomized station (D, y): obstacle center =
    # (D + dx, y + dy, z), target = (D, y, target_z). Sizes/quats are the authored fixed values.
    tmpl = load_phase_template(phase)
    for slot, ((hx, hy, hz), (dx, dy, dz), q) in enumerate(
        zip(tmpl["halves"], tmpl["rel_centers"], tmpl["quats"])
    ):
        s.centers[eid, slot] = torch.stack([D + dx, y + dy, torch.full_like(D, dz)], dim=-1)
        s.half[eid, slot] = torch.tensor([hx, hy, hz], device=device).expand(num_envs, 3)
        s.quat[eid, slot] = torch.tensor(q, device=device).expand(num_envs, 4)
        s.valid[eid, slot] = True
    s.target = torch.stack([D, y, torch.full_like(D, tmpl["target_z"])], dim=-1)
    return s


# --------------------------------------------------------------------------- scene JSON (eval)
def load_scene_json(path) -> dict:
    """Load an authored scene (scene_editor library JSON)."""
    d = json.loads(Path(path).read_text())
    return {
        "name": d.get("name", Path(path).stem),
        "obstacles": [
            {"pos": [float(v) for v in o["pos"]],
             "size": [float(v) for v in o["size"]],
             "quat": [float(v) for v in o.get("quat", _IDENT_QUAT)]}
            for o in d.get("obstacles", [])
        ],
        "target": ([float(v) for v in d["target"]] if d.get("target") else None),
    }


def scene_from_json(path, num_envs: int, device) -> ObstacleScene:
    """Build a (replicated) ObstacleScene for ``num_envs`` from an authored JSON file."""
    j = load_scene_json(path)
    s = ObstacleScene.empty(num_envs, device)
    for k, o in enumerate(j["obstacles"][:MAX_OBSTACLES]):
        s.centers[:, k] = torch.tensor(o["pos"], device=device)
        s.half[:, k] = torch.tensor(o["size"], device=device) / 2.0   # full extents -> half
        s.quat[:, k] = torch.tensor(o["quat"], device=device)
        s.valid[:, k] = True
    if j["target"] is not None:
        s.target[:] = torch.tensor(j["target"], device=device)
    return s


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    dev = "cpu"
    # OBB sanity
    centers = torch.zeros(1, 1, 3); half = torch.full((1, 1, 3), 0.2)
    quat = torch.tensor([[[1.0, 0, 0, 0]]]); valid = torch.ones(1, 1, dtype=torch.bool)
    pts = torch.tensor([[[0, 0, 0], [0.1, 0, 0], [0.3, 0, 0]]], dtype=torch.float32)
    pen = obb_penetration(pts, centers, half, quat, valid)[0]
    assert abs(pen[0] - 0.2) < 1e-5 and abs(pen[1] - 0.1) < 1e-5 and pen[2] == 0, pen
    # rotated box: 45deg about z, half (0.3,0.05,0.3); point (0.2,0.2,0) is inside rotated, outside AABB
    import math
    q45 = torch.tensor([[[math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8)]]])
    pen_r = obb_penetration(torch.tensor([[[0.18, 0.18, 0.0]]]), torch.zeros(1, 1, 3),
                            torch.tensor([[[0.3, 0.06, 0.3]]]), q45, valid)[0]
    assert pen_r[0] > 0, pen_r
    # sampling per phase
    for ph in (0, 1, 2, 4):
        sc = sample_phase(ph, 8, dev)
        nval = int(sc.valid.sum(dim=1).float().mean().item())
        print(f"phase {ph}: avg valid obstacles={nval}, target z range "
              f"[{sc.target[:,2].min():.2f},{sc.target[:,2].max():.2f}]")
        if ph == 2:  # target must be inside the container footprint, above the floor
            assert (sc.target[:, 2] > 0).all()
            pen_tgt = sc.penetration(sc.target[:, None, :])[:, 0]
            print(f"   target-in-floor penetration (should be ~0): {pen_tgt.max():.3f}")
    print("obstacle_reach self-test OK")
