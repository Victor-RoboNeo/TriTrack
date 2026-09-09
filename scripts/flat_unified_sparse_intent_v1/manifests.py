"""Generate and freeze FLAT_DEV_V1 / FLAT_TEST_V1. Do not regenerate after campaign start."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from flat_locomani.reference_suite import (
    FPS,
    H0_M,
    TrajectoryMeta,
    family_a,
    family_b,
    family_c,
    family_e,
    family_f,
    min_jerk,
    save_suite,
)
from flat_locomani.reference_suite import _alloc, _standing_pose

from .constants import DEV_MANIFEST_RNG_SEED, RESULTS, TEST_MANIFEST_RNG_SEED
from .io_util import atomic_write_json, sha256_file


def geometry_ok(world: np.ndarray) -> bool:
    """Reachability sanity for dataset construction only. Not a runtime controller."""
    z = world[:, :, 2]
    if z.min() < 0.25 or z.max() > 1.40:
        return False
    torso = world[:, 0]
    for j in (1, 2):
        dxy = np.linalg.norm(world[:, j, :2] - torso[:, :2], axis=-1)
        if float(dxy.max()) > 0.90:
            return False
        dz = np.abs(world[:, j, 2] - torso[:, 2])
        if float(dz.max()) > 0.55:
            return False
    return True


def family_static(seed: int, n: int = 12) -> list:
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        h = float(np.clip(0.78 + rng.uniform(-0.05, 0.03), 0.70, 0.85))
        pose = _standing_pose(h)
        pose[1, 0] += rng.uniform(-0.05, 0.12)
        pose[1, 1] += rng.uniform(-0.05, 0.08)
        pose[2, 0] += rng.uniform(-0.05, 0.12)
        pose[2, 1] += rng.uniform(-0.08, 0.05)
        pose[1, 2] += rng.uniform(-0.08, 0.05)
        pose[2, 2] += rng.uniform(-0.08, 0.05)
        T = 150
        traj = _alloc(T, pose)
        meta = TrajectoryMeta(
            traj_id=f"F1_static_{i:02d}",
            family="F1_STATIC",
            hand_mode="body_follow",
            source_semantics="reachable_static_3pt",
            h0_m=h,
            target_height_drop_m=0.0,
            transition_duration_s=0.0,
            low_hold_s=3.0,
            horizontal_speed_mps=0.0,
            is_stress=False,
            n_frames=T,
            fps=FPS,
        )
        items.append((traj, meta))
    return items


def family_reach(seed: int) -> list:
    items = []
    for world, meta in family_a(seed):
        meta.family = "F2_REACH"
        meta.traj_id = f"F2_{meta.traj_id}"
        items.append((world, meta))
    return items


def family_height_moderate(seed: int) -> list:
    items = []
    for world, meta in family_b():
        drop = float(meta.target_height_drop_m)
        if drop > 0.15:
            # compress to moderate Δh ∈ [-0.15, +0.05]
            scale = 0.15 / max(drop, 1e-6)
            h0 = world[0, 0, 2]
            world = world.copy()
            world[:, :, 2] = h0 + (world[:, :, 2] - h0) * scale
            meta.target_height_drop_m = 0.15
        meta.family = "F3_BASIC_HEIGHT"
        meta.traj_id = f"F3_{meta.traj_id}"
        items.append((world, meta))
    return items


def family_translate(seed: int, n: int = 10) -> list:
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        speed = float(rng.uniform(0.15, 0.40))
        dist = float(rng.uniform(0.4, 1.2))
        T = int(max(80, round(dist / speed * FPS)))
        pose = _standing_pose(0.78)
        traj = _alloc(T, pose)
        t = np.arange(T) / FPS
        x = np.minimum(speed * t, dist)
        traj[:, :, 0] += x[:, None]
        meta = TrajectoryMeta(
            traj_id=f"F4_trans_{i:02d}",
            family="F4_HORIZONTAL",
            hand_mode="body_follow",
            source_semantics="joint_xy_translation",
            h0_m=0.78,
            target_height_drop_m=0.0,
            transition_duration_s=float(T / FPS),
            low_hold_s=0.0,
            horizontal_speed_mps=speed,
            is_stress=False,
            n_frames=T,
            fps=FPS,
        )
        items.append((traj, meta))
    return items


def _filter(items: list) -> tuple[list, int, int]:
    kept = []
    n_in = len(items)
    for w, m in items:
        if geometry_ok(w):
            kept.append((w, m))
    return kept, n_in, len(kept)


def _write_jsonl(path: Path, items: list, extra: dict) -> None:
    rows = []
    for world, meta in items:
        rec = asdict(meta)
        rec["phases"] = [asdict(p) if hasattr(p, "__dataclass_fields__") else p for p in rec.get("phases") or []]
        rec["mask"] = [1, 1, 1]
        rec["n_frames"] = int(world.shape[0])
        rec["z_min"] = float(world[:, :, 2].min())
        rec["z_max"] = float(world[:, :, 2].max())
        rec.update(extra)
        rows.append(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def build_split(kind: str, seed: int, n_scale: int) -> dict:
    items = []
    items += family_static(seed, n=8 * n_scale)
    items += family_reach(seed)
    items += family_height_moderate(seed)
    items += family_translate(seed, n=6 * n_scale)
    # Keep a few original families for P0 compatibility diagnostics.
    items += [(w, m) for w, m in family_c()[: 2 * n_scale]]
    items += [(w, m) for w, m in family_e()[: 2 * n_scale]]
    items += [(w, m) for w, m in family_f()[: 2 * n_scale]]
    try:
        from flat_locomani.live_anchor_v2.d1_d2 import generate_d2

        items += generate_d2(seed)[: 4 * n_scale]
    except Exception as exc:
        print(f"[manifests] skip generate_d2: {exc}", flush=True)
    kept, n_in, n_out = _filter(items)
    suite_dir = RESULTS / "manifests" / f"suite_{kind.lower()}"
    if suite_dir.exists():
        # Frozen: never regenerate if already present.
        man = suite_dir / "manifest.json"
        if man.exists():
            return {"skipped": True, "reason": "already exists", "root": str(suite_dir), "n": json.loads(man.read_text()).get("n")}
    info = save_suite(suite_dir, items=kept)
    jsonl = RESULTS / "manifests" / f"FLAT_{kind}_V1.jsonl"
    _write_jsonl(jsonl, kept, {"split": kind, "manifest_rng_seed": seed, "geometry_in": n_in, "geometry_out": n_out})
    digest = sha256_file(jsonl)
    (RESULTS / "manifests" / f"FLAT_{kind}_V1.sha256").write_text(digest + "\n")
    rec = {
        "split": kind,
        "seed": seed,
        "suite": str(suite_dir),
        "jsonl": str(jsonl),
        "sha256": digest,
        "geometry_in": n_in,
        "geometry_out": n_out,
        "n": info["n"],
    }
    atomic_write_json(RESULTS / "manifests" / f"FLAT_{kind}_V1_meta.json", rec)
    return rec


def generate() -> dict:
    if (RESULTS / "manifests" / "FLAT_DEV_V1.sha256").exists() and (RESULTS / "manifests" / "FLAT_TEST_V1.sha256").exists():
        print("[manifests] already frozen; not regenerating")
        return {"skipped": True}
    dev = build_split("DEV", DEV_MANIFEST_RNG_SEED, n_scale=1)
    test = build_split("TEST", TEST_MANIFEST_RNG_SEED, n_scale=2)
    out = {"DEV": dev, "TEST": test}
    atomic_write_json(RESULTS / "manifests" / "generation.json", out)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    generate()
