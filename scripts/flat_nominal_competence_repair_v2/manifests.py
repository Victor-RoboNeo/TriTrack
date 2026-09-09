"""Fixed FLAT_3PT_DEV_V2 / TEST_V2. Never regenerate after first write."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from flat_locomani.reference_suite import FPS, H0_M, min_jerk, save_suite, TrajectoryMeta, _alloc, _standing_pose

from .constants import DEV_N, DEV_SEED, RESULTS, TEST_N, TEST_SEED, TRAIN_MIX_SEED
from .io_util import atomic_write_json, sha256_file


def geometry_ok(world: np.ndarray) -> bool:
    z = world[:, :, 2]
    if float(z.min()) < 0.30 or float(z.max()) > 1.35:
        return False
    torso = world[:, 0]
    for j in (1, 2):
        if float(np.linalg.norm(world[:, j, :2] - torso[:, :2], axis=-1).max()) > 0.85:
            return False
    return True


def _static(rng, i, prefix) -> tuple[np.ndarray, TrajectoryMeta]:
    h = float(np.clip(0.78 + rng.uniform(-0.08, 0.05), 0.62, 0.90))
    pose = _standing_pose(h)
    kind = i % 6
    if kind == 1:
        pose[1, 1] += rng.uniform(0.04, 0.10)
        pose[2, 1] -= rng.uniform(0.04, 0.10)
    elif kind == 2:
        pose[1, 0] += rng.uniform(0.06, 0.14)
        pose[2, 0] += rng.uniform(0.06, 0.14)
    elif kind == 3:
        pose[1, 0] += rng.uniform(0.08, 0.16)
        pose[2, 0] += rng.uniform(-0.02, 0.04)
    elif kind == 4:
        pose[0, 2] += rng.uniform(-0.08, 0.04)
        pose[1, 2] += rng.uniform(-0.10, 0.06)
        pose[2, 2] += rng.uniform(-0.10, 0.06)
    elif kind == 5:
        pose[1, 2] += rng.uniform(-0.12, 0.08)
        pose[2, 2] += rng.uniform(-0.12, 0.08)
    T = 160
    traj = _alloc(T, pose)
    meta = TrajectoryMeta(
        traj_id=f"{prefix}_F1_{i:03d}",
        family="F1_STATIC",
        hand_mode="static",
        source_semantics="canonical_chest_lh_rh_static",
        h0_m=h,
        target_height_drop_m=0.0,
        transition_duration_s=0.0,
        low_hold_s=T / FPS,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _reach(rng, i, prefix) -> tuple[np.ndarray, TrajectoryMeta]:
    h = 0.78
    pose0 = _standing_pose(h)
    T = 180
    traj = _alloc(T, pose0)
    u = min_jerk(np.linspace(0, 1, T // 2))
    kind = i % 8
    d = np.zeros(3)
    if kind == 0:
        d = np.array([0.16, 0.0, 0.0])
        traj[: T // 2, 1] += u[:, None] * d
        traj[T // 2 :, 1] = traj[T // 2 - 1, 1]
    elif kind == 1:
        d = np.array([0.16, 0.0, 0.0])
        traj[: T // 2, 2] += u[:, None] * d
        traj[T // 2 :, 2] = traj[T // 2 - 1, 2]
    elif kind == 2:
        d = np.array([0.14, 0.0, 0.0])
        traj[: T // 2, 1] += u[:, None] * d
        traj[: T // 2, 2] += u[:, None] * d
        traj[T // 2 :, 1:] = traj[T // 2 - 1, 1:]
    elif kind == 3:
        traj[: T // 2, 1, 1] += u * 0.12
        traj[T // 2 :, 1] = traj[T // 2 - 1, 1]
    elif kind == 4:
        traj[: T // 2, 2, 1] -= u * 0.12
        traj[T // 2 :, 2] = traj[T // 2 - 1, 2]
    elif kind == 5:
        traj[: T // 2, 1] += u[:, None] * np.array([0.12, 0.06, 0.0])
        traj[: T // 2, 2] += u[:, None] * np.array([0.04, -0.02, 0.0])
        traj[T // 2 :, 1:] = traj[T // 2 - 1, 1:]
    elif kind == 6:
        traj[: T // 2, 1, 2] += u * 0.12
        traj[: T // 2, 2, 2] += u * 0.12
        traj[T // 2 :, 1:] = traj[T // 2 - 1, 1:]
    else:
        traj[: T // 2, 1, 2] -= u * 0.10
        traj[: T // 2, 2, 2] -= u * 0.10
        traj[T // 2 :, 1:] = traj[T // 2 - 1, 1:]
    # chest mostly static with tiny shift
    traj[:, 0, 0] += rng.uniform(-0.02, 0.02)
    meta = TrajectoryMeta(
        traj_id=f"{prefix}_F2_{i:03d}",
        family="F2_REACH",
        hand_mode="reach",
        source_semantics="canonical_reach",
        h0_m=h,
        target_height_drop_m=0.0,
        transition_duration_s=T / 2 / FPS,
        low_hold_s=T / 2 / FPS,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _height(rng, i, prefix) -> tuple[np.ndarray, TrajectoryMeta]:
    dzs = (-0.05, -0.10, -0.15, 0.05)
    dz = dzs[i % 4]
    pose = _standing_pose(0.78)
    T = 180
    traj = _alloc(T, pose)
    u = min_jerk(np.linspace(0, 1, T // 2))
    traj[: T // 2, 0, 2] += u * dz
    traj[T // 2 :, 0, 2] = traj[T // 2 - 1, 0, 2]
    meta = TrajectoryMeta(
        traj_id=f"{prefix}_F3_{i:03d}",
        family="F3_BASIC_HEIGHT_DIAG",
        hand_mode="body_follow",
        source_semantics="chest_delta_z_diag",
        h0_m=0.78,
        target_height_drop_m=max(0.0, -dz),
        transition_duration_s=T / 2 / FPS,
        low_hold_s=T / 2 / FPS,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _trans(rng, i, prefix) -> tuple[np.ndarray, TrajectoryMeta]:
    speed = float(rng.uniform(0.15, 0.30))
    dist = float(rng.uniform(0.4, 0.9))
    T = int(max(100, round(dist / speed * FPS)))
    pose = _standing_pose(0.78)
    traj = _alloc(T, pose)
    t = np.arange(T) / FPS
    traj[:, :, 0] += np.minimum(speed * t, dist)[:, None]
    meta = TrajectoryMeta(
        traj_id=f"{prefix}_F4_{i:03d}",
        family="F4_HORIZONTAL_DIAG",
        hand_mode="body_follow",
        source_semantics="low_speed_xy",
        h0_m=0.78,
        target_height_drop_m=0.0,
        transition_duration_s=T / FPS,
        low_hold_s=0.0,
        horizontal_speed_mps=speed,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _build(kind: str, seed: int, n_each: int) -> list:
    rng = np.random.default_rng(seed)
    items = []
    gens = (_static, _reach, _height, _trans)
    for gen in gens:
        k = 0
        tries = 0
        while k < n_each and tries < n_each * 20:
            tries += 1
            w, m = gen(rng, k + tries, kind)
            if geometry_ok(w):
                items.append((w, m))
                k += 1
        if k < n_each:
            raise RuntimeError(f"{gen.__name__} only kept {k}/{n_each}")
    return items


def _write_jsonl(path: Path, items: list, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for world, meta in items:
            rec = asdict(meta)
            rec["phases"] = [asdict(p) if hasattr(p, "__dataclass_fields__") else p for p in rec.get("phases") or []]
            rec["mask"] = [1, 1, 1]
            rec["canonical_slots"] = ["C", "LH", "RH"]
            rec["initial_root_pose"] = {"xyz": [0.0, 0.0, 0.8], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
            rec["initial_joint_state_source"] = "eval_clip_seed_start_frame_10_zero_pose_range"
            rec["evaluation_horizon"] = int(world.shape[0])
            rec["rng_seed"] = extra.get("seed")
            rec["z_min"] = float(world[:, :, 2].min())
            rec["z_max"] = float(world[:, :, 2].max())
            rec.update(extra)
            f.write(json.dumps(rec, default=str) + "\n")


def generate_if_needed() -> dict:
    lock = RESULTS / "manifests" / "FLAT_3PT_DEV_V2.sha256"
    if lock.exists() and (RESULTS / "manifests" / "FLAT_3PT_TEST_V2.sha256").exists():
        print("[manifests] already frozen; not regenerating")
        return {"skipped": True, "sha256_dev": lock.read_text().strip()}
    out = {}
    for kind, seed, n_each in (("DEV", DEV_SEED, 64), ("TEST", TEST_SEED, 128)):
        items = _build(kind, seed, n_each)
        suite = RESULTS / "manifests" / f"suite_{kind.lower()}"
        info = save_suite(suite, items=items)
        jsonl = RESULTS / "manifests" / f"FLAT_3PT_{kind}_V2.jsonl"
        _write_jsonl(jsonl, items, {"split": kind, "seed": seed})
        digest = sha256_file(jsonl)
        (RESULTS / "manifests" / f"FLAT_3PT_{kind}_V2.sha256").write_text(digest + "\n")
        counts = {}
        for _, m in items:
            counts[m.family] = counts.get(m.family, 0) + 1
        rec = {"split": kind, "seed": seed, "n": info["n"], "sha256": digest, "counts": counts, "suite": str(suite)}
        atomic_write_json(RESULTS / "manifests" / f"FLAT_3PT_{kind}_V2_meta.json", rec)
        out[kind] = rec
        assert info["n"] == (256 if kind == "DEV" else 512), info
        assert all(v == n_each for v in counts.values()), counts
    # training mix disjoint from DEV/TEST
    mix_items = _build("TRAIN", TRAIN_MIX_SEED, 80)
    mix_items = [(w, m) for w, m in mix_items if m.family in ("F1_STATIC", "F2_REACH")]
    mix_root = RESULTS / "manifests" / "train_mix_f1f2"
    from flat_locomani.live_anchor_v2.motion_convert import load_seed_and_index, world_to_motion_absolute

    seed0, idxs = load_seed_and_index()
    for sub in ("static", "reach"):
        (mix_root / sub).mkdir(parents=True, exist_ok=True)
    n_s = n_r = 0
    for w, m in mix_items:
        mot = world_to_motion_absolute(w, seed0, idxs)
        sub = "static" if m.family == "F1_STATIC" else "reach"
        if sub == "static":
            n_s += 1
        else:
            n_r += 1
        path = mix_root / sub / f"{m.traj_id}.npz"
        np.savez_compressed(path, **mot)
    out["TRAIN_MIX"] = {"root": str(mix_root), "n_static": n_s, "n_reach": n_r}
    atomic_write_json(RESULTS / "manifests" / "generation.json", out)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    generate_if_needed()
