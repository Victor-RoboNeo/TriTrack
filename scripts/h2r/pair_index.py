"""Pair Bones-SEED SOMA Uniform BVH with G1 npz via metadata filename."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .constants import G1_NPZ_ROOT, META_CSV, SOMA_ROOT


def _split_of(key: str, salt: str = "h2r0") -> str:
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    v = int(h[:8], 16) % 100
    if v < 80:
        return "train"
    if v < 90:
        return "val"
    return "test"


def iter_meta_rows(path: Path = META_CSV):
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            yield row


def g1_npz_path(filename: str, date: str) -> Path:
    return G1_NPZ_ROOT / str(date) / f"bones_seed_g1_{filename}.npz"


def soma_bvh_path(rel: str) -> Path:
    return SOMA_ROOT.parent / rel if not rel.startswith("/") else Path(rel)
    # metadata path is soma_uniform/bvh/... relative to bones-seed
    # SOMA_ROOT is bones-seed/soma_uniform so parent is bones-seed


def build_index() -> list[dict]:
    rows = []
    for row in iter_meta_rows():
        fn = row["filename"]
        rel = row["move_soma_uniform_path"]
        bvh = Path("/data/home/chenxiangyu/robotics/Anybody/datasets/bones-seed") / rel
        date = Path(rel).parts[2] if len(Path(rel).parts) >= 3 else str(row.get("take_date", ""))
        npz = g1_npz_path(fn, date)
        rec = {
            "filename": fn,
            "move_name": row["move_name"],
            "take_name": row["take_name"],
            "actor_uid": row["actor_uid"],
            "is_mirror": row.get("is_mirror", ""),
            "package": row.get("package", ""),
            "bvh": str(bvh),
            "npz": str(npz),
            "date": date,
            "split_key": row["take_name"] or fn.rsplit("_M", 1)[0],
            "bvh_ok": bvh.is_file(),
            "npz_ok": npz.is_file(),
        }
        rec["split"] = _split_of(rec["split_key"])
        rows.append(rec)
    return rows
