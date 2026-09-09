"""Compare official vs fast CSV->NPZ outputs (no Isaac needed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def quat_max_err(a: np.ndarray, b: np.ndarray) -> float:
    """Max geodesic-style error allowing q ~ -q."""
    d1 = np.linalg.norm(a - b, axis=-1)
    d2 = np.linalg.norm(a + b, axis=-1)
    return float(np.min(np.stack([d1, d2], axis=0), axis=0).max())


def compare_pair(ref_path: Path, new_path: Path) -> dict:
    ref = np.load(ref_path)
    new = np.load(new_path)
    row = {"ref": str(ref_path), "new": str(new_path), "ok": True, "keys": {}}
    for k in KEYS:
        if k not in ref.files or k not in new.files:
            row["ok"] = False
            row["keys"][k] = {"error": "missing key"}
            continue
        ra, na = ref[k], new[k]
        if ra.shape != na.shape:
            row["ok"] = False
            row["keys"][k] = {"error": f"shape {ra.shape} vs {na.shape}"}
            continue
        if k == "body_quat_w":
            max_abs = quat_max_err(ra.astype(np.float64), na.astype(np.float64))
        else:
            max_abs = float(np.max(np.abs(ra.astype(np.float64) - na.astype(np.float64))))
        mean_abs = float(np.mean(np.abs(ra.astype(np.float64) - na.astype(np.float64))))
        row["keys"][k] = {"max_abs": max_abs, "mean_abs": mean_abs, "shape": list(ra.shape)}
    if "fps" in ref.files and "fps" in new.files:
        row["fps_ref"] = np.array(ref["fps"]).tolist()
        row["fps_new"] = np.array(new["fps"]).tolist()
        if not np.array_equal(np.array(ref["fps"]), np.array(new["fps"])):
            row["ok"] = False
            row["fps_mismatch"] = True
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Compare official vs fast NPZ conversions.")
    p.add_argument("--ref_dir", type=str, required=True, help="Official NPZ root.")
    p.add_argument("--new_dir", type=str, required=True, help="Fast NPZ root.")
    p.add_argument("--pairs", type=str, default=None, help="Optional text file: ref_npz<TAB>new_npz per line.")
    p.add_argument("--atol_joint", type=float, default=1e-5)
    p.add_argument("--atol_body_pos", type=float, default=1e-4)
    p.add_argument("--atol_quat", type=float, default=1e-4)
    p.add_argument("--atol_vel", type=float, default=1e-3)
    p.add_argument("--report", type=str, default=None, help="Write JSON report to this path.")
    args = p.parse_args()

    thresholds = {
        "joint_pos": args.atol_joint,
        "joint_vel": args.atol_vel,
        "body_pos_w": args.atol_body_pos,
        "body_quat_w": args.atol_quat,
        "body_lin_vel_w": args.atol_vel,
        "body_ang_vel_w": args.atol_vel,
    }

    pairs: list[tuple[Path, Path]] = []
    if args.pairs:
        with open(args.pairs, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split("\t")
                pairs.append((Path(a), Path(b)))
    else:
        new_dir = Path(args.new_dir)
        ref_dir = Path(args.ref_dir)
        for new_path in sorted(new_dir.rglob("*.npz")):
            rel = new_path.relative_to(new_dir)
            ref_path = ref_dir / rel
            if not ref_path.is_file():
                raise SystemExit(f"[ERROR] missing official npz: {ref_path}")
            pairs.append((ref_path, new_path))

    if not pairs:
        raise SystemExit("[ERROR] no NPZ pairs to compare")

    report = {"n": len(pairs), "pass": 0, "fail": 0, "clips": []}
    print(f"{'clip':<70} {'status':<6} " + " ".join(f"{k:>16}" for k in KEYS))
    for ref_path, new_path in pairs:
        row = compare_pair(ref_path, new_path)
        failed_keys = []
        for k, thr in thresholds.items():
            info = row["keys"].get(k, {})
            if "max_abs" not in info or info["max_abs"] > thr:
                failed_keys.append(k)
                row["ok"] = False
        status = "PASS" if row["ok"] and not failed_keys else "FAIL"
        if status == "PASS":
            report["pass"] += 1
        else:
            report["fail"] += 1
            row["failed_keys"] = failed_keys
        report["clips"].append(row)
        name = new_path.stem
        vals = []
        for k in KEYS:
            info = row["keys"].get(k, {})
            vals.append(f"{info.get('max_abs', float('nan')):16.3e}")
        print(f"{name:<70} {status:<6} " + " ".join(vals))

    print("=" * 80)
    print(f"PASS {report['pass']}/{report['n']}  FAIL {report['fail']}/{report['n']}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] wrote {args.report}")
    if report["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
