#!/usr/bin/env python3
"""H2R-1: causal live-human three-point stream → analytic adapter → G1 clips.

No physical tracker on this server. The live pipeline is driven by timestamped
SOMA Head/LHand/RHand packets (causal, delayed, jittered, droppable), then the
already-frozen analytic H2R-0 adapter. Recovery OFF. THB OFF.

Compares:
  G1-GT replay (existing)
  SOMA → Analytic replay (existing, full-clip offline)
  REAL-stream simulation → Analytic (this script)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2r.adapter import analytic_map, calibrate, resample_traj, to_u
from h2r.bvh_fk import extract_hlr
from h2r.constants import MAPPER_B, OUT_HZ, P1_ROOT, TASKS
from h2r.g1_io import load_g1_tlr, overlay_npz
from h2r.overlay_p1 import match_p1, p1_stem
from h2r.pair_index import build_index

NEXT = Path("/data/home/chenxiangyu/robotics/Anybody/results/next_phase/h2r1_live_human")
DT = 1.0 / OUT_HZ


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def apply_variation(hlr: np.ndarray, kind: str, fps: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Causal-safe human-stream variations. Does not look at robot state."""
    h = np.asarray(hlr, dtype=np.float64)
    if kind in ("normal", "live"):
        return h, fps
    if kind == "slow":
        return resample_traj(h, fps, fps / 1.5), fps
    if kind == "fast":
        return resample_traj(h, fps, fps * 1.4), fps
    if kind == "stop":
        t0 = max(int(0.5 * h.shape[0]), 2)
        out = h.copy()
        out[t0:] = h[t0 - 1]
        return out, fps
    if kind == "dir_change":
        t0 = max(int(0.4 * h.shape[0]), 2)
        out = h.copy()
        c = out[t0, 0].copy()
        out[t0:, :, 0] = 2.0 * c[0] - out[t0:, :, 0]
        out[t0:, :, 1] = 2.0 * c[1] - out[t0:, :, 1]
        return out, fps
    if kind == "excursion_large":
        out = h.copy()
        out[:, 1] = out[:, 0] + 1.35 * (h[:, 1] - h[:, 0])
        out[:, 2] = out[:, 0] + 1.35 * (h[:, 2] - h[:, 0])
        return out, fps
    if kind == "excursion_small":
        out = h.copy()
        out[:, 1] = out[:, 0] + 0.70 * (h[:, 1] - h[:, 0])
        out[:, 2] = out[:, 0] + 0.70 * (h[:, 2] - h[:, 0])
        return out, fps
    raise ValueError(kind)


def live_sample(
    hlr: np.ndarray,
    fps_h: float,
    n_out: int,
    latency_s: float,
    jitter_s: float,
    drop_p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """At output tick k, only packets with t_pkt <= k*DT - delay are visible."""
    t_in = (hlr.shape[0] - 1) / max(fps_h, 1e-8)
    xin = np.linspace(0.0, t_in, hlr.shape[0])
    out = np.zeros((n_out, 3, 3), dtype=np.float64)
    last = hlr[0].copy()
    dropped = 0
    used_t = []
    for k in range(n_out):
        jitter = float(rng.normal(0.0, jitter_s)) if jitter_s > 0 else 0.0
        delay = max(0.0, latency_s + jitter)
        t_vis = max(0.0, k * DT - delay)
        if rng.random() < drop_p:
            dropped += 1
            out[k] = last
            used_t.append(t_vis)
            continue
        # hold-last / linear sample of visible past only
        t_vis = min(t_vis, t_in)
        flat = hlr.reshape(hlr.shape[0], -1)
        samp = np.stack([np.interp(t_vis, xin, flat[:, j]) for j in range(flat.shape[1])])
        last = samp.reshape(3, 3)
        out[k] = last
        used_t.append(t_vis)
    dt_used = np.diff(np.asarray(used_t, dtype=np.float64))
    stats = {
        "dropped_frames": int(dropped),
        "drop_frac": float(dropped / max(n_out, 1)),
        "latency_s": float(latency_s),
        "jitter_s": float(jitter_s),
        "input_dt_std": float(np.std(dt_used)) if dt_used.size else 0.0,
        "n_out": int(n_out),
    }
    return out, stats


def adapt_live(h_live: np.ndarray, g50: np.ndarray, calib_s: float = 1.5) -> tuple[np.ndarray, dict]:
    hc = calibrate(h_live, OUT_HZ, calib_s=calib_s)
    gc = calibrate(g50, OUT_HZ, calib_s=calib_s)
    uh = to_u(h_live, hc)
    pred = analytic_map(uh, gc, gc.s)
    meta = {
        "s_h": float(hc.s),
        "s_g1": float(gc.s),
        "rest_l": hc.rest_l.tolist(),
        "rest_r": hc.rest_r.tolist(),
        "scale_ratio": float(gc.s / max(hc.s, 1e-8)),
    }
    return pred.astype(np.float32), meta


def mapper_latency_us(n: int = 200) -> dict:
    try:
        from tritrack.intent.mapper import FutureIntentMapper, IN_DIM_INTENT

        ckpt = torch.load(str(MAPPER_B), map_location="cpu", weights_only=False)
        m = FutureIntentMapper(in_dim=IN_DIM_INTENT)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=False)
        m.eval()
        x = torch.zeros(1, IN_DIM_INTENT)
        cur = torch.zeros(1, 9)
        with torch.no_grad():
            for _ in range(20):
                m(x, cur)
            t0 = time.perf_counter()
            for _ in range(n):
                m(x, cur)
            dt = (time.perf_counter() - t0) / n
        return {"mapper_b_mean_ms": float(dt * 1e3), "n": int(n), "device": "cpu"}
    except Exception as e:
        return {"mapper_b_mean_ms": float("nan"), "n": 0, "device": "cpu", "error": f"{type(e).__name__}: {e}"}


def canon_latency_us(hlr: np.ndarray, g50: np.ndarray, n: int = 50) -> dict:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        adapt_live(hlr[: min(400, len(hlr))], g50[: min(400, len(g50))])
        times.append(time.perf_counter() - t0)
    a = np.asarray(times) * 1e3
    return {"canonicalizer_mean_ms": float(a.mean()), "canonicalizer_p90_ms": float(np.percentile(a, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(NEXT))
    ap.add_argument("--variants", type=str, default="live")
    ap.add_argument("--tasks", type=str, default=",".join(TASKS))
    ap.add_argument("--latency_ms", type=float, default=40.0)
    ap.add_argument("--jitter_ms", type=float, default=8.0)
    ap.add_argument("--drop_p", type=float, default=0.02)
    ap.add_argument("--calib_s", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    index = build_index()
    rng = _rng(int(args.seed))
    actors = {}
    man = {"variants": {}, "latency": {}, "users": {}}
    t_map = mapper_latency_us()
    man["latency"]["mapper_b"] = t_map
    mm = t_map.get("mapper_b_mean_ms")
    print(f"[h2r1] mapper-B {mm} ms CPU {t_map.get('error','')}", flush=True)

    first_pair = None
    for variant in variants:
        man["variants"][variant] = {}
        for task in tasks:
            clips = sorted((P1_ROOT / task).glob("*.npz"))
            rows = []
            dst_dir = out / "clips" / variant / task
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p1 in clips:
                rec = match_p1(p1, index)
                row = {"p1": p1.name, "ok": False, "variant": variant, "task": task}
                dst = dst_dir / p1.name
                if dst.is_file() and dst.stat().st_size > 1000:
                    row["ok"] = True
                    row["skipped"] = True
                    rows.append(row)
                    continue
                if rec is None:
                    row["reason"] = "no_meta_match"
                    rows.append(row)
                    continue
                try:
                    hlr, fps_h = extract_hlr(rec["bvh"])
                    tlr, fps_g = load_g1_tlr(p1)
                    n_out = int(tlr.shape[0])
                    h_var, fps_v = apply_variation(hlr, variant if variant != "live" else "normal", fps_h, rng)
                    live, st = live_sample(
                        h_var,
                        fps_v,
                        n_out,
                        latency_s=args.latency_ms * 1e-3,
                        jitter_s=args.jitter_ms * 1e-3,
                        drop_p=float(args.drop_p),
                        rng=rng,
                    )
                    # freeze calib from the first 1.5 s of THIS live stream only
                    pred, meta = adapt_live(live, tlr, calib_s=float(args.calib_s))
                    overlay_npz(p1, dst_dir / p1.name, pred)
                    if first_pair is None:
                        first_pair = (live, tlr)
                    uid = str(rec.get("actor_uid") or rec.get("take_name") or "?")
                    actors.setdefault(uid, 0)
                    actors[uid] += 1
                    drift = float(np.linalg.norm(pred[-1] - pred[0], axis=-1).mean())
                    row.update(
                        {
                            "ok": True,
                            "filename": rec["filename"],
                            "actor_uid": uid,
                            "bvh": rec["bvh"],
                            "stream": st,
                            **meta,
                            "end_start_drift_m": drift,
                        }
                    )
                except Exception as e:
                    row["reason"] = f"{type(e).__name__}: {e}"
                rows.append(row)
                print(f"[{variant}/{task}] {p1.name} {row.get('ok')} {row.get('reason', row.get('s_h'))}", flush=True)
            man["variants"][variant][task] = rows
            n_ok = sum(1 for r in rows if r.get("ok"))
            print(f"[h2r1] {variant}/{task} ok={n_ok}/{len(rows)}", flush=True)

    if first_pair is not None:
        man["latency"]["canonicalizer"] = canon_latency_us(*first_pair)
        man["latency"]["tracker_sim_ms"] = float(args.latency_ms)
        mb = t_map.get("mapper_b_mean_ms")
        mb = float(mb) if mb is not None and np.isfinite(mb) else 0.0
        man["latency"]["total_front_ms"] = (
            float(args.latency_ms)
            + man["latency"]["canonicalizer"]["canonicalizer_mean_ms"]
            + mb
        )
    man["users"] = {
        "n_actor_uids": len(actors),
        "counts": actors,
        "note": "SOMA actor_uid as morphology stand-in; no live headset on this host.",
    }
    man["policy"] = {
        "recovery": False,
        "thb": False,
        "learned_h2r_mlp": False,
        "dt": DT,
        "no_physical_tracker": True,
        "input": "timestamped SOMA Head/LeftHand/RightHand",
    }
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    (out / "latency").mkdir(parents=True, exist_ok=True)
    (out / "raw_stream").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / "live_clips.json").write_text(json.dumps(man, indent=2, default=str))
    (out / "latency" / "front_end.json").write_text(json.dumps(man["latency"], indent=2))
    (out / "calibration" / "users.json").write_text(json.dumps(man["users"], indent=2))
    print(json.dumps({"latency": man["latency"], "users": man["users"]["n_actor_uids"]}, indent=2), flush=True)
    print(f"[h2r1] wrote {out / 'manifests' / 'live_clips.json'}", flush=True)


if __name__ == "__main__":
    main()
