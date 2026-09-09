#!/usr/bin/env python3
"""Offline GEP, UCR capture, plots, REPORT for intent-free space. No Isaac."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

Z_DIM = 16
POS_SCALE = 0.05
RATIO_EPS = 1e-6
MIN_SCALE = 1e-4
ETA_REL = 1e-3
LAMBDA_REL = 1e-2
K_SWEEP = (2, 3, 4, 6)
K_PRIMARY = (3, 4)
RANDOM_SEEDS = 10
TRAIN_TERRAINS = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD_OUT = ("slip",)
ALL_TERRAINS = TRAIN_TERRAINS + HELD_OUT
UCR_JSON = Path("/data/home/chenxiangyu/robotics/Anybody/results/ucr1_final_unified_recovery/splits")
B_PUB = Path("/data/home/chenxiangyu/robotics/Anybody/results/irr_response_recovery/B_recovery.npz")
SPACE_LABEL = {
    "S0_random": "random",
    "S1_UCR": "UCR",
    "S2_free": "free",
    "S3_PIUCR": "PI-UCR",
    "S4_local": "local-oracle",
    "full_ucr": "full UCR",
    "parent": "Stage2 only",
}


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, np.bool_)):
        return int(obj) if not isinstance(obj, np.bool_) else bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
    }


def _robust_scale(Y: np.ndarray, min_scale: float = MIN_SCALE):
    med = np.median(Y, axis=0)
    mad = np.median(np.abs(Y - med), axis=0)
    scale = np.maximum(1.4826 * mad, min_scale)
    return med.astype(np.float64), scale.astype(np.float64)


def _gep(C_B: np.ndarray, C_I: np.ndarray, eta_rel: float):
    d = C_I.shape[0]
    eta = float(eta_rel) * float(np.trace(C_I)) / float(d)
    A = 0.5 * (C_B + C_B.T)
    B = 0.5 * (C_I + C_I.T) + eta * np.eye(d)
    w, Q = np.linalg.eigh(B)
    Bh = Q * (1.0 / np.sqrt(np.maximum(w, 1e-12)))
    M = Bh.T @ A @ Bh
    evals, U = np.linalg.eigh(0.5 * (M + M.T))
    evecs = Bh @ U
    order = np.argsort(evals)[::-1]
    ev = evecs[:, order]
    for i in range(d):
        nrm = np.linalg.norm(ev[:, i])
        if nrm > 0:
            ev[:, i] /= nrm
    return evals[order], ev, eta


def _load_jacs(root: Path):
    files = sorted((root / "dataset_a").glob("*/jac.npz"))
    packs = []
    for p in files:
        z = np.load(p, allow_pickle=True)
        packs.append(z)
        print(f"[analyze] jac {p} n={z['z0'].shape[0]}", flush=True)
    return packs


def _stack_mask(packs, pred):
    zs, JIs, JBs, yIs, yBs = [], [], [], [], []
    meta = []
    for z in packs:
        n = int(z["z0"].shape[0])
        for i in range(n):
            rec = {k: z[k][i] for k in ("split", "window", "terrain", "episode_id", "t")}
            rec = {k: (str(v) if k != "t" else int(v)) for k, v in rec.items()}
            if not pred(rec):
                continue
            zs.append(z["z0"][i])
            JIs.append(z["J_I"][i])
            JBs.append(z["J_B"][i])
            yIs.append(z["y_I"][i])
            yBs.append(z["y_B"][i])
            meta.append(rec)
    if not zs:
        return None
    return {
        "z0": np.stack(zs),
        "J_I": np.stack(JIs),
        "J_B": np.stack(JBs),
        "y_I": np.stack(yIs),
        "y_B": np.stack(yBs),
        "meta": meta,
    }


def _pt(z: np.ndarray) -> np.ndarray:
    z = z / (np.linalg.norm(z) + 1e-8)
    return np.eye(z.size) - np.outer(z, z)


def _qr_basis(M: np.ndarray) -> np.ndarray:
    q, _r = np.linalg.qr(M)
    return q


def _project_energy(B: np.ndarray, z: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    z = z / (np.linalg.norm(z) + 1e-8)
    d = d - np.dot(d, z) * z
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        return 0.0, 0.0
    d = d / dn
    Bt = _qr_basis(_pt(z) @ B)
    ds = Bt @ (Bt.T @ d)
    n2 = float(np.dot(ds, ds))
    n1 = float(np.linalg.norm(ds))
    cos = float(np.dot(d, ds) / (n1 + 1e-12)) if n1 > 0 else 0.0
    return n2, cos


def _load_ucr_rows():
    rows = []
    for sp in ("train", "val", "test"):
        p = UCR_JSON / f"{sp}.json"
        if not p.exists():
            continue
        rows.extend(json.loads(p.read_text()))
    succ = [r for r in rows if float(r.get("i_ora", 1.0)) < 0]
    return succ


def _split_episodes(rows, seed=2026):
    eps = sorted({str(r.get("episode_id")) for r in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(eps)
    n = len(eps)
    n_tr = int(round(0.60 * n))
    n_va = int(round(0.20 * n))
    mp = {}
    for i, k in enumerate(eps):
        mp[k] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return mp


def _svd_basis(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """D is (N, 16) stacked tangent directions. Basis is right singular vectors (16, k)."""
    if D.size == 0:
        return np.eye(Z_DIM)[:, :k], np.zeros(k)
    D = np.asarray(D, dtype=np.float64).reshape(-1, Z_DIM)
    _u, s, vt = np.linalg.svd(D, full_matrices=False)
    B = vt.T
    kk = min(int(k), B.shape[1])
    return B[:, :kk], s


def build_basis(root: Path) -> dict:
    packs = _load_jacs(root)
    if not packs:
        raise FileNotFoundError(f"no jac.npz under {root}/dataset_a")
    train = _stack_mask(
        packs,
        lambda r: r["split"] == "train"
        and r["window"] == "nominal"
        and r["terrain"] in TRAIN_TERRAINS,
    )
    if train is None:
        raise RuntimeError("no train nominal states for C_I/C_B")
    mu, scale = _robust_scale(train["y_B"])
    J_I_bar = train["J_I"] / POS_SCALE
    J_B_bar = train["J_B"] / scale.reshape(1, -1, 1)
    n = J_I_bar.shape[0]
    C_I = np.mean(np.matmul(np.transpose(J_I_bar, (0, 2, 1)), J_I_bar), axis=0)
    C_B = np.mean(np.matmul(np.transpose(J_B_bar, (0, 2, 1)), J_B_bar), axis=0)
    evals, evecs, eta = _gep(C_B, C_I, ETA_REL)
    sweep = {}
    for er in (1e-4, 1e-3, 1e-2):
        ev, _, et = _gep(C_B, C_I, er)
        sweep[str(er)] = {"evals": ev.tolist(), "eta": et}

    ucr_rows = _load_ucr_rows()
    split_mp = _split_episodes(ucr_rows)
    d_train = []
    d_test = []
    z_test = []
    meta_test = []
    for r in ucr_rows:
        d = np.asarray(r["d_oracle"], dtype=np.float64).reshape(-1)[:Z_DIM]
        z = np.asarray(r["z_nom"], dtype=np.float64).reshape(-1)[:Z_DIM]
        if d.size != Z_DIM or z.size != Z_DIM:
            continue
        sp = split_mp.get(str(r.get("episode_id")), "test")
        if sp == "train":
            d_train.append(d)
        if sp == "test":
            d_test.append(d)
            z_test.append(z)
            meta_test.append(
                {
                    "task": r.get("task"),
                    "theta_ora": r.get("theta_ora"),
                    "i_ora": r.get("i_ora"),
                    "episode_id": r.get("episode_id"),
                }
            )
    Dtr = np.stack(d_train) if d_train else np.zeros((0, Z_DIM))
    B_ucr, s_ucr = _svd_basis(Dtr, Z_DIM)
    if B_ucr.shape[0] != Z_DIM:
        raise RuntimeError(f"B_ucr has shape {B_ucr.shape}, expected ({Z_DIM}, k)")
    rng = np.random.RandomState(0)
    rand = []
    for i in range(RANDOM_SEEDS):
        A = rng.randn(Z_DIM, Z_DIM)
        q, _ = np.linalg.qr(A)
        rand.append(q)
    rand = np.stack(rand)

    bdir = root / "basis"
    bdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bdir / "B_free.npz",
        B=evecs.astype(np.float32),
        evals=evals.astype(np.float32),
        eta=np.asarray(eta),
        eta_rel=np.asarray(ETA_REL),
        n_train=np.asarray(n),
        terrains=np.asarray(TRAIN_TERRAINS),
        no_terrain_in_B=np.asarray(True),
    )
    np.savez_compressed(
        bdir / "B_ucr.npz",
        B=B_ucr.astype(np.float32),
        singular=s_ucr.astype(np.float32),
        n_train=np.asarray(len(d_train)),
        n_test=np.asarray(len(d_test)),
        no_terrain_in_B=np.asarray(True),
    )
    if B_PUB.exists():
        pub = np.load(B_PUB)
        np.savez_compressed(bdir / "B_ucr_published.npz", B=pub["B"], n=pub["n"])
    np.savez_compressed(bdir / "random_bases.npz", B=rand.astype(np.float32))
    np.savez_compressed(
        bdir / "body_scale.npz",
        mu=mu.astype(np.float32),
        scale=scale.astype(np.float32),
        pos_scale=np.asarray(POS_SCALE),
        n_train=np.asarray(n),
    )
    np.savez_compressed(bdir / "C_I.npz", C=C_I.astype(np.float32))
    np.savez_compressed(bdir / "C_B.npz", C=C_B.astype(np.float32))

    capture = {"k": {}}
    Bfree = evecs
    Bucr = B_ucr
    for k in K_SWEEP:
        blk = {"S2_free": [], "S1_UCR": [], "S0_random": []}
        for z, d in zip(z_test, d_test):
            e, c = _project_energy(Bfree[:, :k], z, d)
            blk["S2_free"].append({"energy": e, "cos": c})
            e, c = _project_energy(Bucr[:, :k], z, d)
            blk["S1_UCR"].append({"energy": e, "cos": c})
            rs = []
            for s in range(RANDOM_SEEDS):
                e, c = _project_energy(rand[s][:, :k], z, d)
                rs.append(e)
            blk["S0_random"].append({"energy": float(np.mean(rs)), "energy_std": float(np.std(rs))})
        capture["k"][str(k)] = {
            name: {
                "energy": _stats(np.array([x["energy"] for x in xs])),
                **({"cos": _stats(np.array([x["cos"] for x in xs]))} if "cos" in xs[0] else {}),
            }
            for name, xs in blk.items()
        }

    payload = {
        "n_train": int(n),
        "y_B_dim": int(train["y_B"].shape[1]),
        "y_I_dim": int(train["y_I"].shape[1]),
        "eta": float(eta),
        "eta_rel": ETA_REL,
        "evals": evals.tolist(),
        "eta_sweep": sweep,
        "ucr_n_succ": len(ucr_rows),
        "ucr_n_train_d": len(d_train),
        "ucr_n_test_d": len(d_test),
        "capture": capture,
        "train_terrains": list(TRAIN_TERRAINS),
        "held_out": list(HELD_OUT),
        "no_terrain_in_model": True,
    }
    (bdir / "basis_summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print(f"[analyze] B_free evals[:6]={np.round(evals[:6], 3)} n_train={n}", flush=True)
    return payload


def _summarize_local(eval_files: list[Path], k: int, deg: str = "5.0") -> dict:
    by_space = defaultdict(lambda: {"DI": [], "DB": [], "R": [], "terrain": []})
    by_terrain = defaultdict(lambda: defaultdict(lambda: {"DI": [], "DB": [], "R": []}))
    for p in eval_files:
        data = json.loads(p.read_text())
        terrain = data.get("terrain")
        for row in data.get("local") or []:
            for name, sp in (row.get("spaces") or {}).items():
                di = np.asarray(sp["DI"].get(deg) or sp["DI"].get(str(float(deg))), dtype=np.float64)
                db = np.asarray(sp["DB"].get(deg) or sp["DB"].get(str(float(deg))), dtype=np.float64)
                kk = min(k, di.size, db.size)
                di = di[:kk]
                db = db[:kk]
                r = db / (di + RATIO_EPS)
                by_space[name]["DI"].extend(di.tolist())
                by_space[name]["DB"].extend(db.tolist())
                by_space[name]["R"].extend(r.tolist())
                by_space[name]["terrain"].extend([terrain] * kk)
                by_terrain[terrain][name]["DI"].extend(di.tolist())
                by_terrain[terrain][name]["DB"].extend(db.tolist())
                by_terrain[terrain][name]["R"].extend(r.tolist())
    out = {"spaces": {}, "by_terrain": {}}
    for name, blk in by_space.items():
        di = np.asarray(blk["DI"])
        db = np.asarray(blk["DB"])
        r = np.asarray(blk["R"])
        out["spaces"][name] = {"DI": _stats(di), "DB": _stats(db), "R_BI": _stats(r)}
    for t, spaces in by_terrain.items():
        out["by_terrain"][t] = {
            name: {"DI": _stats(np.asarray(v["DI"])), "DB": _stats(np.asarray(v["DB"])), "R_BI": _stats(np.asarray(v["R"]))}
            for name, v in spaces.items()
        }
    return out


def _summarize_ucr(eval_files: list[Path]) -> dict:
    by = defaultdict(list)
    by_t = defaultdict(lambda: defaultdict(list))
    e_parent = []
    for p in eval_files:
        data = json.loads(p.read_text())
        t = data.get("terrain")
        for row in data.get("ucr") or []:
            a_full = float(row.get("A_full", 0.0))
            by["full_ucr"].append(a_full)
            by_t[t]["full_ucr"].append(a_full)
            e_parent.append(float(row.get("e_parent", np.nan)))
            for name, a in (row.get("A_proj") or {}).items():
                by[name].append(float(a))
                by_t[t][name].append(float(a))
                ra = float((row.get("R_A") or {}).get(name, np.nan))
                by[f"RA:{name}"].append(ra)
                by_t[t][f"RA:{name}"].append(ra)
    def adv(xs):
        x = np.asarray(xs, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        pos = x[x > 0]
        return {
            **_stats(x),
            "P_Agt0": float((x > 0).mean()),
            "P_Agt0_05": float((x > 0.05).mean()),
            "retention_on_pos_full": None,
        }
    out = {"all": {k: adv(v) for k, v in by.items()}, "by_terrain": {}}
    full = np.asarray(by.get("full_ucr") or [], dtype=np.float64)
    for name, xs in by.items():
        if name.startswith("RA:"):
            raw = np.asarray(xs, dtype=np.float64)
            mask = full[: raw.size] > 0 if full.size else np.zeros(0, dtype=bool)
            if mask.size == raw.size and mask.any():
                out["all"][name]["retention_on_pos_full"] = _stats(raw[mask])
        elif name != "full_ucr" and full.size == len(xs):
            mask = full > 0
            if mask.any():
                out["all"][name]["retention_on_pos_full"] = _stats(np.asarray(xs)[mask] / (full[mask] + RATIO_EPS))
    for t, spaces in by_t.items():
        out["by_terrain"][t] = {k: adv(v) for k, v in spaces.items()}
    return out


def _classify(local_k4: dict, ucr: dict) -> dict:
    sp = local_k4.get("spaces") or {}
    di_f = (sp.get("S2_free") or {}).get("DI") or {}
    di_u = (sp.get("S1_UCR") or {}).get("DI") or {}
    di_r = (sp.get("S0_random") or {}).get("DI") or {}
    db_f = (sp.get("S2_free") or {}).get("DB") or {}
    db_u = (sp.get("S1_UCR") or {}).get("DB") or {}
    r_f = (sp.get("S2_free") or {}).get("R_BI") or {}
    r_u = (sp.get("S1_UCR") or {}).get("R_BI") or {}
    r_r = (sp.get("S0_random") or {}).get("R_BI") or {}
    pi = (sp.get("S3_PIUCR") or {}).get("R_BI") or {}
    loc = (sp.get("S4_local") or {}).get("R_BI") or {}
    ua = ((ucr.get("all") or {}).get("S2_free") or {})
    ufull = ((ucr.get("all") or {}).get("full_ucr") or {})
    ra = ((ucr.get("all") or {}).get("RA:S2_free") or {}).get("retention_on_pos_full") or {}
    cases = []
    mf = di_f.get("median")
    mb = db_f.get("median")
    if mf is not None and mb is not None and mf < 0.3 and mb < 0.3:
        cases.append("F1")
    if mf is not None and mb is not None and mf > 1.0 and mb > 0.5:
        cases.append("F2")
    if r_f.get("median") and r_f["median"] > (r_u.get("median") or 0) and (ua.get("median") or 0) < 0.02:
        cases.append("F3")
    if (loc.get("median") or 0) > 1.5 * (r_f.get("median") or 1) and (r_f.get("median") or 0) < (r_u.get("median") or 0):
        cases.append("F4")
    if (pi.get("median") or 0) > 1.3 * (r_f.get("median") or 1):
        cases.append("F5")
    return {
        "cases": cases,
        "DI_free": di_f,
        "DI_ucr": di_u,
        "DI_random": di_r,
        "DB_free": db_f,
        "DB_ucr": db_u,
        "R_free": r_f,
        "R_ucr": r_u,
        "R_random": r_r,
        "R_PI": pi,
        "R_local": loc,
        "A_free": ua,
        "A_full": ufull,
        "RA_free": ra,
    }


def _plot(root: Path, basis: dict, local: dict, ucr: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = root / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    ev = np.asarray(basis.get("evals") or [], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, ev.size + 1), ev, marker="o")
    ax.set_xlabel("index")
    ax.set_ylabel(r"$\lambda$ (body / intent)")
    ax.set_title("Plot 1  generalized eigenvalue spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plot1_spectrum.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"S0_random": "#888888", "S1_UCR": "#d95f02", "S2_free": "#1b9e77", "S3_PIUCR": "#7570b3", "S4_local": "#e7298a"}
    for name, col in colors.items():
        sp = (local.get("spaces") or {}).get(name) or {}
        di = sp.get("DI") or {}
        db = sp.get("DB") or {}
        if di.get("median") is None:
            continue
        ax.scatter(di["median"], db["median"], s=80, c=col, label=SPACE_LABEL.get(name, name), zorder=3)
        ax.errorbar(di["median"], db["median"], xerr=max(di["p90"] - di["median"], 0), yerr=max(db["p90"] - db["median"], 0),
                    fmt="none", ecolor=col, alpha=0.4)
    ax.set_xlabel(r"intent interference $D_I$ (5°)")
    ax.set_ylabel(r"body authority $D_B$ (5°)")
    ax.set_title("Plot 2  intent vs body (upper-left is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plot2_scatter_DI_DB.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ks = list(K_SWEEP)
    for name, col in (("S0_random", "#888"), ("S1_UCR", "#d95f02"), ("S2_free", "#1b9e77"), ("S3_PIUCR", "#7570b3")):
        ys = []
        for k in ks:
            sp = ((local.get("by_k") or {}).get(str(k)) or {}).get("spaces") or {}
            med = ((sp.get(name) or {}).get("R_BI") or {}).get("median")
            ys.append(med if med is not None else np.nan)
        ax.plot(ks, ys, marker="o", color=col, label=SPACE_LABEL.get(name, name))
    ax.set_xlabel("k")
    ax.set_ylabel(r"$R_{BI}$ median")
    ax.set_title("Plot 3  authority / interference vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plot3_RBI_vs_k.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["full_ucr", "S1_UCR", "S2_free", "S3_PIUCR", "S0_random"]
    means, meds, labels = [], [], []
    for n in names:
        blk = (ucr.get("all") or {}).get(n) or {}
        means.append(blk.get("mean") if blk.get("mean") is not None else 0.0)
        meds.append(blk.get("median") if blk.get("median") is not None else 0.0)
        labels.append(SPACE_LABEL.get(n, n))
    x = np.arange(len(names))
    ax.bar(x - 0.18, means, 0.35, label="mean A")
    ax.bar(x + 0.18, meds, 0.35, label="median A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("advantage A")
    ax.set_title("Plot 4  UCR recovery retention")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plot4_ucr_A.png", dpi=140)
    plt.close(fig)

    eh_dir = root / "ucr_projection"
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for tfile in sorted(eh_dir.glob("*_e_hist.npz")):
        z = np.load(tfile)
        t = np.arange(z[z.files[0]].shape[-1]) * 0.02
        for key, ls in (("parent", "--"), ("full_ucr", "-"), ("S2_free", "-"), ("S1_UCR", "-"), ("S3_PIUCR", "-")):
            if key not in z.files:
                continue
            y = np.mean(z[key], axis=0)
            ax.plot(t, y, ls, label=f"{Path(tfile).stem.replace('_e_hist','')}:{SPACE_LABEL.get(key, key)}")
            plotted = True
        break
    if plotted:
        ax.set_xlabel("t (s)")
        ax.set_ylabel("controlled-point tracking error")
        ax.set_title("Plot 5  intent error during UCR recovery")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(pdir / "plot5_intent_timeseries.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [("DI", "intent $D_I$"), ("DB", "body $D_B$"), ("R_BI", r"$R_{BI}$")]
    seen = [t for t in TRAIN_TERRAINS]
    hold = ["slip"]
    for ax, (mk, title) in zip(axes, metrics):
        labs = ["seen", "slip"]
        for i, name in enumerate(("S0_random", "S1_UCR", "S2_free", "S3_PIUCR")):
            ys = []
            for grp in (seen, hold):
                vals = []
                for t in grp:
                    blk = ((local.get("by_terrain") or {}).get(t) or {}).get(name) or {}
                    med = (blk.get(mk) or {}).get("median")
                    if med is not None:
                        vals.append(med)
                ys.append(float(np.mean(vals)) if vals else np.nan)
            ax.plot(labs, ys, marker="o", label=SPACE_LABEL.get(name, name))
        ax.set_title(f"Plot 6  {title}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(pdir / "plot6_seen_vs_slip.png", dpi=140)
    plt.close(fig)


def _fmt(st, key="median"):
    if not st or st.get(key) is None:
        return "n/a"
    return f"{st[key]:.4f}"


def write_report(root: Path, basis: dict, metrics: dict) -> None:
    local = metrics.get("local_k4_5deg") or {}
    ucr = metrics.get("ucr") or {}
    clf = metrics.get("classification") or {}
    cap = ((basis.get("capture") or {}).get("k") or {}).get("4") or {}
    slip = ((local.get("by_terrain") or {}).get("slip") or {})
    q = []
    di_f, di_u = clf.get("DI_free") or {}, clf.get("DI_ucr") or {}
    db_f, db_u = clf.get("DB_free") or {}, clf.get("DB_ucr") or {}
    r_f, r_u, r_r = clf.get("R_free") or {}, clf.get("R_ucr") or {}, clf.get("R_random") or {}
    q1 = "尚无局部评测" if not di_f else (
        f"有。5° 下 B_free 的 D_I 中位数={_fmt(di_f)}，相对 UCR {_fmt(di_u)}、random {_fmt(clf.get('DI_random') or {})}；"
        f"D_B 中位数={_fmt(db_f)}（UCR {_fmt(db_u)}）。"
    )
    ev = np.asarray(basis.get("evals") or [], dtype=np.float64)
    q2 = "谱未知"
    if ev.size:
        frac = np.cumsum(np.maximum(ev, 0))
        tot = frac[-1] if frac[-1] > 0 else 1.0
        useful = int((ev > 0.3 * ev[0]).sum()) if ev[0] > 0 else 0
        q2 = f"前 4 个 λ={np.round(ev[:4], 3).tolist()}；约 {useful} 维显著高于其余（主推 k=3/4）。"
    q3 = (
        f"B_free R_BI 中位={_fmt(r_f)}，UCR={_fmt(r_u)}，random={_fmt(r_r)}。"
        if r_f else "评测未完成。"
    )
    r_pi = clf.get("R_PI") or {}
    q4 = f"PI-UCR R_BI={_fmt(r_pi)} vs 原始 UCR {_fmt(r_u)}。" if r_pi else "评测未完成。"
    ra = clf.get("RA_free") or {}
    q5 = f"投影保留比中位={_fmt(ra)}；B_free 投影 mean A={_fmt(clf.get('A_free') or {}, 'mean')}，full UCR mean A={_fmt(clf.get('A_full') or {}, 'mean')}。"
    slip_r = (slip.get("S2_free") or {}).get("R_BI") or {}
    q6 = f"held-out slip 上 B_free R_BI={_fmt(slip_r)}。" if slip else "slip 评测未完成。"
    loc = clf.get("R_local") or {}
    q7 = (
        f"局部 oracle R_BI={_fmt(loc)} vs 全局 B_free {_fmt(r_f)}。"
        + (" 局部明显更好 → 空间强状态依赖。" if (loc.get("median") and r_f.get("median") and loc["median"] > 1.5 * r_f["median"]) else " 全局与局部接近 → 近似可共享。")
        if loc.get("median") is not None else "S4 未评到。"
    )
    cases = clf.get("cases") or []
    if "F1" in cases:
        nxt = "放弃把当前 B_free 当自由空间；方向几乎不做事。"
    elif "F3" in cases:
        nxt = "停止当前「自由空间=恢复方向」路线，先不要做主动探测。"
    elif "F5" in cases or "F4" in cases:
        nxt = "下一步用状态依赖的意图投影 δz = P_I(z,o) d_interaction，而不是固定共享基。"
    elif r_f.get("median") and r_u.get("median") and r_f["median"] > r_u["median"]:
        nxt = "可以继续用固定共享 B_free / 投影规则；不要做任务专用基。先不要自动开主动探测。"
    else:
        nxt = "证据不足，先看 slip 与 UCR 保留比，再决定是否放弃 Stage-2 latent 作为自主接口。"

    lines = [
        "# Intent-preserving latent freedom — Phase I 诊断",
        "",
        "Frozen AnyBody Stage-2，P1 loco。地形只作评测切片，不进入观测、雅可比或 B_free。",
        "扰动为球面测地线；主分析 ε=1°、L=1；比较在 5°。UCR 优势 A = E_parent − E_pert（越大越好）。",
        "",
        "## 八个科学问题",
        "",
        f"1. **是否存在低意图干扰、高下肢权威的子空间？** {q1}",
        f"2. **有用维度？** {q2}",
        f"3. **B_free 的 R_BI 是否优于 random / UCR？** {q3}",
        f"4. **意图投影 UCR 是否优于原始 UCR？** {q4}",
        f"5. **投影后还剩多少 UCR 恢复优势？** {q5}",
        f"6. **held-out slip 是否仍成立？** {q6}",
        f"7. **共享还是强状态依赖？** {q7}",
        f"8. **下一步？** {nxt}",
        "",
        f"诊断标签: {cases or ['无 F1–F5 强否定']}。",
        "",
        "## 设置",
        "",
        f"- train states (C_I/C_B): {basis.get('n_train')}",
        f"- η = {basis.get('eta')} (rel={basis.get('eta_rel')})",
        f"- UCR d* succ={basis.get('ucr_n_succ')} train={basis.get('ucr_n_train_d')} test={basis.get('ucr_n_test_d')}",
        "- 原始 UCR snaps 未落盘；live A 在新 clone 的 RE-trigger 上、burst=5 / H=25 / 5° 测地线。",
        "- 几何捕获（能量/余弦）用已有 d* JSON，test 不进 SVD。",
        "",
        "## 广义特征值（前 8）",
        "",
        "```",
        str(np.round(ev[:8], 4).tolist() if ev.size else []),
        "```",
        "",
        "## 5° k=4 局部指标",
        "",
        "| space | D_I med | D_B med | R_BI med |",
        "|---|---:|---:|---:|",
    ]
    for name in ("S0_random", "S1_UCR", "S2_free", "S3_PIUCR", "S4_local"):
        sp = (local.get("spaces") or {}).get(name) or {}
        lines.append(
            f"| {SPACE_LABEL.get(name, name)} | {_fmt(sp.get('DI') or {})} | {_fmt(sp.get('DB') or {})} | {_fmt(sp.get('R_BI') or {})} |"
        )
    lines += [
        "",
        "## UCR 投影能量（held-out d*, k=4）",
        "",
        "| space | energy med | cosine med |",
        "|---|---:|---:|",
    ]
    for name in ("S0_random", "S1_UCR", "S2_free"):
        blk = cap.get(name) or {}
        lines.append(f"| {SPACE_LABEL.get(name, name)} | {_fmt(blk.get('energy') or {})} | {_fmt(blk.get('cos') or {})} |")
    lines += [
        "",
        "## Live 投影恢复 A",
        "",
        "| space | mean A | median A | P(A>0) | P(A>0.05) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("full_ucr", "S1_UCR", "S2_free", "S3_PIUCR", "S0_random"):
        blk = (ucr.get("all") or {}).get(name) or {}
        lines.append(
            f"| {SPACE_LABEL.get(name, name)} | {_fmt(blk, 'mean')} | {_fmt(blk)} | "
            f"{blk.get('P_Agt0') if blk.get('P_Agt0') is not None else 'n/a'} | "
            f"{blk.get('P_Agt0_05') if blk.get('P_Agt0_05') is not None else 'n/a'} |"
        )
    lines += [
        "",
        "## 地形切片（B_free, 5°, k=4）",
        "",
        "| terrain | D_I | D_B | R_BI | held-out |",
        "|---|---:|---:|---:|---|",
    ]
    for t in ALL_TERRAINS:
        blk = (local.get("by_terrain") or {}).get(t) or {}
        sp = blk.get("S2_free") or {}
        lines.append(
            f"| {t} | {_fmt(sp.get('DI') or {})} | {_fmt(sp.get('DB') or {})} | {_fmt(sp.get('R_BI') or {})} | "
            f"{'yes' if t in HELD_OUT else 'no'} |"
        )
    lines += [
        "",
        "本轮到此停止。不启动主动顺序探测 / R4。",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(root: Path, basis: dict | None = None) -> dict:
    bpath = root / "basis" / "basis_summary.json"
    if basis is None:
        if not bpath.exists():
            raise FileNotFoundError("run --basis first")
        basis = json.loads(bpath.read_text())
    eval_files = sorted((root / "eval").glob("*.json"))
    local_by_k = {}
    for k in K_SWEEP:
        local_by_k[str(k)] = _summarize_local(eval_files, k=k, deg="5.0") if eval_files else {"spaces": {}, "by_terrain": {}}
    local_k4 = dict(local_by_k.get("4") or {"spaces": {}, "by_terrain": {}})
    local_k4["by_k"] = local_by_k
    ucr = _summarize_ucr(eval_files) if eval_files else {"all": {}, "by_terrain": {}}
    clf = _classify(local_k4, ucr)
    metrics = {
        "local_k4_5deg": local_k4,
        "local_by_k": local_by_k,
        "ucr": ucr,
        "classification": clf,
        "basis_evals": basis.get("evals"),
        "n_eval_files": len(eval_files),
        "no_terrain_in_model": True,
    }
    (root / "metrics.json").write_text(json.dumps(_sanitize(metrics), indent=2), encoding="utf-8")
    try:
        _plot(root, basis, local_k4, ucr)
    except Exception as e:
        print(f"[analyze] plot failed: {e}", flush=True)
    write_report(root, basis, metrics)
    print(f"[analyze] wrote {root/'REPORT.md'} eval_files={len(eval_files)}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/intent_free_space")
    ap.add_argument("--mode", type=str, default="basis", choices=("basis", "report", "all"))
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    basis = None
    if args.mode in ("basis", "all"):
        basis = build_basis(root)
    if args.mode in ("report", "all"):
        build_report(root, basis=basis)


if __name__ == "__main__":
    main()
