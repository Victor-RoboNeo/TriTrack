#!/usr/bin/env python3
"""IRR R0/R2 report. R1 is ICR residual (already failed)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _pm(x) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _f(x, nd=4) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def _pair_sign(a: np.ndarray, names: list[str]) -> dict:
    """Oracle ±-pair: if +v helps, does −v hurt?"""
    idx = {n: i for i, n in enumerate(names)}
    n_pair, opp, both_pos = 0, 0, 0
    for n in names:
        if not n.startswith("+"):
            continue
        neg = "-" + n[1:]
        if neg not in idx:
            continue
        i, j = idx[n], idx[neg]
        ap, am = a[:, i], a[:, j]
        n_pair += int(a.shape[0])
        opp += int((((ap > 0) & (am < 0)) | ((ap < 0) & (am > 0))).sum())
        both_pos += int(((ap > 0) & (am > 0)).sum())
    if n_pair == 0:
        return {"n": 0, "P_opposite_sign": None, "P_both_positive": None}
    return {
        "n": n_pair,
        "P_opposite_sign": opp / n_pair,
        "P_both_positive": both_pos / n_pair,
    }


def _r0_extra(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    a = z["a"].astype(np.float64)
    star = z["star"].astype(np.int32)
    win = np.asarray(z["window"]).astype(str)
    names = [str(x) for x in z["names"].tolist()]
    rec = win == "recovery"
    out = {
        "P_star_nonzero": float((star != 0).mean()) if star.size else None,
        "P_star_nonzero_recovery": float((star[rec] != 0).mean()) if rec.any() else None,
        "pairs": _pair_sign(a, names),
        "pairs_recovery": _pair_sign(a[rec], names) if rec.any() else {"n": 0},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/irr_response_recovery")
    args = ap.parse_args()
    root = Path(args.root)
    md = ["# IRR — Interaction Response Recovery\n\n"]
    md.append(
        "R1 (`h→Δz`) is ICR M1–M3: failed. Residual became a second tracker; "
        "that path is closed.\n\n"
        "This round does **not** train a residual policy. Frozen Stage-2 only. "
        "Question: can we *rank candidate latent corrections* relative to `δz=0`?\n\n"
    )
    md.append("## R0 oracle candidate search\n\n")
    md.append(
        "| Tag | Task | Terrain | window | n | P(A*>0) | P(A*>0.05) | A*/std | mean A* | p50 A* |\n"
    )
    md.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    extras = {}
    rec_p, rec_a, rec_std, rec_p05 = [], [], [], []
    nom_a = []
    for p in sorted(root.glob("r0*/**/*/summary.json")):
        d = json.loads(p.read_text())
        npz = p.parent / "r0.npz"
        extra = _r0_extra(npz) if npz.is_file() else {}
        extras[str(p.parent.relative_to(root))] = extra
        tag = d.get("r0_tag") or p.parts[-4]
        for win in ("recovery", "nominal", "all"):
            b = d.get(win) or {}
            if not b.get("n"):
                continue
            md.append(
                f"| {tag} | {d.get('task')} | {d.get('terrain')} | {win} | {b.get('n')} | "
                f"{_pm(b.get('P_Agt0'))} | {_pm(b.get('P_Agt0_05'))} | "
                f"{_f(b.get('Astar_over_std'), 2)} | "
                f"{_f(b.get('mean_Astar'))} | {_f(b.get('p50_Astar'))} |\n"
            )
            if "r0_full" in str(p) and win == "recovery" and b.get("mean_Astar") is not None:
                rec_p.append(float(b.get("P_Agt0") or 0.0))
                rec_a.append(float(b["mean_Astar"]))
                if b.get("Astar_over_std") is not None:
                    rec_std.append(float(b["Astar_over_std"]))
                if b.get("P_Agt0_05") is not None:
                    rec_p05.append(float(b["P_Agt0_05"]))
            if "r0_full" in str(p) and win == "nominal" and b.get("mean_Astar") is not None:
                nom_a.append(float(b["mean_Astar"]))
    md.append("\nOracle ±-pair sign (recovery windows):\n\n")
    md.append("| Cell | P(+v and −v opposite) | P(both A>0) |\n")
    md.append("|---|---:|---:|\n")
    for k, extra in extras.items():
        pr = extra.get("pairs_recovery") or {}
        md.append(
            f"| {k} | {_pm(pr.get('P_opposite_sign'))} | {_pm(pr.get('P_both_positive'))} |\n"
        )
    r2 = root / "r2.json"
    r2n = root / "r2_nohist.json"
    md.append("\n## R2 response model (held-out clips)\n\n")
    md.append("| Model | n | Spearman | Spearman(rec) | top-1 | regret | sign | P(Â*→A>0) |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    r2_spearman_rec = None
    for lab, fp in (("history+δz", r2), ("δz only", r2n)):
        if not fp.is_file():
            md.append(f"| {lab} | — | not trained | — | — | — | — | — |\n")
            continue
        d = json.loads(fp.read_text())
        t = d.get("test") or {}
        if lab == "history+δz":
            r2_spearman_rec = t.get("spearman_recovery")
        md.append(
            f"| {lab} | {t.get('n_states')} | {_f(t.get('spearman'))} | "
            f"{_f(t.get('spearman_recovery'))} | {_pm(t.get('top1'))} | "
            f"{_f(t.get('oracle_regret'))} | {_pm(t.get('sign_acc'))} | "
            f"{_pm(t.get('P_pred_star_gt0'))} |\n"
        )
        by_t = t.get("by_terrain") or {}
        if by_t:
            md.append("\n")
            for terr, b in by_t.items():
                md.append(
                    f"  - {lab} / {terr}: n={b.get('n')} Spearman={_f(b.get('spearman'))} "
                    f"top1={_pm(b.get('top1'))}\n"
                )
            md.append("\n")
    mean_rec_p = float(np.mean(rec_p)) if rec_p else None
    mean_rec_a = float(np.mean(rec_a)) if rec_a else None
    mean_nom_a = float(np.mean(nom_a)) if nom_a else None
    mean_std = float(np.mean(rec_std)) if rec_std else None
    mean_p05 = float(np.mean(rec_p05)) if rec_p05 else None
    r2h = json.loads(r2.read_text()) if r2.is_file() else {}
    r2n_d = json.loads(r2n.read_text()) if r2n.is_file() else {}
    t_h = (r2h.get("test") or {}) if r2h else {}
    t_n = (r2n_d.get("test") or {}) if r2n_d else {}
    r2_spearman_rec = t_h.get("spearman_recovery")
    r2_sign = t_h.get("sign_acc_recovery") or t_h.get("sign_acc")
    nohist_s = t_n.get("spearman_recovery") or t_n.get("spearman")
    hist_gain = None
    if r2_spearman_rec is not None and nohist_s is not None:
        hist_gain = float(r2_spearman_rec) - float(nohist_s)
    r3p = root / "r3_lite.json"
    r3 = json.loads(r3p.read_text()) if r3p.is_file() else {}
    r2d_p = root / "r2_direction.json"
    r2d = json.loads(r2d_p.read_text()) if r2d_p.is_file() else {}
    has_full = any((root / "r0_full").glob("*/*/summary.json"))
    r0_pos = bool(
        has_full
        and (
            (mean_p05 is not None and mean_p05 >= 0.15)
            or (mean_rec_a is not None and mean_rec_a >= 0.03)
        )
    )
    r2_dir = bool(
        r2_sign is not None
        and float(r2_sign) >= 0.65
        and hist_gain is not None
        and hist_gain >= 0.15
    )
    r2_done = r2.is_file()
    r3_pos = bool(r3.get("sign_1deg_to_5deg") is not None and float(r3["sign_1deg_to_5deg"]) >= 0.75)
    if not has_full and not r0_pos:
        case = (
            "R0-narrow (6-axis, 1–2°) is extreme-value noise (A*/std≈2). "
            "Not Case C yet — expand to 15-axis × {1,2,5}° first."
        )
    elif not r0_pos:
        case = "C (latent recovery freedom weak on this candidate set)"
    elif not r2_done:
        case = "R0 looks non-C; R2 not trained yet"
    elif r2_dir:
        case = "A (passive history ranks recovery direction, not just an axis prior)"
    elif r3_pos:
        case = "B (freedom exists; passive history does not identify direction; 1° intervention does)"
    else:
        case = "B (freedom exists; passive history does not identify direction → need a cheaper probe)"
    md.append("\n## Direction diagnostics\n\n")
    if r2d:
        md.append(
            f"Held-out recovery, 5° subset: Spearman={_f(r2d.get('spearman_5deg_only'))}, "
            f"top1={_pm(r2d.get('top1_5deg'))} (chance {_pm(r2d.get('chance_top1_5deg'))}), "
            f"±pair acc={_pm(r2d.get('pair_sign_acc_5deg'))}, "
            f"sign vs zero={_pm(r2d.get('sign_acc'))}. "
            f"History gain vs δz-only Spearman={_f(hist_gain)}.\n\n"
        )
    if r3:
        md.append(
            f"R3-lite (oracle 1° response → 5° sign): 1°→5° sign transfer="
            f"{_pm(r3.get('sign_1deg_to_5deg'))}, pair from 1°={_pm(r3.get('pair_sign_from_1deg'))}, "
            f"mean A(probe→5°)={_f(r3.get('mean_probe5_from_1deg'))} vs A*={_f(r3.get('mean_Astar'))}. "
            f"Caveat: the 1° 'probe' here is a full K=10 rollout, not a 1–2 step pulse.\n\n"
        )
    md.append("\n## Case call\n\n")
    md.append(
        f"R0-full recovery: mean P(A*>0)={_pm(mean_rec_p)}, P(A*>0.05)={_pm(mean_p05)}, "
        f"mean A*={_f(mean_rec_a)}, A*/std={_f(mean_std, 2)}; "
        f"nominal mean A*={_f(mean_nom_a)}.\n\n"
    )
    md.append(f"**{case}**\n\n")
    md.append(
        "Case A: R0≫0 and R2 Spearman≫0 → passive history ranks candidates.\n\n"
        "Case B: R0≫0, R2≈0 → need active probe (R3).\n\n"
        "Case C: R0≈0 → latent lacks recovery freedom here.\n"
    )
    text = "".join(md)
    (root / "REPORT.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
