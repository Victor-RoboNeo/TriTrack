#!/usr/bin/env python3
"""P2-R Step 4 smoke tests. No Isaac, no PPO.

A  α=0 → exact parent
B  α=1 + zero-init r_η → exact parent
C  freeze audit (only r_η + critic trainable)
D  mask audit (visible intent only)
E  gate causality (≤ t, no 0.5 s self-recover)
F  replay Step 3 loco rollouts through RecoveryRiskGate
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import torch
import yaml

ROOT = Path("/data/home/chenxiangyu/robotics/Anybody")
sys.path.insert(0, str(ROOT / "source" / "rsl_rl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsl_rl.modules.intent_recovery import (  # noqa: E402
    DT,
    E_DIM,
    Q50_E,
    Q90_E,
    RecoveryRiskGate,
    RecoveryResidualMLP,
    extract_visible_task_error,
    risk_scores,
)
from rsl_rl.modules.latent_rl_actor_critic import LatentRLActorCritic  # noqa: E402

from analyze_p2r_step3 import (  # noqa: E402
    HEALTHY_TERRAINS,
    MATRIX,
    WARMUP,
    _first_trigger,
    _lead_sweep,
    _load_root,
    _pct,
)

CKPT = Path(
    "/data/home/chenxiangyu/robotics/Anybody/logs/rsl_rl/"
    "g1_flat_muse_kp_latent_rl/2026-08-26_00-38-10_tritrack_headhands_locomani_from35000/"
    "model_50000.pt"
)
AGENT_YAML = CKPT.parent / "params" / "agent.yaml"
OUT = Path("/data/home/chenxiangyu/robotics/Anybody/results/p2r_step4")
ATOL = 1e-6


def _policy_kwargs() -> dict:
    with AGENT_YAML.open() as f:
        agent = yaml.safe_load(f)
    kw = dict(agent["policy"])
    kw.pop("class_name", None)
    kw.pop("actor_hidden_dims", None)
    kw.pop("noise_std_type", None)
    return kw


def _build(intent_recovery: bool) -> LatentRLActorCritic:
    kw = _policy_kwargs()
    kw["intent_recovery"] = bool(intent_recovery)
    kw["terrain_scan_dim"] = 0
    pol = LatentRLActorCritic(
        num_actor_obs=750,
        num_critic_obs=895,
        num_actions=29,
        **kw,
    )
    pol.eval()
    return pol


def _load(pol: LatentRLActorCritic, strict: bool) -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    pol.load_state_dict(ck["model_state_dict"], strict=strict)


def _rand_obs(n: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, 750, generator=g)
    # Valid HeadHands loco mask: torso visible, others masked (1=masked).
    L, N = 15, 5
    kp = L * N * 3
    mask = obs[:, kp : kp + L * N].reshape(n, L, N)
    mask.fill_(1.0)
    mask[:, :, 0] = 0.0
    obs[:, kp : kp + L * N] = mask.reshape(n, -1)
    return obs


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def test_ab() -> dict:
    parent = _build(False)
    rec = _build(True)
    _load(parent, strict=True)
    _load(rec, strict=False)
    obs = _rand_obs()
    with torch.no_grad():
        a0 = parent.act_inference(obs)
        z0 = parent.inspect_inference(obs)

        rec._recovery_force_alpha = 0.0
        rec.reset()
        a_a = rec.act_inference(obs)
        z_a = rec.inspect_inference(obs)

        rec._recovery_force_alpha = 1.0
        rec.reset()
        a_b = rec.act_inference(obs)
        z_b = rec.inspect_inference(obs)

    d_a = _max_abs(a_a, a0)
    d_b = _max_abs(a_b, a0)
    dz_a = _max_abs(z_a["z_exec"], z0["z_exec"])
    dz_b = _max_abs(z_b["z_exec"], z0["z_exec"])
    dzn_a = _max_abs(z_a["z_nom"], z0["z_nom"])
    r_b = float(z_b["u"].abs().max().item())
    ok_a = d_a < ATOL and dz_a < ATOL
    ok_b = d_b < ATOL and dz_b < ATOL and r_b < ATOL
    return {
        "A_alpha0_action_maxabs": d_a,
        "A_z_exec_maxabs": dz_a,
        "A_z_nom_vs_parent": dzn_a,
        "A_pass": ok_a,
        "B_alpha1_zeroinit_action_maxabs": d_b,
        "B_z_exec_maxabs": dz_b,
        "B_r_raw_maxabs": r_b,
        "B_pass": ok_b,
    }


def test_c() -> dict:
    rec = _build(True)
    _load(rec, strict=False)
    trainable = [n for n, p in rec.named_parameters() if p.requires_grad]
    frozen_must = []
    for n, p in rec.named_parameters():
        if n.startswith("muse.transformer_encoder.") and p.requires_grad:
            frozen_must.append(n)
        if n.startswith("muse.decoder.") and p.requires_grad:
            frozen_must.append(n)
        if n.startswith("residual_corrector.") and p.requires_grad:
            frozen_must.append(n)
        if n == "latent_log_std" and p.requires_grad:
            frozen_must.append(n)
    prefixes = sorted({n.split(".")[0] for n in trainable})
    rec_only = all(
        n.startswith("intent_recovery_net.") or n.startswith("critic.") for n in trainable
    )
    last = rec.intent_recovery_net.net[-1]
    w0 = float(last.weight.abs().max().item())
    b0 = float(last.bias.abs().max().item())
    return {
        "n_trainable": len(trainable),
        "trainable_prefixes": prefixes,
        "only_recovery_and_critic": rec_only,
        "illegal_trainable": frozen_must,
        "last_W_max": w0,
        "last_b_max": b0,
        "C_pass": rec_only and not frozen_must and w0 == 0.0 and b0 == 0.0,
    }


def test_d() -> dict:
    L, n_b = 15, 5
    slot0 = 7  # KP_LAYOUT_SYM_SPARSE_0P5S index of offset 0
    modes = {
        "torso": (1, 0, 0),
        "torso_L": (1, 1, 0),
        "torso_R": (1, 0, 1),
        "torso_L_R": (1, 1, 1),
    }
    torso = torch.tensor([0.20, 0.0, 0.0])
    lw = torch.tensor([0.50, 0.0, 0.0])
    rw = torch.tensor([0.40, 0.0, 0.0])
    rows = {}
    ok = True
    for name, vis_bit in modes.items():
        kp = torch.zeros(1, L, n_b, 3)
        kp[0, slot0, 0] = torso
        kp[0, slot0, 1] = lw
        kp[0, slot0, 2] = rw
        mask = torch.ones(1, n_b)
        for i, v in enumerate(vis_bit):
            mask[0, i] = 0.0 if v else 1.0
        # ankles always masked
        mask[0, 3:] = 1.0
        e, vis, e_rms = extract_visible_task_error(kp, mask, slot0)
        e9 = e.reshape(3, 3)
        for i, v in enumerate(vis_bit):
            if v == 0 and float(e9[i].abs().sum()) > 0:
                ok = False
        nvis = sum(vis_bit)
        expect = (vis_bit[0] * 0.20 + vis_bit[1] * 0.50 + vis_bit[2] * 0.40) / nvis
        err = abs(float(e_rms) - expect)
        rows[name] = {
            "vis": [int(x) for x in vis.reshape(-1).tolist()],
            "E": float(e_rms),
            "E_expect": expect,
            "E_err": err,
            "e_lw_norm": float(e9[1].norm()),
            "e_rw_norm": float(e9[2].norm()),
        }
        if err > 1e-6:
            ok = False
        if vis_bit[1] == 0 and rows[name]["e_lw_norm"] > 1e-8:
            ok = False
        if vis_bit[2] == 0 and rows[name]["e_rw_norm"] > 1e-8:
            ok = False
    return {"modes": rows, "D_pass": ok}


def test_e() -> dict:
    src = (ROOT / "source/rsl_rl/rsl_rl/modules/intent_recovery.py").read_text()
    forbidden = ["fail_t", "future_e", "e[t+", "self_rec"]
    leaks = [w for w in forbidden if w in src]
    gate = RecoveryRiskGate()
    T = 40
    e_a = torch.zeros(T)
    e_b = torch.zeros(T)
    e_a[10:30] = 0.20
    e_b[10:30] = 0.20
    e_b[30:] = 1.0  # future differs
    s = torch.zeros(T)
    al_a, al_b = [], []
    gate.reset()
    for t in range(T):
        o = gate.step(e_a[t : t + 1], s[t : t + 1])
        al_a.append(float(o["alpha"]))
    gate.reset()
    for t in range(T):
        o = gate.step(e_b[t : t + 1], s[t : t + 1])
        al_b.append(float(o["alpha"]))
    prefix = 30
    match = max(abs(x - y) for x, y in zip(al_a[:prefix], al_b[:prefix])) < 1e-12
    # R_E >= 1 at E=0.13; persist 3 → first ON at t=12 (10,11,12)
    first_on = next(t for t, a in enumerate(al_a) if a > 0)
    at_trigger = al_a[first_on]
    return {
        "source_leaks": leaks,
        "prefix_match": match,
        "first_on_t": first_on,
        "alpha_at_trigger": at_trigger,
        "alpha_at_trigger_gt0": at_trigger > 0,
        "E_pass": match and not leaks and at_trigger > 0 and first_on == 12,
    }


def _replay_set(rows: list[dict], use_s: bool) -> dict:
    gate = RecoveryRiskGate()
    healthy = [r for r in rows if (not r["fail"]) and r["terrain"] in HEALTHY_TERRAINS]
    fails = [r for r in rows if r["fail"]]
    n_h = d_hyst = d_persist = 0
    leads_hyst = []
    leads_p3 = []
    hits_hyst = hits_p3 = 0
    for r in healthy:
        e = r["feats"]["E"]
        s = r["feats"]["abs_v_root_z"] if use_s else np.zeros_like(e)
        sl_e = e[WARMUP:]
        sl_s = s[WARMUP:]
        n_h += sl_e.size
        gate.reset()
        for t in range(sl_e.size):
            o = gate.step(
                torch.tensor([sl_e[t]], dtype=torch.float32),
                torch.tensor([sl_s[t]], dtype=torch.float32),
            )
            if bool(o["active"]):
                d_hyst += 1
        # persist-3 on R>=1, no hysteresis (matches Step 3 first-trigger duty)
        rsc, _, _ = risk_scores(
            torch.tensor(sl_e, dtype=torch.float32),
            torch.tensor(sl_s, dtype=torch.float32),
        )
        rnp = rsc.numpy()
        run = 0
        for v in rnp:
            run = run + 1 if v >= 1.0 else 0
            if run >= 3:
                d_persist += 1
    for r in fails:
        e = r["feats"]["E"]
        s = r["feats"]["abs_v_root_z"] if use_s else np.zeros_like(e)
        ft = int(r["fail_t"])
        gate.reset()
        first = None
        for t in range(0, ft + 1):
            o = gate.step(
                torch.tensor([float(e[t])], dtype=torch.float32),
                torch.tensor([float(s[t])], dtype=torch.float32),
            )
            if bool(o["active"]) and first is None and t >= WARMUP:
                first = t
        if first is not None:
            hits_hyst += 1
            leads_hyst.append((ft - first) * DT)
        rsc, _, _ = risk_scores(
            torch.tensor(e, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32),
        )
        t0 = _first_trigger(rsc.numpy(), 1.0, 3, ft)
        if t0 is not None:
            hits_p3 += 1
            leads_p3.append((ft - t0) * DT)
    def _pack(hits, leads):
        lp = _pct(np.array(leads), qs=(10, 50, 90))
        return {
            "recall": hits / len(fails) if fails else float("nan"),
            "n_fail": len(fails),
            "n_triggered": hits,
            "lead_median_s": lp["p50"],
            "lead_p10_s": lp["p10"],
        }
    return {
        "healthy_duty_hysteresis": d_hyst / n_h if n_h else float("nan"),
        "healthy_duty_persist3": d_persist / n_h if n_h else float("nan"),
        "hysteresis": _pack(hits_hyst, leads_hyst),
        "persist3": _pack(hits_p3, leads_p3),
    }


def test_f() -> dict:
    rows = []
    for t in ("plane", "light_rough", "slope", "steps"):
        rows += _load_root(MATRIX, "loco", t, "torso")
    e_only = _replay_set(rows, use_s=False)
    both = _replay_set(rows, use_s=True)
    step3 = _lead_sweep(rows, "E", persist=3)
    ref = next(x for x in step3 if x["th_cm"] == 13)
    rec_ok = abs(e_only["persist3"]["recall"] - ref["prefail_recall"]) < 1e-9
    duty_ok = abs(e_only["healthy_duty_persist3"] - ref["healthy_duty"]) < 5e-3
    lead_ok = abs(e_only["persist3"]["lead_median_s"] - ref["lead_median_s"]) < 0.05
    return {
        "step3_E13_persist3": {
            "healthy_duty": ref["healthy_duty"],
            "recall": ref["prefail_recall"],
            "lead_median_s": ref["lead_median_s"],
            "lead_p10_s": ref["lead_p10_s"],
            "n_triggered": ref["n_triggered"],
            "n_fail": ref["n_fail"],
        },
        "gate_R_E_only": e_only,
        "gate_max_RE_RS": both,
        "match_step3_persist3": rec_ok and duty_ok and lead_ok,
        "F_pass": rec_ok and duty_ok and lead_ok,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "ckpt": str(CKPT),
        "theta_max_deg": 5.0,
        "r_max": math.tan(math.radians(5.0)),
        "Q50_E": Q50_E,
        "Q90_E": Q90_E,
    }
    print("== Test A/B: load parent vs zero-init recovery ==")
    report["AB"] = test_ab()
    print(json.dumps(report["AB"], indent=2))
    print("== Test C: freeze audit ==")
    report["C"] = test_c()
    print(json.dumps(report["C"], indent=2))
    print("== Test D: mask audit ==")
    report["D"] = test_d()
    print(json.dumps(report["D"], indent=2))
    print("== Test E: gate causality ==")
    report["E"] = test_e()
    print(json.dumps(report["E"], indent=2))
    print("== Test F: Step 3 replay (no Isaac) ==")
    report["F"] = test_f()
    print(json.dumps(report["F"], indent=2, default=str))
    report["all_pass"] = all(
        report[k][f"{k}_pass"] if k != "AB" else (report["AB"]["A_pass"] and report["AB"]["B_pass"])
        for k in ("AB", "C", "D", "E", "F")
    )
    # flatten pass flags
    report["all_pass"] = bool(
        report["AB"]["A_pass"]
        and report["AB"]["B_pass"]
        and report["C"]["C_pass"]
        and report["D"]["D_pass"]
        and report["E"]["E_pass"]
        and report["F"]["F_pass"]
    )
    from rsl_rl.modules.normalizer import EmpiricalNormalization

    nrm = EmpiricalNormalization(shape=9, eps=0.0)
    nrm._mean[:] = 0
    nrm._std[:] = 0.05
    nrm._var[:] = 0.05**2
    raw = torch.tensor([[0.046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    z = (raw - nrm._mean) / (nrm._std + nrm.eps)
    back = nrm.inverse(z)
    g_ok = float((back - raw).abs().max()) < 1e-6
    report["G"] = {"G_pass": g_ok, "roundtrip_max": float((back - raw).abs().max())}
    report["all_pass"] = bool(report["all_pass"] and g_ok)
    outp = OUT / "smoke_report.json"
    outp.write_text(json.dumps(report, indent=2, default=str))
    print("ALL PASS" if report["all_pass"] else "FAIL")
    print("wrote", outp)


if __name__ == "__main__":
    main()
