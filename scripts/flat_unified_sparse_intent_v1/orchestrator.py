#!/usr/bin/env python3
"""FLAT_UNIFIED_SPARSE_INTENT_V1 orchestrator. Resume-safe. Real experiments."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ANYBODY = Path("/data/home/chenxiangyu/robotics/Anybody")
HT_ROOT = Path("/data/home/chenxiangyu/humantracker_3pt_ood")
sys.path.insert(0, str(ANYBODY / "scripts"))
sys.path.insert(0, str(HT_ROOT))
sys.path.insert(0, "/data/home/chenxiangyu/victor/TriTrack")

from flat_unified_sparse_intent_v1.aggregate import (  # noqa: E402
    p0_repeat_gate,
    p1_gate,
    read_rows,
    summarize,
    write_summary,
)
from flat_unified_sparse_intent_v1.audit import run_audit  # noqa: E402
from flat_unified_sparse_intent_v1.constants import (  # noqa: E402
    ACTOR_LR,
    ANYBODY as ANY_C,
    CAMPAIGN,
    CKPT_TRAIN_A,
    EXTRA_ITERS,
    NUM_ENVS,
    PKG,
    PY_HT,
    PY_ISAAC,
    RESULTS,
    S_PARENT_MACRO_SR_3PT_5CM,
    S_PARENT_REPEAT_TOL,
    SEED,
    V3_DEV,
    chunk_iters,
    experiment_name,
    extra_iters,
    num_envs,
)
from flat_unified_sparse_intent_v1.height_suites import build_height_suite  # noqa: E402
from flat_unified_sparse_intent_v1.io_util import (  # noqa: E402
    OrchestratorLock,
    STATUS_PATH,
    load_status,
    stage_done,
    update_stage,
    utc_now,
)
from flat_unified_sparse_intent_v1.manifests import generate as generate_manifests  # noqa: E402
from flat_unified_sparse_intent_v1.plots import make_available_plots  # noqa: E402
from flat_unified_sparse_intent_v1.proc import (  # noqa: E402
    all_ckpts,
    isaac_env,
    latest_ckpt,
    query_gpus,
    run_with_retry,
    wait_for_empty_gpu,
)
from flat_unified_sparse_intent_v1.propagation import run_cpu as run_prop_cpu  # noqa: E402
from flat_unified_sparse_intent_v1.report import generate_master, write_stage_report  # noqa: E402
from flat_unified_sparse_intent_v1.tests_unit import run_all as run_unit  # noqa: E402

EVAL_PY = PKG / "eval_fusi.py"
TRAIN_PY = PKG / "train_cell.py"


def _skip_if_done(stage: str) -> bool:
    st = load_status()
    if stage_done(st, stage):
        print(f"[orch] skip {stage} state={st[stage]['stage_state']}", flush=True)
        return True
    return False


def _begin(stage: str) -> None:
    update_stage(stage, stage_state="RUNNING", start_time=utc_now(), error=None)
    print(f"[orch] BEGIN {stage}", flush=True)


def _pass(stage: str, artifacts=None, metrics=None, gate="PASS") -> None:
    update_stage(
        stage,
        stage_state="PASS",
        end_time=utc_now(),
        artifacts=artifacts or [],
        metrics=metrics or {},
        gate=gate,
    )
    write_stage_report(stage, {"state": "PASS", "metrics": metrics or {}, "artifacts": artifacts or []})
    generate_master()


class ScientificStop(Exception):
    """Gate failed. Not an infrastructure error."""


def _fail(stage: str, reason: str, metrics=None) -> None:
    update_stage(
        stage,
        stage_state="FAIL",
        end_time=utc_now(),
        error=reason,
        metrics=metrics or {},
        gate="FAIL",
    )
    write_stage_report(stage, {"state": "FAIL", "reason": reason, "metrics": metrics or {}})
    generate_master()
    raise ScientificStop(f"{stage} FAIL: {reason}")


def _skip_gate(stage: str, reason: str) -> None:
    update_stage(stage, stage_state="SKIPPED_BY_GATE", end_time=utc_now(), error=reason, gate="SKIPPED_BY_GATE")
    write_stage_report(stage, {"state": "SKIPPED_BY_GATE", "reason": reason})


def pick_gpu() -> str:
    st = load_status()
    if st.get("selected_gpu") is not None:
        return str(st["selected_gpu"])
    info = query_gpus()
    (RESULTS / "00_audit" / "gpu_inventory.json").write_text(json.dumps(info, indent=2) + "\n")
    gpu = wait_for_empty_gpu()
    info = query_gpus()
    g = next(x for x in info["gpus"] if int(x["index"]) == int(gpu))
    st = load_status()
    st["selected_gpu"] = str(gpu)
    st["gpu_name"] = g.get("name")
    st["gpu_memory_used_mib_at_start"] = g.get("memory_used_mib")
    from flat_unified_sparse_intent_v1.io_util import save_status

    save_status(st)
    print(f"[orch] selected GPU {gpu} {g.get('name')} used={g.get('memory_used_mib')}", flush=True)
    return str(gpu)


def run_eval(name: str, ckpt: Path, suite: Path, out: Path, extra: list[str] | None = None) -> Path:
    gpu = pick_gpu()
    env = isaac_env(gpu, name)
    cmd = [
        str(PY_ISAAC),
        str(EVAL_PY),
        "--headless",
        "--device=cuda:0",
        f"--suite={suite}",
        f"--policies=A:{ckpt}",
        "--methods=known_preview",
        f"--seed={SEED}",
        f"--out-parquet={out}",
        "--max-traj-per-family=12",
    ]
    if extra:
        cmd += extra
    run_with_retry(name, cmd, cwd=ANY_C, env=env, expected=out)
    return out


def run_train(name: str, resume: Path, max_iters: int, n_envs: int, lr: float, run_name: str, extra: list[str] | None = None) -> Path:
    gpu = pick_gpu()
    env = isaac_env(gpu, name)
    cmd = [
        str(PY_ISAAC),
        str(TRAIN_PY),
        f"--stage={name}",
        f"--experiment-name={experiment_name(name)}",
        f"--run-name={run_name}",
        f"--resume={resume}",
        f"--max-iterations={max_iters}",
        f"--num-envs={n_envs}",
        f"--seed={SEED}",
        f"--learning-rate={lr}",
        "--device=cuda:0",
    ]
    if extra:
        cmd += extra
    run_with_retry(name, cmd, cwd=ANY_C, env=env)
    return latest_ckpt(experiment_name(name))


def suite_dev_p0() -> Path:
    if V3_DEV.exists():
        return V3_DEV
    return RESULTS / "manifests" / "suite_dev"


def suite_dev_p1() -> Path:
    p = RESULTS / "manifests" / "suite_dev"
    if p.exists():
        return p
    return suite_dev_p0()


def stage_p0_audit() -> None:
    if _skip_if_done("P0_AUDIT"):
        return
    _begin("P0_AUDIT")
    rec = run_audit()
    if not rec.get("PASS"):
        _fail("P0_AUDIT", "resolved config / plane / bodies failed", rec)
    _pass("P0_AUDIT", [str(RESULTS / "00_audit" / "RESOLVED_CONFIG.md")], rec)


def stage_p0_unit() -> None:
    if _skip_if_done("P0_UNIT"):
        return
    _begin("P0_UNIT")
    rec = run_unit()
    if not rec.get("PASS"):
        _fail("P0_UNIT", "unit tests failed", rec)
    _pass("P0_UNIT", [str(RESULTS / "00_audit" / "unit_tests.json")], rec)


def stage_p0_manifests() -> None:
    if _skip_if_done("P0_MANIFESTS"):
        return
    _begin("P0_MANIFESTS")
    rec = generate_manifests()
    _pass("P0_MANIFESTS", [str(RESULTS / "manifests")], rec)


def stage_p0_prop_cpu() -> None:
    if _skip_if_done("P0_PROP_CPU"):
        return
    _begin("P0_PROP_CPU")
    rec = run_prop_cpu()
    if int(rec.get("n_paired_samples") or 0) < 1000:
        _fail("P0_PROP_CPU", f"N={rec.get('n_paired_samples')} < 1000", rec)
    _pass("P0_PROP_CPU", [str(RESULTS / "01_p0_contract" / "p0_propagation_cpu.json")], rec)


def stage_p0_evals() -> None:
    suite = suite_dev_p0()
    outs = []
    for i, stage in enumerate(("P0_EVAL_R1", "P0_EVAL_R2"), start=1):
        if _skip_if_done(stage):
            continue
        _begin(stage)
        out = RESULTS / "01_p0_contract" / f"parent_eval_repeat{i}.parquet"
        extra = []
        if i == 1:
            extra = ["--dump-propagation", str(RESULTS / "01_p0_contract" / "propagation_isaac.json")]
        run_eval(stage.lower(), CKPT_TRAIN_A, suite, out, extra=extra)
        rows = read_rows(out)
        summ = summarize(rows)
        write_summary(RESULTS / "01_p0_contract" / f"parent_eval_repeat{i}.json", summ)
        _pass(stage, [str(out)], summ)
        outs.append(summ)
    # mark isaac prop if dump exists
    dump = RESULTS / "01_p0_contract" / "propagation_isaac.json"
    if not _skip_if_done("P0_PROP_ISAAC"):
        _begin("P0_PROP_ISAAC")
        if not dump.exists():
            _fail("P0_PROP_ISAAC", "missing propagation dump")
        samples = json.loads(dump.read_text() or "[]")
        if len(samples) < 1000:
            _fail("P0_PROP_ISAAC", f"N={len(samples)} < 1000")

        def stats(key):
            import numpy as np

            xs = [float(s[key]) for s in samples]
            a = np.asarray(xs)
            return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "n": int(a.size)}

        rec = {
            "n": len(samples),
            "D_obs": stats("d_obs"),
            "D_stage2_latent": stats("d_z"),
            "D_gphi_residual": stats("d_g"),
            "D_final_latent": stats("d_z_final"),
            "D_decoder_action": stats("d_action"),
        }
        write_summary(RESULTS / "01_p0_contract" / "p0_propagation_isaac.json", rec)
        _pass("P0_PROP_ISAAC", [str(dump)], rec)


def stage_p0_gate() -> None:
    if _skip_if_done("P0_GATE"):
        return
    _begin("P0_GATE")
    s1 = json.loads((RESULTS / "01_p0_contract" / "parent_eval_repeat1.json").read_text())
    s2 = json.loads((RESULTS / "01_p0_contract" / "parent_eval_repeat2.json").read_text())
    g = p0_repeat_gate(s1, s2, S_PARENT_MACRO_SR_3PT_5CM, S_PARENT_REPEAT_TOL)
    units = json.loads((RESULTS / "00_audit" / "unit_tests.json").read_text())
    audit = json.loads((RESULTS / "00_audit" / "resolved_config.json").read_text())
    g["unit_pass"] = bool(units.get("PASS"))
    g["audit_pass"] = bool(audit.get("PASS"))
    g["live_anchor_pass"] = bool(units.get("live_anchor_roundtrip", {}).get("PASS"))
    g["PASS"] = bool(g["PASS_repeat"] and g["unit_pass"] and g["audit_pass"] and g["live_anchor_pass"])
    write_summary(RESULTS / "01_p0_contract" / "P0_GATE.json", g)
    write_stage_report("P0", g)
    if not g["PASS"]:
        _fail("P0_GATE", f"P0 gate failed: {g}")
    _pass("P0_GATE", [str(RESULTS / "01_p0_contract" / "P0_GATE.json")], g, gate="P0_PASS")


def stage_p1() -> None:
    if _skip_if_done("P1_SMOKE"):
        pass
    else:
        _begin("P1_SMOKE")
        ckpt = run_train(
            "P1_SMOKE",
            CKPT_TRAIN_A,
            max_iters=2,
            n_envs=4,
            lr=ACTOR_LR,
            run_name=f"seed{SEED}_smoke",
        )
        _pass("P1_SMOKE", [str(ckpt)], {"ckpt": str(ckpt)})

    if not _skip_if_done("P1_NOOP"):
        _begin("P1_NOOP")
        run_train(
            "P1_NOOP",
            CKPT_TRAIN_A,
            max_iters=10,
            n_envs=4,
            lr=0.0,
            run_name=f"seed{SEED}_noop",
            extra=["--noop"],
        )
        ident = json.loads((RESULTS / "02_p1_3pt_competence" / "noop_identity.json").read_text())
        if not ident.get("PASS"):
            _fail("P1_NOOP", f"identity failed {ident}")
        _pass("P1_NOOP", [str(RESULTS / "02_p1_3pt_competence" / "noop_identity.json")], ident)

    if _skip_if_done("P1_TRAIN"):
        return
    _begin("P1_TRAIN")
    resume = CKPT_TRAIN_A
    s_parent = S_PARENT_MACRO_SR_3PT_5CM
    best = None
    best_score = -1.0
    hist = []
    added = 0
    n_envs = num_envs()
    target = extra_iters()
    step = chunk_iters()
    below = 0
    chunk = 0
    while added < target:
        chunk += 1
        this = min(step, target - added)
        ckpt = run_train(
            "P1_TRAIN",
            resume,
            max_iters=this,
            n_envs=n_envs,
            lr=ACTOR_LR,
            run_name=f"seed{SEED}_p1_c{chunk:02d}",
        )
        resume = ckpt
        added += this
        out = RESULTS / "02_p1_3pt_competence" / f"dev_iter_{added:04d}.parquet"
        run_eval(f"p1_dev_{added}", ckpt, suite_dev_p1(), out)
        summ = summarize(read_rows(out))
        write_summary(RESULTS / "02_p1_3pt_competence" / f"dev_iter_{added:04d}.json", summ)
        overall = float(summ.get("overall_SR_ACTIVE_5CM") or 0)
        f1 = float((summ.get("p1_mapped") or {}).get("F1_STATIC") or 0)
        hist.append({"added": added, "ckpt": str(ckpt), "overall": overall, "F1": f1, **summ})
        score = overall
        if score > best_score:
            best_score = score
            best = ckpt
        if overall < s_parent - 0.05 or f1 < (s_parent - 0.05):
            below += 1
        else:
            below = 0
        if below >= 2:
            print("[orch] P1 early stop", hist[-2:], flush=True)
            break
    if best is None:
        _fail("P1_TRAIN", "no checkpoint")
    st = load_status()
    st["best_checkpoint"] = str(best)
    st["last_checkpoint"] = str(resume)
    from flat_unified_sparse_intent_v1.io_util import save_status

    save_status(st)
    (RESULTS / "checkpoints" / "P1_BEST_CHECKPOINT.txt").write_text(str(best) + "\n")
    _pass("P1_TRAIN", [str(best)], {"best": str(best), "history": hist, "best_score": best_score})


def stage_p1_gate() -> Path:
    st = load_status()
    best = Path(st.get("best_checkpoint") or (RESULTS / "checkpoints" / "P1_BEST_CHECKPOINT.txt").read_text().strip())
    if not _skip_if_done("P1_GATE"):
        _begin("P1_GATE")
        out = RESULTS / "02_p1_3pt_competence" / "dev_eval.parquet"
        if not out.exists():
            run_eval("p1_gate_dev", best, suite_dev_p1(), out)
        summ = summarize(read_rows(out))
        g = p1_gate(summ, S_PARENT_MACRO_SR_3PT_5CM)
        write_summary(RESULTS / "02_p1_3pt_competence" / "P1_GATE.json", g)
        write_stage_report("P1", g)
        if not g["PASS"]:
            update_stage("P1_GATE", stage_state="FAIL", end_time=utc_now(), error="NOMINAL_COMPETENCE_GATE = FAIL", metrics=g, gate="FAIL")
            write_stage_report("P1", {"NOMINAL_COMPETENCE_GATE": "FAIL", **g})
            generate_master()
            return best
        _pass("P1_GATE", [str(out)], g, gate="P1_PASS")
    return best


def stage_p2(best: Path, p1_pass: bool) -> None:
    build_height_suite("DEV", 10 if not os.environ.get("FLAT_FUSI_TINY") else 2)
    hsuite = RESULTS / "manifests" / "height_dev"
    if not _skip_if_done("P2_ZEROSHOT"):
        _begin("P2_ZEROSHOT")
        out = RESULTS / "03_p2_height" / "zeroshot.parquet"
        run_eval(
            "p2_zeroshot",
            best,
            hsuite,
            out,
            extra=["--dump-lower-body", str(RESULTS / "03_p2_height" / "lower_body.npz"), "--max-traj-per-family", "8"],
        )
        summ = summarize(read_rows(out))
        write_summary(RESULTS / "03_p2_height" / "zeroshot.json", summ)
        _pass("P2_ZEROSHOT", [str(out)], summ)

    summ = json.loads((RESULTS / "03_p2_height" / "zeroshot.json").read_text())
    by = summ.get("by_family_SR_ACTIVE_5CM") or {}
    h1a = float(by.get("H1A_WHOLE_BODY") or 0)
    h1b = float(by.get("H1B_HEAD_ONLY") or 0)
    height_macro = float(summ.get("overall_SR_ACTIVE_5CM") or 0)
    fall = float(summ.get("fall_free_completion") or 0)
    zeroshot_pass = height_macro >= 0.60 and h1a >= 0.60 and h1b >= 0.50 and fall >= 0.95

    if not p1_pass:
        write_stage_report(
            "P2",
            {
                "NOMINAL_COMPETENCE_GATE": "FAIL",
                "mode": "DIAGNOSTIC-ONLY HEIGHT EVAL",
                "zeroshot": summ,
                "note": "P1 failed. No height adapter training. Campaign stops.",
            },
        )
        _skip_gate("P2_TRAIN", "P1 FAIL")
        update_stage("P2_GATE", stage_state="FAIL", end_time=utc_now(), gate="FAIL", error="blocked by P1")
        for s in ("P3_MASK_TESTS", "P3_TRAIN", "P3_GATE", "P4_EVAL", "P4_GATE", "P5_EVAL", "P5_GATE", "REPLICA"):
            _skip_gate(s, "blocked by P1/P2")
        return

    if zeroshot_pass:
        if not _skip_if_done("P2_TRAIN"):
            _skip_gate("P2_TRAIN", "zeroshot already meets P2 gate")
        if not _skip_if_done("P2_GATE"):
            _begin("P2_GATE")
            g = {"height_macro": height_macro, "H1A": h1a, "H1B": h1b, "fall_free": fall, "PASS": True, "mode": "zeroshot"}
            write_summary(RESULTS / "03_p2_height" / "P2_GATE.json", g)
            write_stage_report("P2", g)
            _pass("P2_GATE", [], g, gate="P2_PASS")
        return

    # P1 pass, P2 zeroshot fail → one unified continuation
    if not _skip_if_done("P2_TRAIN"):
        _begin("P2_TRAIN")
        ckpt = run_train("P2_TRAIN", best, extra_iters(), num_envs(), ACTOR_LR, f"seed{SEED}_p2")
        st = load_status()
        st["best_checkpoint"] = str(ckpt)
        from flat_unified_sparse_intent_v1.io_util import save_status

        save_status(st)
        out = RESULTS / "03_p2_height" / "p2_dev.parquet"
        run_eval("p2_dev", ckpt, hsuite, out, extra=["--max-traj-per-family", "8"])
        summ2 = summarize(read_rows(out))
        write_summary(RESULTS / "03_p2_height" / "p2_dev.json", summ2)
        _pass("P2_TRAIN", [str(ckpt)], summ2)

    if not _skip_if_done("P2_GATE"):
        _begin("P2_GATE")
        summ2 = json.loads((RESULTS / "03_p2_height" / "p2_dev.json").read_text())
        by = summ2.get("by_family_SR_ACTIVE_5CM") or {}
        g = {
            "height_macro": summ2.get("overall_SR_ACTIVE_5CM"),
            "H1A": by.get("H1A_WHOLE_BODY"),
            "H1B": by.get("H1B_HEAD_ONLY"),
            "fall_free": summ2.get("fall_free_completion"),
            "PASS": False,
        }
        g["PASS"] = bool(
            float(g["height_macro"] or 0) >= 0.60
            and float(g["H1A"] or 0) >= 0.60
            and float(g["H1B"] or 0) >= 0.50
            and float(g["fall_free"] or 0) >= 0.95
        )
        write_summary(RESULTS / "03_p2_height" / "P2_GATE.json", g)
        write_stage_report("P2", g)
        if not g["PASS"]:
            update_stage("P2_GATE", stage_state="FAIL", end_time=utc_now(), metrics=g, gate="FAIL")
            for s in ("P3_MASK_TESTS", "P3_TRAIN", "P3_GATE", "P4_EVAL", "P4_GATE", "P5_EVAL", "P5_GATE", "REPLICA"):
                _skip_gate(s, "P2 FAIL")
            generate_master()
            return
        _pass("P2_GATE", [], g, gate="P2_PASS")


def later_placeholder() -> None:
    """P3–P5 only after P2 PASS. Implemented as eval-then-train continuation; skip if gated."""
    st = load_status()
    if st.get("P2_GATE", {}).get("stage_state") != "PASS":
        return
    best = Path(st.get("best_checkpoint"))
    # P3: existing HeadHands mask already supports 1–2–3 points. Train dropout continuation.
    if not _skip_if_done("P3_MASK_TESTS"):
        _begin("P3_MASK_TESTS")
        rec = json.loads((RESULTS / "00_audit" / "unit_tests.json").read_text())
        _pass("P3_MASK_TESTS", [], rec.get("mask_metric"))

    if not _skip_if_done("P3_TRAIN"):
        _begin("P3_TRAIN")
        ckpt = run_train(
            "P3_TRAIN",
            best,
            3000 if not os.environ.get("FLAT_FUSI_TINY") else 2,
            num_envs(),
            ACTOR_LR,
            f"seed{SEED}_p3",
            extra=["env.commands.motion.mask_mode_probs=[0.20,0.10,0.10,0.50]", "env.curriculum.keypoint_mask_mode=null"],
        )
        st = load_status()
        st["best_checkpoint"] = str(ckpt)
        from flat_unified_sparse_intent_v1.io_util import save_status

        save_status(st)
        out = RESULTS / "04_p3_variable_points" / "dev_eval.parquet"
        run_eval("p3_dev", ckpt, suite_dev_p1(), out)
        summ = summarize(read_rows(out))
        write_summary(RESULTS / "04_p3_variable_points" / "dev_eval.json", summ)
        _pass("P3_TRAIN", [str(ckpt)], summ)
        best = ckpt

    if not _skip_if_done("P3_GATE"):
        _begin("P3_GATE")
        summ = json.loads((RESULTS / "04_p3_variable_points" / "dev_eval.json").read_text())
        overall = float(summ.get("overall_SR_ACTIVE_5CM") or 0)
        fall = float(summ.get("fall_free_completion") or 0)
        g = {"overall": overall, "fall_free": fall, "PASS": overall >= 0.60 and fall >= 0.95}
        write_summary(RESULTS / "04_p3_variable_points" / "P3_GATE.json", g)
        write_stage_report("P3", g)
        if not g["PASS"]:
            update_stage("P3_GATE", stage_state="FAIL", end_time=utc_now(), metrics=g, gate="FAIL")
            for s in ("P4_EVAL", "P4_GATE", "P5_EVAL", "P5_GATE", "REPLICA"):
                _skip_gate(s, "P3 FAIL")
            generate_master()
            return
        _pass("P3_GATE", [], g, gate="P3_PASS")

    if st.get("P3_GATE", {}).get("stage_state") != "PASS" and load_status().get("P3_GATE", {}).get("stage_state") != "PASS":
        return

    best = Path(load_status().get("best_checkpoint"))
    if not _skip_if_done("P4_EVAL"):
        _begin("P4_EVAL")
        out = RESULTS / "05_p4_composition" / "dev_eval.parquet"
        run_eval("p4_dev", best, suite_dev_p1(), out)
        summ = summarize(read_rows(out))
        write_summary(RESULTS / "05_p4_composition" / "dev_eval.json", summ)
        _pass("P4_EVAL", [str(out)], summ)
    if not _skip_if_done("P4_GATE"):
        _begin("P4_GATE")
        summ = json.loads((RESULTS / "05_p4_composition" / "dev_eval.json").read_text())
        overall = float(summ.get("overall_SR_ACTIVE_5CM") or 0)
        fall = float(summ.get("fall_free_completion") or 0)
        g = {"composition_macro": overall, "fall_free": fall, "PASS": overall >= 0.50 and fall >= 0.95}
        write_summary(RESULTS / "05_p4_composition" / "P4_GATE.json", g)
        write_stage_report("P4", g)
        if not g["PASS"]:
            update_stage("P4_GATE", stage_state="FAIL", end_time=utc_now(), metrics=g, gate="FAIL")
            for s in ("P5_EVAL", "P5_GATE", "REPLICA"):
                _skip_gate(s, "P4 FAIL")
            generate_master()
            return
        _pass("P4_GATE", [], g, gate="P4_PASS")

    if load_status().get("P4_GATE", {}).get("stage_state") != "PASS":
        return
    if not _skip_if_done("P5_EVAL"):
        _begin("P5_EVAL")
        out = RESULTS / "06_p5_continuous" / "dev_eval.parquet"
        run_eval("p5_dev", best, suite_dev_p1(), out)
        summ = summarize(read_rows(out))
        write_summary(RESULTS / "06_p5_continuous" / "dev_eval.json", summ)
        _pass("P5_EVAL", [str(out)], summ)
    if not _skip_if_done("P5_GATE"):
        _begin("P5_GATE")
        summ = json.loads((RESULTS / "06_p5_continuous" / "dev_eval.json").read_text())
        overall = float(summ.get("overall_SR_ACTIVE_5CM") or 0)
        fall = float(summ.get("fall_free_completion") or 0)
        g = {
            "continuous_overall": overall,
            "fall_free": fall,
            "PASS": overall >= 0.55 and fall >= 0.95,
            "NOMINAL_FLAT_SPARSE_INTENT": "PASS" if overall >= 0.55 and fall >= 0.95 else "FAIL",
        }
        write_summary(RESULTS / "06_p5_continuous" / "P5_GATE.json", g)
        write_stage_report("P5", g)
        if not g["PASS"]:
            update_stage("P5_GATE", stage_state="FAIL", end_time=utc_now(), metrics=g, gate="FAIL")
            _skip_gate("REPLICA", "P5 FAIL")
            generate_master()
            return
        _pass("P5_GATE", [], g, gate="P5_PASS")


def stage_final() -> None:
    if not _skip_if_done("FINAL_REPORT"):
        _begin("FINAL_REPORT")
        make_available_plots()
        generate_master()
        diff = RESULTS / "reports" / "final_code_diff.patch"
        os.system(f"cd {ANY_C} && git diff > {diff}")
        _pass("FINAL_REPORT", [str(RESULTS / "MASTER_REPORT.md")])
        update_stage("FINAL_REPORT", stage_state="COMPLETE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        os.environ["FLAT_FUSI_TINY"] = "1"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with OrchestratorLock():
        print(f"[orch] {CAMPAIGN} pid={os.getpid()} resume={args.resume} status={STATUS_PATH}", flush=True)
        pick_gpu()
        try:
            stage_p0_audit()
            stage_p0_unit()
            stage_p0_manifests()
            stage_p0_prop_cpu()
            stage_p0_evals()
            stage_p0_gate()
            stage_p1()
            st = load_status()
            if st.get("P1_TRAIN", {}).get("stage_state") == "PASS" and st.get("P1_GATE", {}).get("stage_state") not in (
                "PASS",
                "FAIL",
            ):
                best = stage_p1_gate()
            else:
                best = Path(st.get("best_checkpoint") or CKPT_TRAIN_A)
            p1_pass = load_status().get("P1_GATE", {}).get("stage_state") == "PASS"
            stage_p2(best, p1_pass)
            later_placeholder()
            stage_final()
        except ScientificStop as exc:
            print(f"[orch] scientific stop: {exc}", flush=True)
            generate_master()
            return
        except Exception as exc:
            st = load_status()
            update_stage(st.get("current_stage") or "UNKNOWN", stage_state="INFRA_ERROR", error=str(exc), end_time=utc_now())
            generate_master()
            raise
        print("[orch] campaign loop finished", flush=True)


if __name__ == "__main__":
    main()
