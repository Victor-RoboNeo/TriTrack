"""Stage reports and MASTER_REPORT. Claims follow gates; never inflate."""
from __future__ import annotations

import json
from pathlib import Path

from .constants import CAMPAIGN, CKPT_TRAIN_A, RESULTS
from .io_util import atomic_write_json, atomic_write_text, load_status, utc_now


def write_stage_report(stage: str, payload: dict) -> Path:
    md = RESULTS / "reports" / f"{stage}_REPORT.md"
    lines = [f"# {stage} REPORT", "", f"updated: {utc_now()}", "", "```json", json.dumps(payload, indent=2, default=str), "```", ""]
    atomic_write_text(md, "\n".join(lines))
    atomic_write_json(RESULTS / "reports" / f"{stage}_REPORT.json", payload)
    return md


def generate_master(st: dict | None = None) -> Path:
    st = st or load_status()
    claims = {
        "height_changes": "NOT SUPPORTED" if st.get("P2_GATE", {}).get("stage_state") != "PASS" else "SUPPORTED",
        "variable_1_3_point": "NOT SUPPORTED" if st.get("P3_GATE", {}).get("stage_state") != "PASS" else "SUPPORTED",
        "unified_composition_and_transitions": "NOT SUPPORTED"
        if not (st.get("P4_GATE", {}).get("stage_state") == "PASS" and st.get("P5_GATE", {}).get("stage_state") == "PASS")
        else "SUPPORTED",
    }
    md = RESULTS / "reports" / "MASTER_REPORT.md"
    body = f"""# MASTER_REPORT

CAMPAIGN: {CAMPAIGN}
TERRAIN: FLAT PLANE ONLY
HUMAN INPUT: 1–3 sparse Head/LH/RH task-space targets
HUMAN LOWER-BODY INPUT: NONE
POLICY COUNT: ONE
TASK ID INPUT: NONE
ROBOT LOWER-BODY: AUTONOMOUS

updated: {utc_now()}
current_stage: {st.get('current_stage')}
stage_state: {st.get('stage_state')}
gate: {st.get('gate')}
best_checkpoint: {st.get('best_checkpoint')}
last_checkpoint: {st.get('last_checkpoint')}
selected_gpu: {st.get('selected_gpu')}

## 1. Executive Summary

This campaign is an autonomous execution of FLAT_UNIFIED_SPARSE_INTENT_V1.
Do not treat incomplete stages as scientific success.

Supported claims: {json.dumps(claims)}

## 2. Exact Scientific Question

Can one unified flat-ground humanoid policy autonomously realize whole-body motion from 1–3 sparse human task-space targets, including continuous height changes, without human lower-body supervision?

## 3. Final Controller Architecture

WORLD sparse targets → live-anchor transform every control tick → frozen Stage-2 encoder → 16-D latent → g_phi → frozen Stage-2 decoder → 29-DoF G1 action.

Parent: `{CKPT_TRAIN_A}`

## 4. Flat-Only Scope Verification

See `00_audit/RESOLVED_CONFIG.md` and `00_audit/observation_audit.md`.

## 5. Human Input Audit

HUMAN_LOWER_BODY_INPUT = NONE
ROBOT_LOWER_BODY_REALIZATION = AUTONOMOUS

## 6. Parent Checkpoint

`{CKPT_TRAIN_A}`

## 7–24. Stage payloads

See `reports/P*_REPORT.json` and `STATUS.json`.

```json
{json.dumps({k: st.get(k) for k in st if k.startswith('P') or k in ('REPLICA','FINAL_REPORT','current_stage','gate','best_checkpoint')}, indent=2, default=str)}
```

## Supported Claims

{json.dumps(claims, indent=2)}

## Unsupported Claims

Any claim not listed as SUPPORTED above is NOT SUPPORTED.

## Recommendation

Follow STATUS.json. If a gate is FAIL, stop dependent stages. Do not invent modules.
"""
    atomic_write_text(md, body)
    atomic_write_json(
        RESULTS / "reports" / "MASTER_REPORT.json",
        {"campaign": CAMPAIGN, "status": st, "claims": claims, "updated_at": utc_now()},
    )
    # convenience copies
    atomic_write_text(RESULTS / "MASTER_REPORT.md", body)
    return md
