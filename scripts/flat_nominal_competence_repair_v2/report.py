"""Stage reports and MASTER_REPORT. Never treat report_generation as scientific PASS."""
from __future__ import annotations

import json
from pathlib import Path

from .constants import CAMPAIGN, PARENT_A, RESULTS
from .io_util import atomic_write_json, atomic_write_text, load_status, utc_now


def write_stage_report(stage: str, payload: dict) -> Path:
    md = RESULTS / "reports" / f"{stage}_REPORT.md"
    lines = [f"# {stage} REPORT", "", f"updated: {utc_now()}", "", "```json", json.dumps(payload, indent=2, default=str), "```", ""]
    atomic_write_text(md, "\n".join(lines))
    atomic_write_json(RESULTS / "reports" / f"{stage}_REPORT.json", payload)
    return md


def generate_master(st: dict | None = None) -> Path:
    st = st or load_status()
    sci = str(st.get("scientific_gate") or "PENDING")
    p1_ok = sci == "PASS"
    claims = {
        "both_frontends_canonicalize_to_chest_hands": "SUPPORTED"
        if str(st.get("R1_CANONICALIZATION", {}).get("stage_state")) == "PASS"
        else "NOT SUPPORTED",
        "robust_real_human_head_hands_teleoperation": "NOT SUPPORTED",
        "canonical_3pt_flat_nominal_static_reach_competence": "SUPPORTED" if p1_ok else "NOT SUPPORTED",
        "height_supported": "NOT SUPPORTED",
        "variable_1_3_point": "NOT SUPPORTED",
    }
    body = f"""# MASTER_REPORT

CAMPAIGN:
{CAMPAIGN}

TERRAIN:
FLAT PLANE ONLY

EXTERNAL HUMAN INPUT MODES:
1. HEAD + LH + RH
2. CHEST + LH + RH

CANONICAL POLICY INPUT:
CHEST + LH + RH

ROBOT TARGET LINKS:
torso_link + left_wrist_yaw_link + right_wrist_yaw_link

HUMAN LOWER-BODY INPUT:
NONE

POLICY COUNT:
ONE

TASK ID:
NONE

updated: {utc_now()}
execution_status: {st.get('execution_status')}
report_generation: {st.get('report_generation')}
scientific_gate: {st.get('scientific_gate')}
blocked_at_stage: {st.get('blocked_at_stage')}
current_stage: {st.get('current_stage')}
best_checkpoint: {st.get('best_checkpoint')}
selected_gpu: {st.get('selected_gpu')}

## 1. Status semantics

`report_generation` is not a scientific result. Only `scientific_gate` answers Q4.

HEAD_MODE_INTERFACE = IMPLEMENTED
HEAD_MODE_REAL_HUMAN_CALIBRATION = NOT VALIDATED
EXTERNAL_HEAD_HANDS_INTERFACE = SUPPORTED_BY_ADAPTER (unit tests)
REAL_HUMAN_HEAD_HANDS_CONTROL = NOT VALIDATED IN THIS CAMPAIGN

H_LEGACY_ALIAS == CANONICAL_CHEST_SLOT == torso_link. Do not read H as robot head.

Parent: `{PARENT_A}`

## 2. Exact questions

Q1 Can both human input modes canonicalize to Chest+LH+RH?
Q2 Were previous P1 conclusions affected by unstable/non-fixed evaluation?
Q3 Did existing continuation checkpoints genuinely improve F1/F2, and where did they lose stability?
Q4 Can a conservative Static+Reach continuation produce a competent flat nominal 3PT policy?

## 3. Stage payloads

```json
{json.dumps({k: st.get(k) for k in st if str(k).startswith('R') or k in ('FINAL_REPORT','execution_status','scientific_gate','blocked_at_stage','best_checkpoint')}, indent=2, default=str)}
```

## 4. Supported claims

{json.dumps(claims, indent=2)}

## 5. Unsupported claims

Any claim not listed as SUPPORTED above is NOT SUPPORTED.
Do not claim height support, real-human Head+Hands teleoperation, or scientific PASS from FINAL_REPORT.
"""
    atomic_write_text(RESULTS / "reports" / "MASTER_REPORT.md", body)
    atomic_write_json(
        RESULTS / "reports" / "MASTER_REPORT.json",
        {"campaign": CAMPAIGN, "status": st, "claims": claims, "updated_at": utc_now()},
    )
    atomic_write_text(RESULTS / "MASTER_REPORT.md", body)
    return RESULTS / "reports" / "MASTER_REPORT.md"
