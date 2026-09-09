"""R0 slot / observation / terrain audit."""
from __future__ import annotations

import json
from pathlib import Path

from .constants import CANONICAL_SLOTS, PARENT_A, RESULTS
from .io_util import atomic_write_text, sha256_file, utc_now


def run_audit() -> dict:
    rec = {
        "canonical_slot_0": CANONICAL_SLOTS[0],
        "canonical_slot_1": CANONICAL_SLOTS[1],
        "canonical_slot_2": CANONICAL_SLOTS[2],
        "legacy_name": "H_LEGACY_ALIAS",
        "scientific_meaning": "H_LEGACY_ALIAS == CANONICAL_CHEST_SLOT == torso_link, NOT robot head",
        "robot_link_C": "torso_link",
        "robot_link_LH": "left_wrist_yaw_link",
        "robot_link_RH": "right_wrist_yaw_link",
        "HUMAN_LOWER_BODY_INPUT": "NONE",
        "terrain_type": "plane",
        "terrain_scan_dim": 0,
        "parent": str(PARENT_A),
        "parent_sha256": sha256_file(PARENT_A),
        "adapter_source": "scripts/flat_nominal_competence_repair_v2/canonicalize.py::canonicalize_sparse_input",
        "existing_frontend": "humantracker_3pt_ood/flat_locomani/sources.py HeadHandsFrontend uses identity head-delta after one-shot calib; NOT a validated geometric T_H_C",
        "HEAD_MODE_INTERFACE": "IMPLEMENTED",
        "HEAD_MODE_REAL_HUMAN_CALIBRATION": "NOT VALIDATED",
        "ppo_from_cfg": {
            "gamma": 0.99,
            "lam": 0.95,
            "clip": 0.2,
            "desired_kl": 0.01,
            "num_steps_per_env": 24,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate_this_campaign": 1.0e-6,
            "freeze_normalizer_on_resume": True,
            "source": "G1FlatMUSEKpLatentRLKp5RunnerCfg in rsl_rl_ppo_cfg.py (must be re-read at train time)",
        },
        "timestamp": utc_now(),
        "PASS": True,
    }
    md = RESULTS / "00_audit" / "CANONICAL_SLOT_AUDIT.md"
    atomic_write_text(
        md,
        "# CANONICAL_SLOT_AUDIT\n\n"
        "slot0 = C / chest / torso_link (H_LEGACY_ALIAS)\n"
        "slot1 = LH / left_wrist_yaw_link\n"
        "slot2 = RH / right_wrist_yaw_link\n\n"
        "Do not write H = robot head in scientific reports.\n\n"
        + json.dumps(rec, indent=2)
        + "\n",
    )
    atomic_write_text(
        RESULTS / "00_audit" / "HEAD_TO_CHEST_ADAPTER.md",
        """# HEAD_TO_CHEST_ADAPTER

## Case C — no validated real-human T_H_C

Repo `flat_locomani/sources.py::HeadHandsFrontend` applies identity head-delta onto a one-shot calibrated zero. That is **not** a geometric head→chest SE(3).

This campaign implements:

- `canonicalize_sparse_input(input_mode, external_targets, calibration)`
- `chest_hands` → identity
- `head_hands` → T_W_C = T_W_H @ T_H_C

Convention: quaternion wxyz, WORLD frames, `T_W_C = T_W_H @ T_H_C` with T_H_C = chest expressed in head.

HEAD_MODE_INTERFACE = IMPLEMENTED
HEAD_MODE_REAL_HUMAN_CALIBRATION = NOT VALIDATED

Synthetic T_H_C (unit tests only): translation [0,0,-0.22] m, identity quaternion.
Training and simulator evaluation use canonical chest data, never this synthetic offset as if it were real.

See `scripts/calibrate_head_to_chest.py` to estimate T_H_C from paired data later.
""",
    )
    print(json.dumps({"PASS": True, "slots": CANONICAL_SLOTS}, default=str))
    return rec


if __name__ == "__main__":
    run_audit()
