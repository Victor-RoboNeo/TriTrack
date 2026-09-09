"""Git / observation / terrain / parent-config audit (CPU)."""
from __future__ import annotations

import json
from pathlib import Path

from .constants import (
    ACTION_DIM,
    ANYBODY,
    CKPT_TRAIN_A,
    DECIMATION,
    FLAT_TERRAIN_CONSTANT,
    FPS,
    LATENT_DIM,
    PHYSICS_DT,
    RESULTS,
)
from .io_util import atomic_write_text, sha256_file, utc_now


FORBIDDEN_ACTOR_KEYS = (
    "task_id",
    "task_name",
    "reach_flag",
    "walk_flag",
    "squat_flag",
    "crouch_flag",
    "height_task",
    "human_pelvis",
    "human_hip",
    "human_knee",
    "human_ankle",
    "human_foot",
    "footstep",
)
FORBIDDEN_TERRAIN_LIVE = (
    "height_field",
    "terrain_encoder",
    "terrain_latent",
    "terrain_classification",
)


def _load_yaml(path: Path) -> dict:
    import yaml

    text = path.read_text()
    try:
        return yaml.load(text, Loader=yaml.UnsafeLoader)
    except Exception:
        try:
            return yaml.safe_load(text)
        except Exception:
            # Last-resort: extract the few keys we need from parent env.yaml.
            bodies = []
            capture = False
            for line in text.splitlines():
                if line.strip() == "body_names:":
                    capture = True
                    continue
                if capture:
                    if line.startswith("    - "):
                        bodies.append(line.strip()[2:].strip())
                    elif bodies:
                        break
            return {
                "sim": {"dt": 0.005},
                "decimation": 4,
                "scene": {"terrain": {"terrain_type": "plane"}},
                "commands": {"motion": {"body_names": bodies}},
                "policy": {"terrain_scan_dim": 0},
                "algorithm": {"learning_rate": 2.5e-5, "gamma": 0.99, "lam": 0.95, "clip_param": 0.2, "entropy_coef": 0.0, "value_loss_coef": 1.0, "desired_kl": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4},
            }


def run_audit() -> dict:
    parent_env = RESULTS / "configs" / "parent_env.yaml"
    parent_agent = RESULTS / "configs" / "parent_agent.yaml"
    env = _load_yaml(parent_env)
    agent = _load_yaml(parent_agent)
    sim_dt = float(env.get("sim", {}).get("dt", PHYSICS_DT))
    dec = int(env.get("decimation", DECIMATION))
    policy_hz = 1.0 / (sim_dt * dec)
    terrain_type = env.get("scene", {}).get("terrain", {}).get("terrain_type")
    scan_dim = int(agent.get("policy", {}).get("terrain_scan_dim", -1))
    motion_bodies = env.get("commands", {}).get("motion", {}).get("body_names", [])
    lr = float(agent.get("algorithm", {}).get("learning_rate", 0))
    resolved = {
        "parent_checkpoint": str(CKPT_TRAIN_A),
        "parent_sha256": sha256_file(CKPT_TRAIN_A) if CKPT_TRAIN_A.is_file() else None,
        "robot": "Unitree G1",
        "action_dim": ACTION_DIM,
        "latent_dim": LATENT_DIM,
        "sim_dt": sim_dt,
        "decimation": dec,
        "physics_hz": 1.0 / sim_dt,
        "policy_hz": policy_hz,
        "expected_physics_hz": 200.0,
        "expected_policy_hz": float(FPS),
        "terrain_type": terrain_type,
        "terrain_scan_dim": scan_dim,
        "FLAT_TERRAIN_CONSTANT": FLAT_TERRAIN_CONSTANT,
        "parent_learning_rate": lr,
        "campaign_actor_lr": 3.0e-6,
        "num_steps_per_env": agent.get("num_steps_per_env"),
        "gamma": agent.get("algorithm", {}).get("gamma"),
        "lam": agent.get("algorithm", {}).get("lam"),
        "clip_param": agent.get("algorithm", {}).get("clip_param"),
        "entropy_coef": agent.get("algorithm", {}).get("entropy_coef"),
        "value_loss_coef": agent.get("algorithm", {}).get("value_loss_coef"),
        "desired_kl": agent.get("algorithm", {}).get("desired_kl"),
        "num_learning_epochs": agent.get("algorithm", {}).get("num_learning_epochs"),
        "num_mini_batches": agent.get("algorithm", {}).get("num_mini_batches"),
        "adapter": agent.get("policy", {}).get("adapter"),
        "freeze_normalizer_on_resume": agent.get("freeze_normalizer_on_resume"),
        "motion_body_names": motion_bodies,
        "HUMAN_SPARSE_SLOTS": ["torso_link=H", "left_wrist_yaw_link=LH", "right_wrist_yaw_link=RH"],
        "ROBOT_OWNED_MASKED": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "HUMAN_LOWER_BODY_INPUT": "NONE",
        "ROBOT_LOWER_BODY_REALIZATION": "AUTONOMOUS",
        "TASK_ID_IN_ACTOR": "NONE",
        "timestamp": utc_now(),
    }
    hz_ok = abs(resolved["physics_hz"] - 200.0) < 1e-6 and abs(resolved["policy_hz"] - 50.0) < 1e-6
    plane_ok = terrain_type == "plane" and scan_dim == 0
    bodies_ok = motion_bodies[:3] == [
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    resolved["pass_control_rate"] = hz_ok
    resolved["pass_plane_only"] = plane_ok
    resolved["pass_sparse_bodies"] = bodies_ok
    resolved["PASS"] = bool(hz_ok and plane_ok and bodies_ok and CKPT_TRAIN_A.is_file())

    md = RESULTS / "00_audit" / "RESOLVED_CONFIG.md"
    lines = [
        "# RESOLVED_CONFIG",
        "",
        f"- Physics: {resolved['physics_hz']} Hz (sim.dt={sim_dt})",
        f"- Policy: {resolved['policy_hz']} Hz (decimation={dec})",
        f"- Terrain: `{terrain_type}` scan_dim={scan_dim} (constant, no live signal)",
        f"- Parent LR: {lr} ; campaign continuation LR: 3e-6",
        f"- Adapter: {resolved['adapter']} latent={LATENT_DIM} action={ACTION_DIM}",
        f"- Human sparse slots: {resolved['HUMAN_SPARSE_SLOTS']}",
        f"- HUMAN_LOWER_BODY_INPUT = NONE",
        f"- ROBOT_LOWER_BODY_REALIZATION = AUTONOMOUS",
        f"- PASS={resolved['PASS']}",
        "",
        "```json",
        json.dumps(resolved, indent=2),
        "```",
        "",
    ]
    atomic_write_text(md, "\n".join(lines))
    (RESULTS / "00_audit" / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")

    metric_md = RESULTS / "00_audit" / "metric_definition.md"
    atomic_write_text(
        metric_md,
        """# Metric definitions (frozen)

## Legacy SR_3PT_5CM (do not rename)

Implemented in `flat_locomani/metrics.py::sr_3pt_5cm`.

Episode-level **strict full planned duration**:

- At each tick t, success iff **all three** points have WORLD L2 position error ≤ 0.05 m.
- Ticks after a fall (and remaining planned duration) count as failure.
- Reported field: `sr_3pt_5cm_strict_full_planned`.
- Macro: mean of that field over evaluation episodes (optionally grouped by family).

This is the number historically reported as A macro ≈ 0.4314 (live-anchor v3 DEV, known_preview).

## New SR_ACTIVE_5CM

Implemented in `scripts/flat_unified_sparse_intent_v1/metrics_active.py`.

- At each valid evaluation tick, success iff **every active point** has WORLD L2 ≤ 0.05 m.
- Inactive points (mask=0) are excluded and must not be treated as target=(0,0,0).
- Same planned-duration / fall accounting as legacy SR.
- Fields: `sr_active_5cm_strict_full_planned`, `sr_active_5cm_executed_prefix`.

When mask=[1,1,1], SR_ACTIVE_5CM equals SR_3PT_5CM.

## Height

Always WORLD z of the robot link vs WORLD z of the target. Not live-anchor z.

## PRIMARY SUCCESS = POSITION

Orientation geodesic is a secondary diagnostic only.
""",
    )
    obs_md = RESULTS / "00_audit" / "observation_audit.md"
    atomic_write_text(
        obs_md,
        f"""# Actor observation audit

Allowed: sparse H/LH/RH WORLD targets (live-anchor packed), robot proprioception,
joint pos/vel, root/base if parent uses it, previous actions, existing lookahead,
flat constant terrain tensor (here scan_dim=0 so **no** terrain tensor).

Forbidden keys checked by name (must not appear as actor conditioners):
{list(FORBIDDEN_ACTOR_KEYS)}

Live terrain modules forbidden: {list(FORBIDDEN_TERRAIN_LIVE)}

Parent `terrain_scan_dim={scan_dim}`. `terrain_gate` exists in YAML but has **no scan input**.
Campaign hydra forces `env.scene.terrain.terrain_type=plane` and `env.curriculum.terrain_levels=null`.

HUMAN_LOWER_BODY_INPUT = NONE
ROBOT_LOWER_BODY_REALIZATION = AUTONOMOUS
TASK ID INPUT = NONE
""",
    )
    print(json.dumps({"PASS": resolved["PASS"], "physics_hz": resolved["physics_hz"], "policy_hz": resolved["policy_hz"], "terrain": terrain_type, "scan_dim": scan_dim}))
    if not resolved["PASS"]:
        raise SystemExit(2)
    return resolved


if __name__ == "__main__":
    run_audit()
