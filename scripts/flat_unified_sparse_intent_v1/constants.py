"""FLAT_UNIFIED_SPARSE_INTENT_V1 frozen paths and gates. Do not retune gates."""
from __future__ import annotations

import os
from pathlib import Path

ANYBODY = Path("/data/home/chenxiangyu/robotics/Anybody")
HT_ROOT = Path("/data/home/chenxiangyu/humantracker_3pt_ood")
TRITRACK = Path("/data/home/chenxiangyu/victor/TriTrack")
ISAACLAB = Path("/data/home/chenxiangyu/robotics/IsaacLab_v2.1")
PY_ISAAC = Path("/data/home/chenxiangyu/miniconda3/envs/isaaclab/bin/python")
PY_HT = Path("/data/home/chenxiangyu/miniconda3/envs/humantracker/bin/python")
X11LIB = Path("/data/home/chenxiangyu/tools/x11libs/lib")

CAMPAIGN = "FLAT_UNIFIED_SPARSE_INTENT_V1"
RESULTS = Path(
    os.environ.get(
        "FLAT_FUSI_ROOT",
        str(ANYBODY / "results" / "flat_unified_sparse_intent_v1"),
    )
)
PKG = Path(__file__).resolve().parent
TRAIN_LOG_ROOT = ANYBODY / "logs" / "rsl_rl" / "flat_unified_sparse_intent_v1"

CKPT_TRAIN_A = (
    ANYBODY
    / "logs/rsl_rl/flat_locomani_flat_continuation"
    / "2026-09-06_15-16-59_seed2026_pilot/model_51999.pt"
)
SONIC_TRAIN = ANYBODY / "datasets" / "SONIC_npzs" / "g1" / "npz_splits_loco_manip" / "train"
CLIP_LOCO = ANYBODY / "logs" / "tritrack" / "infer_clips" / "batch4_locomani"
BODY_JSON = (
    ANYBODY
    / "source/whole_body_tracking/whole_body_tracking/synth/cache/g1_body_names.json"
)
SEED_NPZ = CLIP_LOCO / "04_walk_forward.npz"
V3_DEV = Path("/tmp/flat_locomani_live_anchor_v3/data/dev")

TASK = "MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0"
FPS = 50.0
DT = 1.0 / FPS
PHYSICS_DT = 0.005
DECIMATION = 4
NUM_ENVS = 512
SEED = 2026
REPLICA_SEEDS = (2027, 2028)
PARENT_ITER = 51999
EXTRA_ITERS = 2000
P3_ITERS = 3000
ACTOR_LR = 3.0e-6
CHECKPOINT_INTERVAL = 250
MAX_INFRA_RETRY = 2
ACTION_DIM = 29
LATENT_DIM = 16

DEV_MANIFEST_RNG_SEED = 7301
TEST_MANIFEST_RNG_SEED = 9301

# Historical v3 live-anchor WORLD eval of TRAIN A (known_preview).
S_PARENT_MACRO_SR_3PT_5CM = 0.4314
S_PARENT_REPEAT_TOL = 0.02

FLAT_TERRAIN_CONSTANT = {
    "terrain_type": "plane",
    "terrain_scan_dim": 0,
    "terrain_scan_fill": 0.0,
    "note": "Parent checkpoint has terrain_scan_dim=0; no elevation tensor is fed. "
    "If a compatibility pad is present it is the constant 0.0 and never varies.",
}

PLANE_HYDRA = [
    "env.scene.terrain.terrain_type=plane",
    "env.curriculum.terrain_levels=null",
]
PRIOR_A = [
    "env.rewards.motion_body_pos.weight=0.1",
    "env.rewards.motion_body_ori.weight=0.1",
    "env.rewards.motion_body_lin_vel.weight=0.15",
    "env.rewards.motion_body_ang_vel.weight=0.15",
]
# P1: mask = [H,LH,RH] all active. HeadHands spec order: torso, head_left, head_right, vr.
P1_MASK_HYDRA = [
    "env.commands.motion.mask_mode_probs=[0.0,0.0,0.0,1.0]",
    "env.curriculum.keypoint_mask_mode=null",
]

# Gates — frozen.
P1_OVERALL = 0.55
P1_F1 = 0.70
P1_F2 = 0.60
P1_FALL_FREE = 0.95
P2_HEIGHT_MACRO = 0.60
P2_H1A = 0.60
P2_H1B = 0.50
P2_CONT_FALL_FREE = 0.95
P2_OVERALL = 0.55
P2_F1 = 0.65
P3_SR_1PT = 0.65
P3_SR_2PT = 0.60
P3_SR_3PT = 0.60
P3_HEIGHT = 0.55
P4_MACRO = 0.50
P4_C123 = 0.45
P4_C4 = 0.40
P5_OVERALL = 0.55
P5_HEIGHT = 0.55
P5_MASK_TRANS = 0.50

STAGES = (
    "P0_AUDIT",
    "P0_UNIT",
    "P0_MANIFESTS",
    "P0_PROP_CPU",
    "P0_EVAL_R1",
    "P0_EVAL_R2",
    "P0_PROP_ISAAC",
    "P0_GATE",
    "P1_SMOKE",
    "P1_NOOP",
    "P1_TRAIN",
    "P1_GATE",
    "P2_ZEROSHOT",
    "P2_TRAIN",
    "P2_GATE",
    "P3_MASK_TESTS",
    "P3_TRAIN",
    "P3_GATE",
    "P4_EVAL",
    "P4_GATE",
    "P5_EVAL",
    "P5_GATE",
    "REPLICA",
    "FINAL_REPORT",
)


def experiment_name(stage: str) -> str:
    base = "flat_unified_sparse_intent_v1"
    if os.environ.get("FLAT_FUSI_TINY"):
        return f"{base}_{stage}_smoke"
    return f"{base}_{stage}"


def extra_iters() -> int:
    if os.environ.get("FLAT_FUSI_TINY"):
        return 4
    return EXTRA_ITERS


def num_envs() -> int:
    if os.environ.get("FLAT_FUSI_TINY"):
        return 4
    return NUM_ENVS


def chunk_iters() -> int:
    if os.environ.get("FLAT_FUSI_TINY"):
        return 2
    return CHECKPOINT_INTERVAL
