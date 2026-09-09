"""FLAT_NOMINAL_COMPETENCE_REPAIR_V2 frozen paths and gates."""
from __future__ import annotations

import os
from pathlib import Path

ANYBODY = Path("/data/home/chenxiangyu/robotics/Anybody")
HT_ROOT = Path("/data/home/chenxiangyu/humantracker_3pt_ood")
TRITRACK = Path("/data/home/chenxiangyu/victor/TriTrack")
PY_ISAAC = Path("/data/home/chenxiangyu/miniconda3/envs/isaaclab/bin/python")
PY_HT = Path("/data/home/chenxiangyu/miniconda3/envs/humantracker/bin/python")
X11LIB = Path("/data/home/chenxiangyu/tools/x11libs/lib")

CAMPAIGN = "FLAT_NOMINAL_COMPETENCE_REPAIR_V2"
RESULTS = Path(
    os.environ.get(
        "FLAT_NCR_ROOT",
        str(ANYBODY / "results" / "flat_nominal_competence_repair_v2"),
    )
)
PKG = Path(__file__).resolve().parent

PARENT_A = (
    ANYBODY
    / "logs/rsl_rl/flat_locomani_flat_continuation"
    / "2026-09-06_15-16-59_seed2026_pilot/model_51999.pt"
)
EXISTING = {
    "PARENT": PARENT_A,
    "c01": ANYBODY
    / "logs/rsl_rl/flat_unified_sparse_intent_v1_P1_TRAIN/2026-09-09_03-51-47_seed2026_p1_c01/model_52248.pt",
    "c02": ANYBODY
    / "logs/rsl_rl/flat_unified_sparse_intent_v1_P1_TRAIN/2026-09-09_04-28-39_seed2026_p1_c02/model_52497.pt",
    "c03": ANYBODY
    / "logs/rsl_rl/flat_unified_sparse_intent_v1_P1_TRAIN/2026-09-09_05-04-37_seed2026_p1_c03/model_52746.pt",
    "c04": ANYBODY
    / "logs/rsl_rl/flat_unified_sparse_intent_v1_P1_TRAIN/2026-09-09_05-40-35_seed2026_p1_c04/model_52995.pt",
}

SONIC_TRAIN = ANYBODY / "datasets" / "SONIC_npzs" / "g1" / "npz_splits_loco_manip" / "train"
CLIP_LOCO = ANYBODY / "logs" / "tritrack" / "infer_clips" / "batch4_locomani"
SEED_NPZ = CLIP_LOCO / "04_walk_forward.npz"
BODY_JSON = (
    ANYBODY
    / "source/whole_body_tracking/whole_body_tracking/synth/cache/g1_body_names.json"
)

TASK = "MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0"
FPS = 50.0
NUM_ENVS = 512
SEED = 2026
PARENT_ITER = 51999
ACTOR_LR = 1.0e-6
EXTRA_ITERS = 1500
CHUNK_ITERS = 100
MAX_INFRA_RETRY = 2
DEV_N = 256
TEST_N = 512
DEV_SEED = 8301
TEST_SEED = 10301
TRAIN_MIX_SEED = 2026

CANONICAL_SLOTS = {
    0: {"name": "C", "legacy_alias": "H_LEGACY_ALIAS", "meaning": "chest/torso", "link": "torso_link"},
    1: {"name": "LH", "legacy_alias": None, "meaning": "left hand/wrist", "link": "left_wrist_yaw_link"},
    2: {"name": "RH", "legacy_alias": None, "meaning": "right hand/wrist", "link": "right_wrist_yaw_link"},
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
P1_MASK_HYDRA = [
    "env.commands.motion.mask_mode_probs=[0.0,0.0,0.0,1.0]",
    "env.curriculum.keypoint_mask_mode=null",
]

GATE_F1 = 0.70
GATE_F2 = 0.60
GATE_FALL = 0.95
GATE_MACRO = 0.65

STAGES = (
    "R0_AUDIT",
    "R1_CANONICALIZATION",
    "R2_MANIFEST",
    "R3_PAIRED_REEVAL",
    "R4_POLICY_DRIFT",
    "R5_CLEAN_P1_TRAIN",
    "R6_CLEAN_P1_TEST",
    "R7_HEIGHT_PREVIEW",
    "FINAL_REPORT",
)


def experiment_name(stage: str) -> str:
    return f"flat_nominal_competence_repair_v2_{stage}"
