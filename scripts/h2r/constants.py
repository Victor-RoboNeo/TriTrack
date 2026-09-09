"""H2R-0 paths and G1 KP indices. Frozen Parent / Mapper-B. No SMPL."""

from pathlib import Path

ROOT = Path("/data/home/chenxiangyu/robotics/Anybody")
BONES = ROOT / "datasets/bones-seed"
SOMA_ROOT = BONES / "soma_uniform"
META_CSV = BONES / "metadata/seed_metadata_v004.csv"
G1_NPZ_ROOT = ROOT / "datasets/SONIC_npzs/g1/npz_splits_loco_manip/train"
P1_ROOT = ROOT / "logs/tritrack/infer_clips/p1"
OUT = ROOT / "results/h2r_three_point_adapter"

PARENT_CKPT = (
    ROOT / "logs/rsl_rl/g1_flat_muse_kp_latent_rl/"
    "2026-08-26_00-38-10_tritrack_headhands_locomani_from35000/model_50000.pt"
)
MAPPER_B = Path("/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt")
TASK = "MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0"

# 30-body Isaac npz layout (TriTrack replay_npz.BODY_IDX)
G1_BODY_IDX = {
    "pelvis": 0,
    "torso_link": 9,
    "left_ankle_roll_link": 18,
    "right_ankle_roll_link": 19,
    "left_wrist_yaw_link": 28,
    "right_wrist_yaw_link": 29,
}
TORSO_I = G1_BODY_IDX["torso_link"]
LW_I = G1_BODY_IDX["left_wrist_yaw_link"]
RW_I = G1_BODY_IDX["right_wrist_yaw_link"]

SOMA_JOINTS = ("Head", "LeftHand", "RightHand")
OUT_HZ = 50.0
CALIB_S = 1.5
DUR_TOL = 0.08
MIN_FRAMES = 40
SCALE_SWEEP = (0.75, 0.90, 1.00, 1.10, 1.25, 1.30)
SCALE_SWEEP_EVAL = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30)

TASKS = ("loco", "stoop", "reach", "carry")
TASK_MASK = {
    "loco": ("torso", "--no_s_enabled"),
    "stoop": ("vr", "--s_enabled"),
    "reach": ("head_right", "--s_enabled"),
    "carry": ("vr", "--s_enabled"),
}
