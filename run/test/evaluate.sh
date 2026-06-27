# Evaluation: visualize teacher model (one-stage GMT)
# Checkpoints under logs/rsl_rl/g1_flat_mosaic_hybrid/<timestamp>_GMT_MOSAIC_GMT/ (e.g. model_20000.pt)
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
    --num_envs=1 \
    --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
    --motion MOSAIC_Dataset/visualize \
    --load_run=2026-03-04_22-05-38_GMT_MOSAIC_GMT \
    --checkpoint=model_20000.pt \
    --headless \
    --video \
    --video_length=200 \
    --disable_motion_group_sampling
# bash run/evaluate.sh
