# 1st stage: MOSAIC GMT training
# Checkpoints are saved under logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/ (e.g. model_1000.pt, model_2000.pt).
# For this task experiment_name is g1_flat_mosaic_hybrid; run_name is GMT_MOSAIC_GMT. So look in logs/rsl_rl/g1_flat_mosaic_hybrid/.
#export CUDA_VISIBLE_DEVICES=
HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train.py \
    --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
    --distributed \
    --num_envs=4096 \
    --motion /home/lsn/Datasets/SONIC_npzs/g1/npz_splits_loco_manip/train \
    --headless \
    --logger wandb \
    --log_project_name MUSE_Distill \
    --run_name sonic_102k_teacher \
    --resume True \
    --load_run 2026-05-17_08-20-11_sonic_102k_teacher \
    --checkpoint model_85000.pt \

# bash run/train/GMT/run_mosaic_gmt.sh
