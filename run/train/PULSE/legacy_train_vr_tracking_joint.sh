# VR-Tracking-Joint G1 PPO (joint actions; LatentBottleneckAnyBodyActorCritic + PULSE init + latent VR actor/critic warmstart).
# Split RL obs normalizer is loaded from --latent_rl_checkpoint (not --prior_checkpoint). On resume, it loads from the resume .pt.
# Checkpoints: logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/ (e.g. model_1000.pt).
# experiment_name defaults to g1_flat_vr_tracking_joint_pulse (see vr_tracking joint agent cfg).
export CUDA_VISIBLE_DEVICES=0,1,2,3

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/rsl_rl/train.py \
    --task=VR-Tracking-Joint-Flat-G1-v0 \
    --distributed \
    --num_envs=4096 \
    --motion /home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/train \
    --prior_checkpoint "logs/rsl_rl/g1_flat_pulse_distillation/2026-04-21_21-35-33_kl_0.01_regu_0.005_prior_std_0.5/model_10000.pt" \
    --latent_rl_checkpoint "logs/rsl_rl/g1_flat_vr_tracking_residual_latent/2026-04-30_21-34-38_actor_512_256_128_critic_1024_1024_512_512_256_12_critic_warmup_1000_critic_priviledged_obs/model_20000.pt" \
    --headless \
    --logger wandb \
    --log_project_name latent_joint_vr_tracking_rl \
    --run_name joint_pulse_finetune \
    --vr_compact_goal_obs false \

# bash run/train/train_vr_tracking_joint.sh
