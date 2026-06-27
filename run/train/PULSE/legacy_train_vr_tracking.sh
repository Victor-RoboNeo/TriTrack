# VR-Tracking G1 PPO (residual latent in env + command-manager EE mask sampling).
# Checkpoints: logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/ (e.g. model_1000.pt).
# experiment_name defaults to g1_flat_vr_tracking_residual_latent (see vr_tracking agent cfg).
export CUDA_VISIBLE_DEVICES=3,4,5

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=3 scripts/rsl_rl/train.py \
    --task=VR-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs=4096 \
    --motion /home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/train \
    --prior_checkpoint "logs/rsl_rl/g1_flat_pulse_distillation/2026-04-21_21-35-33_kl_0.01_regu_0.005_prior_std_0.5/model_10000.pt" \
    --headless \
    --logger wandb \
    --log_project_name latent_residual_vr_tracking_rl \
    --run_name actor_1024_512_256_128_critic_1024_1024_512_512_256_12_critic_warmup_200_critic_priviledged_obs \
    --warmstart_from_masked_partial_kp_tracker \
    --warmstart_checkpoint "logs/rsl_rl/g1_flat_vae_latent_distillation_2b_partial_mask/2026-05-01_11-58-04_initial_curriculum_encoder_1024_512_256_128/model_16000.pt" \
    --critic_warmup_itrs 200 \
    --vr_compact_goal_obs false \
    #--warmstart_lock_full_normalizer \

# bash run/train/train_vr_tracking.sh
