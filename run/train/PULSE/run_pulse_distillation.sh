export CUDA_VISIBLE_DEVICES=5

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/rsl_rl/train.py \
    --task=PULSE-Distill-General-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs=4096 \
    --motion /home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/train \
    --teacher_checkpoint "logs/rsl_rl/g1_flat_mosaic_hybrid/2026-04-15_17-11-44_medium_5_step_1024_1024_512_512_256_256/model_55000.pt" \
    --headless \
    --logger wandb \
    --log_project_name PULSE_Distill_mdp \
    --run_name regu_0.005_prior_std_0.5_kl_0.02_latent_28\
    #--resume_student_checkpoint "logs/rsl_rl/g1_flat_pulse_distillation/2026-04-17_19-19-38_medium_5_step_pulse_regu_16_upper_1_init_std_0.5/model_12500.pt"
    

# bash run/run_pulse_distillation.sh

# sweep:
# 1. regu weight: 0.05 0.10 0.20 0.50 
# 2. fixed std: 0.10 0.05 

# 1. fixed std: from 0.2 to 0.9
# 2. latent space: (16), 20,24,28,32