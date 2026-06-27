export CUDA_VISIBLE_DEVICES=3

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/rsl_rl/train.py \
    --task=PULSE-AdvDistill-General-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs=4096 \
    --motion /home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/train \
    --teacher_checkpoint "logs/rsl_rl/g1_flat_mosaic_hybrid/2026-04-15_17-11-44_medium_5_step_1024_1024_512_512_256_256/model_55000.pt" \
    --headless \
    --logger wandb \
    --log_project_name PULSE_Distill_mdp \
    --run_name regu_0.005_prior_std_0.5_kl_0.01_latent_24_decoder_no_proprio \
    #--resume_student_checkpoint ""

# bash run/run_adv_pulse_distillation.sh
