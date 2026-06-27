# Synthetic torso-only motions for partial-keypoint trackers

Tests a partial-KP tracker (KP5/KP6) on **out-of-distribution torso commands** —
"if I move only the torso downward, does the policy squat? horizontally, does
it walk?" — by generating motion `.npz` files in which only `torso_link` has a
non-NaN trajectory after frame 0.

The output drops straight into `play.py` and any motion-dir-based PPO /
distillation training **with no code changes** — the npz schema matches SONIC
clips exactly. So this same pipeline supports:

1. **Zero-shot probing**: visualize whether a checkpoint already follows synthetic
   torso commands (the v1 use case).
2. **RL fine-tuning**: if zero-shot is poor, point existing training at a synth
   recipe directory and finetune. The world-POI metric (visible-only) computes
   correctly because the only visible POI is torso.

## Pipeline at a glance

```
seed clip (real SONIC walk_forward, frame 0 only)
        │
        ▼   recenter pelvis to (0,0)
synth/cache/g1_body_names.json   ←   one-time dump (needs Isaac)
        │
        ▼
   recipe (squat | walk | …)
        │
        ▼
  synth/<recipe>/<recipe>.npz      ◄── identical schema to SONIC clips
        │
        ▼
  play.py --motion <synth_dir> --start_frame 0 --mask_modes kp6_torso --synth_eval --video
```

## One-time setup

```bash
# Dumps robot.body_names → synth/cache/g1_body_names.json. Needs a free GPU
# (Isaac requires CUDA). Only re-run if the URDF changes. The cache file is
# small (~2KB) and meant to be committed to the repo.
python scripts/synth_dump_body_names.py --robot g1
```

> ⚠️ The dumper bypasses `whole_body_tracking.robots.robot_registry` because that
> module's `from .h1_2 import ...` and `from .adam import ...` lines reference
> Python files that are missing from the working tree (only stale `.pyc`s remain
> under `__pycache__/`). The dumper imports `G1_CYLINDER_CFG` directly. If you
> add other robots, restore those files first or extend the dumper's per-robot
> dispatch.

## Generate recipes

```bash
# List the registry (offline, no Isaac). 54 recipes as of v2:
#   6 squats: squat_{10,15,20}cm × {0p5,1p0}hz
#   48 walks: walk_{000,045,090,135,180,225,270,315}deg × {0p1,0p3,0p6,1p0,1p5,2p0}ms
python scripts/synth_cli.py list

# Generate all recipes into per-recipe subdirectories.
python scripts/synth_cli.py generate-all \
    --seed-dir /home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/test/loco/walk_forward \
    --seed-rng-seed 42 \
    --out-root /home/lsn/Datasets/SONIC_npzs/g1/npz_synth

# Or just one recipe.
python scripts/synth_cli.py generate \
    --recipe walk_045deg_1p0ms \
    --seed-dir /home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/test/loco/walk_forward \
    --out-dir /home/lsn/Datasets/SONIC_npzs/g1/npz_synth/walk_045deg_1p0ms
```

### Recipe families

- **Squat (6 recipes)** — one-sided drop-from-standing via `SquatDown`. Three
  depths × two freqs. Trajectory starts at the seed's torso z (top), drops by
  `depth_m`, returns. Never rises above the start (real squats can't rise without
  a hop). Recipe name = depth in cm: `squat_15cm_1p0hz` drops 15cm at 1Hz.
- **Walk (48 recipes)** — `LinearTrans` at 8 world-frame directions (cardinal +
  diagonals, every 45°) × 6 speeds (0.1, 0.3, 0.6, 1.0, 1.5, 2.0 m/s). Naming:
  `walk_<deg>deg_<speed>ms`, with `<deg>` measured CCW from +x. **Facing**
  direction is locked to the seed's frame-0 torso quat (yaw anchor below) — so
  `walk_180deg` walks *backwards* relative to seed-facing, `walk_090deg` strafes
  left, `walk_315deg` is forward-right diagonal, etc.

## Wrist writing (in-air letter tracing)

A second family of recipes makes the **right wrist** trace a string of letters
in a vertical plane in front of the robot. Static **red dots** stay fixed in
the air showing the letter shape; the **green dot** is the robot's actual wrist
following the target. Robot faces the camera frontally (`viewer.eye` shifted
under `--synth_eval`).

- `write_MUSE_rwrist_chest`, `write_HI_rwrist_chest`, `write_MUSE_lwrist_chest`
  — small (12 cm letters) body-pinned variants. Suggested mask: `kp5_full`.
- `write_MUSE_rwrist_size{12,20,30,45}cm` — **size sweep**. At 30 cm the word
  width is 1.5 m, at 45 cm it's 2.25 m (over arm reach). Suggested mask:
  `right_wrist_only` so the policy is free to step/lean to reach the larger
  letters.

```bash
# Polished demo (body-pinned, small letters):
LOAD_RUN=2026-05-19_13-19-35_muse_kp5_latent_distill \
TASK=MUSE-Kp-LatentDistill-General-Tracking-Flat-G1-v0 \
MASK_MODE=kp5_full ONLY="write_MUSE_rwrist_chest,write_HI_rwrist_chest" \
GPUS="0 1" bash run/test/synth/synth_suite.sh

# Locomotion probe (let the policy step to reach):
LOAD_RUN=... TASK=... \
MASK_MODE=right_wrist_only \
ONLY="write_MUSE_rwrist_size12cm,write_MUSE_rwrist_size20cm,write_MUSE_rwrist_size30cm,write_MUSE_rwrist_size45cm" \
GPUS="0 1 2 3" bash run/test/synth/synth_suite.sh
```

The sphere radius auto-adjusts: 0.025 m for wrist recipes (so the green dot
doesn't overwhelm the red trail) and 0.05 m for torso recipes.

## Run the suite

The suite **wave-schedules** jobs across the GPUs you nominate via `GPUS=...`,
mirroring `run/test/eval_cotrain/indomain_suite.sh`: each wave launches one
play.py per GPU in parallel, waits for all to finish, then starts the next wave.
On three GPUs the 54-recipe set runs in ~18 waves ≈ 25–35 min wall-clock.

```bash
# All 54 recipes, parallel across GPUs 0/1/2 (verified end-to-end 2026-05-20):
LOAD_RUN=2026-05-19_13-19-35_muse_kp5_latent_distill \
TASK=MUSE-Kp-LatentDistill-General-Tracking-Flat-G1-v0 \
MASK_MODE=kp5_torso \
GPUS="0 1 2" \
bash run/test/synth/synth_suite.sh

# Curated subset (faster) — only cardinal walks @ 1 m/s + a squat:
ONLY="walk_000deg_1p0ms,walk_090deg_1p0ms,walk_180deg_1p0ms,walk_270deg_1p0ms,squat_15cm_1p0hz" \
LOAD_RUN=... TASK=... MASK_MODE=... GPUS="0 1 2" \
bash run/test/synth/synth_suite.sh

# KP6 distill checkpoint (canonical recipe, when one is available with the
# current 810-dim obs layout):
# LOAD_RUN=<run_dir> TASK=MUSE-Kp-Distill-General-Tracking-Flat-G1-v0 \
# MASK_MODE=kp6_torso GPUS="0 1 2"  bash run/test/synth/synth_suite.sh
```

`CHECKPOINT` defaults to the latest `model_*.pt` in the run dir, `EXP_NAME` is
guessed from `TASK` (override either explicitly if your run lives elsewhere).
Common filters: `ONLY`, `SKIP`, `MAX_RECIPES`, `RECIPE_SAMPLE_SEED`,
`VIDEO_LENGTH`, `EXTRA_PLAY_ARGS` — see the script header for the full list.

Outputs one video per recipe under
`logs/rsl_rl/<task_dir>/<run>/videos/synth_video/<recipe_name>/rl-video-step-0.mp4`
(set `VIDEO_DIR_TAG=...` to override). Synth videos are kept in this dedicated
subfolder so they don't mix with the `indom_*` directories.

**Pre-flight test (single recipe)** — equivalent to one shell-suite cell:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/play.py \
    --num_envs=1 \
    --task=MUSE-Kp-LatentDistill-General-Tracking-Flat-G1-v0 \
    --motion /home/lsn/Datasets/SONIC_npzs/g1/npz_synth/walk_000deg_1p0ms \
    --video_dir_tag synth_video \
    --load_run=2026-05-19_13-19-35_muse_kp5_latent_distill \
    --checkpoint=model_9500.pt \
    --start_frame=0 \
    --mask_modes kp5_torso \
    --synth_eval \
    --headless --video --video_length=300
```

## The npz contract

| Frame range | Field          | Value                                        |
|-------------|----------------|----------------------------------------------|
| frame 0     | all keys       | wholesale copy of seed-clip frame 0 (recentered so pelvis XY = 0) |
| 1..T-1      | `joint_pos`    | repeat of frame-0 pose (not consumed at t>0) |
| 1..T-1      | `joint_vel`    | zeros                                        |
| 1..T-1      | `body_*_w[torso_idx]` | synthetic trajectory                  |
| 1..T-1      | `body_*_w[!torso_idx]` | **NaN** (sentinel for masked obs)    |

### Why this works without breaking anything

- **Reset** reads `motion.root_pos_w[t]` (which is `body_pos_w[t, 0]` = pelvis). At
  reset, `t = start_frame = 0` because `--start_frame 0` forces
  `start_from_beginning = True`. Frame 0 of pelvis is valid → root reset works.
  Frames 1..T-1 of pelvis are NaN but never read for reset.
- **Observation** uses `--mask_modes kp{5,6}_torso`: only the torso slot is visible;
  every other slot is masked to NaN by the existing masked-obs pipeline. The npz
  NaN at non-torso bodies is invisible to the policy because the mask hides those
  slots anyway.
- **Termination**: `--video` already strips non-timeout terminations by default
  (`play_video_disable_non_timeout_terminations=True`). The timeout itself is also
  cleared during play. Motion-relative termination on ankles is never reached, so
  NaN ankle positions are fine.
- **Marker rendering**: `--synth_eval` (a) pins `video_debug_vis_body_names` to
  `["torso_link"]` so the visualizer doesn't render goal markers at NaN
  positions for the other 29 bodies, (b) hides the anchor frame triad (which
  would otherwise draw a coordinate-axes gizmo at the torso, since torso is the
  KP5/KP6 anchor body), and (c) bumps the body-visualizer sphere radius to
  0.05 m so the torso reads as a *torso* dot (vs the 0.025 m default used for
  KP markers in the rest of the codebase). Green = robot's actual torso, red =
  the goal torso position from the synth trajectory.

### Centering

The writer subtracts seed-frame-0 pelvis XY from every body's XY in frame 0.
Combined with how the env applies `motion.root_pos_w[t] + env_origins` at reset,
the robot spawns exactly at the env origin — centered in the camera frame, no
extra `--no_reset_base_xy_to_origin` machinery needed.

### Position and yaw anchors

Two motion-writer anchors keep the policy from seeing artificial discontinuities
between the seed's frame 0 and the primitive-driven frame 1+:

- **`anchor_torso_to_seed=True`** (default) — shifts the primitive so
  `primitive.at(0)` lines up exactly with the seed's frame-0 torso position. The
  recipe is written against a nominal standing torso z (~1.07m); the anchor
  brings it down to whatever the seed's actual torso z is (~0.82m for
  bones-seed walks). Without this, frame 0→1 would jump 25cm in z.
- **`anchor_yaw_to_seed=True`** (default) — overrides the primitive's
  quaternion across all frames with the seed's frame-0 torso quat. The policy
  sees a *constant facing* equal to the seed; the primitive only drives
  translation direction. Without this, `walk_180deg` (which the primitive
  internally writes as identity-yaw + (-1,0,0) direction) would force a sharp
  facing flip between frame 0 (seed quat) and frame 1 (identity).

Set `anchor_yaw_to_seed=False` only if you add a primitive that drives yaw
explicitly (none of the current primitives do).

## Extending

Add a new primitive in [primitives.py](primitives.py) (subclass `TorsoPrimitive`),
or a new recipe in [recipes.py](recipes.py) (declarative; the factory is called
once per build).

The writer is robot-agnostic by design: the JSON sidecar carries the body
ordering. To support a new robot, run `synth_dump_body_names.py --robot <name>`
and pass `--robot <name>` to the CLI.

## RL fine-tuning hookup

The output directory is a valid `--motion` argument for any RL training script
in this repo. Train PPO / latent-RL on synth distributions exactly as you'd
train on SONIC:

```bash
torchrun ... scripts/rsl_rl/train.py \
    --task=MUSE-Kp-LatentRL-General-Tracking-Flat-G1-v0 \
    --motion /home/lsn/Datasets/SONIC_npzs/g1/npz_synth \
    ...
```

Reward = `error_body_pos_visible` over the visible mask (torso only), which is
the world-POI accuracy already used by the latent-RL plan.
