# SIRAC Phase 0 — Controller Audit

Status: **verified from source** where a local or fetched file is cited.
Labels used below:

- **Verified** — read from code / config / URDF in this workspace or from the official HTD / Decoupled-WBC sources listed.
- **Implementation choice** — SIRAC Phase 1 decision, not a fact about the original controller.
- **Remaining hypothesis** — not frozen until a unit or sim test on the installed Isaac Lab stack resolves it.

Human-owned intent for SIRAC is only:

`head SE(3)`, `left-hand SE(3)`, `right-hand SE(3)`.

The 7-D command audited here is **not** a human interface. It is the HTD lower-body student's command vector, which SIRAC treats as a robot-owned realization variable.

---

## 0. Repositories inspected

| Tree | Path / origin | Role |
| --- | --- | --- |
| AnyBody (sparse-intent Stage-2) | `/data/home/chenxiangyu/robotics/Anybody` | Existing head/hand tracker, Stage-2 29-DoF decoder, eval stack |
| Isaac Lab (AnyBody runtime) | `/data/home/chenxiangyu/robotics/IsaacLab_v2.1` | Sim 4.5 / Lab 2.1 used by AnyBody |
| humanoid-touch-dream | Official repo is a thin wrapper; submodule `htd_wbc/isaaclab_decoupled_wbc` | Public teleop + WBC packaging |
| IsaacLab-Decoupled-WBC | https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC | **Actual lower-body controller** |

**Verified:** HTD itself does not reimplement the student; deployment lives in Decoupled-WBC `deploy/deploy_student_htd.py` + `deploy/configs/g1_student_htd.yaml`. Training lives in `legged_lab/`.

Local clones of HTD / Decoupled-WBC were **not** present under `robotics/` at audit time (GitHub SSL/443 failures). Command / obs / joint facts below are from official source files fetched for this audit, cross-checked against AnyBody's G1 URDF and `robot_registry.py`.

---

## 1. Lower-body policy observation vector

### 1.1 Deployed student (the SIRAC frozen transplant)

**Verified** from `deploy/deploy_student_htd.py` and `deploy/configs/g1_student_htd.yaml`:

Single-frame observation, **58-D**, no feet contact, no height scan, no terrain id:

| slice | dim | content | scale in YAML |
| --- | --- | --- | --- |
| `[0:3]` | 3 | IMU angular velocity (pelvis), rad/s | `ang_vel: 1.0` |
| `[3:6]` | 3 | projected gravity in IMU/base frame | `gravity: 1.0` |
| `[6:13]` | 7 | realization command (see §2) | `command_scale: all 1.0` |
| `[13:28]` | 15 | `q_lower - q_default_htd` | `dof_pos: 1.0` |
| `[28:43]` | 15 | lower/waist joint velocity | `dof_vel: 1.0` |
| `[43:58]` | 15 | last action | `action: 1.0` |

`history_length: 2` → policy input **116-D** (two concatenated 58-D frames).

`imu_type: "pelvis"`. Empirical observation normalization is **off** in the G1 agent cfg (`empirical_normalization: False`). Obs/action clip in training: ±100.

**Verified:** joint positions in the observation are offsets from **HTD** default pose, not AnyBody Beyond-Mimic defaults.

### 1.2 Teacher actor (training only)

**Verified** from `legged_lab/envs/base/base_env.py` `compute_current_observations`:

```
ang_vel_b (3) | projected_gravity_b (3) | command (7) | joint_pos_err (15)
| joint_vel (15) | last_action (15) | feet_contact (2)
```

= **60-D**. Docs that say “student is 58-D without feet contact” match the **deploy student**, not the teacher.

Teacher critic adds `root_lin_vel_b` and, on rough terrains, a height scan. **SIRAC student must not receive terrain identity or height maps at inference.**

### 1.3 History / recurrence / privileged

| item | G1-flat deploy student | G1-flat teacher | rough teacher |
| --- | --- | --- | --- |
| history | 2 frames (40 ms @ 50 Hz) | `actor_obs_history_length: 1` | longer / LSTM |
| recurrence | MLP `[512, 256, 128]`, no RNN | MLP | `ActorCriticRecurrent` |
| contact | **no** | feet_contact in actor | yes |
| height scan | **no** | no on flat | yes |
| terrain id | **no** | **no** | **no** as a class label |

**Implementation choice (Phase 1A):** keep the public student's 2-frame history so the frozen JIT is a valid transplant.

**Implementation choice (Phase 1B, untrained):** optional 200–400 ms proprio history (e.g. 16 frames @ 50 Hz = 320 ms) for a *new* student. That is a different policy; it cannot be dropped onto the public JIT.

---

## 2. Command-vector definition and ordering

**Verified** order, identical in deploy YAML, teacher observation slice `[6:13]`, and `UniformVelocityCommand`:

```
c = [lin_vel_x, lin_vel_y, ang_vel_z, height, body_roll, body_pitch, body_yaw]
```

### 2.1 Semantics — follow the *rewards*, not the class comment

The command class comment says “in the base frame”. **Rewards disagree.** SIRAC extractors use reward frames:

| command | tracked quantity (reward) | frame |
| --- | --- | --- |
| `lin_vel_x, lin_vel_y` | `quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w)[:, :2]` | **yaw-aligned world**, not `root_lin_vel_b` |
| `ang_vel_z` | `root_ang_vel_w[:, 2]` | **world yaw rate**, not body-z |
| `height` | `torso_link` world z | not pelvis z |
| `body_roll/pitch/yaw` | torso orientation vs `yaw_quat(pelvis)` | pelvis-yaw / virtual-anchor heading; yaw wrapped to π |

Heading command: `heading_command=True` by default. Heading error is converted to yaw-rate with `heading_control_stiffness=0.5`. Standing environments zero **only the first 3 dims** (velocities), not torso pose.

Default standing height on keyboard reset: **0.72 m**.

### 2.2 Ranges

Deploy clip (`g1_student_htd.yaml` `command_range`):

| name | lo | hi |
| --- | --- | --- |
| lin_vel_x/y | -0.55 | 0.55 |
| ang_vel_z | -1.57 | 1.57 |
| height | 0.35 | 0.8 |
| body_roll | -0.5 | 0.5 |
| body_pitch | -0.52 | 1.22 |
| body_yaw | -1.27 | 1.27 |

Training ranges in `g1_config.py` / `G1FlatEnvCfg` are slightly **wider** (roll ±0.7, pitch to 1.57, yaw ±1.57, lin_vel ±0.5). Extra hardware clip: `waist_pitch_limits: [-0.60, 0.60]` on the **action target**, not on the command.

**Implementation choice:** Phase 1A clips to **deploy** ranges so a frozen student is not queried out of distribution.

### 2.3 Remaining hypothesis — torso/pelvis body index

Reward code uses `SceneEntityCfg(..., body_names=["torso_link", "pelvis"])` then:

```
quat_w_torso  = body_quat_w[:, body_ids[1]]  # comment: torso
quat_w_pelvis = body_quat_w[:, body_ids[0]]  # comment: pelvis
```

If Isaac Lab `preserve_order=False` (typical default), bodies are in asset order: pelvis before torso → `body_ids[0]=pelvis`, `body_ids[1]=torso`, matching the comments.

If `preserve_order=True`, names would map torso→0, pelvis→1 and the comments would be **wrong**.

SIRAC extractors **do not use those integer ids**. They take named `torso_*` and `pelvis_*` tensors. A live Isaac test should still print `body_ids` on the installed Lab 2.1 stack before claiming relative-yaw tracking of the *original* HTD student.

---

## 3. Controlled-joint list and ordering

**Verified** 29-DoF canonical order (`target_joint_order` in `base_env.py` = AnyBody `ROBOT_PLATFORMS["g1"].joint_names` = URDF revolute order in `assets/unitree_description/urdf/g1/main.urdf`):

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
waist_yaw, waist_roll, waist_pitch,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

HTD `num_actions = n_joints - 14` → **15** (12 legs + 3 waist). Deploy `joint2motor_idx: [0..14]`.

Isaac Lab USD joint index may differ from this name order. HTD remaps via `custom_joint_ids` / `inverse_joint_ids`. **SIRAC maps by name only** (`lower_indices_in`). Automated tests freeze this (`tests/sirac/test_mappings.py`).

---

## 4. Policy output interpretation

**Verified** (deploy + `env.step`):

```
q_target = q_default_htd + clip(action, ±100) * action_scale
action_scale = 0.25   # uniform over 15 DoF
```

This is a **position offset** from HTD defaults, **not** torque, **not** velocity, **not** absolute position, **not** AnyBody `G1_ACTION_SCALE = 0.25 * effort / stiffness`.

HTD default lower pose (rad):

```
[-0.20, 0, 0, 0.42, -0.23, 0,  -0.20, 0, 0, 0.42, -0.23, 0,  0, 0, 0]
```

AnyBody `G1_CYLINDER_CFG` uses a different squat (`hip_pitch=-0.312`, `knee=0.669`, `ankle_pitch=-0.363`, init root z 0.76 vs HTD 0.80). **Do not reuse AnyBody action scale or defaults inside the HTD observation.**

PD (HTD deploy):  
`kp = [150,150,100,150,40,40, 150,150,100,150,40,40, 200,150,150]`  
`kd = [3,3,2,4,2,2, 3,3,2,4,2,2, 4,4,4]`

AnyBody splits waist across actuator groups with Beyond-Mimic scales. Running the HTD student **inside** AnyBody's PD plant is a remaining sim-to-sim risk (Lab 2.1 vs HTD's Lab 2.2 / Sim 5.0 docs).

---

## 5. Control and simulation frequencies

| layer | HTD G1-flat | AnyBody tracking env |
| --- | --- | --- |
| physics `sim.dt` | 0.005 s (200 Hz) | 0.005 s |
| decimation | 4 | 4 |
| policy / command | **50 Hz** (`control_dt: 0.02`) | **50 Hz** |
| deploy command-publish thread | 500 Hz | n/a |

Phase 1A can share the AnyBody 50 Hz tick without resampling.

---

## 6. Coordinate frames (summary)

| quantity | frame used by HTD rewards / SIRAC extractor |
| --- | --- |
| base linear velocity command | pelvis **yaw** frame of world linear velocity |
| base yaw rate command | **world** wz |
| torso height | `torso_link` world z |
| torso roll/pitch/yaw | torso RPY in **pelvis yaw** frame (virtual-anchor heading) |
| student IMU ang vel / gravity | pelvis IMU / base |

Global heading is therefore **not** a human intent channel; it is absorbed into the yaw-aligned velocity and the relative torso yaw.

---

## 7. Checkpoint loading and normalization

Public JIT paths in Decoupled-WBC:

- `deploy/policy/g1_student/student_policy_jit.pt`
- `example/student_checkpoints/student_policy_jit.pt`

Load: `torch.jit.load(path, map_location=...)`; eval mode; **no** running-mean normalizer on the maintained G1-flat student.

**Verified locally:** public G1-flat student JIT is on disk after a later zipball fetch:

- `/data/home/chenxiangyu/robotics/IsaacLab-Decoupled-WBC/example/student_checkpoints/student_policy_jit.pt`
- `/data/home/chenxiangyu/robotics/IsaacLab-Decoupled-WBC/deploy/policy/g1_student/student_policy_jit.pt`
- copy: `Anybody/vendor/htd/student_policy_jit.pt`

SHA256: `c09f87a81a49f1e8ca4826f0db15251973e63222c685c7ea3d2fcac7cdffcc35`

`torch.jit.load` succeeds; a batched dummy proprio+command query returns finite **(B, 15)** actions (`is_dummy=False`). This confirms checkpoint loading and action dimension only. It is **not** a Baseline A/B/C Isaac eval.

---

## 8. What the deployed student uses

Only: current (and 1-step) proprioception, 7-D command, last action.  
No privileged contact, no terrain height, no recurrence, no human joystick inside the network (joystick is how the *public deploy script* writes the command).

---

## 9. Public deploy script vs full HTD / vs AnyBody teleop

| | `deploy_student_htd.py` | Full HTD paper teleop | AnyBody sparse-intent Stage-2 |
| --- | --- | --- | --- |
| 7-D command source | joystick / buttons | joystick base vel + mapped human | **SIRAC: extracted from Stage-2** (robot-owned) |
| arms | held at **zeros** | IK / retargeted | Stage-2 decoder / sparse tracker |
| VR head/hands | no | yes | yes (head + two wrists) |
| HTD transformer | no | yes | no (AnyBody encoder/decoder) |
| touch | no | yes | AnyBody contact stack, separate |

Phase 1 **must not** add joystick velocity to the human interface. The deploy script's joystick is how HTD *authors* drove the student; SIRAC replaces that with analytic extraction.

---

## 10. G1 joint-order mismatch table

| source | 29-DoF name order | notes |
| --- | --- | --- |
| URDF `g1/main.urdf` revolute | matches §3 | unit-tested |
| AnyBody `robot_registry.py` | matches §3 | unit-tested |
| HTD `target_joint_order` | matches §3 | from official source |
| Isaac Lab USD indices | **may differ** | map by name |
| Unitree HG motor index | 0–14 lower/waist, 15–28 arms | HTD `joint2motor_idx` |
| AnyBody Stage-2 action | all 29, `joint_names=[".*"]`, `use_default_offset=True` | scale ≠ HTD 0.25 |
| HTD student action | 15 DoF, scale 0.25, HTD defaults | |

Name order agrees. **Scale, default pose, PD, and arm residual do not.**

---

## 11. AnyBody Stage-2 facts needed by the extractor

- Decoder: `policy.muse._decode(z, proprio)` → full-body joint **position offsets** (29-D), then Isaac Lab `JointPositionAction` adds AnyBody defaults.
- Sparse intent bodies (HeadHands): `torso_link`, `left_wrist_yaw_link`, `right_wrist_yaw_link`.
- Future slots already in actor obs (`scripts/rsl_rl/causal_future.py`): log-spaced offsets including **+10 steps = 0.20 s** and **+20 steps = 0.40 s** at 50 Hz. Phase 1B horizons `{0.0, 0.2, 0.4}` s are therefore already in the AnyBody buffer.
- Policy 50 Hz matches HTD.

---

## 12. What this audit does *not* freeze

1. Isaac Lab `SceneEntityCfg` body_id order on Lab 2.1 (`preserve_order`).
2. JIT file checksum / that the public student was trained with the 58-D layout above (layout is from deploy source; weights unverified until the file is loaded).
3. Whether AnyBody's PD + Lab 2.1 plant is close enough to HTD Lab 2.2 for a zero-shot transplant.
4. Whether Stage-2 torso yaw relative to pelvis stays inside deploy `body_yaw` clip during loco-manipulation.

Those are remaining hypotheses, not implementation bugs.
