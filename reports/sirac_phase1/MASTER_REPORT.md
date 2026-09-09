# SIRAC Phase 1 — Master Report

**Sparse-Intent Realization and Adaptation Controller**  
Phase 1 isolates one question: is a low-dimensional, robot-owned lower-body realization command a useful bottleneck between sparse head/hand intent and whole-body execution?

This report separates four kinds of statement:

1. **Verified facts** from repository inspection  
2. **Implementation choices** made for Phase 1  
3. **Experimental findings** (runtime evidence)  
4. **Remaining hypotheses**

A successful dummy smoke test is **not** evidence that the bottleneck works.

---

## 1. Question and method constraints

Sparse human intent is only:

- head SE(3)
- left-hand SE(3)
- right-hand SE(3)

Lower body, pelvis/base, and waist are robot-owned. No joystick velocity, human torso tracking, footstep commands, terrain labels, or task-mode switches are added to the human interface. One unified controller across tasks and terrains: no per-terrain policy, no per-task adapter, no terrain-class routing.

Phase 1 does **not** train ADAPT, diffusion, PPO recovery, or a terrain-specific module.

---

## 2. Architecture

```
                    human-owned (unchanged)
                 ┌─────────────────────────┐
                 │ head / L-hand / R-hand  │
                 │      SE(3) intent       │
                 └───────────┬─────────────┘
                             │
                             v
                 existing AnyBody Stage-2
                 encoder / decoder (29-DoF
                 nominal rollout)
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
              v              v                  v
        upper 14 DoF    pelvis + torso     (Baseline A)
        sparse tracker  kinematics         full 29-DoF
        (Baselines B/C)      │             joint targets
                             v
                  extract_realization_command
                  c_t ∈ R^7  (robot-owned)
                  [vx, vy, wz, h, r, p, yaw]
                  pelvis-yaw / virtual-anchor frame
                             │
                             v
                  xi_t = 0   (navigator disabled)
                  delta_c = B @ xi = 0
                  governor = identity
                             │
                             v
                  frozen HTD student LBC
                  obs history (2×58) + c_t
                             │
                             v
                  15-D position-offset actions
                  q = q_htd + 0.25 * a
                  → legs + waist
```

Phase 1B (interface only, untrained): `z_intent = SparseIntentEncoder(head/hand trajectory at t, t+0.2, t+0.4)` in the same yaw frame. `IntentConditionedLBC` refuses to run with `use_intent=True` until a student exists.

---

## 3. Verified facts (from `CONTROLLER_AUDIT.md`)

Short form; details and citations are in the audit.

| item | verified value |
| --- | --- |
| Student obs | 58-D, no feet contact; history 2 → 116-D |
| Teacher actor obs | 60-D, includes feet contact |
| Command order | `[lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch, yaw]` |
| Command frames | yaw-frame vx/vy; world wz; torso z; torso RPY in pelvis-yaw |
| Actions | 15-D, `q = q_htd + 0.25 * clip(a)` (position offset) |
| Joint name order | 29-DoF URDF = AnyBody registry = HTD `target_joint_order` |
| Frequencies | policy 50 Hz, physics 200 Hz, both stacks |
| Student extras | no RNN, no height scan, no terrain id |
| Deploy vs AnyBody | public deploy holds arms at 0 and takes joystick command; AnyBody tracks head/hands |

**Verified mismatch (not a joint-order bug):** HTD vs AnyBody default pose, PD, and action scale.

---

## 4. Implementation choices

All new code is additive. Frozen AnyBody eval scripts and P4 artifacts are not modified.

| choice | what | why |
| --- | --- | --- |
| Package location | `source/whole_body_tracking/.../sirac/` | keep original baselines runnable |
| Mapping | by joint **name**, never assumed index | USD vs URDF |
| Extractor frames | HTD **reward** frames | class comment is misleading |
| Command clip | deploy ranges | frozen student OOD protection |
| Obs defaults | HTD `q_default`, not AnyBody | student was trained that way |
| History (Phase 1A) | 2 frames | match public JIT |
| History (Phase 1B API) | 320 ms option | user request 200–400 ms; needs a new student |
| Navigator / governor | present, `enabled=False`, `xi=0` | ADAPT-ready, not trained |
| Dummy policy | zeros if JIT missing | tests/CI without GitHub weights |
| Baseline C arms | HTD zeros | matches public deploy arm hold; isolates LBC |

Configs: `scripts/sirac/configs/baseline_{a,b,c}.yaml` and `ablations.yaml`.

---

## 5. Baselines and ablations (wired, not yet Isaac-measured)

| id | pipeline |
| --- | --- |
| **A** | Stage-2 direct 29-DoF (existing) |
| **B** | Stage-2 upper + extracted `c_t` + frozen HTD LBC lower/waist |
| **C** | B with static/default arms |

Ablation slots (configs only; no terrain modules):

1. Direct joints vs 7-D bottleneck (A vs B)  
2. Frozen vs fine-tuned LBC (fine-tune **not trained**)  
3. No future intent vs future intent (encoder exists; `use_intent` locked off)  
4. 2-frame vs 200–400 ms history (16-frame student not trained)  
5. Static / AMASS / sparse-intent / Stage-2 upper distributions (eval harness later)  
6. World-frame vs pelvis-yaw torso features (extractor default is yaw-relative)

Causal teleoperation corruptions (40 ms latency, jitter, drop, actuator delay) are specified as randomized disturbances for a later Isaac eval loop, not as separate policies.

---

## 6. Tests and smoke

Isaac-free suite: `python tests/sirac/run_all.py` — **24/24 passed** (`isaaclab` Python 3.10 / NumPy 1.x; `np.arctan2` not `np.atan2`).

Covered:

- URDF / AnyBody registry / HTD name order  
- name-based remap under permutation  
- yaw-frame velocity, finite difference, relative RPY, wrap-to-π  
- height = torso z  
- command clip  
- 58-D obs, action scale 0.25, waist-pitch clip  
- batched dummy controller  
- Baseline A passthrough; B replaces lower, keeps arms; C zeros arms  
- navigator/governor identity  

Smoke: `python scripts/sirac/smoke_realization.py`  
Eval (dummy): `python scripts/sirac/eval_baselines.py --seed 42`  
Joint-map dump: `python scripts/sirac/convert_joint_order.py`

Metrics schema (JSON/CSV): head/hand pos+rot error, success, fall, SR@5cm, foot slip, contact impact, command tracking, actuator saturation, action smoothness, CoM/capture-point, `D_intent` and `D_intent_excess`, runtime, policy latency. Dummy rows leave tracking fields as NaN on purpose.

---

## 7. Experimental findings

First Isaac soak (2026-09-03, seed 42, loco, 10 envs × 400 steps, Mapper-B + `model_50000`, VR mask):

| cell | head pos | L hand | R hand | SR@5cm | D_intent |
| --- | --- | --- | --- | --- | --- |
| **A** plane (parent) | 0.074 m | 0.088 m | 0.088 m | 0.61 | 0.17 |
| **B** plane (7D LBC) | 2.06 m | 1.93 m | 2.06 m | 0.024 | 2.33 |
| **C** plane (static arms) | 2.16 m | 2.10 m | 2.13 m | 0.012 | 2.44 |
| **B** light_rough | 2.14 m | 2.11 m | 2.20 m | 0.023 | 2.52 |

`D_intent_excess(B−A) ≈ 2.16`. JIT was real (`dummy=false`).

**Do not use `fell=1.0` as the story.** Baseline A also records `fell=1` because the eval uses a 0.25 m wrist-z fail bit (`ee_z`); the parent still tracks at ~7–9 cm. Compare tracking and `D_intent`.

**Provisional Case C** (not closed): the frozen 7-D transplant destroys head/hand tracking even with static arms. That is an interface / frame / PD / action-scale / command-range problem until proven otherwise. Do **not** add ADAPT, PPO, or a large residual yet.

Machine-readable: `results/sirac_phase1/isaac/COMPARE.json`.

Unit tests remain **24/24**. Dummy smoke from before the JIT is not evidence. This Isaac soak **is** evidence, but only for the frozen transplant as currently wired — not for ADAPT.

---

## 8. Remaining hypotheses

1. `SceneEntityCfg` body_id order on Isaac Lab 2.1 vs HTD comments (audit §2.3). SIRAC extractors use named bodies; the *original* student may still have been trained with swapped ids if `preserve_order` differs.  
2. Zero-shot HTD student on AnyBody PD / Lab 2.1 / Sim 4.5 vs the student's Lab 2.2 plant.  
3. Stage-2 loco-manipulation commands may saturate deploy clips (height 0.8 m, yaw ±1.27). Saturation would look like “bottleneck failure” but is a range problem (Case C diagnostic).  
4. Upper-body disturbance from sparse-intent arms may be outside the HTD student's AMASS training (Case B diagnostic, Baseline C vs B).  
5. Fine-tuning a unified LBC on AnyBody rollouts may be required even if the 7-D *interface* is right (Case D).  

---

## 9. Decision rule (to be applied after real eval)

| case | evidence | next step |
| --- | --- | --- |
| **A** | B/C more stable than A, small `D_intent_excess` | proceed to ADAPT 3–5D residual on `c_t` + intent governor |
| **B** | C stable, B fails with moving arms | keep 7-D interface; train future-intent / task-matched upper-body disturbance |
| **C** | B/C degrade head/hand tracking even with static arms and passing unit tests | debug frames, height convention, torso ownership, feasibility — **do not** add residual nets |
| **D** | frozen student fails, fine-tuned student recovers | keep 7-D realization; train a new unified LBC |

Current status: **provisional Case C**. Required next: diagnose command/frame/PD/action-scale mismatch on the frozen transplant (Baseline C still ~2 m). Do not start ADAPT-style residual steering until B/C tracking is in the same order of magnitude as A.

---

## 10. File index

| path | role |
| --- | --- |
| `reports/sirac_phase1/CONTROLLER_AUDIT.md` | Phase 0 |
| `reports/sirac_phase1/MASTER_REPORT.md` | this file |
| `source/.../sirac/mappings.py` | joints, command ranges |
| `source/.../sirac/frames.py` | yaw frame, RPY, FD |
| `source/.../sirac/command_extract.py` | `c_t^0` |
| `source/.../sirac/lower_body_controller.py` | frozen LBC wrapper |
| `source/.../sirac/pipeline.py` | Baselines A/B/C |
| `source/.../sirac/intent_encoder.py` | Phase 1B stub |
| `source/.../sirac/adapt_interfaces.py` | navigator, governor, CF logs |
| `source/.../sirac/metrics.py` | JSON/CSV |
| `scripts/sirac/` | smoke, eval, train stub, configs |
| `tests/sirac/` | unit tests |
