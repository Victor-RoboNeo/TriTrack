ALL_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


UPPER_ONLY = [
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

LOWER_ONLY = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]

# Both wrist keypoints (typical end-effector targets for manipulation / reach).
UPPER_END_EFFECTOR_ONLY = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# Single wrist only (one keypoint each).
LEFT_END_EFFECTOR_ONLY = [
    "left_wrist_yaw_link",
]

RIGHT_END_EFFECTOR_ONLY = [
    "right_wrist_yaw_link",
]

# Pelvis-only visibility (single root keypoint). Used by eval-time single-mode overrides
# (see ``eval_single_mode_spec``); NOT part of the training mask_mode list.
PELVIS_ONLY = [
    "pelvis",
]

END_EFFECTOR_ONLY = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

UPPER_LEFT_ONLY = [
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
]

UPPER_RIGHT_ONLY = [
    "torso_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

UPPER_LEFT_WITHOUT_TORSO = [
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
]

UPPER_RIGHT_WITHOUT_TORSO = [
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# Left/right half body without center links (torso/pelvis), 6 keypoints each.
LEFT_HALF_ONLY = [
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
]

RIGHT_HALF_ONLY = [
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# Order matches ``mode_spec`` keys in ``G1FlatPartialMaskedAnyBodyLatentDistillationRunnerCfg``
# (``rsl_rl_ppo_cfg.py``) and the mask matrix row order from ``build_mask_matrix``.
PARTIAL_MASKED_2B_G1_MODE_NAMES = (
    "full",
    "upper",
    "upper_left",
    "upper_right",
    "left_half",
    "right_half",
    "end_effector",
    "left_end_effector",
    "right_end_effector",
    "upper_end_effector",
    "bernoulli",
)


def partial_masked_2b_g1_mode_spec() -> dict[str, list[str]]:
    """Same keys/order as :obj:`PARTIAL_MASKED_2B_G1_MODE_NAMES` and training ``mask_cfg.mode_spec``."""
    return {
        "full": list(ALL_BODIES),
        "upper": list(UPPER_ONLY),
        "upper_left": list(UPPER_LEFT_ONLY),
        "upper_right": list(UPPER_RIGHT_ONLY),
        "left_half": list(LEFT_HALF_ONLY),
        "right_half": list(RIGHT_HALF_ONLY),
        "end_effector": list(END_EFFECTOR_ONLY),
        "left_end_effector": list(LEFT_END_EFFECTOR_ONLY),
        "right_end_effector": list(RIGHT_END_EFFECTOR_ONLY),
        "upper_end_effector": list(UPPER_END_EFFECTOR_ONLY),
        # Special mode: Bernoulli per-body visibility is sampled at runtime (see PartialMaskedMultiMotionCommand).
        # We still include all bodies here so "compact_goal_observation=True" sees every keypoint as potentially visible.
        "bernoulli": list(ALL_BODIES),
    }


_EVAL_SINGLE_MODES: dict[str, list[str]] = {
    "full": list(ALL_BODIES),
    "upper": list(UPPER_ONLY),
    "lower": list(LOWER_ONLY),
    "upper_left": list(UPPER_LEFT_ONLY),
    "upper_right": list(UPPER_RIGHT_ONLY),
    "left_half": list(LEFT_HALF_ONLY),
    "right_half": list(RIGHT_HALF_ONLY),
    "end_effector": list(END_EFFECTOR_ONLY),
    "left_end_effector": list(LEFT_END_EFFECTOR_ONLY),
    "right_end_effector": list(RIGHT_END_EFFECTOR_ONLY),
    "upper_end_effector": list(UPPER_END_EFFECTOR_ONLY),
    "pelvis_only": list(PELVIS_ONLY),
    # Phase-4 demo modes (5-body interactive-drag distribution). Defined inline rather than
    # referencing COTRAIN_KP5_BODIES — that constant is declared further down in this file.
    "kp5_full": [
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    "kp5_bernoulli": [
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    "kp5_torso": ["torso_link"],
    "kp5_wrists": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    "kp5_ankles": ["left_ankle_roll_link", "right_ankle_roll_link"],
    # VR deploy mode (headset + 2 hand controllers): wrists + torso. Matches the ``vr`` entry
    # in :func:`muse_kp5_latent_demo_mode_spec` (latent-distill 8-mode demo spec).
    "vr": ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"],
    # Eval-only single-/dual-wrist drag modes (diagnostic for one-hand vs two-hand following).
    "wrist_only": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    "left_wrist_only": ["left_wrist_yaw_link"],
    "right_wrist_only": ["right_wrist_yaw_link"],
}


# Co-train training mode list: same as PARTIAL_MASKED_2B_G1_MODE_NAMES but with ``pelvis_only``
# inserted before ``bernoulli``. 12 modes total. Used by the MUSE co-train env/curriculum so
# the KP-tracker sees pelvis-only training; other configs (MUSEKp, SAGEKp, PartialMaskedVAE)
# keep the original 11-mode list to avoid invalidating their length-11 mode_probs tuples.
COTRAIN_MODE_NAMES = (
    "full",
    "upper",
    "upper_left",
    "upper_right",
    "left_half",
    "right_half",
    "end_effector",
    "left_end_effector",
    "right_end_effector",
    "upper_end_effector",
    "pelvis_only",
    "bernoulli",
)


def cotrain_mode_spec() -> dict[str, list[str]]:
    """12-mode spec used only by the MUSE co-train training config.

    Key order matches :obj:`COTRAIN_MODE_NAMES`. ``pelvis_only`` is inserted at index 10
    (right before ``bernoulli``); existing semantic modes keep indices 0–9 so any external
    code reading them stays valid.
    """
    return {
        "full": list(ALL_BODIES),
        "upper": list(UPPER_ONLY),
        "upper_left": list(UPPER_LEFT_ONLY),
        "upper_right": list(UPPER_RIGHT_ONLY),
        "left_half": list(LEFT_HALF_ONLY),
        "right_half": list(RIGHT_HALF_ONLY),
        "end_effector": list(END_EFFECTOR_ONLY),
        "left_end_effector": list(LEFT_END_EFFECTOR_ONLY),
        "right_end_effector": list(RIGHT_END_EFFECTOR_ONLY),
        "upper_end_effector": list(UPPER_END_EFFECTOR_ONLY),
        "pelvis_only": list(PELVIS_ONLY),
        "bernoulli": list(ALL_BODIES),
    }


# --------------------------------------------------------------------------------------
# Reduced 5-body cotrain spec (2026-05-14 onward).
#
# Drops the 14-body spec down to one keypoint per limb end plus the torso anchor.
# Rationale: with only 5 candidate bodies, the 10 semantic modes (upper, left_half,
# end_effector, …) become near-duplicates of bernoulli samples, so the curriculum
# collapses to {full, bernoulli}. The interactive-drag demo only ever needs these 5
# targets anyway. Pelvis intentionally dropped — torso covers root pose for the demo.
# --------------------------------------------------------------------------------------
COTRAIN_KP5_BODIES = [
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


COTRAIN_KP5_MODE_NAMES = ("full", "bernoulli")


def cotrain_kp5_mode_spec() -> dict[str, list[str]]:
    """2-mode spec for the 5-body cotrain setup. ``full`` and ``bernoulli`` both cover
    every body; ``bernoulli`` triggers per-body visibility sampling at runtime."""
    return {
        "full": list(COTRAIN_KP5_BODIES),
        "bernoulli": list(COTRAIN_KP5_BODIES),
    }


# --------------------------------------------------------------------------------------
# Dense 14-body 2-mode spec — minimal ablation against the 5-body sparse default.
#
# Same {full, bernoulli} structure as :func:`cotrain_kp5_mode_spec`, but over all 14
# tracked G1 bodies (:data:`ALL_BODIES`). Kept for any caller that wants the simplest
# dense recipe. The MUSE-Kp default curriculum now uses the richer 7-mode spec below
# (:func:`muse_kp_dense_curriculum_mode_spec`).
# --------------------------------------------------------------------------------------
def dense_kp14_mode_spec() -> dict[str, list[str]]:
    """2-mode spec: {full, bernoulli} both over all 14 G1 tracked bodies."""
    return {
        "full": list(ALL_BODIES),
        "bernoulli": list(ALL_BODIES),
    }


# --------------------------------------------------------------------------------------
# MUSE-Kp 4-phase dense+demo-mix curriculum mode spec (2026-05-14).
#
# 7 modes:
#   - ``full``         : all 14 G1 bodies (phase-1 bootstrap)
#   - ``bernoulli``    : per-body bernoulli over all 14 bodies (phase-2 ramp + phase-3 hold;
#                        also ~40% of phase-4 to preserve 14-body generalization)
#   - ``kp5_bernoulli``: per-body bernoulli over only the 5 demo bodies; other 9 fixed-masked
#                        (phase-4: stochastic coverage of the 31 non-empty 5-body subsets)
#   - ``kp5_full``     : all 5 demo bodies visible, other 9 fixed-masked (phase-4)
#   - ``kp5_torso``    : torso only (phase-4: single-body drag)
#   - ``kp5_wrists``   : both wrists (phase-4: upper end-effector drag)
#   - ``kp5_ankles``   : both ankles (phase-4: lower end-effector drag)
#
# Bernoulli-suffix mode names trigger per-body sampling (via PartialMaskedMultiMotionCommand);
# they share the global ``bernoulli_keep_prob`` set by the curriculum.
# --------------------------------------------------------------------------------------
MUSE_KP_DENSE_CURRICULUM_MODE_NAMES = (
    "full",
    "bernoulli",
    "kp5_bernoulli",
    "kp5_full",
    "kp5_torso",
    "kp5_wrists",
    "kp5_ankles",
)


def muse_kp_dense_curriculum_mode_spec() -> dict[str, list[str]]:
    """7-mode spec for the MUSE-Kp 4-phase dense+demo-mix curriculum.

    Key order matches :data:`MUSE_KP_DENSE_CURRICULUM_MODE_NAMES`. Phase-1 uses ``full``;
    phase-2/3 use ``bernoulli`` (over 14); phase-4 mixes ``bernoulli`` (broad 14-body coverage)
    with the 5 demo modes (focused on the interactive-drag distribution).
    """
    return {
        "full": list(ALL_BODIES),
        "bernoulli": list(ALL_BODIES),
        "kp5_bernoulli": list(COTRAIN_KP5_BODIES),
        "kp5_full": list(COTRAIN_KP5_BODIES),
        "kp5_torso": ["torso_link"],
        "kp5_wrists": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "kp5_ankles": ["left_ankle_roll_link", "right_ankle_roll_link"],
    }


# --------------------------------------------------------------------------------------
# MUSE-Kp5 from-scratch 2-phase curriculum mode spec (2026-05-15).
#
# 5-body-NATIVE spec (encoder kp_n_bodies=5; body_names = COTRAIN_KP5_BODIES). Unlike
# :func:`muse_kp_dense_curriculum_mode_spec` (14-body encoder, scopes down to 5 via
# fixed-masking the other 9), here the encoder only has the 5 demo tokens, so every mode's
# body list is a subset of the 5 kp5 bodies. Used by the from-scratch pilot-anneal run.
#
# 5 modes (order = :data:`MUSE_KP5_CURRICULUM_MODE_NAMES`):
#   - ``bernoulli``  : per-body bernoulli over all 5 (M1 ramp p_keep 1.0→0.4; M2 broad coverage)
#   - ``kp5_full``   : all 5 visible (M2)
#   - ``kp5_torso``  : torso only (M2: single-body drag)
#   - ``kp5_wrists`` : both wrists (M2: upper end-effector drag)
#   - ``kp5_ankles`` : both ankles (M2: lower end-effector drag)
#
# ``bernoulli`` MUST stay at index 0 — the keypoint_mask_mode curriculum finds it by name for
# the p_keep ramp, and a length-5 mode_probs tuple is positional.
# --------------------------------------------------------------------------------------
MUSE_KP5_CURRICULUM_MODE_NAMES = (
    "bernoulli",
    "kp5_full",
    "kp5_torso",
    "kp5_wrists",
    "kp5_ankles",
)


def muse_kp5_curriculum_mode_spec() -> dict[str, list[str]]:
    """5-mode, 5-body-native spec for the MUSE-Kp5 from-scratch 2-phase curriculum.

    Key order matches :data:`MUSE_KP5_CURRICULUM_MODE_NAMES`. M1 uses ``bernoulli`` only
    (p_keep ramps 1.0→0.4); M2 mixes ``bernoulli`` (broad 5-body coverage) with the explicit
    demo modes. All body lists are subsets of :data:`COTRAIN_KP5_BODIES`.
    """
    return {
        "bernoulli": list(COTRAIN_KP5_BODIES),
        "kp5_full": list(COTRAIN_KP5_BODIES),
        "kp5_torso": ["torso_link"],
        "kp5_wrists": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "kp5_ankles": ["left_ankle_roll_link", "right_ankle_roll_link"],
    }


# --------------------------------------------------------------------------------------
# MUSE-Kp6 from-scratch 2-phase curriculum — KP5 + pelvis (2026-05-16).
#
# The 5-body-native set (:data:`COTRAIN_KP5_BODIES`) leads with ``torso_link`` and drops
# pelvis. The motion-command reset writes the robot's root-link (pelvis) state, so a tracked
# set without pelvis previously caused the base to spawn at the torso reference every reset
# (a near-perfect teacher then read ~0.80 instead of ~0.99). ``_resolve_root_body_index``
# now hard-asserts the root link is tracked; this 6-body set adds ``pelvis`` (kept FIRST so
# it is also index 0 / the resolved root) while keeping torso + the 4 limb-ends.
#
# Same 5 modes as KP5 (mode count unchanged ⇒ curriculum/mask_mode_probs tuples stay
# length-5). pelvis participates only in ``bernoulli`` / ``kp6_full``; the explicit demo
# modes (torso/wrists/ankles) deliberately exclude it, so the interactive-drag deployment
# distribution is unchanged — pelvis is a maskable root-pose token, masked out at deploy.
# --------------------------------------------------------------------------------------
KP6_NATIVE_BODIES = [
    "pelvis",
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


MUSE_KP6_CURRICULUM_MODE_NAMES = (
    "bernoulli",
    "kp6_full",
    "kp6_torso",
    "kp6_wrists",
    "kp6_ankles",
)


def muse_kp6_curriculum_mode_spec() -> dict[str, list[str]]:
    """5-mode, 6-body-native spec (KP5 + pelvis) for the from-scratch 2-phase curriculum.

    Key order matches :data:`MUSE_KP6_CURRICULUM_MODE_NAMES`. ``bernoulli`` / ``kp6_full``
    span all 6 bodies (incl. pelvis); the demo modes match :func:`muse_kp5_curriculum_mode_spec`
    exactly (pelvis intentionally excluded there). ``bernoulli`` MUST stay index 0 — the
    keypoint_mask_mode curriculum finds it by name for the p_keep ramp.
    """
    return {
        "bernoulli": list(KP6_NATIVE_BODIES),
        "kp6_full": list(KP6_NATIVE_BODIES),
        "kp6_torso": ["torso_link"],
        "kp6_wrists": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "kp6_ankles": ["left_ankle_roll_link", "right_ankle_roll_link"],
    }


# --------------------------------------------------------------------------------------
# MUSE-Kp6 single-mode (pure-bernoulli) spec — the simplified canonical MUSE-Kp recipe
# (2026-05-16). No mode mixing: one ``bernoulli`` mode over all 6 KP6 bodies, per-body
# visibility sampled at the curriculum's ``p_keep`` (p_see). The 3-phase
# :class:`MUSEKpDistillationCurriculumCfg` only ramps p_see (1.0 → 0.4); ``mode_probs`` is
# always ``(1.0,)``. Used by ``MUSE-Kp-Distill-General-Tracking-Flat-G1-v0``.
# --------------------------------------------------------------------------------------
MUSE_KP6_BERNOULLI_ONLY_MODE_NAMES = ("bernoulli",)


def muse_kp6_bernoulli_only_mode_spec() -> dict[str, list[str]]:
    """1-mode spec: a single per-body ``bernoulli`` over all 6 :data:`KP6_NATIVE_BODIES`.

    ``mask_mode_probs`` must be length-1 ``(1.0,)``. The keypoint_mask_mode curriculum finds
    ``bernoulli`` by name and ramps its shared p_keep; there is no mode mixing.
    """
    return {"bernoulli": list(KP6_NATIVE_BODIES)}


# --------------------------------------------------------------------------------------
# MUSE-Kp6 OOD-avoidance mix spec — adds explicit single-point deploy modes (2026-05-16).
#
# The canonical KP6 recipe trains pure-bernoulli over the 6 bodies. At deploy the
# interactive-drag demo runs *single-point-visible* (e.g. R-wrist-only); pure-bernoulli
# rarely samples an exactly-one-body-visible mask at p_see=0.4, so those single-point
# configurations are out-of-distribution. This 5-mode spec mixes the broad ``bernoulli``
# coverage with the four deploy-relevant single-point modes so the student sees them
# directly during the final (phase-4) mask-mode-sampling phase.
#
# 5 modes (order = :data:`MUSE_KP6_OOD_MIX_MODE_NAMES`; ``bernoulli`` MUST stay index 0 —
# the keypoint_mask_mode curriculum finds it by name for the p_keep ramp):
#   - ``bernoulli``        : per-body bernoulli over all 6 KP6 bodies (broad coverage)
#   - ``kp6_left_wrist``   : left wrist only   (single-point deploy)
#   - ``kp6_right_wrist``  : right wrist only  (single-point deploy / demo headline)
#   - ``kp6_torso``        : torso only        (single-point deploy)
#   - ``kp6_pelvis``       : pelvis only       (single-point deploy / root-pose drag)
#
# All body lists are subsets of :data:`KP6_NATIVE_BODIES`, so the 6-token KP encoder
# (``kp_n_bodies=6``) is unchanged — only the mode count (1 → 5) and the curriculum's
# ``mode_probs`` tuple length change.
# --------------------------------------------------------------------------------------
MUSE_KP6_OOD_MIX_MODE_NAMES = (
    "bernoulli",
    "kp6_left_wrist",
    "kp6_right_wrist",
    "kp6_torso",
    "kp6_pelvis",
)


def muse_kp6_ood_mix_mode_spec() -> dict[str, list[str]]:
    """5-mode spec: broad ``bernoulli`` + the 4 single-point deploy modes (KP6 OOD mix).

    Key order matches :data:`MUSE_KP6_OOD_MIX_MODE_NAMES`. ``bernoulli`` spans all 6
    :data:`KP6_NATIVE_BODIES`; the four explicit modes are deterministic single-body
    visibility (left wrist / right wrist / torso / pelvis). ``mask_mode_probs`` must be
    length-5; ``bernoulli`` stays index 0 for the curriculum's p_keep ramp.
    """
    return {
        "bernoulli": list(KP6_NATIVE_BODIES),
        "kp6_left_wrist": ["left_wrist_yaw_link"],
        "kp6_right_wrist": ["right_wrist_yaw_link"],
        "kp6_torso": ["torso_link"],
        "kp6_pelvis": ["pelvis"],
    }


# ======================================================================================
# CANONICAL unified MUSE-Kp6 mask spec (2026-05-17). SINGLE SOURCE OF TRUTH.
#
# ALL MUSE-kp training envs — Kp distill, Kp-aux, Kp5-fromscratch, sym-sweep, co-train —
# use this 6-body native set + this 6-mode spec + these probs constants. Curricula and env
# post_inits reference the constants below instead of hand-typing tuples, so a mode_probs
# tuple cannot silently drift out of sync with the spec. (Backstop:
# MotionCommand.set_mask_mode_probs_tuple raises if len(probs) != n_modes.)
#
# 6 modes (``bernoulli`` MUST stay index 0 — the keypoint_mask_mode curriculum finds it by
# name for the p_keep ramp):
#   bernoulli        per-body bernoulli over all 6 KP6 bodies (broad; p_see-scheduled)
#   kp6_wrists       both wrists only            (group deploy)
#   kp6_ankles       both ankles only            (group deploy)
#   kp6_torso        torso only                  (single-point deploy)
#   kp6_left_wrist   left wrist only             (single-point deploy)
#   kp6_right_wrist  right wrist only            (single-point deploy / demo headline)
# pelvis stays in KP6_NATIVE_BODIES (root-link reset + spanned by bernoulli) but is NOT a
# deploy mode. Supersedes muse_kp6_ood_mix / muse_kp6_curriculum / muse_kp6_bernoulli_only
# / muse_kp_dense / muse_kp5 for MUSE-kp *training* runs (old fns kept for eval tooling).
# ======================================================================================
MUSE_KP6_UNIFIED_MODE_NAMES = (
    "bernoulli",
    "kp6_wrists",
    "kp6_ankles",
    "kp6_torso",
    "kp6_left_wrist",
    "kp6_right_wrist",
)

MUSE_KP6_UNIFIED_BERNOULLI_PROBS = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
"""Warmup-phase mode_probs (bernoulli-only). Length == len(MUSE_KP6_UNIFIED_MODE_NAMES)=6."""

MUSE_KP6_UNIFIED_MIX_PROBS = (0.5, 0.1, 0.1, 0.1, 0.1, 0.1)
"""Final-phase mode_probs: 0.5 bernoulli + 0.1 each of the 5 deploy modes (sums to 1.0)."""


def muse_kp6_unified_mode_spec() -> dict[str, list[str]]:
    """CANONICAL 6-mode MUSE-Kp6 spec — the single spec all MUSE-kp envs unify onto.

    Key order == :data:`MUSE_KP6_UNIFIED_MODE_NAMES`. ``bernoulli`` spans all 6
    :data:`KP6_NATIVE_BODIES`; the five explicit modes are deterministic only-visible body
    sets. Pair with :data:`MUSE_KP6_UNIFIED_BERNOULLI_PROBS` (warmup phases) /
    :data:`MUSE_KP6_UNIFIED_MIX_PROBS` (final mix phase). ``bernoulli`` stays index 0.
    """
    return {
        "bernoulli":       list(KP6_NATIVE_BODIES),
        "kp6_wrists":      ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "kp6_ankles":      ["left_ankle_roll_link", "right_ankle_roll_link"],
        "kp6_torso":       ["torso_link"],
        "kp6_left_wrist":  ["left_wrist_yaw_link"],
        "kp6_right_wrist": ["right_wrist_yaw_link"],
    }


def muse_kp5_unified_mode_spec() -> dict[str, list[str]]:
    """5-body (no-pelvis) analogue of :func:`muse_kp6_unified_mode_spec`.

    Identical mode set and ORDER (== :data:`MUSE_KP6_UNIFIED_MODE_NAMES`, length 6) so the
    existing length-6 curriculum / ``mask_mode_probs`` tuples
    (:data:`MUSE_KP6_UNIFIED_BERNOULLI_PROBS` / :data:`MUSE_KP6_UNIFIED_MIX_PROBS`) and
    :class:`MUSEKpDistillationCurriculumCfg` apply UNCHANGED. The only difference: ``bernoulli``
    spans :data:`COTRAIN_KP5_BODIES` (torso + L/R wrist + L/R ankle, **no pelvis**) instead of
    :data:`KP6_NATIVE_BODIES`. The five explicit deploy modes are already pelvis-free and are
    subsets of the 5 bodies, so they are reused verbatim. Pair with encoder ``kp_n_bodies=5``
    and ``commands.motion.body_names = COTRAIN_KP5_BODIES``. The robot base is still reset
    correctly without a pelvis keypoint — see ``commands._resolve_root_full_index`` (the reset
    reads a dedicated full-axis root channel, independent of the tracked body set).
    """
    return {
        "bernoulli":       list(COTRAIN_KP5_BODIES),
        "kp6_wrists":      ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "kp6_ankles":      ["left_ankle_roll_link", "right_ankle_roll_link"],
        "kp6_torso":       ["torso_link"],
        "kp6_left_wrist":  ["left_wrist_yaw_link"],
        "kp6_right_wrist": ["right_wrist_yaw_link"],
    }


# ======================================================================================
# MUSE-Kp5 LATENT-DISTILL demo mask spec (2026-05-22).
#
# 8 deploy-shaped modes for the latent-distillation pretrain AND the KP5 latent-RL finetune
# that warmstarts from it. Same 5 bodies as :data:`COTRAIN_KP5_BODIES` (torso + L/R wrist +
# L/R ankle, NO pelvis) — only the masking patterns change, so the warmstarted KP5 encoder
# (mode-agnostic; visibility is encoded purely via NaN-masked obs) is unaffected. The modes
# are picked to cover the interactive-demo deploy distribution:
#   full        all 5 bodies visible           (full-body mocap drag)
#   vr          wrists + torso                 (VR headset + 2 hand controllers)
#   torso       torso only                     (single-point root drag)
#   left_wrist  left wrist only                (single hand)
#   right_wrist right wrist only               (single hand / writing headline)
#   wrists      both wrists                     (two-hand reach)
#   ankles      both ankles                    (foot targets)
#   bernoulli   per-body bernoulli over all 5  (broad coverage; p_see-scheduled)
#
# Mode ORDER follows the user's deploy list; ``bernoulli`` may sit anywhere (it is index 7
# here) — the keypoint_mask_mode curriculum locates it BY NAME for the p_see ramp. The
# ``mask_mode_probs`` tuples are positional, so pair the spec ONLY with the matching *_PROBS
# constants below. Supersedes :func:`muse_kp5_unified_mode_spec` for the latent-distill +
# KP5-latent-RL training envs (the 6-mode unified spec stays the default for the KP6 envs).
# ======================================================================================
MUSE_KP5_LATENT_DEMO_MODE_NAMES = (
    "full",
    "vr",
    "torso",
    "left_wrist",
    "right_wrist",
    "wrists",
    "ankles",
    "bernoulli",
)

MUSE_KP5_LATENT_DEMO_BERNOULLI_PROBS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
"""Warmup-phase mode_probs (bernoulli-only). Length == len(MUSE_KP5_LATENT_DEMO_MODE_NAMES)=8."""

MUSE_KP5_LATENT_DEMO_MIX_PROBS = (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)
"""Final-phase mode_probs: 0.1 each of the 7 explicit demo modes + 0.3 bernoulli (sums to 1.0)."""

MUSE_KP5_LATENT_DEMO_MIX_PROBS_NO_BERNOULLI = (1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0)


def muse_kp5_latent_demo_mode_spec() -> dict[str, list[str]]:
    """8-mode deploy-shaped spec for latent-distill + KP5 latent-RL (5 bodies, no pelvis).

    Key order == :data:`MUSE_KP5_LATENT_DEMO_MODE_NAMES`. Every body list is a subset of
    :data:`COTRAIN_KP5_BODIES`. ``bernoulli`` spans all 5 bodies; the other seven are
    deterministic only-visible sets. Pair with :data:`MUSE_KP5_LATENT_DEMO_BERNOULLI_PROBS`
    (warmup phases) / :data:`MUSE_KP5_LATENT_DEMO_MIX_PROBS` (final mix phase).
    """
    return {
        "full":        list(COTRAIN_KP5_BODIES),
        "vr":          ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"],
        "torso":       ["torso_link"],
        "left_wrist":  ["left_wrist_yaw_link"],
        "right_wrist": ["right_wrist_yaw_link"],
        "wrists":      ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "ankles":      ["left_ankle_roll_link", "right_ankle_roll_link"],
        "bernoulli":   list(COTRAIN_KP5_BODIES),
    }


def eval_single_mode_spec(name: str) -> tuple[dict[str, list[str]], tuple[float, ...]]:
    """Build a single-mode ``(mask_mode_spec, mask_mode_probs)`` pair for eval-time overrides.

    The cotrain KP encoder is mode-agnostic (visibility is encoded in NaN-masked obs only), so
    replacing the 11-mode training spec with a 1-mode spec at eval time is safe and yields
    deterministic per-env body visibility under ``mask_mode_probs=(1.0,)``.
    """
    if name not in _EVAL_SINGLE_MODES:
        raise ValueError(
            f"Unknown eval mask mode {name!r}. Available: {sorted(_EVAL_SINGLE_MODES.keys())}."
        )
    return {name: list(_EVAL_SINGLE_MODES[name])}, (1.0,)

