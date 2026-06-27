from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Policy cfg: 
# 1. actor-critic teacher:
@configclass
class RslRlPpoActorCriticWithRefVelSkipCfg(RslRlPpoActorCriticCfg):
    """
    Actor-Critic configuration with ref_vel skip connection support.

    When enabled, estimated ref_vel skips the first layer of the policy network
    and connects directly to the second layer.

    Architecture:
        policy_obs → layer1 → layer1_out
        ref_vel ─────────────────────┘
                                      ↓
        [layer1_out, ref_vel] → layer2 → ... → output
    """
    ref_vel_skip_first_layer: bool = False
    """Enable ref_vel skip connection (default: False)."""
    ref_vel_dim: int = 3
    """Dimension of estimated ref_vel (default: 3)."""

@configclass
class RslRlPpoActorCriticTransformerCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticTransformer"
    seq_len: int = 1
    d_model: int = 512
    nhead: int = 4
    num_layers: int = 2
    activation_transformer: str = "gelu"

@configclass
class RslRlPpoActorCriticFSQCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticFSQ"
    num_actor_proprio: int = 1
    encoder_hidden_dims: list[int] = [1024, 1024]
    activation_fsq: str = "elu"
    latent_dim: int = 8
    num_levels: int = 5

@configclass
class RslRlPpoActorCriticVQCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticVQ"
    num_actor_proprio: int = 1
    encoder_hidden_dims: list[int] = [1024, 1024]
    encoder_output_dim: int = 256
    activation_vq: str = "elu"
    num_embeddings: int = 512
    embedding_dim: int = 32
    commitment_weight: float = 0.25
    vq_loss_coef: float = 0.1

@configclass
class RslRlPpoActorCriticAttentionCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticAttention"
    num_actor_proprio: int = 1
    encoder_hidden_dims: list[int] = [1024, 1024]
    activation_attn: str = "elu"
    attention_dim: int = 256
    nhead: int = 4

@configclass
class RslRlResidualActorCriticCfg(RslRlPpoActorCriticCfg):
    """
    Residual Actor-Critic configuration for ResMimic-style residual learning.

    Architecture:
    - GMT policy (frozen): Loaded from checkpoint, provides base actions
    - Residual network (trainable): Learns corrections Δa
    - Final action: a_final = a_gmt + Δa_residual
    """
    class_name: str = "ResidualActorCritic"

    # Residual network configuration
    residual_hidden_dims: list[int] = [512, 256, 128]
    """Hidden dimensions for residual network."""
    residual_last_layer_gain: float = 0.01
    """Xavier initialization gain for last layer (small value for near-zero initial output)."""

    # GMT configuration
    gmt_checkpoint_path: str = MISSING
    """Path to GMT policy checkpoint (.pt file). Required."""
    gmt_policy_cfg: dict | None = None
    """Optional GMT policy architecture config. If None, auto-inferred from checkpoint."""
    init_critic_from_gmt: bool = False
    """Initialize residual critic weights from the GMT checkpoint if dimensions match."""

    # Ref vel estimator configuration
    num_ref_vel_estimator_obs: int | None = None
    """Dimension of ref_vel_estimator observations (e.g., 305). If None, estimator is not used."""
    ref_vel_estimator_checkpoint_path: str | None = None
    """Path to ref_vel estimator checkpoint (.pt file). If None, zero padding is used."""
    ref_vel_estimator_type: str = "mlp"
    """Type of estimator: 'mlp' or 'transformer'."""


# 2. distillation student-teacher: 
@configclass
class RslRlDistillationCfg(RslRlPpoActorCriticCfg):
    class_name: str = "StudentTeacher"
    student_hidden_dims: list[int] = [256, 256, 256]
    teacher_hidden_dims: list[int] = [256, 256, 256]

@configclass
class RslRlKLDistillationAlgorithmCfg:
    """
    KL Distillation algorithm configuration.

    Improved distillation using KL divergence instead of MSE loss.
    This matches MOSAIC's teacher BC approach for better distribution matching.
    """
    class_name: str = "KLDistillation"

    num_learning_epochs: int = 5
    """Number of passes through the dataset per update."""
    gradient_length: int = 15
    """Number of steps to accumulate gradients before optimizer step."""
    learning_rate: float = 1.0e-3
    """Learning rate for student optimizer."""
    loss_type: str = "kl"
    """Loss function type: 'kl' (recommended), 'mse', or 'huber'."""
    kl_reduction: str = "mean"
    """How to reduce KL loss: 'mean' or 'sum'."""

# 3. PULSE: 
@configclass
class RslRlPULSEDistillationCfg(RslRlPpoActorCriticCfg):
    """
    Configuration for PULSE-style VAE distillation.

    Uses LatentBottleneckPULSE: encoder E, decoder D, prior R. R(proprio) -> μ^p, σ^p
    for KL(encoder || prior) in the supervised update .
    """

    class_name: str = "LatentBottleneckPULSE"

    # Observation split: obs = [goal, proprio], goal_dim = num_student_obs - proprio_dim
    proprio_dim: int = MISSING

    # Latent bottleneck (student) hyperparameters
    latent_dim: int = 16
    initialize_std: float = -1.0
    """If >= 0, init encoder/prior std heads near zero weight with bias for this std; -1 disables."""
    latent_predict_std_min: float = 0.01
    """Minimum σ for encoder and prior diagonal latents (log σ is clamped accordingly)."""
    latent_predict_std_max: float = 1.0
    """Maximum σ for encoder and prior diagonal latents (log σ is clamped accordingly)."""
    fixed_prior_std: float | None = None
    """If set, prior diagonal std in KL and default prior rollout uses this fixed σ per dim; the prior MLP still predicts σ for logging."""
    fixed_encoder_std: float | None = None
    """If set, encoder diagonal std σ^e in KL and sampling uses this fixed σ per dim; the encoder log-σ head is omitted (like ``fixed_prior_std`` for the prior)."""
    decoder_use_proprio: bool = True
    """If True, decoder input is ``[z, proprio]`` (default). If False, decoder input is latent ``z`` only."""
    encoder_hidden_dims: list[int] = [256, 128]
    decoder_hidden_dims: list[int] = [256, 128]
    prior_hidden_dims: list[int] = [256, 128]

    # Teacher MLP hyperparameters
    teacher_hidden_dims: list[int] = [512, 256, 128]

@configclass
class RslRlPULSEAdvDistillationCfg(RslRlPULSEDistillationCfg):
    """PULSE student + MLP discriminator on concat(action, proprio) for encoder vs prior decode diagnostics."""

    class_name: str = "LatentBottleneckPULSEAdv"
    disc_hidden_dims: list[int] = [256, 128]
    """Hidden sizes for :class:`~rsl_rl.modules.pulse_action_discriminator.PulseActionDiscriminator`."""

# 4. AnyBody latent distillation:
@configclass
class RslRlAnyBodyLatentDistillationCfg(RslRlPpoActorCriticCfg):
    """
    Configuration for residual-latent AnyBody latent distillation.

    Policy uses `LatentBottleneckAnyBody` in `pulse_residual` mode:
    - frozen teacher encoder (for latent targets),
    - frozen prior and decoder from pretrained PULSE,
    - trainable residual encoder producing delta_mu.
    """

    class_name: str = "LatentBottleneckAnyBody"

    # Observation split: obs = [goal, proprio], proprio_dim = num_student_obs - goal_dim
    proprio_dim: int = MISSING
    # Frozen PULSE ``teacher_core`` input size (goal + proprio). If None, use env ``teacher`` obs dim.
    # Set when the checkpoint encoder width differs from the current ``teacher`` observation vector.
    teacher_encoder_obs_dim: int | None = None
    # Hidden layers of the frozen PULSE encoder inside ``teacher_core`` (e.g. [256, 128]). If None, reuse
    # ``encoder_hidden_dims``. Trainable residual encoder always uses ``encoder_hidden_dims``.
    teacher_encoder_hidden_dims: list[int] | None = None

    latent_dim: int = 16
    encoder_hidden_dims: list[int] = [512, 256, 128]
    decoder_hidden_dims: list[int] = [512, 256, 128]
    prior_hidden_dims: list[int] = [512, 256, 128]

    # Must match the pretrained PULSE ``LatentBottleneckPULSE`` run (see that run's ``params/agent.yaml``).
    # When the teacher used fixed diagonal σ, the encoder / prior omit ``encoder_log_sigma`` / ``log_sigma``
    # in the checkpoint; set these to the same values so ``teacher_core`` / ``prior`` architecture matches.
    latent_predict_std_min: float | None = 0.001
    latent_predict_std_max: float | None = 1.0
    fixed_encoder_std: float | None = None
    fixed_prior_std: float | None = None


# Algorithm cfg: 
# 1. MOSAIC: 
@configclass
class RslRlMOSAICAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """
    MOSAIC algorithm configuration.

    MOSAIC is a plugin-style extension of PPO that adds hybrid learning:
    1. PPO: Standard reinforcement learning (optional)
    2. Offline expert BC: Learn from pre-collected expert trajectories (.npy file)
    3. Online teacher BC: Learn from teacher policy with privileged observations

    This configuration supports all modes:
    - PPO only: use_ppo=True, expert_trajectory_path=None, lambda_teacher_init=0.0
    - PPO + Expert BC: use_ppo=True, expert_trajectory_path=set, lambda_teacher_init=0.0
    - PPO + Teacher BC: use_ppo=True, expert_trajectory_path=None, lambda_teacher_init>0.0
    - Pure Teacher BC: use_ppo=False, expert_trajectory_path=None, lambda_teacher_init>0.0
    - Full MOSAIC: use_ppo=True, expert_trajectory_path=set, lambda_teacher_init>0.0
    """
    class_name: str = "MOSAIC"

    # Mode selection
    hybrid: bool = True
    """True = hybrid mode (random mini-batches, per-batch updates), False = pure BC mode (sequential data, gradient accumulation)."""

    # PPO switch
    use_ppo: bool = True
    """Enable PPO reinforcement learning. Set to False for pure BC mode."""

    # Offline Expert BC parameters
    expert_trajectory_path: str | None = None
    """Path to expert trajectory .npy file for offline BC. Set to None to disable."""
    lambda_off_policy: float = 0.3
    """Initial weight for offline expert BC loss."""
    lambda_off_policy_decay: float = 0.995
    """Decay rate for offline BC weight (1.0 = no decay, 0.995 = slow decay)."""
    lambda_off_policy_min: float = 0.01
    """Minimum offline BC weight after decay."""
    off_policy_batch_size: int = 256
    """Batch size for sampling expert trajectories."""
    expert_allow_repeat_sampling: bool = False
    """Allow sampling with replacement if batch_size > dataset_size."""
    expert_loss_type: str = "mse"
    """Loss function for expert BC: 'kl' (KL divergence) or 'mse' (MSE on action means)."""
    expert_normalize_obs: bool = True
    """Whether to normalize expert observations with student's normalizer."""
    expert_update_normalizer: bool = False
    """Whether expert observations should update normalizer statistics (False=recommended)."""

    # Online Teacher BC parameters
    teacher_checkpoint_path: str | dict[str, str] | None = None
    """Path to teacher checkpoint .pt file. Supports single teacher (str) or multi-teacher (dict: group_name -> path). Required if lambda_teacher_init > 0.0."""
    teacher_obs_source_mapping: dict[str, str] | None = None
    """Maps teacher group names to observation sources for multi-teacher mode. Options: 'policy', 'teacher', 'critic'. Example: {'lafan': 'teacher', 'fld': 'policy'}"""
    teacher_critic_checkpoint_path: str | None = None
    """Path to separate teacher critic checkpoint .pt file. If provided, loads critic weights from this checkpoint."""
    teacher_critic_frozen: bool = True
    """Whether to freeze teacher critic (True=frozen, False=allow fine-tuning). Only applies when teacher_critic_checkpoint_path is provided."""
    train_critic_during_distillation: bool = False
    """Whether to train critic during distillation (use_ppo=False). If True, critic is trained via value loss even when PPO is disabled."""
    lambda_teacher_init: float = 1.0
    """Initial weight for online teacher BC loss. Set to 0.0 to disable."""
    lambda_teacher_decay: float = 0.995
    """Decay rate for teacher BC weight (0.995 = slow decay to encourage early learning)."""
    lambda_teacher_min: float = 0.1
    """Minimum teacher BC weight after decay."""
    teacher_loss_type: str = "mse"
    """Loss function for teacher BC: 'kl' (KL divergence) or 'mse' (MSE on action means)."""

    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    """Number of mini-batches to accumulate gradients before optimizer step. 1 = no accumulation."""

    # Reference Velocity Estimator
    use_estimate_ref_vel: bool = False
    """Whether to use learned reference velocity estimator."""
    ref_vel_estimator_checkpoint_path: str | None = None
    """Path to reference velocity estimator checkpoint (.pt file). Required if use_estimate_ref_vel=True."""
    ref_vel_estimator_type: str = "mlp"
    """Type of velocity estimator: 'mlp' or 'transformer'."""

# 2. PULSE: 
@configclass
class RslRlPULSEAlgorithmCfg:
    """
    PULSE distillation algorithm configuration .

    Loss = action (BC) + weight_kl * KL(encoder || prior) + weight_regularization * regularization.
    Prior R is trained jointly with E and D so encoder produces legitimate latents.
    """

    class_name: str = "PulseDistillation"

    num_learning_epochs: int = 5
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0 # TODO: finetune this
    loss_type: str = "mse"

    weight_kl: float = 0.01
    """Initial β for L_KL (paper: 0.01). Constant if KL annealing disabled."""
    weight_kl_end: float = 0.001
    """Final β after annealing (paper: 0.001)."""
    kl_anneal_start_iter: int = 4000 # TODO: finetune this
    """Iter at which to start decreasing β. 0 with kl_anneal_end_iter=0 → no annealing."""
    kl_anneal_end_iter: int = 8000 # TODO: finetune this
    """Iter at which β reaches weight_kl_end. Paper uses sample counts; we use iters. Set > kl_anneal_start_iter to enable."""
    weight_regularization: float = 0.005 #TODO: finetune this
    """Fixed weight for L_regu = ||μ_t^e - μ_{t-1}^e||_2^2 (temporal smoothness)."""
    kl_loss_upper_bound: float | None = None
    """Cap KL in the training objective; no gradient through KL when raw batch KL exceeds this. None disables capping."""
    teacher_checkpoint_path: str | None = None
    """Path to teacher policy checkpoint (.pt). Set via --teacher_checkpoint or --load_teacher_run. Required for PULSE distillation."""

@configclass
class RslRlAdvPulseAlgorithmCfg(RslRlPULSEAlgorithmCfg):
    """PULSE distillation plus discriminator-only training (no adversarial gradient into E/D/R)."""

    class_name: str = "AdvPulseDistillation"
    disc_lr: float = 1e-3
    """Learning rate for ``policy.action_discriminator``."""
    disc_update_interval: int = 1
    """Run one discriminator update every this many micro-batches (same micro-batches as PULSE)."""

@configclass
class RslRlPriorOnlyPULSEAlgorithmCfg(RslRlPULSEAlgorithmCfg):
    """
    Prior-only PULSE: train only R(proprio) to match encoder latents; E and D are not updated.

    Uses the same rollout collection as PULSE; each iteration only minimizes KL(q_encoder || p_prior)
    with respect to prior parameters (encoder outputs detached).
    """

    class_name: str = "PriorOnlyPulseDistillation"
    weight_prior_fit: float = 1.0
    """Scales the prior-only KL(encoder || prior) loss."""
    num_prior_fit_epochs: int = 5
    """Passes over the buffer per iteration (mirrors ``num_learning_epochs`` for PULSE)."""
    prior_fit_learning_rate: float | None = None
    """Adam LR for the prior; when None, the algorithm uses the same value as ``learning_rate``."""

# 3. MUSE distillation:
@configclass
class RslRlMUSEDistillationCfg(RslRlPpoActorCriticCfg):
    """Configuration for MUSE distillation policy.

    Uses ``LatentBottleneckMUSE``: encoder E and decoder D, no separate prior. The "encoder-as-prior"
    role is learned implicitly via random masking of the goal block (delta-command zero,
    anchor-orientation identity) during training.
    """

    class_name: str = "LatentBottleneckMUSE"

    proprio_dim: int = MISSING

    latent_dim: int = 16
    initialize_std: float = -1.0
    """If >= 0, init encoder log-σ head near zero weight with bias for this std; -1 disables.
    Ignored when ``fixed_encoder_std`` is set."""
    latent_predict_std_min: float = 0.001
    latent_predict_std_max: float = 1.0
    fixed_encoder_std: float | None = None
    """If set, encoder σ is broadcast as this constant per latent dim and the σ head is omitted.
    Recommended for MUSE since there is no KL term to regularize a learned σ — without one, the
    σ head collapses to the lower clamp and wastes capacity. Typical values 0.2-0.5."""
    decoder_use_proprio: bool = True
    encoder_hidden_dims: list[int] = [512, 256, 128]
    decoder_hidden_dims: list[int] = [512, 256, 128]
    teacher_hidden_dims: list[int] = [1024, 1024, 512, 512, 256, 256]


@configclass
class RslRlMUSEAlgorithmCfg:
    """MUSE distillation algorithm: BC + temporal latent regularization. No KL term."""

    class_name: str = "MuseDistillation"

    num_learning_epochs: int = 5
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_type: str = "mse"

    smoothness_type: str = "l2"
    """Which temporal smoothness term to use on encoder μ. Options:
      - "l2"     : ``‖μ_t − μ_{t-1}‖²`` (legacy). NOT scale-invariant — smoothing pressure pushes the
        encoder to grow ‖μ‖ to amplify per-step BC signal; under Kendall this self-reinforces.
      - "cosine" : ``1 − cos(μ_t, μ_{t-1})``. Scale-invariant, bounded in [0, 2]. Penalizes
        directional change only; encoder is free to use any scale.
    Use "cosine" unless reproducing legacy runs."""

    weight_regularization: float = 0.005
    """Fixed weight for the temporal smoothness loss. Ignored when ``use_adaptive_regularization=True``."""

    use_adaptive_regularization: bool = False
    """Kendall-style adaptive weighting for the temporal latent regularization.
    When True, total loss adds ``exp(-s) * reg_loss + α * s`` (s learned, α below) instead of
    ``weight_regularization * reg_loss``. Behavior loss keeps weight 1, so encoder/decoder
    gradient scale is unchanged from the unweighted MUSE recipe."""

    regularization_alpha: float = 0.01
    """Target equilibrium contribution of the regularization term to the total loss when
    ``use_adaptive_regularization=True``. Pick relative to behavior-loss scale; e.g. 0.01 makes
    the reg term contribute ~0.01 to total loss at convergence regardless of raw L_reg scale."""

    regularization_log_var_init: float = 3.0
    """Initial value of s = log(σ²) for the adaptive weight. s=3 → effective weight exp(-3)≈0.05,
    so the regularizer enters gently and the optimizer ramps it as the encoder converges."""

    regularization_log_var_min: float = -3.0
    """Lower clamp on s. Prevents runaway: s→-∞ would make effective weight exp(-s) explode and
    destabilize the encoder. s=-3 caps the weight at exp(3)≈20."""

    regularization_log_var_max: float = 8.0
    """Upper clamp on s. Prevents s→+∞ which would make the weight vanish and silently disable
    the regularization. s=8 floors the effective weight at exp(-8)≈3e-4."""

    # --- KL-against-N(0, I) -------------------------------------------------------
    # Anchors μ at 0 and σ at 1, so the encoder distribution stays bounded. Unlike Kendall on
    # temporal-reg (which can be defeated by μ-drift), Kendall on KL has a well-defined equilibrium
    # because KL's gradient direction is always corrective. The two adaptive terms cooperate.

    weight_kl: float = 0.0
    """Fixed β for L_KL = 0.5 * Σ_d (μ² + σ² − 1 − 2·log σ). Ignored when ``use_adaptive_kl=True``."""

    use_adaptive_kl: bool = False
    """Kendall-style adaptive weighting for the KL term. When True, total loss adds
    ``exp(-s_KL) * KL + α_KL * s_KL`` (s_KL learned, α_KL below). Recommended when
    ``predict_sigma=True`` on the policy — KL keeps σ non-trivial."""

    kl_alpha: float = 0.01
    """Target equilibrium contribution of the KL term to total loss when ``use_adaptive_kl=True``.
    With behavior loss O(1), α=0.01 makes KL contribute ~0.01 — strong enough to anchor scale,
    weak enough to not dominate BC."""

    kl_log_var_init: float = 3.0
    """Initial s_KL. s=3 → effective weight ≈ 0.05 (KL enters gently)."""

    kl_log_var_min: float = -3.0
    """Lower clamp on s_KL (caps effective weight at exp(3) ≈ 20)."""

    kl_log_var_max: float = 4.0
    """Upper clamp on s_KL. Set lower than the temporal-reg cap (4 vs 8): if KL ever saturates,
    eff_weight floors at exp(-4) ≈ 0.018 — still strong enough to pull μ back from drift. The
    reg-term cap is looser because reg loss is naturally smaller; KL can grow fast when μ leaves
    the prior, so a tighter cap is the safety net."""

    teacher_checkpoint_path: str | None = None
    """Path to teacher policy checkpoint (.pt). Set via --teacher_checkpoint or --load_teacher_run."""


# 3b. MUSE Transformer distillation:
@configclass
class RslRlMUSETransformerDistillationCfg(RslRlPpoActorCriticCfg):
    """Configuration for the transformer-encoder MUSE policy.

    Encoder layout (post-refactor, mirrors SAGE-II): 1 un-masked command token + ``H`` proprio
    tokens (+ [CLS]). Env's ``goal_mask_history`` (1-step) drops the command token from attention
    via ``key_padding_mask`` when set. Decoder takes ``[z, proprio_history]`` (full history
    flattened by default; last-frame only if ``decoder_one_step_proprio=True``).
    """

    class_name: str = "LatentBottleneckMUSETransformer"

    proprio_dim: int = MISSING  # kept for cfg-runner compatibility; encoder ignores it.

    history_length: int = 5
    goal_term_sizes: list[int] = [58, 6]
    """Per-frame dims of the goal terms (delta_command, motion_anchor_ori_b) in obs order. G1: [58, 6]."""
    proprio_term_sizes: list[int] = [29, 29, 3, 29]
    """Per-frame dims of the proprio terms (joint_pos, joint_vel, base_ang_vel, last_action). G1: [29, 29, 3, 29]."""
    mask_term_size: int = 1

    latent_dim: int = 16
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    ffn_dim: int = 256
    decoder_hidden_dims: list[int] = [256, 128]
    teacher_hidden_dims: list[int] = [1024, 1024, 512, 512, 256, 256]

    latent_predict_std_min: float = 0.001
    latent_predict_std_max: float = 1.0
    fixed_encoder_std: float = 0.3
    """Used as σ when ``predict_sigma=False`` and also as the bias-init for the σ-head when
    ``predict_sigma=True`` (so early-training σ matches the fixed-σ recipe)."""
    predict_sigma: bool = False
    """If True, encoder gets a ``log_sigma_head`` (Linear d_model → latent_dim) and σ is per-input.
    Only meaningful when the algorithm carries a KL term — otherwise σ collapses to the lower clamp
    and the head wastes capacity. Pair with ``use_adaptive_kl=True`` on the algorithm cfg."""
    encoder_dropout: float = 0.0
    decoder_one_step_proprio: bool = False
    """If True, decoder consumes only the last frame's proprio (90-d for G1) and the latent z must
    carry all history info. If False (default), decoder consumes the full proprio history flattened
    (5×90 = 450-d for G1), matching the MLP-MUSE decoder shape — z still carries history but the
    decoder gets the redundant raw history as a safety net."""
    latent_normalize: bool = False
    """If True, L2-normalize z onto the unit hypersphere before the decoder (and return normalized
    μ from ``forward_for_update`` so the algorithm's smoothness term operates on the same geometry).
    Decouples direction from magnitude and makes cosine the natural metric. Pair with
    ``smoothness_type="cosine"`` and a fixed regularization weight."""
    deterministic_encoder: bool = False
    """If True, every forward path uses z := μ — no reparameterization noise, on rollout or update.
    Recommended for input-modal conditioning regimes (no output-sampling diversity goal)."""


# 3d. MUSE-Kp distillation: KP-token encoder warmstarted from MUSE-Transformer.
@configclass
class RslRlMUSEKpDistillationCfg(RslRlPpoActorCriticCfg):
    """Configuration for the MUSE-Kp policy: per-body KP-token encoder + MUSE-Transformer body.

    Built on top of the stabilized MUSE-Transformer recipe (deterministic encoder + unit-norm
    latent + cosine smoothness regularization on μ). The encoder ingests N_kp KP tokens (each
    packing one body's L-frame position sequence) plus H proprio tokens. The transformer body /
    proprio projections / decoder are shape-identical to :class:`RslRlMUSETransformerDistillationCfg`
    so a MUSE-Transformer checkpoint warmstarts cleanly via shape-filtered ``load_state_dict``.

    Distillation source is the **MLP teacher** (loaded into ``self.teacher`` from the MUSE
    checkpoint); the loss is BC + cosine smoothness — there is **no** latent matching against the
    MUSE encoder.
    """

    class_name: str = "LatentBottleneckMUSEKp"

    proprio_dim: int = MISSING  # placeholder for cfg-runner compatibility; encoder ignores it.

    history_length: int = 5
    proprio_term_sizes: list[int] = [29, 29, 3, 29]
    """Per-frame dims of the proprio terms (joint_pos, joint_vel, base_ang_vel, last_action). G1: [29, 29, 3, 29]."""

    # KP encoder shape.
    kp_n_bodies: int = 14
    kp_lookahead_steps: int = 12
    """Per-body PACKED slot count consumed by the KP-token Linear projection. Must equal the slot
    count of the selected :attr:`kp_layout`. Defaults to 12 (the log-spaced 0.5s cap layout: 3
    history + 1 abs + 8 future). Each KP token carries kp_lookahead_steps * 3 features."""

    kp_layout: str = "log_0_5s"
    """Named slot layout consumed by the KP encoder. One of:
      - ``"legacy"``: 11 slots, dense layout (1 abs + 10 consecutive future deltas at 50Hz).
        For backwards compat / resuming pre-2026-05-15 MUSE-Kp checkpoints without warmstart.
      - ``"log_0_5s"``: 12 slots, 3 history + 1 abs + 8 future, log-spaced, cap=0.5s lookahead.
        Default. Use for the cursor-drag deployment scenario (test-time buffer ≈ 0.5s).
      - ``"log_1_0s"``: 13 slots, 3 history + 1 abs + 9 future, log-spaced, cap=1.0s lookahead.
        Fallback for scenarios with a longer reliable future buffer.

    The encoder's :meth:`load_state_dict` does slot-aware kp_proj column remapping when loading a
    checkpoint that was trained on a different layout; shared slots copy, new slots stay at the
    current model's small random init."""

    aux_predictor_enabled: bool = False
    """Build the :class:`AuxAnchorPredictor` sibling head off the encoder latent + last-frame
    proprio. Required for Run A (probe) and Run B (encoder-pressure). When False, no aux head
    is created and the algorithm's ``aux_lambda`` is ignored."""

    # Transformer body — defaults match G1FlatMUSETransformerDistillationRunnerCfg.policy so
    # warmstart from a MUSE-Transformer checkpoint is shape-clean.
    latent_dim: int = 16
    d_model: int = 192
    nhead: int = 4
    num_layers: int = 2
    ffn_dim: int = 768
    decoder_hidden_dims: list[int] = [1024, 512, 256, 128]
    teacher_hidden_dims: list[int] = [1024, 1024, 512, 512, 256, 256]

    activation: str = "gelu"
    latent_predict_std_min: float = 0.001
    latent_predict_std_max: float = 10.0
    fixed_encoder_std: float = 1.0
    """Used as σ for log-σ diagnostics; deterministic_encoder=True bypasses sampling regardless."""
    encoder_dropout: float = 0.0
    decoder_one_step_proprio: bool = False
    latent_normalize: bool = True
    deterministic_encoder: bool = True

    freeze_mode: str = "none"
    """One of ``"none"`` (full encoder + decoder train; recommended default per user),
    ``"decoder_only"`` (decoder frozen; encoder trains), or ``"decoder_plus_shared_encoder"``
    (decoder + MUSE-shared encoder pieces frozen; only KP-only layers train — useful as a warmstart
    phase before unfreezing). Teacher MLP is always frozen."""


@configclass
class RslRlMUSEKpAlgorithmCfg:
    """MUSE-Kp distillation algorithm: aliased :class:`MuseDistillation` (BC + cosine smoothness).

    Same loss, same forward contract, same gradient-accumulation logic as
    :class:`RslRlMUSEAlgorithmCfg`. Distinct ``class_name`` so logs / runner branching are explicit.
    """

    class_name: str = "MuseKpDistillation"

    num_learning_epochs: int = 5
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_type: str = "mse"

    smoothness_type: str = "cosine"
    """Cosine smoothness on the unit-norm μ trajectory (matches MUSE-Transformer recipe)."""

    weight_regularization: float = 0.1
    """Fixed weight; with cosine smoothness in [0, 2] the contribution is at most 0.2."""

    use_adaptive_regularization: bool = False
    regularization_alpha: float = 0.01
    regularization_log_var_init: float = 3.0
    regularization_log_var_min: float = -3.0
    regularization_log_var_max: float = 8.0

    weight_kl: float = 0.0
    use_adaptive_kl: bool = False
    kl_alpha: float = 0.01
    kl_log_var_init: float = 3.0
    kl_log_var_min: float = -3.0
    kl_log_var_max: float = 4.0

    # --- Freeze-mode warmup curriculum -----------------------------------------------------------
    # Pin the policy in a partially-frozen mode for the first ``warmup_freeze_iters`` updates so
    # the new KP-input layers can catch up to the warmstarted decoder; then transition to
    # ``post_warmup_freeze_mode`` and rebuild the optimizer over the new param set.

    warmup_freeze_iters: int = 0
    """Number of PPO iterations to hold the policy in ``warmup_freeze_mode`` before unfreezing.
    Set to 0 to disable the warmup (use the policy cfg's ``freeze_mode`` for the entire run)."""

    warmup_freeze_mode: str = "decoder_only"
    """Freeze mode active during the warmup phase. One of ``"decoder_only"``,
    ``"decoder_plus_shared_encoder"``, or ``"none"``."""

    post_warmup_freeze_mode: str = "none"
    """Freeze mode after warmup ends; the optimizer is rebuilt over the corresponding param set."""

    # --- Aux anchor predictor (optional) ---------------------------------------------------------
    # Run A (probe, ``aux_stop_grad=True``) trains the predictor head from a detached latent — no
    # encoder pressure, just diagnoses how much speed/heading info is already in the latent.
    # Run B (encoder-pressure, ``aux_stop_grad=False``) lets aux gradient flow back into z.
    # Requires the paired :attr:`RslRlMUSEKpDistillationCfg.aux_predictor_enabled` = True.

    aux_lambda: float = 0.0
    """Weight on the aux loss (speed_smoothL1 + ori_smoothL1, both on EMA-normalized targets).
    0.0 → aux disabled (no extra forward pass, no contribution). Recommended starting value when
    enabled: 0.1 → contributes ~5-10%% of distillation loss at convergence."""

    aux_stop_grad: bool = True
    """``True`` (Run A, probe): detach z + last-frame proprio before the aux predictor — encoder
    receives no aux gradient. ``False`` (Run B): aux gradient flows back into z via the encoder's
    second forward pass."""

    aux_target_ema_decay: float = 0.99
    """EMA decay for the running mean/std of aux targets (used to z-normalize the loss so a single
    ``aux_lambda`` controls both heads' contribution)."""

    # Teacher-obs layout: required to slice aux targets from ``privileged_observations`` at
    # training time. The defaults match :class:`MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg.TeacherCfg`
    # (sonic_55k teacher contract, per-frame layout):
    #   [command(2J=58), motion_anchor_ori_b(6), joint_pos(J=29), joint_vel(J=29), base_ang_vel(3),
    #    actions(J=29), motion_anchor_pos_b(3), base_lin_vel(3), ref_base_lin_vel(3)] = 163 dims.
    # The aux targets are motion_anchor_ori_b (term index 1) and ref_base_lin_vel (term index 8).
    teacher_term_sizes: list[int] = [58, 6, 29, 29, 3, 29, 3, 3, 3]
    teacher_history_length: int = 5
    aux_ori_term_index: int = 1
    aux_speed_term_index: int = 8

    # --- Passive→active pilot anneal (DAgger-style β schedule) -----------------------------------
    # Per-env per-rollout Bernoulli(teacher_fraction) controls who drives the env; the supervision
    # target stays the teacher throughout (BC loss recomputes student action from stored obs).
    pilot_anneal_enabled: bool = False
    """Enable the teacher→student pilot anneal. False → student pilots from iter 0 (legacy
    on-policy distillation)."""

    pilot_teacher_start_iter: int = 1500
    """teacher_fraction held at 1.0 (pure teacher-pilot bootstrap) until this learning iteration.
    Align with the mask curriculum's M1→M2 transition so the mask is at final difficulty before
    the pilot starts shifting (one new hardness at a time)."""

    pilot_teacher_end_iter: int = 4000
    """teacher_fraction reaches ``pilot_teacher_floor`` by this iteration; held at the floor after."""

    pilot_teacher_floor: float = 0.08
    """Steady fraction of teacher-piloted envs kept to the end — a trickle of clean on-distribution
    states so the student can't drift into a self-confirming distribution."""

    pilot_anneal_power: float = 2.0
    """Anneal shape exponent: teacher_fraction = floor + (1-floor)·(1-t)^power over the window.
    1.0 = linear; >1.0 = convex (stay teacher-heavy through the first part, then drop faster)."""

    # --- Teacher-eval inspection mode (diagnostic) -----------------------------------------------
    teacher_eval_mode: bool = False
    """Diagnostic: when True, the teacher drives every env every step and ``update()`` is a no-op
    (storage clear only) — the student is never trained. The runner's rollout logging
    (``Train/mean_reward``, ``Train/mean_episode_length``, ``Episode/mdp_termination_success_rate``)
    then measures the teacher's own task performance under this env config: the achievable ceiling
    and a sanity check on the loaded teacher checkpoint. Overrides ``warmup_freeze_iters`` and
    ``pilot_anneal_enabled``. Enable from the CLI with
    ``agent.algorithm.teacher_eval_mode=true``. Runner checkpoints saved during such a run hold the
    untrained student — discard them (or set a large ``save_interval``)."""

    teacher_checkpoint_path: str | None = None
    """Path to the **PHC+ stage-1** checkpoint (.pt). Loaded by the runner into:
      - ``self.teacher`` (via the ``actor.*`` path) — the action-target source
      - ``teacher_obs_normalizer`` (via the ckpt's ``obs_norm_state_dict``) — sized for PHC+'s
        770-d teacher input.
    NOT a MUSE-Transformer checkpoint: that one's ``obs_norm_state_dict`` is the MUSE student
    normalizer (515-d) and would shape-mismatch ``teacher_obs_normalizer``. Use
    ``encoder_decoder_warmstart_checkpoint_path`` for the MUSE-T weight warmstart instead."""

    encoder_decoder_warmstart_checkpoint_path: str | None = None
    """Path to a **MUSE-Transformer** checkpoint (.pt). Loaded AFTER ``teacher_checkpoint_path``;
    only the ``model_state_dict`` is consumed (no normalizer touched, so the PHC+-loaded
    ``teacher_obs_normalizer`` survives intact). Warmstarts shared encoder + decoder + re-loads
    ``teacher.*`` (no-op since MUSE-T's ``teacher.*`` IS PHC+ weights). KP-only layers (``kp_proj``,
    ``body_id_emb``) keep their small random init."""


# 3d''. MUSE-Kp LATENT-space distillation: KP encoder distilled against a frozen JC MUSE-T
#       teacher in latent space, with the shared decoder frozen.
@configclass
class RslRlMUSEKpLatentDistillationCfg(RslRlMUSEKpDistillationCfg):
    """Policy cfg for MUSE-Kp latent-space distillation (frozen JC MUSE-T teacher + frozen
    shared decoder).

    Subclasses :class:`RslRlMUSEKpDistillationCfg` so every KP-encoder / backbone / decoder
    shape field stays in lockstep with the MUSE-Kp recipe and the JC MUSE-T warmstart. The JC
    teacher encoder is built shape-identical to the backbone with the JC front-end term sizes.

    KEY (2026-05-18): ``latent_normalize=True`` — the JC MUSE-T teacher now trains with the
    unit-norm cosine recipe, so the frozen decoder consumes the L2-normalized z and the
    latent-alignment target is z_jc = normalize(μ_jc); the student is matched as
    z_kp = normalize(μ_kp). The policy class FORCES this regardless of the cfg.
    """

    class_name: str = "LatentBottleneckMUSEKpLatent"

    latent_normalize: bool = True
    """The JC MUSE-T teacher uses unit-norm z (latent_normalize=True); the frozen decoder
    consumes the L2-normalized z, so student z_kp and target z_jc are matched on the unit
    sphere. MUST equal the JC teacher ckpt's latent_normalize (the policy forces True)."""
    deterministic_encoder: bool = True

    freeze_mode: str = "decoder_only"
    """Decoder is ALWAYS frozen in latent distillation (it's the JC teacher's near-perfect motor
    prior). ``decoder_only`` = decoder frozen, full KP encoder (incl. warmstarted backbone)
    trains. The algorithm's freeze-warmup may tighten this to
    ``decoder_plus_shared_encoder`` for the first N iters, then relax back to ``decoder_only``."""

    # JC teacher front-end term sizes (the standalone MUSE-Transformer obs schema). Must match the
    # JC MUSE-T checkpoint's training obs: delta_command(58) + motion_anchor_ori_b(6) +
    # motion_anchor_pos_b(3) + base_lin_vel(3) + ref_base_lin_vel(3); proprio (29,29,3,29) H5;
    # 1-step goal mask bit (held at 0 — the JC teacher is the privileged clean teacher).
    jc_goal_term_sizes: list[int] = [58, 6, 3, 3, 3]
    jc_proprio_term_sizes: list[int] = [29, 29, 3, 29]
    jc_mask_term_size: int = 1


@configclass
class RslRlMUSEKpLatentDistillationAlgorithmCfg:
    """MUSE-Kp latent-space distillation algorithm.

    Loss = ``weight_latent`` · (1 - cos(z_kp, sg(z_jc)))  [unit-sphere; latent_loss_type] +
           ``weight_behavior`` · mse(decoder(z_kp), sg(action_jc))  [frozen shared decoder].
    Teacher-pilot warmup (JC teacher drives the env early) + freeze-mode warmup.
    """

    class_name: str = "MuseKpLatentDistillation"

    num_learning_epochs: int = 5
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_type: str = "mse"
    """Metric for the BEHAVIOR (action-anchor) term: mse | huber."""

    latent_loss_type: str = "cosine"
    """Metric for the LATENT term: ``cosine`` (default; 1 - cos(z_kp, z_jc)) | mse | huber.
    Under unit-norm latents MSE = (2-2cos)/latent_dim saturates near alignment (vanishing
    gradient exactly where the frozen decoder needs fidelity); cosine is the unit-sphere-
    natural metric (same one the JC teacher trains with), scale ~[0,2] with stable gradient."""

    weight_latent: float = 1.0
    """Primary supervision: match student z_kp to the frozen JC teacher's z_jc (unit sphere)."""

    weight_behavior: float = 0.05
    """Small action anchor through the frozen shared decoder — keeps z_kp from drifting into
    action-irrelevant directions the frozen decoder ignores. Trimmed 0.1→0.05 once the latent
    term was put on the cosine (well-scaled) metric so it stays the dominant signal. 0.0 → pure
    latent."""

    teacher_pilot_warmup_iters: int = 200
    """JC teacher drives the env for the first N iters (on-distribution states while the fresh KP
    front-end catches up), then a hard switch to student-pilot. Supervision is the JC teacher
    throughout. 0 → student pilots from iter 0."""

    # --- Freeze-mode warmup ----------------------------------------------------------------------
    warmup_freeze_iters: int = 200
    """Hold the policy in ``warmup_freeze_mode`` for the first N iters, then switch to
    ``post_warmup_freeze_mode`` and rebuild the optimizer. 0 → no warmup (use the policy cfg's
    ``freeze_mode`` for the whole run). Default aligned with ``teacher_pilot_warmup_iters`` so the
    KP front-end learns under the clean teacher-piloted distribution before the backbone moves."""

    warmup_freeze_mode: str = "decoder_plus_shared_encoder"
    """During warmup: decoder + shared backbone frozen, only kp_proj/body_id_emb/modality train
    (the fresh KP front-end catches up before perturbing the warmstarted backbone)."""

    post_warmup_freeze_mode: str = "decoder_only"
    """After warmup: decoder stays frozen forever; the full KP encoder (incl. backbone) trains."""

    teacher_checkpoint_path: str | None = None
    """Path to the **JC MUSE-Transformer** checkpoint (.pt). The runner loads it into:
      - ``jc_encoder`` (frozen teacher) + shared ``decoder`` (frozen) via the policy's
        ``load_state_dict`` (``transformer_encoder.*`` is replicated onto ``jc_encoder.*`` and
        also warmstarts the KP backbone),
      - ``teacher_obs_normalizer`` (frozen) via the ckpt's ``obs_norm_state_dict``,
      - the student proprio normalizer slice (frozen, copied at the JC-goal offset).
    A single checkpoint drives the whole pipeline — no separate warmstart flag needed."""

    encoder_decoder_warmstart_checkpoint_path: str | None = None
    """Unused by default (the single ``teacher_checkpoint_path`` already warmstarts the KP
    backbone). Present for CLI compatibility; if set, the runner re-applies it as a
    teacher-stripped weight warmstart after the teacher load."""


# 3d'. MUSE-Kp Latent-space RL finetune: PPO over the MUSE-Kp latent (frozen decoder).
@configclass
class RslRlLatentRLActorCriticCfg(RslRlMUSEKpDistillationCfg):
    """Policy cfg for latent-space RL finetuning of a distilled MUSE-Kp policy.

    Subclasses :class:`RslRlMUSEKpDistillationCfg` so every encoder/decoder shape
    field stays in lockstep with the distilled MUSE-Kp checkpoint that is
    warmstarted (the policy wraps a :class:`LatentBottleneckMUSEKp`). Adds the
    latent-RL knobs. The module FORCES ``deterministic_encoder=True`` +
    ``latent_normalize=True`` + the adapter-implied freeze_mode regardless of the
    inherited values, so the inherited distillation defaults are harmless.
    """

    class_name: str = "LatentRLActorCritic"

    adapter: str = "full_ft"
    """Encoder-adaptation method: ``"full_ft"`` (M1/M1b — encoder trains, decoder
    frozen), ``"lora"`` (M3 — base frozen, low-rank adapters train),
    ``"residual"`` (M4, built separately)."""

    # --- LoRA adapter (M3); used only when adapter == "lora" ---
    lora_rank: int = 8
    """LoRA rank r (capacity knob; sweep {4, 8, 16})."""
    lora_alpha: float | None = None
    """LoRA scaling = alpha / r. None ⇒ alpha = r (scaling 1.0)."""
    lora_targets: list[str] = ["attn_qkv", "attn_out", "mu_head"]
    """Encoder weights to LoRA. Default = decision's primary set (attention
    q/k/v + o) plus the high-leverage final latent projection. Ablatable tokens:
    ``"ffn"`` (block FFN linears), ``"kp_proj"`` (per-body KP input proj)."""

    init_latent_std: float = 0.1
    """State-independent Gaussian std over the unit-norm latent (decision D1).
    Small so step-0 ≈ the distilled deterministic policy (safe start)."""

    # --- obstacle obs (option C); 0 = disabled (writing / general latent-RL unchanged) ---
    obstacle_feat_dim: int = 0
    """Total obstacle dims appended LAST to the policy obs (= obstacle_n × per-box; e.g.
    MAX_OBSTACLES(5) × 7 = 35). LatentRLActorCritic strips these before the frozen encoder and
    routes them to the residual corrector. MUST equal the env's obstacle obs-term dim."""
    obstacle_n: int = 0
    """Number of obstacle boxes (per-box tokens) in the obstacle block."""

    num_teacher_obs: int = 815
    """Inner MUSE-Kp teacher-MLP input dim (sonic_55k contract: 163×5). Only so
    the warmstart ckpt's ``teacher.*`` slot loads shape-clean; the teacher MLP is
    unused in RL (no BC)."""

    critic_hidden_dims: list[int] = [512, 256, 128]
    """Fresh value-MLP hidden sizes. M1: symmetric critic on the student obs;
    asymmetric/privileged critic is a later enhancement."""
    critic_activation: str = "elu"

    # --- residual adapter (M4 / decision D2; only used when adapter="residual") ---
    # g_φ is a SHALLOW per-body-token transformer (handles KP masking
    # structurally via key_padding_mask; ~12× fewer params than a flat MLP at
    # the default size). It sees the encoder's split tokens + the encoder's
    # unit-norm μ̂ (as a conditioning token) -> Δz; mean = normalize(μ̂ + α·Δz).
    residual_d_model: int = 64
    residual_num_layers: int = 1
    residual_nhead: int = 4
    residual_ffn: int = 128
    """Default ~45k params. Documented larger combo for more upper-body-under-
    mask capacity: d_model=96, num_layers=2, nhead=4, ffn=256 (~191k, still
    ~3× under the old MLP)."""
    residual_last_layer_gain: float = 0.01
    """Xavier gain on g_φ's Δz head — small ⇒ Δz≈0 at init ⇒ step-0 == distilled
    policy (safe start; residual needs no D3 prior anchor)."""
    residual_alpha: float = 1.0
    """Scalar on Δz in ``normalize(μ̂ + α·Δz)``."""


@configclass
class RslRlLatentPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Latent-space PPO. Stock PPO update; distinct ``class_name`` so the runner
    resolves ``training_type="latent_rl"`` (latent-sized rollout storage + decode
    latent→joint before ``env.step``). The prior-anchor term (decision D3) lands
    here in M1b."""

    class_name: str = "LatentPPO"

    critic_warmup_itrs: int = 200
    """Critic-only warmup before actor updates (decision: no teacher-critic init;
    warm the fresh critic ~200 iters). Consumed by ``PPO.update`` via the runner's
    per-iter ``self.alg.current_learning_iteration``."""

    prior_anchor_coef: float = 0.0
    """Decision D3 prior anchor. ``0.0`` = unanchored (M1 default — identical to
    stock PPO). ``>0`` = the *anchored* full_ft run: subtract
    ``coef·(1 − cos(μ, μ_distilled))`` from the per-step reward (reward-side
    operationalization in ``LatentPPO.process_env_step``). Run BOTH (D3) to keep
    the anchored-vs-unanchored comparison honest. Typical anchored value ~0.05."""

    encoder_decoder_warmstart_checkpoint_path: str | None = None
    """Distilled MUSE-Kp ``.pt`` to warmstart the policy (encoder + decoder +
    frozen teacher slot). Set via the ``--encoder_decoder_warmstart`` CLI flag;
    read by the runner's ``latent_rl`` warmstart block. Absorbed by
    ``LatentPPO.__init__`` (not a stock-PPO param)."""

    teacher_checkpoint_path: str | None = None
    """Unused for RL (no BC); present so shared CLI/cfg plumbing that may set it
    doesn't fail. Absorbed by ``LatentPPO.__init__``."""


# 3e. MUSE Co-Train distillation: dual-frontend (joint-cmd + KP) sharing one transformer backbone.
@configclass
class RslRlMUSECoTrainDistillationCfg(RslRlPpoActorCriticCfg):
    """MUSE co-train policy: shared transformer backbone + JC frontend (``goal_proj``) + KP
    frontend (``kp_proj`` + ``body_id_emb``). Both modalities forward on every batch; 50/50
    random partition picks the env-piloting action source per env per step.

    Backbone shape (``d_model`` / ``nhead`` / ``num_layers`` / ``ffn_dim`` / ``decoder_hidden_dims``)
    must match :class:`RslRlMUSETransformerDistillationCfg` so a MUSE-Transformer checkpoint
    warmstarts cleanly via shape-filtered ``load_state_dict``.

    KP-side defaults: ``kp_layout="log_0_5s"`` ⇒ 12 sparse log-spaced slots (3 history + 1 abs +
    8 future, cap 0.5 s). The per-body KP token packs ``len(layout)*3 = 36`` dims (slot 0 = abs
    ref pos; others = deltas, so current actual is implicit — no separate FK token) and projects
    to ``d_model`` via the single ``kp_proj`` Linear. Mirrors :class:`RslRlMUSEKpDistillationCfg`
    so a cotrained KP encoder transfers to the canonical MUSE-Kp deployment config.
    """

    class_name: str = "LatentBottleneckMUSECoTrain"

    proprio_dim: int = MISSING  # placeholder for cfg-runner compatibility; encoder ignores it.

    history_length: int = 5
    proprio_term_sizes: list[int] = [29, 29, 3, 29]
    """Per-frame dims of the proprio terms (joint_pos, joint_vel, base_ang_vel, last_action)."""

    # JC modality (privileged: + motion_anchor_pos_b so JC is world-frame-faithful — a
    # latent-bottlenecked teacher / gold align target, not a deployable modality).
    jc_goal_term_sizes: list[int] = [58, 6, 3]
    """Per-frame dims of the JC goal terms (delta_command, motion_anchor_ori_b, motion_anchor_pos_b)."""
    jc_mask_term_size: int = 1

    # KP modality (6-body KP6 native set + sparse log-spaced slot pack; mirrors RslRlMUSEKpDistillationCfg).
    kp_n_bodies: int = 6
    kp_lookahead_steps: int = 12
    kp_layout: str = "log_0_5s"

    # Backbone — defaults match G1FlatMUSETransformerDistillationRunnerCfg.policy.
    latent_dim: int = 16
    d_model: int = 192
    nhead: int = 4
    num_layers: int = 2
    ffn_dim: int = 768
    decoder_hidden_dims: list[int] = [1024, 512, 256, 128]
    teacher_hidden_dims: list[int] = [1024, 1024, 512, 512, 256, 256]

    activation: str = "gelu"
    latent_predict_std_min: float = 0.001
    latent_predict_std_max: float = 10.0
    fixed_encoder_std: float = 1.0
    encoder_dropout: float = 0.0
    decoder_one_step_proprio: bool = False
    latent_normalize: bool = True
    deterministic_encoder: bool = True

    pilot_kp_fraction: float = 0.5
    """Fraction of envs piloted by the KP-modality action per step. 0.0 = JC-only piloting,
    1.0 = KP-only piloting; default 0.5 = random per-env per-step partition."""


@configclass
class RslRlMUSECoTrainAlgorithmCfg:
    """MUSE co-train algorithm: 2× BC + 2× cosine smoothness + 1× cross-modal alignment.

    Smoothness defaults follow the locked-in MUSE-Transformer recipe (cosine, fixed weight,
    deterministic encoder + unit-norm latent). The added knob is ``weight_align``: λ on the
    ``1 − cos(μ_jc, μ_kp)`` cross-modal alignment term.
    """

    class_name: str = "MuseCoTrainDistillation"

    num_learning_epochs: int = 5
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_type: str = "mse"

    smoothness_type: str = "cosine"
    weight_regularization_jc: float = 0.1
    """Smoothness weight applied to the JC modality temporal-μ regularization term. The JC
    frontend is warmstarted and stable; keeping it regularized prevents drift."""

    weight_regularization_kp: float = 0.01
    """Smoothness weight applied to the KP modality temporal-μ regularization term. Lower than
    JC so the (still-learning) KP encoder can adapt to mask-mode changes without over-constraint."""

    use_adaptive_regularization: bool = False
    regularization_alpha: float = 0.01
    regularization_log_var_init: float = 3.0
    regularization_log_var_min: float = -3.0
    regularization_log_var_max: float = 8.0

    weight_kl: float = 0.0
    use_adaptive_kl: bool = False
    kl_alpha: float = 0.01
    kl_log_var_init: float = 3.0
    kl_log_var_min: float = -3.0
    kl_log_var_max: float = 4.0

    weight_align: float = 0.1
    """Weight on the **asymmetric** cross-modal alignment term ``1 − cos(sg(μ_jc), μ_kp)`` (μ_jc
    detached in the algorithm — KP chases the privileged JC latent as a fixed gold target). v1:
    fixed 0.1, no mask-gating, no warmup — the floor-only corner of the floor+gated design; 0.1
    doubles as the decoder-freeze-safety manifold keeper. Range [0, 2]; 0.0 disables alignment."""

    teacher_checkpoint_path: str | None = None
    """Path to the **PHC+ stage-1** checkpoint (.pt). Loaded by the runner into ``self.teacher``
    via the ``actor.*`` path and into ``teacher_obs_normalizer``. Same convention as MUSE-KP."""

    encoder_decoder_warmstart_checkpoint_path: str | None = None
    """Path to a **MUSE-Transformer** checkpoint (.pt). Loaded AFTER ``teacher_checkpoint_path``;
    warmstarts the shared backbone + JC ``goal_proj`` + decoder. KP-only layers (``kp_proj``,
    ``body_id_emb``) keep random init; ``modality_emb`` slot [1] (KP) keeps random init while
    slots [0] (JC) and [2] (proprio) are copied from the MUSE-T checkpoint."""


# 4. AnyBody latent distillation:
@configclass
class RslRlAnyBodyLatentDistillationAlgorithmCfg:
    """
    AnyBody latent residual-latent distillation algorithm.

    Main supervision is latent matching (teacher mu_e vs student mu_prior + delta_mu).
    Optional behavior loss can be used as a weak regularizer.
    """

    class_name: str = "AnyBodyLatentDistillation"

    num_learning_epochs: int = 5
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_type: str = "mse"
    weight_latent: float = 1.0
    weight_behavior: float = 0.0
    teacher_checkpoint_path: str | None = None
    """Path to PULSE checkpoint (.pt). student_core becomes teacher. Set via --teacher_checkpoint or --load_teacher_run."""
