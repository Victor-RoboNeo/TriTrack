from __future__ import annotations

import os

import torch
import warnings

from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from rsl_rl.modules.latent_bottleneck_pulse import _PULSEPrior

class ResidualLatentAction(ActionTerm):
    """Predict residual latent, decode via frozen prior+decoder to joint targets."""

    cfg: "ResidualLatentActionCfg"
    @staticmethod
    def _assert_state_dict_finite(state_dict: dict[str, torch.Tensor], prefix: str) -> None:
        """Fail fast if checkpoint tensors contain NaN/Inf."""
        bad_entries: list[str] = []
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            if not torch.isfinite(value).all():
                bad = int((~torch.isfinite(value)).sum().item())
                bad_entries.append(f"{key} (shape={tuple(value.shape)}, bad={bad})")
        if bad_entries:
            preview = ", ".join(bad_entries[:8])
            if len(bad_entries) > 8:
                preview += f", ... (+{len(bad_entries) - 8} more)"
            raise ValueError(
                f"Checkpoint contains non-finite tensors in {prefix}: {preview}. "
                "Use a different checkpoint (or re-export one with finite weights)."
            )

    def __init__(self, cfg: "ResidualLatentActionCfg", env):
        self._action_dim = int(cfg.latent_dim)
        super().__init__(cfg, env)
        self._env = env
        self._joint_action = JointPositionAction(cfg, env)

        self._joint_ids = getattr(self._joint_action, "_joint_ids", None)
        self._num_joint_actions = int(self._joint_action.action_dim)

        self._feature_dim = self._num_joint_actions * 3 + 3   # joint_pos, joint_vel, base_ang_vel, last_action
        self._history_length = int(cfg.proprio_history_length)
        self._proprio_dim = self._feature_dim * self._history_length
        self._proprio_history = torch.zeros((self.num_envs, self._history_length, self._feature_dim), device=self.device)
        self._last_joint_action = torch.zeros((self.num_envs, self._num_joint_actions), device=self.device)

        if not cfg.prior_checkpoint or not os.path.isfile(cfg.prior_checkpoint):
            raise FileNotFoundError(
                f"ResidualLatentAction requires `prior_checkpoint`; got {cfg.prior_checkpoint!r}."
            )

        # Torch 2.6 defaults weights_only=True; these checkpoints store structured payloads.
        checkpoint = torch.load(cfg.prior_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a dict checkpoint for prior+decoder weights.")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint must contain `model_state_dict` or be a raw state dict.")
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

        # Optional frozen proprio normalizer loaded from checkpoint stats.
        self._proprio_norm_mean: torch.Tensor | None = None
        self._proprio_norm_std: torch.Tensor | None = None
        self._proprio_norm_eps = float(getattr(cfg, "proprio_norm_eps", 1.0e-2))

        # 1. initialize prior model
        self._prior = _PULSEPrior(
            proprio_dim=self._proprio_dim,
            latent_dim=int(cfg.latent_dim),
            hidden_dims=list(cfg.prior_hidden_dims),
            activation=cfg.activation,
            latent_sigma_min=cfg.latent_predict_std_min,
            latent_sigma_max=cfg.latent_predict_std_max,
            fixed_prior_std=cfg.fixed_prior_std,
        ).to(self.device)

        # 2. initialize decoder model
        decoder_layers: list[torch.nn.Module] = []
        act_cls = getattr(torch.nn, cfg.activation.upper(), torch.nn.ELU)
        in_dim = int(cfg.latent_dim) + self._proprio_dim
        for hidden in cfg.decoder_hidden_dims:
            decoder_layers.append(torch.nn.Linear(in_dim, int(hidden)))
            decoder_layers.append(act_cls())
            in_dim = int(hidden)
        decoder_layers.append(torch.nn.Linear(in_dim, self._num_joint_actions))
        self._decoder = torch.nn.Sequential(*decoder_layers).to(self.device)

        # 3. load prior model weights
        prior_sd = {k[len("prior.") :]: v for k, v in state_dict.items() if k.startswith("prior.")}
        if not prior_sd:
            raise ValueError("Checkpoint does not contain `prior.*` parameters.")
        self._assert_state_dict_finite(prior_sd, "prior.*")
        self._prior.load_state_dict(prior_sd, strict=True)

        # 4. load decoder model weights
        decoder_sd = {k[len("student_core.decoder.") :]: v for k, v in state_dict.items() if k.startswith("student_core.decoder.")}
        if not decoder_sd:
            raise ValueError("Checkpoint does not contain `student_core.decoder.*` parameters.")
        self._assert_state_dict_finite(decoder_sd, "student_core.decoder.*")
        self._decoder.load_state_dict(decoder_sd, strict=True)

        self._prior.eval()
        self._decoder.eval()
        for module in (self._prior, self._decoder):
            for param in module.parameters():
                param.requires_grad = False

        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._decoded_actions = torch.zeros((self.num_envs, self._num_joint_actions), device=self.device)
        self._prev_decoded_actions = torch.zeros((self.num_envs, self._num_joint_actions), device=self.device)
        self._setup_frozen_proprio_normalizer(checkpoint)
        self._warned_nonfinite_raw_actions = False
        self._warned_nonfinite_proprio = False
        self._warned_nonfinite_decoded_actions = False
        self._warned_nonfinite_processed_actions = False
        self._history_action_clip = float(getattr(self.cfg, "history_action_clip", 1.0))

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_action.processed_actions

    @property
    def decoded_actions(self) -> torch.Tensor:
        return self._decoded_actions

    @property
    def prev_decoded_actions(self) -> torch.Tensor:
        return self._prev_decoded_actions

    def _current_frame(self) -> torch.Tensor:
        if self._joint_ids is None or isinstance(self._joint_ids, slice):
            joint_pos_rel = self._asset.data.joint_pos - self._asset.data.default_joint_pos
            joint_vel_rel = self._asset.data.joint_vel - self._asset.data.default_joint_vel
        else:
            joint_pos_rel = self._asset.data.joint_pos[:, self._joint_ids] - self._asset.data.default_joint_pos[:, self._joint_ids]
            joint_vel_rel = self._asset.data.joint_vel[:, self._joint_ids] - self._asset.data.default_joint_vel[:, self._joint_ids]
        base_ang_vel = self._asset.data.root_ang_vel_b
        # Keep decoder/prior proprio action term consistent with policy observation `mdp.last_action`.
        frame = torch.cat([joint_pos_rel, joint_vel_rel, base_ang_vel, self._decoded_actions], dim=-1)
        return frame

    def _update_proprio_history(self) -> torch.Tensor:
        frame = self._current_frame()
        self._proprio_history[:, :-1] = self._proprio_history[:, 1:].clone()
        self._proprio_history[:, -1] = frame
        j = self._num_joint_actions
        # Convert frame-major history [T, (jp, jv, w, a)] to term-major layout expected by policy tail:
        # [jp(T), jv(T), base_ang_vel(T), actions(T)].
        joint_pos_hist = self._proprio_history[:, :, :j].reshape(self.num_envs, -1)
        joint_vel_hist = self._proprio_history[:, :, j : 2 * j].reshape(self.num_envs, -1)
        base_ang_vel_hist = self._proprio_history[:, :, 2 * j : 2 * j + 3].reshape(self.num_envs, -1)
        action_hist = self._proprio_history[:, :, 2 * j + 3 : 3 * j + 3].reshape(self.num_envs, -1)
        proprio_term_major = torch.cat([joint_pos_hist, joint_vel_hist, base_ang_vel_hist, action_hist], dim=-1)
        return proprio_term_major

    def _setup_frozen_proprio_normalizer(self, checkpoint: dict) -> None:
        if not bool(getattr(self.cfg, "normalize_proprio_for_prior", True)):
            return
        if not isinstance(checkpoint, dict):
            return

        mean_t: torch.Tensor | None = None
        std_t: torch.Tensor | None = None
        split_state = checkpoint.get("student_proprio_obs_norm_state_dict")
        full_state = checkpoint.get("obs_norm_state_dict")

        if isinstance(split_state, dict):
            m = split_state.get("_mean")
            s = split_state.get("_std")
            if isinstance(m, torch.Tensor) and isinstance(s, torch.Tensor) and m.shape[-1] == self._proprio_dim:
                mean_t, std_t = m, s

        if mean_t is None or std_t is None:
            if isinstance(full_state, dict):
                m = full_state.get("_mean")
                s = full_state.get("_std")
                if isinstance(m, torch.Tensor) and isinstance(s, torch.Tensor) and m.shape[-1] >= self._proprio_dim:
                    mean_t = m[..., -self._proprio_dim :]
                    std_t = s[..., -self._proprio_dim :]

        if mean_t is None or std_t is None:
            warnings.warn(
                "[ResidualLatentAction] Could not load proprio normalizer stats from checkpoint; "
                "prior/decoder will receive raw proprio.",
                stacklevel=2,
            )
            return

        mean_t = mean_t.to(device=self.device, dtype=torch.float32)
        std_t = std_t.to(device=self.device, dtype=torch.float32).clamp_min(1.0e-8)
        self._proprio_norm_mean = mean_t
        self._proprio_norm_std = std_t
        if int(__import__("os").environ.get("RANK", "0")) == 0:
            print(
                "[ResidualLatentAction] Loaded frozen proprio normalizer for prior input: "
                f"dim={self._proprio_dim}, eps={self._proprio_norm_eps}"
            )

    def _normalize_proprio_for_prior(self, proprio: torch.Tensor) -> torch.Tensor:
        if self._proprio_norm_mean is None or self._proprio_norm_std is None:
            return proprio
        return (proprio - self._proprio_norm_mean) / (self._proprio_norm_std + self._proprio_norm_eps)

    def _fetch_policy_obs_proprio_tail(self) -> torch.Tensor | None:
        """Last ``_proprio_dim`` dims of raw policy observations (matches encoder MLP tail pre-runner norm)."""
        env_u = getattr(self, "_env", None)
        if env_u is None:
            return None
        core = getattr(env_u, "unwrapped", env_u)
        om = getattr(core, "observation_manager", None)
        if om is None or not callable(getattr(om, "compute", None)):
            return None
        try:
            od = om.compute()
            if not isinstance(od, dict) or "policy" not in od:
                return None
            pol = od["policy"]
            D = int(self._proprio_dim)
            if not isinstance(pol, torch.Tensor) or pol.shape[-1] < D:
                return None
            return pol[..., -D:].to(device=self.device, dtype=torch.float32)
        except Exception:
            return None

    def process_actions(self, actions: torch.Tensor):
        residual = actions.view(self.num_envs, self.action_dim)
        self._raw_actions.copy_(residual)
        if not torch.isfinite(self._raw_actions).all():
            bad = ~torch.isfinite(self._raw_actions)
            bad_dims = torch.nonzero(bad.any(dim=0), as_tuple=False).squeeze(-1)
            bad_envs = torch.nonzero(bad.any(dim=1), as_tuple=False).squeeze(-1)
            if not self._warned_nonfinite_raw_actions:
                warnings.warn(
                    "[ResidualLatentAction] Non-finite raw residual actions detected. "
                    f"shape={tuple(self._raw_actions.shape)}, "
                    f"num_bad_dims={int(bad_dims.numel())}, num_bad_envs={int(bad_envs.numel())}",
                    stacklevel=2,
                )
                self._warned_nonfinite_raw_actions = True
            self._raw_actions.copy_(torch.nan_to_num(self._raw_actions, nan=0.0, posinf=0.0, neginf=0.0))
            residual = self._raw_actions

        proprio_fallback = self._update_proprio_history()
        use_tail = bool(getattr(self.cfg, "use_policy_observation_tail_for_prior", True))
        proprio_tail = self._fetch_policy_obs_proprio_tail()
        if use_tail and proprio_tail is not None:
            proprio = proprio_tail
        else:
            proprio = proprio_fallback
        proprio_for_prior = self._normalize_proprio_for_prior(proprio)

        with torch.no_grad():
            mu_prior, _ = self._prior(proprio_for_prior)
            latent = mu_prior + float(self.cfg.residual_scale) * residual
            decoded = self._decoder(torch.cat([latent, proprio_for_prior], dim=-1))

        if self.cfg.tanh_actions:
            decoded = torch.tanh(decoded)
        if self.cfg.base_scale != 1.0:
            decoded = decoded * float(self.cfg.base_scale)
        if self.cfg.clip_actions:
            decoded = decoded.clamp(float(self.cfg.clip_action_min), float(self.cfg.clip_action_max))

        self._prev_decoded_actions.copy_(self._decoded_actions)
        self._decoded_actions.copy_(decoded)
        history_action = decoded.clamp(-self._history_action_clip, self._history_action_clip)
        self._last_joint_action.copy_(history_action)
        self._joint_action.process_actions(decoded)

    def apply_actions(self):
        processed_actions = self._joint_action.processed_actions
        if isinstance(processed_actions, torch.Tensor) and (not torch.isfinite(processed_actions).all()):
            # Hard guard before writing into simulation.
            self._joint_action.processed_actions.copy_(
                torch.nan_to_num(processed_actions, nan=0.0, posinf=0.0, neginf=0.0)
            )
        self._joint_action.apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._joint_action.reset(env_ids)
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return
        self._raw_actions[env_ids_t] = 0.0
        self._prev_decoded_actions[env_ids_t] = 0.0
        self._decoded_actions[env_ids_t] = 0.0
        self._last_joint_action[env_ids_t] = 0.0
        current_frame = self._current_frame()
        self._proprio_history[env_ids_t] = current_frame[env_ids_t].unsqueeze(1).repeat(1, self._history_length, 1)


@configclass
class ResidualLatentActionCfg(JointPositionActionCfg):
    class_type = ResidualLatentAction

    prior_checkpoint: str | None = None
    proprio_history_length: int = 5
    latent_dim: int = 16
    prior_hidden_dims: tuple[int, ...] = (512, 256, 128)
    decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    latent_predict_std_min: float = 0.001
    latent_predict_std_max: float = 1.0
    fixed_prior_std: float | None = 0.5
    residual_scale: float = 1.0
    base_scale: float = 1.0
    tanh_actions: bool = False
    clip_actions: bool = False
    clip_action_min: float = -1.0
    clip_action_max: float = 1.0
    history_action_clip: float = 1.0
    normalize_proprio_for_prior: bool = True
    proprio_norm_eps: float = 1.0e-2
    #: If True, prior/decoder use the same raw proprio tail as ``observation_manager.compute()["policy"]``
    #: (encoder MLP input, before runner normalization). If False or compute fails, use the internal
    #: history-based proprio (legacy).
    use_policy_observation_tail_for_prior: bool = True

