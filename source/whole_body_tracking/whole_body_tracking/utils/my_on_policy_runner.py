import math
import os
import time

import torch
import numpy as np
import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import SceneEntityCfg

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.kp_mode_metric_logger import KpModeMetricLogger
from whole_body_tracking.utils.kp_pilot_metric_logger import KpPilotMetricLogger
from whole_body_tracking.utils.world_poi_metric_logger import WorldPoiMetricLogger
from whole_body_tracking.utils.prior_rollout_metrics import finalize_streaming_prior_metrics
from whole_body_tracking.tasks.tracking.mdp import terminations as tracking_terminations
from whole_body_tracking.tasks.tracking.mdp.commands import (
    MotionCommand,
    MultiMotionCommand,
    PartialMaskedMultiMotionCommand,
)
from rsl_rl.modules import (
    LatentBottleneck2B,
    LatentBottleneckMUSE,
    LatentBottleneckMUSECoTrain,
    LatentBottleneckMUSETransformer,
    LatentBottleneckMUSEKp,
    LatentBottleneckPULSE,
    LatentBottleneckAnyBody,
    LatentRLActorCritic,
)

class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        # Latent-bottleneck-style policies don't have an MLP actor the standard
        # exporter can index; skip ONNX export (same rationale as MotionOnPolicyRunner).
        if isinstance(self.alg.policy, LatentRLActorCritic):
            return
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"
    ):
        super().__init__(env, train_cfg, log_dir, device)
        # Note: Teacher policy is automatically loaded in MOSAIC.__init__ if teacher_checkpoint_path is set
        # Two mutually-exclusive metric paths:
        #   - KpPilotMetricLogger (MUSE-cotrain only): buckets per-step KP metrics by which
        #     modality (JC vs KP) drove the env step. Pairs with algorithm-side
        #     ``flush_pilot_counts`` for per-modality success rate.
        #   - KpModeMetricLogger (other PartialMasked configs): buckets per-step KP metrics
        #     by the env's currently-active mask mode.
        #   - WorldPoiMetricLogger (any non-PartialMasked MotionCommand, e.g. the JC
        #     MUSE-Transformer teacher): one global bucket of world-frame anchor +
        #     5-POI pos/vel errors, logged under the ``eval_world/`` wandb panel.
        self._kp_mode_logger: KpModeMetricLogger | None = None
        self._kp_pilot_logger: KpPilotMetricLogger | None = None
        self._world_poi_logger: WorldPoiMetricLogger | None = None
        try:
            cmd = self.env.unwrapped.command_manager.get_term("motion")
            if isinstance(cmd, PartialMaskedMultiMotionCommand):
                if isinstance(self.alg.policy, LatentBottleneckMUSECoTrain):
                    self._kp_pilot_logger = KpPilotMetricLogger(cmd, self.alg.policy)
                    self._kp_pilot_logger.attach()
                    # Plumb env access into the cotrain algorithm so its process_env_step can
                    # read reset_terminated (non-timeout terminations) for per-pilot failure
                    # attribution. Without this it would only see ``dones`` which conflates
                    # timeouts and falls.
                    if hasattr(self.alg, "_env_unwrapped"):
                        self.alg._env_unwrapped = self.env.unwrapped
                else:
                    self._kp_mode_logger = KpModeMetricLogger(cmd)
                    self._kp_mode_logger.attach()
        except Exception as e:
            print(f"[MotionOnPolicyRunner] KP metric logger setup skipped: {e!r}", flush=True)
            self._kp_mode_logger = None
            self._kp_pilot_logger = None

        # World-frame anchor + 5-POI panel: only when no per-mode KP logger is
        # attached (i.e. a plain MultiMotionCommand such as the JC MUSE-Transformer
        # teacher). For PartialMasked KP/cotrain tasks the kp_modes panel already
        # carries the world-frame POI signal — keep that path untouched.
        if self._kp_mode_logger is None and self._kp_pilot_logger is None:
            try:
                cmd = self.env.unwrapped.command_manager.get_term("motion")
                if isinstance(cmd, (MotionCommand, MultiMotionCommand)):
                    self._world_poi_logger = WorldPoiMetricLogger(cmd)
                    self._world_poi_logger.attach()
            except Exception as e:
                print(f"[MotionOnPolicyRunner] WorldPoi metric logger setup skipped: {e!r}", flush=True)
                self._world_poi_logger = None

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # When periodic prior eval chunks training into multiple super().learn() calls,
        # keep progress display normalized to the original requested total iterations.
        if hasattr(self, "_progress_start_iter") and hasattr(self, "_progress_tot_iter"):
            locs["start_iter"] = int(self._progress_start_iter)
            locs["tot_iter"] = int(self._progress_tot_iter)
            locs["num_learning_iterations"] = int(self._progress_tot_iter - self._progress_start_iter)
        # Per-mode KP metrics flush (no-op when logger not attached).
        # Rank-0-local means only — avoids collective-op deadlock risk if a mode happens to be
        # absent on some rank. Each rank has hundreds of envs sampling modes uniformly, so the
        # rank-0 local mean is a very close approximation of the global mean for our cadence.
        if self._kp_mode_logger is not None and not getattr(self, "disable_logs", False) and self.writer is not None:
            means, steps = self._kp_mode_logger.flush()
            it = int(locs.get("it", self.current_learning_iteration))
            for k, v in means.items():
                self.writer.add_scalar(k, float(v), it)
            for k, v in steps.items():
                self.writer.add_scalar(k, float(v), it)
        elif self._kp_mode_logger is not None:
            # On non-rank-0, still drain the accumulator so it doesn't grow unboundedly.
            self._kp_mode_logger.flush()

        # Pilot-bucketed metrics (MUSE-cotrain only): same rank-0-local approximation as above.
        if self._kp_pilot_logger is not None and not getattr(self, "disable_logs", False) and self.writer is not None:
            it = int(locs.get("it", self.current_learning_iteration))
            means, steps = self._kp_pilot_logger.flush()
            for k, v in means.items():
                self.writer.add_scalar(k, float(v), it)
            for k, v in steps.items():
                self.writer.add_scalar(k, float(v), it)
            # Algorithm-side per-pilot step/done counters drive the per-modality success rate.
            if hasattr(self.alg, "flush_pilot_counts"):
                for k, v in self.alg.flush_pilot_counts().items():
                    self.writer.add_scalar(k, float(v), it)
        elif self._kp_pilot_logger is not None:
            self._kp_pilot_logger.flush()
            if hasattr(self.alg, "flush_pilot_counts"):
                self.alg.flush_pilot_counts()

        # World-frame anchor + 5-POI panel (eval_world/...): same rank-0-local
        # approximation as the KP loggers above.
        if self._world_poi_logger is not None and not getattr(self, "disable_logs", False) and self.writer is not None:
            means, steps = self._world_poi_logger.flush()
            it = int(locs.get("it", self.current_learning_iteration))
            for k, v in means.items():
                self.writer.add_scalar(k, float(v), it)
            for k, v in steps.items():
                self.writer.add_scalar(k, float(v), it)
        elif self._world_poi_logger is not None:
            self._world_poi_logger.flush()
        return super().log(locs, width=width, pad=pad)

    def _prior_periodic_eval_enabled(self) -> bool:
        return bool(self.cfg.get("prior_eval_enabled", False))

    def _run_periodic_prior_eval(self, train_iter: int) -> None:
        if self.disable_logs:
            return
        rollout_itr = int(self.cfg.get("prior_eval_rollout_itr", 0))  # number of eval episodes
        rollout_steps = int(self.cfg.get("prior_eval_rollout_steps", 400))  # horizon per eval episode
        if rollout_itr <= 0 or rollout_steps <= 0:
            return

        metrics_list: list[dict[str, float]] = []

        for eval_ep_idx in range(rollout_itr):
            metrics, _ = self.collect_prior_rollout_eval(
                rollout_steps,
                log_interval=int(self.cfg.get("prior_eval_log_interval", 24)),
                clip_idx=eval_ep_idx,
                clip_total=rollout_itr,
                motion_path="training_env_motion",
                fall_term_name=str(self.cfg.get("prior_eval_fall_term_name", "fall")),
                fixed_prior_latent_std=self.cfg.get("prior_eval_fixed_latent_std", None),
                video_output_path=None,
            )
            metrics_list.append(metrics)

            print(
                f"[prior_periodic_eval] train_iter={train_iter} eval_episode={eval_ep_idx + 1}/{rollout_itr} "
                f"rollout_steps={rollout_steps} "
                f"fall_rate={float(metrics.get('prior_eval/fall_rate', 0.0)):.4f} "
                f"avg_steps_before_fall={float(metrics.get('prior_eval/steps_survived_mean', 0.0)):.2f} "
                f"action_rate_l2={float(metrics.get('prior_eval/action_rate_l2_mean', 0.0)):.6f}"
            )

        # Also log periodic aggregate (mean across eval episodes) at training iteration index.
        if metrics_list and self.writer is not None:
            keys = set().union(*(m.keys() for m in metrics_list))
            for key in keys:
                vals = [float(m[key]) for m in metrics_list if key in m]
                if not vals:
                    continue
                suffix = key.split("/", 1)[1] if "/" in key else key
                mean_val = float(sum(vals) / len(vals))
                self.writer.add_scalar(
                    f"prior_periodic_eval/{suffix}",
                    mean_val,
                    int(train_iter),
                )

    def _holdout_periodic_eval_enabled(self) -> bool:
        if not bool(self.cfg.get("holdout_eval_enabled", False)):
            return False
        return str(self.cfg.get("holdout_eval_motion") or "").strip() != ""

    def _periodic_eval_chunk_sizes(self) -> tuple[bool, bool, int]:
        """Return (prior_on, holdout_on, chunk_size) for chunked training + periodic eval hooks."""

        prior_on = self._prior_periodic_eval_enabled() and int(self.cfg.get("prior_eval_rollout_every_itr", 0)) > 0
        holdout_on = self._holdout_periodic_eval_enabled() and int(self.cfg.get("holdout_eval_every_itr", 0)) > 0
        if not prior_on and not holdout_on:
            return False, False, 0
        if prior_on and holdout_on:
            pe = int(self.cfg["prior_eval_rollout_every_itr"])
            he = int(self.cfg["holdout_eval_every_itr"])
            g = math.gcd(pe, he)
            chunk_size = max(1, g)
        elif prior_on:
            chunk_size = int(self.cfg["prior_eval_rollout_every_itr"])
        else:
            chunk_size = int(self.cfg["holdout_eval_every_itr"])
        return prior_on, holdout_on, chunk_size

    def _all_reduce_mean_float_dict(self, d: dict[str, float]) -> dict[str, float]:
        try:
            import torch.distributed as dist
        except Exception:
            return d
        if not (dist.is_available() and dist.is_initialized()):
            return d
        ws = int(dist.get_world_size())
        out: dict[str, float] = {}
        for k in sorted(d.keys()):
            t = torch.tensor([float(d[k])], dtype=torch.float64, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out[k] = float((t / float(ws)).item())
        return out

    @staticmethod
    def _mean_motion_command_metrics(cmd: MultiMotionCommand) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, v in cmd.metrics.items():
            if not torch.is_tensor(v):
                continue
            x = v.float() if not v.is_floating_point() else v
            out[key] = float(torch.nanmean(x).item())
        return out

    def _print_holdout_eval_terminal_log(
        self,
        train_iter: int,
        clip_idx: int,
        clip_total: int,
        metrics: dict[str, float],
        *,
        elapsed_s: float,
        num_env_steps: int,
        num_envs: int,
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """One block per holdout pass, layout aligned with :meth:`OnPolicyRunner.log` (distillation-style)."""

        if int(os.environ.get("RANK", "0")) != 0:
            return
        collection_size = int(num_env_steps) * int(num_envs)
        fps = int(collection_size / elapsed_s) if elapsed_s > 0 else 0
        title = (
            f" \033[1m Test iteration {train_iter} "
            f"(holdout {int(clip_idx) + 1}/{int(clip_total)}) \033[0m "
        )
        log_string = (
            f"""{'#' * width}\n"""
            f"""{title.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {elapsed_s:.3f}s, learning 0.000s)\n"""
        )
        for key in sorted(metrics.keys()):
            log_string += f"""{f'{key}:':>{pad}} {metrics[key]:.4f}\n"""
        log_string += f"""{'-' * width}\n"""
        print(log_string)

    def _run_periodic_holdout_eval(self, train_iter: int) -> None:
        if self.disable_logs:
            return
        if not self._policy_supports_holdout_student_eval():
            return
        motion_path = str(self.cfg.get("holdout_eval_motion") or "").strip()
        if not motion_path:
            return
        rollout_itr = int(self.cfg.get("holdout_eval_rollout_itr", 1))
        # Same horizon as training rollout collection: terminations / motion logic match train.
        rollout_steps = int(self.num_steps_per_env)
        if rollout_itr <= 0 or rollout_steps <= 0:
            return

        metrics_list: list[dict[str, float]] = []
        for eval_ep_idx in range(rollout_itr):
            m = self.collect_holdout_student_rollout_eval(
                rollout_steps,
                train_iter=int(train_iter),
                motion_path=motion_path,
                clip_idx=eval_ep_idx,
                clip_total=rollout_itr,
                deterministic_motion=bool(self.cfg.get("holdout_eval_deterministic_motion", True)),
                motion_sample_seed=int(self.cfg.get("holdout_eval_motion_sample_seed", 42)),
            )
            metrics_list.append(self._all_reduce_mean_float_dict(m))

        if metrics_list and self.writer is not None:
            keys = set().union(*(m.keys() for m in metrics_list))
            for key in keys:
                vals = [float(m[key]) for m in metrics_list if key in m]
                if not vals:
                    continue
                mean_val = float(sum(vals) / len(vals))
                suffix = key.split("/", 1)[1] if "/" in key else key
                self.writer.add_scalar(f"holdout_periodic_eval/{suffix}", mean_val, int(train_iter))

    def _policy_supports_holdout_student_eval(self) -> bool:
        return isinstance(self.alg.policy, (LatentBottleneckAnyBody, LatentBottleneck2B))

    def collect_holdout_student_rollout_eval(
        self,
        num_env_steps: int,
        *,
        train_iter: int,
        motion_path: str,
        clip_idx: int = 0,
        clip_total: int = 1,
        deterministic_motion: bool = False,
        motion_sample_seed: int = 42,
    ) -> dict[str, float]:
        """Student ``act_inference`` rollout on holdout motions.

        ``num_env_steps`` should match training (typically ``num_steps_per_env``): same step budget as
        one training env collection, so timeouts/terminations/resets behave like train.
        Logs motion command metrics only; terminal output once per call (same style as training ``log``).
        """

        if not self._policy_supports_holdout_student_eval():
            raise RuntimeError("collect_holdout_student_rollout_eval requires LatentBottleneckAnyBody or LatentBottleneck2B.")

        cmd = self.env.unwrapped.command_manager.get_term("motion")
        if not isinstance(cmd, MultiMotionCommand):
            raise RuntimeError("collect_holdout_student_rollout_eval requires MultiMotionCommand.")

        cmd.push_holdout_motion_dataset(motion_path)
        try:
            self.eval_mode()
            if hasattr(self.alg, "storage") and getattr(self.alg, "storage", None) is not None:
                self.alg.storage.clear()

            with torch.inference_mode():
                self.env.reset()
                if self.training_type == "anybody_latent_distillation" and hasattr(
                    self.alg.policy, "sample_and_set_mask"
                ):
                    from rsl_rl.algorithms.mask_utils import sync_policy_keypoint_mask_from_motion_command

                    mt0 = self.env.unwrapped.command_manager.get_term("motion")
                    synced0 = sync_policy_keypoint_mask_from_motion_command(
                        policy=self.alg.policy,
                        motion_term=mt0,
                        num_envs=self.env.num_envs,
                        fixed_mode_idx=None,
                    )
                    if not synced0:
                        self.alg.policy.sample_and_set_mask(self.env.num_envs, fixed_mode_idx=None)
                if self.training_type == "anybody_latent_distillation":
                    mt = self.env.unwrapped.command_manager.get_term("motion")
                    if hasattr(mt, "set_pulse_vae_ee_env_mode_indices"):
                        if not getattr(mt, "uses_command_manager_mask_sampling", False):
                            pol = self.alg.policy
                            mi = pol.get_goal_mask_mode_indices() if hasattr(pol, "get_goal_mask_mode_indices") else None
                            mt.set_pulse_vae_ee_env_mode_indices(mi)
                if deterministic_motion:
                    self._apply_holdout_eval_deterministic_motion_indices(clip_idx, motion_sample_seed, cmd)
                else:
                    env_ids = list(range(int(cmd.num_envs)))
                    cmd._resample_command(env_ids)

            num_envs = int(self.env.num_envs)
            device = self.device
            step_means: dict[str, list[float]] = {}

            obs, extras = self.env.get_observations()
            obs_dict = extras.get("observations", {})
            if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
                obs = obs_dict[self.policy_obs_type]
            privileged_obs = obs_dict.get(self.privileged_obs_type, obs)
            teacher_obs = obs_dict.get(self.teacher_obs_type)
            obs = obs.to(device)
            privileged_obs = privileged_obs.to(device)
            if teacher_obs is not None:
                teacher_obs = teacher_obs.to(device)
            else:
                teacher_obs = privileged_obs
            self._assert_anybody_latent_proprio_alignment(obs, teacher_obs)
            obs = self._normalize_student_obs(obs)

            t_rollout0 = time.perf_counter()

            for step in range(int(num_env_steps)):
                with torch.inference_mode():
                    actions = self.alg.policy.act_inference(obs)
                    obs, _rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    dones = dones.to(device)
                    if hasattr(self.alg.policy, "reset"):
                        self.alg.policy.reset(dones)

                    m = self._mean_motion_command_metrics(cmd)
                    for k, v in m.items():
                        step_means.setdefault(k, []).append(v)

                    obs_dict = infos.get("observations", {})
                    if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
                        obs = obs_dict[self.policy_obs_type].to(device)
                    else:
                        obs = obs.to(device)
                    teacher_obs_raw = None
                    if self.teacher_obs_type is not None and self.teacher_obs_type in obs_dict:
                        teacher_obs_raw = obs_dict[self.teacher_obs_type].to(device)
                        self._assert_anybody_latent_proprio_alignment(obs, teacher_obs_raw)
                    obs = self._normalize_student_obs(obs)

            out = {k: float(sum(vals) / len(vals)) for k, vals in step_means.items() if vals}
            elapsed = time.perf_counter() - t_rollout0
            self._print_holdout_eval_terminal_log(
                train_iter,
                clip_idx,
                clip_total,
                out,
                elapsed_s=elapsed,
                num_env_steps=int(num_env_steps),
                num_envs=num_envs,
            )
            return out
        finally:
            cmd.pop_holdout_motion_dataset()
            self.train_mode()

    def _apply_holdout_eval_deterministic_motion_indices(
        self, eval_episode_idx: int, seed: int, cmd: MultiMotionCommand
    ) -> None:
        num_motions = int(cmd.num_motions_total)
        if num_motions <= 0:
            return
        n_env = int(cmd.num_envs)
        ep = int(eval_episode_idx)
        gen_seed = int((int(seed) + ep * 1_000_003) % (2**63 - 1))
        g = torch.Generator(device="cpu")
        g.manual_seed(gen_seed)
        idx_cpu = torch.randint(0, num_motions, (n_env,), generator=g, dtype=torch.long)
        cmd.env_motion_indices.copy_(idx_cpu.to(device=cmd.device))

        lut_key = (id(cmd), num_motions)
        if getattr(self, "_holdout_eval_motion_group_lut_key", None) != lut_key:
            default_g = int(
                cmd.group_name_to_idx.get("default", next(iter(cmd.group_name_to_idx.values())))
            )
            lut = torch.empty(num_motions, dtype=torch.long, device=cmd.device)
            for mi in range(num_motions):
                gname = cmd.motion_dir_loader.motion_to_group.get(mi, "default")
                lut[mi] = int(cmd.group_name_to_idx.get(gname, default_g))
            self._holdout_eval_motion_group_lut = lut
            self._holdout_eval_motion_group_lut_key = lut_key
        lut_t = self._holdout_eval_motion_group_lut
        cmd.env_motion_groups.copy_(lut_t[cmd.env_motion_indices])

        cmd._resample_command(list(range(n_env)))

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False, **kwargs):
        eval_mode = bool(kwargs.get("eval_mode", False))
        prior_on, holdout_on, chunk_size = self._periodic_eval_chunk_sizes()
        if eval_mode or (not prior_on and not holdout_on):
            return super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
                **kwargs,
            )

        remaining = int(num_learning_iterations)
        first_chunk = True
        self._progress_start_iter = int(self.current_learning_iteration)
        self._progress_tot_iter = int(self.current_learning_iteration + num_learning_iterations)
        if not hasattr(self, "_global_learning_iteration"):
            self._global_learning_iteration = max(1, int(self.current_learning_iteration) + 1)
        prior_every = int(self.cfg.get("prior_eval_rollout_every_itr", 0)) if prior_on else 0
        holdout_every = int(self.cfg.get("holdout_eval_every_itr", 0)) if holdout_on else 0
        while remaining > 0:
            chunk = min(chunk_size, remaining)
            self.current_learning_iteration = int(self._global_learning_iteration)
            super().learn(
                num_learning_iterations=chunk,
                init_at_random_ep_len=init_at_random_ep_len if first_chunk else False,
                **kwargs,
            )
            first_chunk = False
            remaining -= chunk
            self._global_learning_iteration = int(self.current_learning_iteration) + 1
            it = int(self.current_learning_iteration)
            # PULSE prior eval: run after every train chunk (same as legacy behavior).
            # When combined with holdout, only run on multiples of prior_every so cadences stay aligned.
            if prior_on:
                if holdout_on:
                    if prior_every > 0 and it % prior_every == 0:
                        self._run_periodic_prior_eval(it)
                else:
                    self._run_periodic_prior_eval(it)
            if holdout_on:
                if prior_on:
                    if holdout_every > 0 and it % holdout_every == 0:
                        self._run_periodic_holdout_eval(it)
                else:
                    self._run_periodic_holdout_eval(it)
        del self._progress_start_iter
        del self._progress_tot_iter

    def _apply_prior_eval_motion_index(self, motion_idx: int) -> None:
        """Align all parallel envs on one dataset clip (``MultiMotionCommand``), then resample poses."""
        cmd = self.env.unwrapped.command_manager.get_term("motion")
        if not isinstance(cmd, MultiMotionCommand):
            raise RuntimeError("prior_eval_motion_idx requires MultiMotionCommand (PULSE tracking env).")
        cmd.env_motion_indices.fill_(int(motion_idx))
        env_ids = list(range(int(cmd.num_envs)))
        cmd._resample_command(env_ids)

    def _apply_prior_eval_deterministic_motion_indices(self, eval_episode_idx: int) -> None:
        """Fix per-env motion clip IDs from ``seed + episode`` (reproducible across runs and train iters).

        Uses ``torch.Generator`` only to draw which motion index each env uses; ``_resample_command`` still
        applies the usual pose / velocity / joint randomization from the global RNG (4096 diverse inits).
        """
        cmd = self.env.unwrapped.command_manager.get_term("motion")
        if not isinstance(cmd, MultiMotionCommand):
            return
        num_motions = int(cmd.num_motions_total)
        if num_motions <= 0:
            return
        n_env = int(cmd.num_envs)
        seed = int(self.cfg.get("prior_eval_motion_sample_seed", 12345))
        ep = int(eval_episode_idx)
        gen_seed = int((seed + ep * 1_000_003) % (2**63 - 1))
        g = torch.Generator(device="cpu")
        g.manual_seed(gen_seed)
        idx_cpu = torch.randint(0, num_motions, (n_env,), generator=g, dtype=torch.long)
        cmd.env_motion_indices.copy_(idx_cpu.to(device=cmd.device))

        lut_key = (id(cmd), num_motions)
        if getattr(self, "_prior_eval_motion_group_lut_key", None) != lut_key:
            default_g = int(
                cmd.group_name_to_idx.get("default", next(iter(cmd.group_name_to_idx.values())))
            )
            lut = torch.empty(num_motions, dtype=torch.long, device=cmd.device)
            for mi in range(num_motions):
                gname = cmd.motion_dir_loader.motion_to_group.get(mi, "default")
                lut[mi] = int(cmd.group_name_to_idx.get(gname, default_g))
            self._prior_eval_motion_group_lut = lut
            self._prior_eval_motion_group_lut_key = lut_key
        lut_t = self._prior_eval_motion_group_lut
        cmd.env_motion_groups.copy_(lut_t[cmd.env_motion_indices])

        cmd._resample_command(list(range(n_env)))

    def _render_prior_eval_frame(self):
        """Render from underlying gym env (RslRlVecEnvWrapper itself has no render())."""
        env_obj = self.env
        visited = set()
        for _ in range(8):
            oid = id(env_obj)
            if oid in visited:
                break
            visited.add(oid)
            render_fn = getattr(env_obj, "render", None)
            if callable(render_fn):
                try:
                    frame = render_fn()
                    if frame is not None:
                        return frame
                except Exception:
                    pass
            if hasattr(env_obj, "env"):
                env_obj = env_obj.env
                continue
            if hasattr(env_obj, "unwrapped"):
                env_obj = env_obj.unwrapped
                continue
            break
        return None

    def _set_prior_eval_debug_vis(self, enabled: bool) -> None:
        """Toggle motion command debug markers for prior-eval videos."""
        try:
            cmd = self.env.unwrapped.command_manager.get_term("motion")
            if hasattr(cmd, "set_debug_vis"):
                cmd.set_debug_vis(enabled)
            elif hasattr(cmd, "_set_debug_vis_impl"):
                cmd._set_debug_vis_impl(enabled)
        except Exception:
            pass

    def _disable_non_prior_terminations(self):
        """Best-effort: disable all termination funcs except time_out during prior eval."""
        mgr = getattr(self.env.unwrapped, "termination_manager", None)
        if mgr is None:
            return None
        active_terms = list(getattr(mgr, "active_terms", []))
        term_cfgs = getattr(mgr, "_term_cfgs", None)
        if not active_terms or not isinstance(term_cfgs, list):
            return None

        def _always_false(env, *args, **kwargs):
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        backups = []
        for idx, name in enumerate(active_terms):
            if name == "time_out":
                continue
            if idx >= len(term_cfgs):
                continue
            cfg = term_cfgs[idx]
            if not hasattr(cfg, "func"):
                continue
            backups.append((idx, cfg.func))
            cfg.func = _always_false
        return backups

    def _restore_terminations(self, backups) -> None:
        if not backups:
            return
        mgr = getattr(self.env.unwrapped, "termination_manager", None)
        if mgr is None:
            return
        term_cfgs = getattr(mgr, "_term_cfgs", None)
        if not isinstance(term_cfgs, list):
            return
        for idx, func in backups:
            if 0 <= idx < len(term_cfgs) and hasattr(term_cfgs[idx], "func"):
                term_cfgs[idx].func = func

    def collect_prior_rollout_eval(
        self,
        num_env_steps: int,
        *,
        log_interval: int = 24,
        clip_idx: int = 0,
        clip_total: int = 1,
        motion_path: str = "",
        fall_term_name: str = "fall",
        prior_eval_motion_idx: int | None = None,
        fixed_prior_latent_std: float | None = None,
        video_output_path: str | None = None,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """One clip: prior-sampled actions, same obs/step/update path as ``learn()`` (eval), streaming metrics.

        Returns:
            (metrics dict, ``first_fall_step`` tensor [N] on device, -1 if never fell this horizon).

        Fall counts use ``termination_manager.get_term(fall_term_name)`` **after** each ``env.step``.
        Calling ``fall_to_ground`` on raw body poses **after** ``step`` is wrong when fall termination
        resets envs: state is already upright again, so the heuristic sees no fall. The manager keeps the
        term signal from the same step as the MDP (pre-reset).
        """
        if not isinstance(self.alg.policy, LatentBottleneckPULSE):
            raise RuntimeError("collect_prior_rollout_eval requires LatentBottleneckPULSE.")

        self.eval_mode()
        if hasattr(self.alg, "storage") and getattr(self.alg, "storage", None) is not None:
            self.alg.storage.clear()

        # Prior-eval-only behavior overrides: no tracking terminations and no debug frames.
        term_backups = self._disable_non_prior_terminations()
        self._set_prior_eval_debug_vis(False)
        # Isaac articulation buffers are inference tensors; in-place sim writes must run under
        # inference_mode (same as the step loop below).
        with torch.inference_mode():
            self.env.reset()
            if prior_eval_motion_idx is not None:
                self._apply_prior_eval_motion_index(prior_eval_motion_idx)
            elif bool(self.cfg.get("prior_eval_deterministic_motion", False)):
                self._apply_prior_eval_deterministic_motion_indices(clip_idx)

        num_envs = int(self.env.num_envs)
        device = self.device

        obs, extras = self.env.get_observations()
        obs_dict = extras.get("observations", {})
        if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
            obs = obs_dict[self.policy_obs_type]
        privileged_obs = obs_dict.get(self.privileged_obs_type, obs)
        teacher_obs = obs_dict.get(self.teacher_obs_type)
        obs = obs.to(device)
        privileged_obs = privileged_obs.to(device)
        if teacher_obs is not None:
            teacher_obs = teacher_obs.to(device)
        else:
            teacher_obs = privileged_obs
        self._assert_anybody_latent_proprio_alignment(obs, teacher_obs)
        ref_vel_estimator_obs = obs_dict.get(self.ref_vel_estimator_obs_type)
        if ref_vel_estimator_obs is not None:
            ref_vel_estimator_obs = ref_vel_estimator_obs.to(device)

        obs = self._normalize_student_obs(obs)
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)

        # Prior-eval fall signal: compute directly from body height, independent of training termination terms.
        # This keeps online training terminations unchanged while enabling fall metrics in prior rollout mode.
        fall_body_names = self.cfg.get("prior_eval_fall_body_names", ["torso_link"])
        if isinstance(fall_body_names, str):
            fall_body_names = [fall_body_names]
        fall_asset_cfg = SceneEntityCfg("robot", body_names=list(fall_body_names))
        fall_min_height = float(self.cfg.get("prior_eval_fall_min_height", 0.35))

        already_fallen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        first_fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        action_sum: torch.Tensor | None = None  # [A], pre-fall pooled over envs+steps (GPU)
        action_sumsq: torch.Tensor | None = None  # [A], pre-fall pooled over envs+steps (GPU)
        action_count = 0  # number of pre-fall action samples (env-steps)
        sum_dp = torch.zeros((), dtype=torch.float64, device=device)
        sum_da = torch.zeros((), dtype=torch.float64, device=device)
        n_delta = 0
        prev_proprio: torch.Tensor | None = None
        prev_action: torch.Tensor | None = None
        prev_alive: torch.Tensor | None = None
        action_rate_l2_sum = 0.0
        action_rate_l2_n = 0

        video_writer = None
        if video_output_path and int(os.environ.get("RANK", "0")) == 0:
            try:
                import imageio.v2 as imageio

                os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
                video_writer = imageio.get_writer(video_output_path, fps=30)
            except Exception as e:
                print(f"[collect_prior_rollout_eval] WARNING: failed to create video writer: {e}")
                video_writer = None

        t_rollout0 = time.perf_counter()
        width = 80
        if int(os.environ.get("RANK", "0")) == 0:
            hdr = f" Prior eval clip {clip_idx + 1}/{clip_total} "
            mp = motion_path if len(motion_path) <= width - 4 else motion_path[: width - 7] + "..."
            print(f"\n{'#' * width}\n{hdr.center(width)}\n{mp}\n{'#' * width}")

        for step in range(num_env_steps):
            with torch.inference_mode():
                # Safety check: prior eval must not build any gradients.
                assert not torch.is_grad_enabled(), "Gradients unexpectedly enabled during prior eval."
                st = self.alg.policy.act_prior_sample_with_stats(
                    obs, fixed_latent_std=fixed_prior_latent_std
                )
                actions = st["actions"]
                ac = actions.detach().float()
                if action_sum is None:
                    num_action_dim = int(ac.shape[1])
                    action_sum = torch.zeros(num_action_dim, dtype=torch.float64, device=device)
                    action_sumsq = torch.zeros(num_action_dim, dtype=torch.float64, device=device)
                alive = ~already_fallen
                if bool(alive.any().item()):
                    ac_alive = ac[alive]
                    ad = ac_alive.double()
                    action_sum += ad.sum(dim=0)
                    action_sumsq += (ad * ad).sum(dim=0)
                    action_count += int(ac_alive.shape[0])

                proprio = st["proprio"].detach().float()
                if prev_proprio is not None and prev_action is not None and prev_alive is not None:
                    valid = alive & prev_alive
                    if bool(valid.any().item()):
                        delta_p = torch.norm(proprio[valid] - prev_proprio[valid], dim=-1)
                        delta_a = torch.norm(ac[valid] - prev_action[valid], dim=-1)
                        sum_dp += delta_p.double().sum()
                        sum_da += delta_a.double().sum()
                        n_delta += int(delta_p.numel())
                prev_proprio = proprio
                prev_action = ac
                prev_alive = alive

                obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                rewards, dones = rewards.to(device), dones.to(device)
                # Do not call alg.process_env_step here: rollout storage is sized for num_steps_per_env (~24);
                # long prior-eval horizons would overflow. Transition buffers are only filled by alg.act().
                if hasattr(self.alg.policy, "reset"):
                    self.alg.policy.reset(dones)

                fell_now = tracking_terminations.fall_to_ground(
                    self.env.unwrapped,
                    asset_cfg=fall_asset_cfg,
                    min_height=fall_min_height,
                ).to(device=device, dtype=torch.bool)

                # Track exactly the same action-rate quantity used in tracking rewards.
                action_rate_l2 = isaaclab_mdp.action_rate_l2(self.env.unwrapped)
                action_rate_l2_sum += float(action_rate_l2.sum().item())
                action_rate_l2_n += int(action_rate_l2.numel())

                newly = fell_now & ~already_fallen
                if newly.any():
                    first_fall_step[newly] = step + 1
                already_fallen |= fell_now
                if bool(already_fallen.all().item()):
                    # All envs have fallen; no need to continue rollout horizon.
                    break

                obs_dict = infos.get("observations", {})
                if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
                    obs = obs_dict[self.policy_obs_type].to(device)
                else:
                    obs = obs.to(device)
                teacher_obs_raw = None
                if self.teacher_obs_type is not None and self.teacher_obs_type in obs_dict:
                    teacher_obs_raw = obs_dict[self.teacher_obs_type].to(device)
                    self._assert_anybody_latent_proprio_alignment(obs, teacher_obs_raw)
                obs = self._normalize_student_obs(obs)
                if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
                    privileged_obs = self.privileged_obs_normalizer(
                        obs_dict[self.privileged_obs_type].to(device)
                    )
                else:
                    privileged_obs = obs
                if teacher_obs_raw is not None:
                    teacher_obs = self.teacher_obs_normalizer(teacher_obs_raw)
                else:
                    teacher_obs = privileged_obs
                if self.ref_vel_estimator_obs_type is not None and self.ref_vel_estimator_obs_type in obs_dict:
                    ref_vel_estimator_obs = obs_dict[self.ref_vel_estimator_obs_type].to(device)
                else:
                    ref_vel_estimator_obs = None

                if video_writer is not None:
                    try:
                        frame = self._render_prior_eval_frame()
                        if isinstance(frame, (list, tuple)):
                            frame = frame[0] if len(frame) > 0 else None
                        if frame is not None:
                            if torch.is_tensor(frame):
                                frame = frame.detach().cpu().numpy()
                            frame_np = np.asarray(frame)
                            if frame_np.dtype != np.uint8:
                                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                            video_writer.append_data(frame_np)
                    except Exception as e:
                        print(f"[collect_prior_rollout_eval] WARNING: failed to append frame: {e}")
                        video_writer.close()
                        video_writer = None

            if (
                log_interval > 0
                and int(os.environ.get("RANK", "0")) == 0
                and (step + 1) % log_interval == 0
            ):
                elapsed = time.perf_counter() - t_rollout0
                env_steps_done = (step + 1) * num_envs
                fps = env_steps_done / elapsed if elapsed > 0 else 0.0
                fell_so_far = int(already_fallen.sum().item())
                print(
                    f"  Rollout step {step + 1:5d}/{num_env_steps} | ~{fps:,.0f} env-steps/s (sim+policy) | "
                    f"fallen_envs {fell_so_far}/{num_envs}"
                )

        effective_rollout_steps = int(num_env_steps)
        if bool((first_fall_step > 0).all().item()):
            effective_rollout_steps = int(first_fall_step.max().item())

        if int(os.environ.get("RANK", "0")) == 0:
            elapsed = time.perf_counter() - t_rollout0
            total_es = effective_rollout_steps * num_envs
            fps = total_es / elapsed if elapsed > 0 else 0.0
            trunc_msg = " (truncated: all envs fell)" if effective_rollout_steps < num_env_steps else ""
            print(
                f"  Clip done: {total_es:,} env-steps in {elapsed:.2f}s (~{fps:,.0f} env-steps/s avg){trunc_msg}\n"
            )

        assert action_sum is not None and action_sumsq is not None
        metrics = finalize_streaming_prior_metrics(
            num_envs=num_envs,
            first_fall_step=first_fall_step,
            action_sum=action_sum.detach().cpu(),
            action_sumsq=action_sumsq.detach().cpu(),
            action_count=action_count,
            sum_dp=float(sum_dp.item()),
            sum_da=float(sum_da.item()),
            n_delta=n_delta,
        )
        if action_rate_l2_n > 0:
            metrics["prior_eval/action_rate_l2_mean"] = action_rate_l2_sum / float(action_rate_l2_n)
        else:
            metrics["prior_eval/action_rate_l2_mean"] = 0.0
        if video_writer is not None:
            video_writer.close()
        self._restore_terminations(term_backups)
        self._set_prior_eval_debug_vis(True)
        return metrics, first_fall_step

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        # Skip ONNX export for latent-bottleneck-style policies. The standard
        # exporter assumes an MLP actor with indexable layers, which does not match
        # the VAE-style LatentBottleneck* policies (including LatentBottleneck2B).
        if isinstance(
            self.alg.policy,
            (
                LatentBottleneckAnyBody,
                LatentBottleneckMUSE,
                LatentBottleneckMUSECoTrain,
                LatentBottleneckMUSETransformer,
                LatentBottleneckMUSEKp,
                LatentBottleneckPULSE,
                LatentBottleneck2B,
                LatentRLActorCritic,
            ),
        ):
            return

        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"

            # Get velocity estimator info if available
            ref_vel_estimator = None
            ref_vel_estimator_obs_dim = None

            if hasattr(self.alg, 'ref_vel_estimator') and self.alg.ref_vel_estimator is not None:
                ref_vel_estimator = self.alg.ref_vel_estimator
                if hasattr(self.alg, 'ref_vel_estimator_obs_shape') and self.alg.ref_vel_estimator_obs_shape is not None:
                    ref_vel_estimator_obs_dim = self.alg.ref_vel_estimator_obs_shape[0]

            # if the command is a multi motion command, or a single motion command
            export_motion_policy_as_onnx(
                self.alg.policy,
                normalizer=self.obs_normalizer,
                path=policy_path,
                filename=filename,
                ref_vel_estimator=ref_vel_estimator,
                ref_vel_estimator_obs_dim=ref_vel_estimator_obs_dim,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
