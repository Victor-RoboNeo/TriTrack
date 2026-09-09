"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import shutil
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Recorded video length and prior-metrics rollout length (steps) when --prior_sample is used.",
)
parser.add_argument(
    "--video_dir_tag",
    type=str,
    default="",
    help="Optional suffix tag for the recorded video root directory (e.g. videos_test).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion", type=str, default=None, help="Path to the motion file or motion directory.")
parser.add_argument("--skip_critic", action="store_true", default=False, help="Only load actor weights.")
parser.add_argument("--disable_motion_group_sampling", action="store_true", default=False, help="Disable motion group sampling ratios (use uniform sampling).")
parser.add_argument(
    "--start_frame",
    type=int,
    default=10,
    help="Start frame index (0-based) for motion playback.",
)
parser.add_argument(
    "--random_init_frame",
    action="store_true",
    default=False,
    help="Sample a uniformly random frame across the motion clip on each env reset (overrides "
    "--start_frame). Use to test rollout behavior under non-standing initial states.",
)
parser.add_argument(
    "--no_reset_base_xy_to_origin",
    action="store_true",
    default=False,
    help="[--random_init_frame] Disable the default behavior of overriding root XY to the env "
    "origin on reset. Without this flag, --random_init_frame keeps the sampled frame's pose+"
    "momentum but spawns the robot at the env origin so it stays in the camera frame.",
)
parser.add_argument(
    "--enable_motion_randomization",
    action="store_true",
    default=False,
    help="Keep motion randomization ranges (pose/velocity/joint) instead of zeroing them.",
)
parser.add_argument(
    "--disable_obs_noise",
    action="store_true",
    default=True,
    help="Disable observation corruption/noise during playback.",
)
parser.add_argument(
    "--disable_events",
    action="store_true",
    default=True,
    help="Disable event manager randomizations during playback.",
)
parser.add_argument(
    "--prior_sample",
    action="store_true",
    default=False,
    help="[PULSE only] Roll out using prior R(proprio) → decode (encoder bypassed). "
    "By default z is sampled from the prior; pass --no_prior_latent_sampling for z = μ only. "
    "Use with --video to record.",
)
parser.add_argument(
    "--no_prior_latent_sampling",
    action="store_true",
    default=False,
    help="[prior_sample] Use prior mean μ as latent (no stochastic sampling). "
    "Default without this flag: sample z ~ N(μ, σ²) from the prior.",
)
parser.add_argument(
    "--encoder_as_proprio",
    action="store_true",
    default=False,
    help="[PULSE only] Roll out using encoder(proprio, self_target_goal) → decode. The goal block "
    "is overridden so command = (current joint_pos, current joint_vel) and motion_anchor_ori_b = "
    "identity (6-D), making the encoder act as a proprio-only policy via in-distribution self-targets. "
    "Mutually exclusive with --prior_sample.",
)
parser.add_argument(
    "--encoder_as_proprio_sample",
    action="store_true",
    default=False,
    help="[encoder_as_proprio] Sample z ~ N(μ, σ) from the encoder posterior. Default (off): "
    "deterministic z = μ via act_inference.",
)
parser.add_argument(
    "--prior_rollout_fixed_latent_std",
    type=float,
    default=None,
    help="[prior_sample] If set, use this fixed per-latent-dim std for sampling and entropy metrics "
    "instead of the prior MLP's predicted σ. μ is still from the network. Ignored with "
    "--no_prior_latent_sampling.",
)
parser.add_argument(
    "--prior_video_show_motion_debug_vis",
    action="store_true",
    default=False,
    help="[prior_sample + --video] Show motion-command debug visualization (current vs reference goal "
    "anchor/body frames). Default: off so logged videos omit those paired frame markers.",
)
parser.add_argument(
    "--no_prior_metrics",
    action="store_true",
    default=False,
    help="[prior_sample] Disable fall/diversity/prior-entropy metrics and JSON logging.",
)
parser.add_argument(
    "--prior_fall_min_height",
    type=float,
    default=0.35,
    help="[prior metrics] World z (m) below which torso/bodies count as fallen (G1 standing torso is ~0.8–1.0).",
)
parser.add_argument(
    "--prior_fall_body_names",
    type=str,
    default="torso_link",
    help="[prior metrics] Comma-separated robot body names for fall check (e.g. torso_link,pelvis).",
)
parser.add_argument(
    "--prior_metrics_json",
    type=str,
    default="",
    help="[prior metrics] Optional path to write metrics JSON; default is next to the video folder or log_dir.",
)
parser.add_argument(
    "--fixed_mask_mode",
    type=str,
    default=None,
    help="[2B partial-mask tasks] Pin keypoint mask to this mode name (must match algorithm.mask_cfg.mode_spec). "
    "Mutually exclusive with --fixed_mask_mode_idx. When set with --video, adds a distinct RecordVideo name_prefix.",
)
parser.add_argument(
    "--fixed_mask_mode_idx",
    type=int,
    default=None,
    help="[2B partial-mask tasks] Pin keypoint mask by row index into mode_spec (same order as training). "
    "Mutually exclusive with --fixed_mask_mode.",
)
parser.add_argument(
    "--partial_2b_video_keypoint_vis",
    action="store_true",
    default=False,
    help="[Partial-Masked-2B tracking task only] With --video and --fixed_mask_mode*, motion debug vis shows "
    "anchor frames plus only the visible keypoint bodies for the pinned mask mode; goal body markers use "
    "motion world poses (same convention as goal anchor). Ignored on other tasks.",
)
parser.add_argument(
    "--video_all_bodies_world_frame",
    action="store_true",
    default=False,
    help="With --video: draw the goal frame pair for all 14 motion-tracked bodies in motion world "
    "frame (overrides the default anchor-frame rendering). Useful for visualizing the teacher's "
    "full-body Cartesian tracking on any tracking task.",
)
parser.add_argument(
    "--force_rough_terrain",
    action="store_true",
    default=False,
    help="Replace the scene ground with a 1x1 random-uniform rough tile (3–6 cm, in-distribution "
    "with HeadHands stairs/rough) so play/video is guaranteed not to spawn on a plane. "
    "Also disables the terrain-level curriculum.",
)
parser.add_argument(
    "--future_mode",
    type=str,
    default="oracle",
    choices=("oracle", "hold", "mapper"),
    help="KP5 future slots: oracle=GT (default), hold=copy current target, mapper=causal "
    "FutureIntentMapper from K_<=t only. Identical clips/seed/mask otherwise.",
)
parser.add_argument(
    "--mapper_path",
    type=str,
    default="/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt",
    help="[future_mode=mapper] Path to mapper_best.pt.",
)
parser.add_argument(
    "--poi5_dot_vis",
    action="store_true",
    default=False,
    help="With --video, on any tracking task with a motion command: render ONLY the 5 points "
    "of interest (COTRAIN_KP5_BODIES = torso + L/R wrist + L/R ankle) as world-frame dot "
    "markers (green=robot, red=goal), hide the anchor frame triads, and force the goal fully "
    "visible (disable the goal-mask curriculum + pin motion.p_mask=0.0). Intended for "
    "visualizing the JC MUSE-Transformer teacher with every timestep seen.",
)
parser.add_argument(
    "--kp_vis_local_frame",
    action="store_true",
    default=False,
    help="With --modality=kp --video: render goal body frame pairs in the robot's *anchor* "
    "frame (local) instead of the default motion world frame. Diagnostic option — useful for "
    "checking the policy's anchor-relative tracking when world-frame anchor drift makes the "
    "absolute world-frame markers misleading.",
)
parser.add_argument(
    "--modality",
    type=str,
    default=None,
    choices=("kp", "jc"),
    help="[MUSE co-train] Force the action source to one modality. Sets ``policy.pilot_kp_fraction`` "
    "to 1.0 (kp) or 0.0 (jc). Pair with --mask_modes for KP or --p_mask for JC.",
)
parser.add_argument(
    "--p_mask",
    type=float,
    default=None,
    help="[MUSE co-train --modality=jc] Per-step probability that the JC goal is masked (Bernoulli).",
)
parser.add_argument(
    "--mask_modes",
    type=str,
    default=None,
    help="[MUSE co-train --modality=kp] Pin a single visibility mode (e.g. pelvis_only, end_effector). "
    "Overrides ``mask_mode_spec``/``mask_mode_probs`` on the motion command before env init. Mutually "
    "exclusive with --fixed_mask_mode and --fixed_mask_mode_idx.",
)
parser.add_argument(
    "--fixed_vr_mask_mode",
    type=str,
    default=None,
    help="[VR-Tracking-Flat-G1-v0] Pin EE mask mode by name (must match motion.mask_mode_spec), e.g. left_ee, "
    "right_ee, both_ee. Mutually exclusive with --fixed_vr_mask_mode_idx.",
)
parser.add_argument(
    "--fixed_vr_mask_mode_idx",
    type=int,
    default=None,
    help="[VR-Tracking-Flat-G1-v0] Pin EE mask mode by row index into motion.mask_mode_spec (same order as cfg).",
)
parser.add_argument(
    "--vr_video_keypoint_vis",
    action="store_true",
    default=False,
    help="[VR-Tracking + --video + fixed VR mode] Motion debug vis: anchor frames plus only wrist keypoints for "
    "the pinned mode; goal keypoints drawn in motion world frame.",
)
parser.add_argument(
    "--prior_checkpoint",
    type=str,
    default=None,
    help="Path to PULSE-style .pt with prior.* and student_core.decoder.* (required for VR-Tracking-Flat-G1-v0; "
    "for VR-Tracking-Joint-Flat-G1-v0 policy init only, not obs normalizer).",
)
parser.add_argument(
    "--latent_rl_checkpoint",
    type=str,
    default=None,
    help="[VR-Tracking-Joint-Flat-G1-v0] If set, split obs normalizer is loaded from this latent VR PPO .pt; "
    "otherwise from the play resume checkpoint.",
)
parser.add_argument("--vr_residual_scale", type=float, default=None, help="[VR] Override actions.joint_pos.residual_scale.")
parser.add_argument("--vr_latent_dim", type=int, default=None, help="[VR] Override actions.joint_pos.latent_dim.")
parser.add_argument(
    "--vr_proprio_history_length",
    type=int,
    default=None,
    help="[VR] Override actions.joint_pos.proprio_history_length.",
)
parser.add_argument(
    "--vr_mask_mode_probs",
    type=str,
    default=None,
    help="[VR] Comma-separated mask_mode_probs override (same order as mask_mode_spec).",
)
parser.add_argument(
    "--vr_compact_goal_obs",
    type=str,
    default=None,
    help="[VR] true/false: compact vs full keypoint goal observation layout.",
)
parser.add_argument(
    "--vr_use_full_goal_obs_with_distill_pretrain",
    type=str,
    default="true",
    help="[VR] When true and warmstart-from-distill is set, force full goal obs unless --vr_compact_goal_obs is set.",
)
parser.add_argument(
    "--warmstart_from_masked_partial_kp_tracker",
    action="store_true",
    default=False,
    help="[VR play] Must match training if using --warmstart_lock_full_normalizer (obs normalizer checkpoint).",
)
parser.add_argument(
    "--warmstart_checkpoint",
    type=str,
    default=None,
    help="[VR play] Checkpoint path for warmstart / normalizer lock.",
)
parser.add_argument(
    "--warmstart_lock_full_normalizer",
    action="store_true",
    default=False,
    help="[VR play] Load full obs normalizer from --warmstart_checkpoint and disable split normalizer.",
)
parser.set_defaults(play_video_disable_non_timeout_terminations=True)
parser.add_argument(
    "--no_play_video_disable_non_timeout_terminations",
    dest="play_video_disable_non_timeout_terminations",
    action="store_false",
    help="When recording with --video, keep all termination terms (default: disable non-timeout terms such as "
    "bad anchor / bad EE tracking failures so clips are not cut short).",
)
parser.add_argument(
    "--synth_eval",
    action="store_true",
    default=False,
    help="[synthetic torso-only motions] Restrict goal-marker debug-vis to ``torso_link`` only "
    "(non-torso bodies in synthetic npzs are NaN at t>=1; rendering markers there would crash "
    "the visualizer). Has no other effect — pair with --start_frame 0 (forces start_from_beginning) "
    "and --mask_modes kp5_torso / kp6_torso to pin torso-only visibility. The default --video "
    "behaviour already disables non-timeout terminations, which is what we want for synth probes.",
)
parser.add_argument(
    "--synth_wrist_history",
    action="store_true",
    default=False,
    help="[--synth_eval, wrist-writing recipes only] Render a trail of small green spheres "
    "showing every Nth past position of the writing body. Lets you see *how well* the wrist "
    "tracked the static red letter trail. Subsample is set by --synth_wrist_history_every "
    "(default 3 sim steps).",
)
parser.add_argument(
    "--synth_wrist_history_every",
    type=int,
    default=3,
    help="With --synth_wrist_history: record a green dot every N sim steps (default 3).",
)
parser.add_argument(
    "--prior_debug_fall",
    action="store_true",
    default=False,
    help="[prior_sample] Print play.py fall heuristic each step (or every N with --prior_debug_fall_every); "
    "not an MDP termination—same check as prior metrics.",
)
parser.add_argument(
    "--prior_debug_fall_every",
    type=int,
    default=1,
    help="With --prior_debug_fall, print at least every N env steps; always prints on env0 fallen↔not-fallen edge.",
)
parser.add_argument(
    "--prior_freeze_after_fallen",
    type=int,
    default=5,
    help="[prior_sample + --video] After first fall (play.py heuristic), simulate this many more env steps then pause "
    "physics so the video keeps recording static frames until --video_length. Use -1 to disable.",
)
parser.add_argument(
    "--prior_freeze_max_extra_steps",
    type=int,
    default=-1,
    help="After physics freeze, stop the play loop after this many *additional* env steps (-1 = no limit; run until "
    "--video_length). Use 0 to end immediately after freeze (short MP4; avoids long frozen tail).",
)
parser.add_argument(
    "--prior_freeze_log_every",
    type=int,
    default=50,
    help="While physics is paused for prior video, print progress + wall time every N env steps (0 = off). "
    "Unused when --prior-freeze-pad-tail is on (default).",
)
parser.set_defaults(prior_freeze_pad_tail=True)
parser.add_argument(
    "--no_prior_freeze_pad_tail",
    dest="prior_freeze_pad_tail",
    action="store_false",
    help="[prior_sample + --video] Do not pad MP4 with ffmpeg; keep stepping until --video_length (slow frozen tail) "
    "or use --prior-freeze-max-extra-steps. Default is to pad (clone last frame via ffmpeg tpad).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False):
    ws = getattr(args_cli, "warmstart_checkpoint", None)
    if ws in (None, ""):
        raise ValueError(
            "--warmstart_from_masked_partial_kp_tracker requires --warmstart_checkpoint=/path/to.pt"
        )

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dataclasses
import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.modules import (
    LatentBottleneckMUSE,
    LatentBottleneckMUSETransformer,
    LatentBottleneckPULSE,
)
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.mdp import fall_to_ground
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _resolve_fixed_mask_mode_for_play(
    args_cli: argparse.Namespace, agent_cfg: RslRlOnPolicyRunnerCfg
) -> tuple[int | None, str | None]:
    """Return (mode_index, slug_for_video_prefix) or (None, None) if no fixed mask requested."""
    name_arg = getattr(args_cli, "fixed_mask_mode", None)
    idx_arg = getattr(args_cli, "fixed_mask_mode_idx", None)
    has_name = name_arg is not None and str(name_arg).strip() != ""
    has_idx = idx_arg is not None
    if not has_name and not has_idx:
        return None, None
    if has_name and has_idx:
        raise ValueError("Pass at most one of --fixed_mask_mode and --fixed_mask_mode_idx.")

    alg = getattr(agent_cfg, "algorithm", None)
    mask_cfg = getattr(alg, "mask_cfg", None) if alg is not None else None
    if not mask_cfg or not isinstance(mask_cfg, dict) or not mask_cfg.get("mode_spec"):
        raise ValueError(
            "Fixed mask mode requires algorithm.mask_cfg.mode_spec "
            "(e.g. Partial-Masked-Residual-Latent-Distill-2B-General-Tracking-Flat-G1-v0)."
        )
    mode_spec = mask_cfg["mode_spec"]
    mode_names = list(mode_spec.keys())

    if has_name:
        name = str(name_arg).strip()
        if name not in mode_spec:
            raise ValueError(f"Unknown --fixed_mask_mode {name!r}. Valid modes: {mode_names}")
        idx = mode_names.index(name)
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "mode"
        return idx, slug

    idx = int(idx_arg)
    if idx < 0 or idx >= len(mode_names):
        raise ValueError(
            f"--fixed_mask_mode_idx={idx} out of range (num_modes={len(mode_names)}); modes={mode_names}"
        )
    mn = mode_names[idx]
    slug = f"idx{idx}_{''.join(c if c.isalnum() or c in '-_' else '_' for c in mn)}"
    return idx, slug


# Gym task id for partial-mask 2B distillation (see whole_body_tracking.tasks.tracking.config.g1).
PARTIAL_2B_TRACKING_GYM_TASK_ID = "Partial-Masked-Residual-Latent-Distill-2B-General-Tracking-Flat-G1-v0"


def _is_partial_2b_tracking_task(task: str | None) -> bool:
    return bool(task and str(task).strip() == PARTIAL_2B_TRACKING_GYM_TASK_ID)


VR_TRACKING_GYM_TASK_ID = "VR-Tracking-Flat-G1-v0"


def _is_vr_tracking_task(task: str | None) -> bool:
    t = str(task or "").strip()
    return t == VR_TRACKING_GYM_TASK_ID or t == "VR-Tracking-Joint-Flat-G1-v0"


def _resolve_fixed_vr_mask_mode_for_play(
    args_cli: argparse.Namespace, task: str | None, motion_cfg: object | None
) -> tuple[int | None, str | None]:
    """Return (mode_index, slug_for_video_prefix) for VR EE mask modes from motion.mask_mode_spec."""
    if not _is_vr_tracking_task(task):
        return None, None
    name_arg = getattr(args_cli, "fixed_vr_mask_mode", None)
    idx_arg = getattr(args_cli, "fixed_vr_mask_mode_idx", None)
    has_name = name_arg is not None and str(name_arg).strip() != ""
    has_idx = idx_arg is not None
    if not has_name and not has_idx:
        return None, None
    if has_name and has_idx:
        raise ValueError("Pass at most one of --fixed_vr_mask_mode and --fixed_vr_mask_mode_idx.")
    if motion_cfg is None or not hasattr(motion_cfg, "mask_mode_spec"):
        raise ValueError("VR fixed mask mode requires env_cfg.commands.motion.mask_mode_spec.")
    mode_spec = dict(getattr(motion_cfg, "mask_mode_spec") or {})
    mode_names = list(mode_spec.keys())
    if not mode_names:
        raise ValueError("motion.mask_mode_spec is empty.")

    if has_name:
        name = str(name_arg).strip()
        if name not in mode_spec:
            raise ValueError(f"Unknown --fixed_vr_mask_mode {name!r}. Valid modes: {mode_names}")
        idx = mode_names.index(name)
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "mode"
        return idx, slug

    idx = int(idx_arg)
    if idx < 0 or idx >= len(mode_names):
        raise ValueError(
            f"--fixed_vr_mask_mode_idx={idx} out of range (num_modes={len(mode_names)}); modes={mode_names}"
        )
    mn = mode_names[idx]
    slug = f"idx{idx}_{''.join(c if c.isalnum() or c in '-_' else '_' for c in mn)}"
    return idx, slug


def _vr_visible_body_names_for_mode(motion_cfg: object, mode_idx: int) -> list[str]:
    mode_spec = dict(getattr(motion_cfg, "mask_mode_spec") or {})
    mode_names = list(mode_spec.keys())
    if mode_idx < 0 or mode_idx >= len(mode_names):
        return []
    return list(mode_spec[mode_names[mode_idx]])


def _resolve_motion_for_synth_meta(motion_arg: str | None) -> str | None:
    """Resolve ``--motion`` (file or directory) to a concrete .npz path for metadata reading.

    Returns the file path itself if motion_arg is a file, the first .npz under it
    if a directory, or None if neither. Used only to load ``_synth_*`` keys for the
    --synth_eval visualisation; the env's motion loader handles the actual playback.
    """
    import glob
    import os

    if not motion_arg:
        return None
    if os.path.isfile(motion_arg) and motion_arg.endswith(".npz"):
        return motion_arg
    if os.path.isdir(motion_arg):
        npzs = sorted(glob.glob(os.path.join(motion_arg, "*.npz")))
        return npzs[0] if npzs else None
    return None


def _load_synth_metadata(npz_path: str | None) -> dict | None:
    """Load ``_synth_*`` keys from a synthetic motion npz; None if absent / unreadable."""
    if not npz_path:
        return None
    import numpy as np

    try:
        d = np.load(npz_path)
    except Exception:  # noqa: BLE001
        return None
    if "_synth_primitive_bodies" not in d.files:
        return None
    out: dict = {
        "primitive_bodies": [str(b) for b in d["_synth_primitive_bodies"]],
        "stay_bodies": [str(b) for b in d["_synth_stay_bodies"]] if "_synth_stay_bodies" in d.files else [],
        "face_world_yaw": None,
        "letter_trail_xyz": None,
        "letter_trail_body": None,
    }
    if "_synth_face_world_yaw" in d.files:
        fwy = float(d["_synth_face_world_yaw"][0])
        if not (fwy != fwy):  # NaN check (NaN != NaN is True)
            out["face_world_yaw"] = fwy
    if "_synth_letter_trail_xyz" in d.files:
        out["letter_trail_xyz"] = d["_synth_letter_trail_xyz"].astype(np.float32)
    if "_synth_letter_trail_body" in d.files:
        out["letter_trail_body"] = str(d["_synth_letter_trail_body"][0])
    return out


def _spawn_synth_letter_trail(env, synth_meta: dict) -> None:
    """Spawn static red sphere markers at the letter-trail points; hide moving goal for that body.

    Called AFTER ``env = gym.make(...)`` so the motion command's debug visualizers
    already exist. Static markers are pinned to env_0's origin + the npz trail
    coordinates (which are in the recentered + yaw-rotated world frame the writer
    used). The visualizer reference is stashed on the env to prevent GC.
    """
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    trail_xyz = synth_meta.get("letter_trail_xyz")
    if trail_xyz is None or trail_xyz.shape[0] == 0:
        return

    base = env.unwrapped
    env_origins = base.scene.env_origins
    env_origin_0 = env_origins[0].detach().cpu().numpy()
    trail_world = trail_xyz.astype(np.float32) + env_origin_0[None, :].astype(np.float32)

    trail_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Synth/LetterTrail",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )
    viz = VisualizationMarkers(trail_cfg)
    translations = torch.tensor(trail_world, dtype=torch.float32, device=base.device)
    orientations = torch.zeros((trail_world.shape[0], 4), dtype=torch.float32, device=base.device)
    orientations[:, 0] = 1.0  # WXYZ identity quaternion
    viz.visualize(translations=translations, orientations=orientations)
    viz.set_visibility(True)
    base._synth_letter_trail_viz = viz  # keep alive for the duration of the rollout

    trail_body = synth_meta.get("letter_trail_body")
    if trail_body is not None:
        # Hide the MOVING goal marker for the trail body so we only see the static
        # red dots + the moving green robot wrist. The command's debug_vis was set
        # up at env init with motion_cfg.video_debug_vis_body_names pinned to the
        # primitive bodies (containing trail_body).
        try:
            motion = base.command_manager.get_term("motion")
            video_body_names = list(motion.cfg.video_debug_vis_body_names or [])
            if trail_body in video_body_names:
                idx = video_body_names.index(trail_body)
                if hasattr(motion, "goal_body_visualizers") and idx < len(motion.goal_body_visualizers):
                    motion.goal_body_visualizers[idx].set_visibility(False)
                    print(
                        f"[INFO]: --synth_eval letter trail: hid moving goal marker for "
                        f"{trail_body!r} (only static red trail + moving green wrist visible)."
                    )
        except Exception as _e:  # noqa: BLE001
            print(f"[INFO]: --synth_eval: could not hide moving goal marker for {trail_body!r}: {_e}")

    print(
        f"[INFO]: --synth_eval: spawned {trail_world.shape[0]} static red letter-trail markers "
        f"at body {synth_meta.get('letter_trail_body')!r}."
    )


def _setup_synth_wrist_history(env, synth_meta: dict, subsample_every: int) -> None:
    """Attach a per-step green-dot history recorder to ``env.unwrapped``.

    The rollout loop checks for the attribute ``_synth_wrist_history_state`` and
    appends + re-visualizes each step. We stash the state on the env rather than
    using a class to avoid passing it through play.py's nested control flow.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    trail_body = synth_meta.get("letter_trail_body")
    if not trail_body:
        return
    base = env.unwrapped
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Synth/WristHistory",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.3)),
            )
        },
    )
    viz = VisualizationMarkers(cfg)
    base._synth_wrist_history_state = {
        "viz": viz,
        "positions": [],
        "body_name": trail_body,
        "subsample_every": max(1, int(subsample_every)),
        "step_counter": 0,
    }
    print(
        f"[INFO]: --synth_wrist_history: recording every "
        f"{subsample_every} sim step(s) for body {trail_body!r} (green dots accumulate)."
    )


def _tick_synth_wrist_history(env) -> None:
    """Per-step callback: append current wrist pos + refresh visualizer.

    No-op if --synth_wrist_history wasn't enabled. Called from the rollout loop
    after each env.step().
    """
    base = env.unwrapped
    state = getattr(base, "_synth_wrist_history_state", None)
    if state is None:
        return
    state["step_counter"] += 1
    if state["step_counter"] % state["subsample_every"] != 0:
        return
    try:
        motion = base.command_manager.get_term("motion")
        body_names = list(motion.cfg.body_names or [])
        if state["body_name"] not in body_names:
            return
        idx = body_names.index(state["body_name"])
        # robot_body_pos_w: (num_envs, num_tracked_bodies, 3) in world frame.
        pos = motion.robot_body_pos_w[0, idx].detach().cpu().numpy().copy()
    except Exception:  # noqa: BLE001
        return
    state["positions"].append(pos)

    import numpy as np
    import torch

    arr = np.stack(state["positions"], axis=0).astype(np.float32)
    translations = torch.from_numpy(arr).to(base.device)
    orientations = torch.zeros((arr.shape[0], 4), dtype=torch.float32, device=base.device)
    orientations[:, 0] = 1.0  # WXYZ identity
    state["viz"].visualize(translations=translations, orientations=orientations)


def _strip_non_timeout_termination_terms(terminations_cfg) -> int:
    """Set each termination field with ``time_out=False`` to None. Returns how many were cleared."""
    if terminations_cfg is None:
        return 0
    try:
        fields = dataclasses.fields(terminations_cfg)
    except TypeError:
        return 0
    cleared = 0
    for f in fields:
        name = f.name
        term = getattr(terminations_cfg, name, None)
        if term is None or not hasattr(term, "time_out"):
            continue
        if not bool(getattr(term, "time_out", False)):
            setattr(terminations_cfg, name, None)
            cleared += 1
    return cleared


def _partial_2b_visible_body_names_for_mode(agent_cfg: RslRlOnPolicyRunnerCfg, mode_idx: int) -> list[str]:
    alg = getattr(agent_cfg, "algorithm", None)
    mask_cfg = getattr(alg, "mask_cfg", None) if alg is not None else None
    if not mask_cfg or not isinstance(mask_cfg, dict):
        return []
    mode_spec = mask_cfg.get("mode_spec") or {}
    mode_names = list(mode_spec.keys())
    if mode_idx < 0 or mode_idx >= len(mode_names):
        return []
    return list(mode_spec[mode_names[mode_idx]])


def _get_scene_env(env):
    e = env.unwrapped
    while hasattr(e, "unwrapped") and not hasattr(e, "scene"):
        e = e.unwrapped
    return e


def _pause_physics_if_available(base_env) -> bool:
    sim = getattr(base_env, "sim", None)
    if sim is None:
        return False
    pause = getattr(sim, "pause", None)
    if not callable(pause):
        return False
    try:
        pause()
        return True
    except Exception:
        return False


def _resume_physics_if_available(base_env) -> bool:
    sim = getattr(base_env, "sim", None)
    if sim is None:
        return False
    play = getattr(sim, "play", None)
    if not callable(play):
        return False
    try:
        play()
        return True
    except Exception:
        return False


def _parse_frame_rate(s: str) -> float:
    s = str(s).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def _find_recorded_mp4(video_folder: str) -> str | None:
    p = pathlib.Path(video_folder)
    if not p.is_dir():
        return None
    cands = sorted(p.glob("rl-video-step-*.mp4"))
    if cands:
        return str(cands[-1])
    any_mp4 = sorted(p.glob("*.mp4"))
    return str(any_mp4[-1]) if any_mp4 else None


def _ffprobe_fps_and_frame_count(path: str) -> tuple[float, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe: no video stream in {path}")
    st = streams[0]
    fps_s = st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1"
    fps = _parse_frame_rate(fps_s)
    nbf = st.get("nb_frames")
    if nbf not in (None, "N/A") and str(nbf).isdigit():
        return fps, int(nbf)
    dur_s = st.get("duration") or (data.get("format") or {}).get("duration")
    dur = float(dur_s) if dur_s else 0.0
    if dur <= 0 or fps <= 0:
        return fps, 0
    return fps, int(round(dur * fps))


def _pad_video_clone_last_frame(path: str, target_frames: int) -> bool:
    """Extend video by cloning the last frame so total duration ~= target_frames at detected fps."""
    if shutil.which("ffmpeg") is None:
        print("[prior_freeze] pad: ffmpeg not on PATH; skipping tail pad.")
        return False
    try:
        fps, n_have = _ffprobe_fps_and_frame_count(path)
    except Exception as exc:
        print(f"[prior_freeze] pad: ffprobe failed ({exc}); skipping tail pad.")
        return False
    if n_have <= 0 or fps <= 0:
        print("[prior_freeze] pad: could not infer fps/frame count; skipping tail pad.")
        return False
    need = int(target_frames) - int(n_have)
    if need <= 0:
        print(f"[prior_freeze] pad: already ~{n_have} frames (>= {target_frames}); no pad.")
        return True
    pad_sec = need / fps
    tmp = path + ".pad_tmp.mp4"
    vf = f"tpad=stop_mode=clone:stop_duration={pad_sec:.6f}"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        tmp,
    ]
    try:
        subprocess.check_call(cmd)
        os.replace(tmp, path)
    except Exception as exc:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        print(f"[prior_freeze] pad: ffmpeg failed ({exc}); leaving unpadded video.")
        return False
    print(
        f"[prior_freeze] pad: extended ~{n_have} -> ~{target_frames} frames "
        f"(+{need} cloned-frame seconds, +{pad_sec:.3f}s at {fps:.3f} fps): {path}"
    )
    return True


def _prior_metrics_motion_key(motion: str) -> str:
    """Stable dict key for merging metrics per motion (absolute normalized path)."""
    return os.path.normpath(os.path.abspath(motion))


def _merge_write_prior_rollout_metrics(out_json: str, metrics_payload: dict) -> int:
    """Merge one motion into prior_rollout_metrics.json without dropping other motions."""
    motion_key = _prior_metrics_motion_key(str(metrics_payload.get("motion", "")))
    motions: dict = {}

    if os.path.isfile(out_json):
        try:
            with open(out_json, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[Prior metrics] WARNING: could not read {out_json} ({exc}); starting a new metrics file.")
            existing = None
        if isinstance(existing, dict):
            if "motions" in existing and isinstance(existing["motions"], dict):
                motions = dict(existing["motions"])
            elif "motion" in existing:
                leg_key = _prior_metrics_motion_key(str(existing["motion"]))
                motions[leg_key] = existing

    motions[motion_key] = metrics_payload
    out_data = {"schema_version": 1, "motions": motions}

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    tmp_path = out_json + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    os.replace(tmp_path, out_json)
    return len(motions)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if bool(getattr(args_cli, "force_rough_terrain", False)):
        from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg

        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            curriculum=False,
            sub_terrains={
                "rough": HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(0.03, 0.06),
                    noise_step=0.01,
                ),
            },
        )
        env_cfg.scene.terrain.max_init_terrain_level = None
        if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
            if hasattr(env_cfg.curriculum, "terrain_levels"):
                env_cfg.curriculum.terrain_levels = None
                print("[play] --force_rough_terrain: disabled curriculum term: terrain_levels.")
        print("[play] --force_rough_terrain: 1x1 HfRandomUniform tile, noise 3–6 cm.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

    else:
        direct_resume_path = getattr(agent_cfg, "resume_checkpoint_path", None)
        if direct_resume_path:
            resume_path = os.path.abspath(str(direct_resume_path))
            print(f"[INFO]: Loading model checkpoint from direct path: {resume_path}")
        else:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Load policy configuration from checkpoint's params/agent.yaml to ensure compatibility
    # This overrides the default configuration with the checkpoint's actual configuration
    checkpoint_dir = os.path.dirname(resume_path)
    params_yaml_path = os.path.join(checkpoint_dir, "params", "agent.yaml")
    if os.path.exists(params_yaml_path):
        import yaml
        with open(params_yaml_path, 'r') as f:
            checkpoint_cfg = yaml.safe_load(f)

        # Override policy configuration from checkpoint
        if 'policy' in checkpoint_cfg:
            policy_cfg = checkpoint_cfg['policy']
            if 'ref_vel_skip_first_layer' in policy_cfg:
                agent_cfg.policy.ref_vel_skip_first_layer = policy_cfg['ref_vel_skip_first_layer']
                print(f"[Play] Loaded ref_vel_skip_first_layer={policy_cfg['ref_vel_skip_first_layer']} from checkpoint")
            if 'ref_vel_dim' in policy_cfg:
                agent_cfg.policy.ref_vel_dim = policy_cfg['ref_vel_dim']

            # Latent-RL (MUSE-Kp-LatentRL): the adapter choice changes the policy
            # architecture (LoRA injects parametrized weights; residual adds g_phi),
            # but parse_rsl_rl_cfg() above rebuilt agent_cfg from the registry default
            # (adapter="full_ft") and Hydra overrides do not survive that. Without
            # restoring these from the run's own params, ppo_runner.load() loads a
            # full_ft policy and silently drops the checkpoint's LoRA/residual weights
            # -> the video shows the BASE distilled policy, not the RL-finetuned one.
            # Same "ensure compatibility" intent as ref_vel_* above; guarded by
            # hasattr so non-latent-RL tasks (whose policy cfg lacks these) are
            # untouched, and by membership so only saved fields are applied.
            _latent_rl_arch_keys = (
                "adapter", "lora_rank", "lora_alpha", "lora_targets",
                "residual_d_model", "residual_num_layers", "residual_nhead",
                "residual_ffn", "residual_last_layer_gain", "residual_alpha",
                # ObstacleReach: obstacle obs is appended to the policy obs; these size the
                # actor-critic's obstacle strip + the corrector's obstacle head. Must be
                # restored or play rebuilds the policy without them (encoder dim mismatch).
                "obstacle_feat_dim", "obstacle_n",
                "terrain_scan_dim", "terrain_r_max", "terrain_scan_zero",
            )
            _restored = {}
            for _k in _latent_rl_arch_keys:
                if _k in policy_cfg and hasattr(agent_cfg.policy, _k):
                    setattr(agent_cfg.policy, _k, policy_cfg[_k])
                    _restored[_k] = policy_cfg[_k]
            if _restored:
                print(f"[Play] Restored latent-RL policy arch from checkpoint params: {_restored}")

    _task = getattr(args_cli, "task", None)
    _fm = getattr(args_cli, "fixed_mask_mode", None)
    _fmi = getattr(args_cli, "fixed_mask_mode_idx", None)
    if (_fm is not None and str(_fm).strip() != "") or _fmi is not None:
        if _is_vr_tracking_task(_task):
            raise ValueError(
                "VR-Tracking uses --fixed_vr_mask_mode / --fixed_vr_mask_mode_idx, not --fixed_mask_mode* "
                "(those are for partial-mask 2B only)."
            )

    if _is_partial_2b_tracking_task(_task):
        fixed_mask_mode_idx, fixed_mask_video_slug = _resolve_fixed_mask_mode_for_play(args_cli, agent_cfg)
    else:
        fixed_mask_mode_idx, fixed_mask_video_slug = None, None

    motion_cfg_for_vr = getattr(env_cfg.commands, "motion", None)
    fixed_vr_mode_idx, fixed_vr_video_slug = _resolve_fixed_vr_mask_mode_for_play(
        args_cli, _task, motion_cfg_for_vr
    )

    want_partial_2b_keypoint_vis = (
        bool(getattr(args_cli, "partial_2b_video_keypoint_vis", False))
        and _is_partial_2b_tracking_task(_task)
        and bool(args_cli.video)
        and fixed_mask_mode_idx is not None
    )
    want_vr_video_keypoint_vis = (
        bool(getattr(args_cli, "vr_video_keypoint_vis", False))
        and _is_vr_tracking_task(_task)
        and bool(args_cli.video)
        and fixed_vr_mode_idx is not None
    )
    if bool(getattr(args_cli, "partial_2b_video_keypoint_vis", False)):
        if not _is_partial_2b_tracking_task(_task):
            print(
                "[Play] WARNING: --partial_2b_video_keypoint_vis applies only to "
                f"{PARTIAL_2B_TRACKING_GYM_TASK_ID}; ignoring."
            )
        elif not bool(args_cli.video):
            print("[Play] WARNING: --partial_2b_video_keypoint_vis requires --video; ignoring.")
        elif fixed_mask_mode_idx is None:
            print(
                "[Play] WARNING: --partial_2b_video_keypoint_vis requires --fixed_mask_mode or "
                "--fixed_mask_mode_idx; ignoring."
            )
    if bool(getattr(args_cli, "vr_video_keypoint_vis", False)):
        if not _is_vr_tracking_task(_task):
            print(
                "[Play] WARNING: --vr_video_keypoint_vis applies only to "
                f"{VR_TRACKING_GYM_TASK_ID}; ignoring."
            )
        elif not bool(args_cli.video):
            print("[Play] WARNING: --vr_video_keypoint_vis requires --video; ignoring.")
        elif fixed_vr_mode_idx is None:
            print(
                "[Play] WARNING: --vr_video_keypoint_vis requires --fixed_vr_mask_mode or "
                "--fixed_vr_mask_mode_idx; ignoring."
            )

    # Co-train modality / mask override: apply env_cfg overrides BEFORE ``gym.make`` so the
    # command term is built with the single-mode spec. Mutually exclusive with the
    # ``--fixed_mask_mode`` / ``--fixed_mask_mode_idx`` flags (those look up names in the
    # training mode list, which doesn't include eval-only modes like ``pelvis_only``).
    play_cotrain_modality = getattr(args_cli, "modality", None)
    play_cotrain_mask_modes = getattr(args_cli, "mask_modes", None)
    play_cotrain_p_mask = getattr(args_cli, "p_mask", None)

    def _apply_kp_single_mode_env_overrides(mode_name: str) -> None:
        """Pin env.commands.motion to a single KP mask mode + configure video kp-vis + disable curricula."""
        from whole_body_tracking.tasks.tracking.config.g1.mask_modes import eval_single_mode_spec

        spec, probs = eval_single_mode_spec(mode_name)
        if not hasattr(env_cfg.commands.motion, "mask_mode_spec"):
            raise RuntimeError(
                "--mask_modes requires a partial-masked command term "
                "(env_cfg.commands.motion must have mask_mode_spec)."
            )
        env_cfg.commands.motion.mask_mode_spec = spec
        env_cfg.commands.motion.mask_mode_probs = probs
        print(f"[play] KP mask pinned: mask_modes={mode_name!r} → single-mode spec {spec}.")
        if bool(args_cli.video):
            motion_cfg = env_cfg.commands.motion
            body_names_all = list(getattr(motion_cfg, "body_names", []) or [])
            visible_in_spec = list(spec[mode_name])
            visible = [n for n in body_names_all if n in set(visible_in_spec)]
            if not visible:
                print(
                    f"[play] WARNING: KP keypoint-vis: no intersection between motion.body_names "
                    f"and mode bodies {visible_in_spec!r}; using default motion debug vis."
                )
            else:
                motion_cfg.video_debug_vis_body_names = visible
                use_world = not bool(getattr(args_cli, "kp_vis_local_frame", False))
                motion_cfg.video_debug_vis_goal_bodies_world_frame = use_world
                print(
                    f"[play] KP video keypoint-vis: anchor + body frame pairs for {visible} "
                    f"(goal bodies in {'motion world' if use_world else 'robot anchor (local)'} frame)."
                )
        if hasattr(env_cfg, "curriculum"):
            for _term_name in ("keypoint_mask_mode", "goal_mask_p"):
                if hasattr(env_cfg.curriculum, _term_name) and getattr(env_cfg.curriculum, _term_name) is not None:
                    setattr(env_cfg.curriculum, _term_name, None)
                    print(f"[play] Disabled curriculum term: {_term_name} (eval has manual override).")

    if play_cotrain_modality is not None:
        if (
            getattr(args_cli, "fixed_mask_mode", None) is not None
            or getattr(args_cli, "fixed_mask_mode_idx", None) is not None
        ):
            raise ValueError(
                "--modality is mutually exclusive with --fixed_mask_mode / --fixed_mask_mode_idx."
            )
        if play_cotrain_modality == "jc":
            if play_cotrain_mask_modes is not None:
                raise ValueError("--mask_modes is only meaningful for --modality=kp.")
            # JC path still wants curricula disabled — same as KP.
            if hasattr(env_cfg, "curriculum"):
                for _term_name in ("keypoint_mask_mode", "goal_mask_p"):
                    if hasattr(env_cfg.curriculum, _term_name) and getattr(env_cfg.curriculum, _term_name) is not None:
                        setattr(env_cfg.curriculum, _term_name, None)
                        print(f"[play] Disabled curriculum term: {_term_name} (eval has manual override).")
        else:  # kp
            if play_cotrain_p_mask is not None:
                raise ValueError("--p_mask is only meaningful for --modality=jc.")
            if play_cotrain_mask_modes is None:
                play_cotrain_mask_modes = "pelvis_only"
            _apply_kp_single_mode_env_overrides(play_cotrain_mask_modes)
    elif play_cotrain_mask_modes is not None:
        # Single-modality KP tasks (e.g. MUSE-Kp-Distill) accept --mask_modes directly: no
        # --modality flag because there's only one modality. Apply the same env-side override
        # as the cotrain --modality=kp branch; skip the policy-side pilot/act_inference patch
        # (the policy has no pilot_kp_fraction attribute).
        if (
            getattr(args_cli, "fixed_mask_mode", None) is not None
            or getattr(args_cli, "fixed_mask_mode_idx", None) is not None
        ):
            raise ValueError(
                "--mask_modes is mutually exclusive with --fixed_mask_mode / --fixed_mask_mode_idx."
            )
        _apply_kp_single_mode_env_overrides(play_cotrain_mask_modes)

    # All-bodies world-frame goal markers (e.g. for teacher rollout visualization). Independent
    # of cotrain modality / mask_mode logic — applies on any tracking task with a motion command.
    if bool(getattr(args_cli, "video_all_bodies_world_frame", False)):
        if not bool(args_cli.video):
            print("[play] WARNING: --video_all_bodies_world_frame requires --video; ignoring.")
        elif hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(
            env_cfg.commands.motion, "video_debug_vis_goal_bodies_world_frame"
        ):
            env_cfg.commands.motion.video_debug_vis_goal_bodies_world_frame = True
            # Leave video_debug_vis_body_names = None (resolves to all 14 bodies, the default).
            env_cfg.commands.motion.video_debug_vis_body_names = None
            print(
                "[play] All-bodies world-frame goal markers enabled "
                "(video_debug_vis_goal_bodies_world_frame=True, body_names=ALL)."
            )

    # Only-5-POI world-frame dot visualization (e.g. JC MUSE-Transformer teacher,
    # every timestep visible). Takes precedence over --video_all_bodies_world_frame.
    if bool(getattr(args_cli, "poi5_dot_vis", False)):
        if not bool(args_cli.video):
            print("[play] WARNING: --poi5_dot_vis requires --video; ignoring.")
        elif hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(
            env_cfg.commands.motion, "video_debug_vis_goal_bodies_world_frame"
        ):
            from whole_body_tracking.tasks.tracking.config.g1.mask_modes import COTRAIN_KP5_BODIES

            motion_cfg = env_cfg.commands.motion
            motion_cfg.video_debug_vis_body_names = list(COTRAIN_KP5_BODIES)
            motion_cfg.video_debug_vis_goal_bodies_world_frame = True
            if hasattr(motion_cfg, "video_debug_vis_hide_anchor"):
                motion_cfg.video_debug_vis_hide_anchor = True
            # Force the goal fully visible: drop the goal-mask curriculum so it can't
            # ramp p_mask at episode resets (p_mask is also pinned post-env-creation).
            if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
                for _term_name in ("keypoint_mask_mode", "goal_mask_p"):
                    if hasattr(env_cfg.curriculum, _term_name) and getattr(env_cfg.curriculum, _term_name) is not None:
                        setattr(env_cfg.curriculum, _term_name, None)
                        print(f"[play] --poi5_dot_vis: disabled curriculum term: {_term_name}.")
            print(
                f"[play] POI-5 world-frame dot vis enabled: {list(COTRAIN_KP5_BODIES)} "
                "(anchor frame hidden, goal fully visible)."
            )
        else:
            print(
                "[play] WARNING: --poi5_dot_vis: motion command has no video_debug_vis_* cfg; "
                "ignoring."
            )

    if args_cli.motion is not None:
        print(f"[INFO]: Using motion directory or file from CLI: {args_cli.motion}")
        env_cfg.commands.motion.motion = args_cli.motion
        # Optionally disable motion group sampling ratios for evaluation
        if args_cli.disable_motion_group_sampling and hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
            env_cfg.commands.motion.motion_group_sampling_ratios = None
            print("[INFO]: Disabled motion group sampling ratios for evaluation (uniform sampling).")
        if args_cli.random_init_frame and hasattr(env_cfg.commands.motion, "random_init_frame"):
            env_cfg.commands.motion.random_init_frame = True
            reset_xy = not args_cli.no_reset_base_xy_to_origin
            if hasattr(env_cfg.commands.motion, "reset_base_xy_to_origin"):
                env_cfg.commands.motion.reset_base_xy_to_origin = reset_xy
            print(
                f"[INFO]: Random init frame enabled (uniform across each motion clip); "
                f"reset_base_xy_to_origin={reset_xy}."
            )
        elif args_cli.start_frame is not None and hasattr(env_cfg.commands.motion, "start_frame"):
            env_cfg.commands.motion.start_from_beginning = True
            env_cfg.commands.motion.start_frame = args_cli.start_frame
            print(f"[INFO]: Forcing motion playback to start from frame {args_cli.start_frame}.")
        if args_cli.synth_eval and hasattr(env_cfg.commands.motion, "video_debug_vis_body_names"):
            # Read metadata stashed by the synth motion writer (``_synth_*`` keys) so
            # we know which body the recipe drives and whether to render a static
            # letter-trail. Two cases handled:
            #   (1) Torso recipe (squat/walk): primitive_bodies = ['torso_link']; no
            #       letter trail. Render the torso as a bigger sphere; hide anchor.
            #   (2) Wrist-writing recipe: primitive_bodies = ['right_wrist_yaw_link'];
            #       letter trail present. Render the wrist's *current* (green)
            #       sphere big, hide the moving goal (red) for that body, and spawn
            #       static red trail markers post-env (see below).
            #   (3) face_world_yaw set: override viewer.eye to a frontal pose so the
            #       camera sees the robot head-on.
            motion_cfg = env_cfg.commands.motion
            _synth_meta = _load_synth_metadata(_resolve_motion_for_synth_meta(args_cli.motion))
            if _synth_meta is None:
                # Backwards-compat: old torso recipes without the new metadata still work.
                _synth_meta = {
                    "primitive_bodies": ["torso_link"],
                    "stay_bodies": [],
                    "face_world_yaw": None,
                    "letter_trail_xyz": None,
                    "letter_trail_body": None,
                }
            # Stash for the post-env hook to access.
            _synth_meta_for_post_env = _synth_meta

            # Only render markers for the body the recipe is actually driving (NaN
            # elsewhere). Multiple primitive bodies are supported (rare).
            motion_cfg.video_debug_vis_body_names = list(_synth_meta["primitive_bodies"])
            print(
                f"[INFO]: --synth_eval: pinning motion debug-vis bodies to "
                f"{motion_cfg.video_debug_vis_body_names!r} (synthetic recipe driven by these)."
            )
            if hasattr(motion_cfg, "video_debug_vis_hide_anchor"):
                motion_cfg.video_debug_vis_hide_anchor = True
            try:
                import isaaclab.sim as _sim_utils
                from isaaclab.markers import VisualizationMarkersCfg as _VizMarkersCfg

                def _synth_sphere_cfg(rgb: tuple, radius: float = 0.05) -> _VizMarkersCfg:
                    return _VizMarkersCfg(
                        prim_path="/Visuals/Command/pose",
                        markers={
                            "sphere": _sim_utils.SphereCfg(
                                radius=radius,
                                visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                            )
                        },
                    )

                # Wrist-writing recipes (recipes with a static letter trail) use a smaller
                # green wrist marker so it doesn't overwhelm the trail. Torso recipes keep
                # the bigger 0.05 m sphere since the torso is a larger body part.
                _is_writing = _synth_meta.get("letter_trail_xyz") is not None
                _current_radius = 0.025 if _is_writing else 0.05
                if hasattr(motion_cfg, "current_body_visualizer_cfg"):
                    motion_cfg.current_body_visualizer_cfg = _synth_sphere_cfg(
                        (0.0, 1.0, 0.0), radius=_current_radius
                    )
                if hasattr(motion_cfg, "goal_body_visualizer_cfg"):
                    # Goal marker is hidden for the trail body anyway (in
                    # _spawn_synth_letter_trail) so radius here only affects non-trail
                    # bodies — match the current-body radius for consistency.
                    motion_cfg.goal_body_visualizer_cfg = _synth_sphere_cfg(
                        (1.0, 0.0, 0.0), radius=_current_radius
                    )
                print(
                    f"[INFO]: --synth_eval: body markers rendered as spheres "
                    f"(r={_current_radius:.3f} m, green=robot, red=goal); anchor frame triad hidden."
                )
            except Exception as _e:  # noqa: BLE001
                print(f"[INFO]: --synth_eval: could not override body-visualizer to sphere ({_e}); using default.")

            # Frontal camera that FOLLOWS the robot. The base env sets
            # origin_type="env" (camera pinned to the env origin) → as the robot
            # steps sideways to write wide words it walks out of frame. Switch to
            # origin_type="asset_root": the eye stays at the same world-frame offset
            # (same angle + distance) but relative to the robot's root, so the robot
            # — and the letters it's currently writing — stay in frame. Distance is
            # auto-zoomed so a good span of the word is visible around the robot.
            if _synth_meta["face_world_yaw"] is not None and hasattr(env_cfg, "viewer"):
                fwy = float(_synth_meta["face_world_yaw"])
                import math as _math
                _trail = _synth_meta.get("letter_trail_xyz")
                if _trail is not None and _trail.shape[0] > 0:
                    word_width = float(_trail[:, 1].max() - _trail[:, 1].min())
                    word_height = float(_trail[:, 2].max() - _trail[:, 2].min())
                    # Assume ~60° horizontal FOV (Isaac default-ish): a horizontal
                    # extent E fills the view at distance ≈ E / (2 · tan(30°)) =
                    # E · 0.866. Pick the bigger of width / (height × 16/9) so
                    # both axes fit, then add a small buffer. With follow-cam this
                    # only needs to frame the robot + the local stretch of the word,
                    # but a wider distance keeps more of the word visible at once.
                    extent = max(word_width, word_height * (16.0 / 9.0))
                    cam_dist = max(3.5, extent * 1.0 + 1.5)
                else:
                    cam_dist = 3.5
                    word_width = 0.0
                cam_x = cam_dist * _math.cos(fwy)
                cam_y = cam_dist * _math.sin(fwy)
                cam_z = 1.5
                env_cfg.viewer.eye = (cam_x, cam_y, cam_z)
                env_cfg.viewer.lookat = (0.0, 0.0, 0.0)  # look at the robot root (asset-relative)
                env_cfg.viewer.origin_type = "asset_root"  # FOLLOW the robot, not the env origin
                if hasattr(env_cfg.viewer, "asset_name"):
                    env_cfg.viewer.asset_name = "robot"
                print(
                    f"[INFO]: --synth_eval: follow-cam eye=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) "
                    f"origin_type=asset_root (tracks robot), face_world_yaw={fwy:.3f} rad, "
                    f"dist={cam_dist:.2f} m for word_width={word_width:.2f} m."
                )
        if want_partial_2b_keypoint_vis:
            motion_cfg = env_cfg.commands.motion
            body_names = list(getattr(motion_cfg, "body_names", []) or [])
            visible = _partial_2b_visible_body_names_for_mode(agent_cfg, int(fixed_mask_mode_idx))
            vis_set = set(visible)
            ordered = [n for n in body_names if n in vis_set]
            if not ordered:
                print(
                    "[Play] WARNING: --partial_2b_video_keypoint_vis: no intersection between motion.body_names "
                    f"and mode visible bodies {visible!r}; using default motion debug vis."
                )
            else:
                motion_cfg.video_debug_vis_body_names = ordered
                motion_cfg.video_debug_vis_goal_bodies_world_frame = True
                print(
                    "[Play] Partial-2B video keypoint vis: anchor + body frame pairs for "
                    f"{ordered} (goal bodies in motion world frame)."
                )
        if want_vr_video_keypoint_vis:
            motion_cfg = env_cfg.commands.motion
            body_names = list(getattr(motion_cfg, "body_names", []) or [])
            visible = _vr_visible_body_names_for_mode(motion_cfg, int(fixed_vr_mode_idx))
            vis_set = set(visible)
            ordered = [n for n in body_names if n in vis_set]
            if not ordered:
                print(
                    "[Play] WARNING: --vr_video_keypoint_vis: no intersection between motion.body_names "
                    f"and mode bodies {visible!r}; using default motion debug vis."
                )
            else:
                motion_cfg.video_debug_vis_body_names = ordered
                motion_cfg.video_debug_vis_goal_bodies_world_frame = True
                print(
                    "[Play] VR video keypoint vis: anchor + EE frame pairs for "
                    f"{ordered} (goal bodies in motion world frame)."
                )
        # Prior / encoder-as-proprio rollout videos: by default hide motion-command debug frames
        # (current vs goal reference markers).
        if (
            (args_cli.prior_sample or args_cli.encoder_as_proprio)
            and args_cli.video
            and not args_cli.prior_video_show_motion_debug_vis
            and not want_partial_2b_keypoint_vis
            and not want_vr_video_keypoint_vis
        ):
            if hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "debug_vis"):
                env_cfg.commands.motion.debug_vis = False
                print(
                    "[INFO]: Disabled motion command debug_vis for prior/encoder-as-proprio video "
                    "(omit goal/current frame markers). Pass --prior_video_show_motion_debug_vis to keep them."
                )
    else:
        raise ValueError("Motion file or motion directory is required for evaluation.")

    if not args_cli.enable_motion_randomization and hasattr(env_cfg, "commands"):
        motion_cfg = getattr(env_cfg.commands, "motion", None)
        if motion_cfg is not None:
            zero_ranges = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            if hasattr(motion_cfg, "pose_range"):
                motion_cfg.pose_range = dict(zero_ranges)
            if hasattr(motion_cfg, "velocity_range"):
                motion_cfg.velocity_range = dict(zero_ranges)
            if hasattr(motion_cfg, "joint_position_range"):
                motion_cfg.joint_position_range = (0.0, 0.0)
            print("[INFO]: Zeroed motion randomization ranges for evaluation.")

    if args_cli.disable_obs_noise and hasattr(env_cfg, "observations"):
        for group_name in ("policy", "teacher", "critic", "ref_vel_estimator"):
            if hasattr(env_cfg.observations, group_name):
                group_cfg = getattr(env_cfg.observations, group_name)
                if hasattr(group_cfg, "enable_corruption"):
                    group_cfg.enable_corruption = False
        print("[INFO]: Disabled observation corruption/noise for evaluation.")

    if args_cli.disable_events and hasattr(env_cfg, "events"):
        env_cfg.events = None
        print("[INFO]: Disabled event manager for evaluation.")

    if args_cli.encoder_as_proprio and args_cli.prior_sample:
        raise ValueError("--encoder_as_proprio is mutually exclusive with --prior_sample.")

    # ``--encoder_as_proprio`` dispatches by task family:
    #   - MUSE-Distill (masked-goal pathway): force motion command ``p_mask=1.0``
    #     (env already has maskable obs terms). The encoder receives the same fully-masked-goal
    #     input it saw under p_mask during training. The goal-mask curriculum is cleared so it
    #     cannot overwrite p_mask at episode resets.
    #   - PULSE-Distill / others: swap policy obs `command` -> command_self_target (absolute
    #     "self-target") and `motion_anchor_ori_b` -> motion_anchor_ori_b_identity (6-D identity).
    _task_str = str(getattr(args_cli, "task", "") or "")
    encoder_as_proprio_task_is_masked_goal = (
        args_cli.encoder_as_proprio and ("MUSE" in _task_str)
    )
    if args_cli.encoder_as_proprio:
        if encoder_as_proprio_task_is_masked_goal:
            # Drop the curriculum on the env cfg side. p_mask is force-set post-env-creation.
            if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
                env_cfg.curriculum = type(env_cfg.curriculum)()
                for _attr in list(vars(env_cfg.curriculum).keys()):
                    setattr(env_cfg.curriculum, _attr, None)
            print(
                "[INFO] encoder_as_proprio (masked-goal): cleared goal-mask curriculum; will force "
                "motion.p_mask = 1.0 after env creation."
            )
        else:
            from whole_body_tracking.tasks.tracking.mdp.observations import (
                command_self_target as _command_self_target,
                motion_anchor_ori_b_identity as _motion_anchor_ori_b_identity,
            )
            policy_obs_cfg = getattr(env_cfg.observations, "policy", None)
            if policy_obs_cfg is None:
                raise ValueError("--encoder_as_proprio requires env_cfg.observations.policy.")
            if not (hasattr(policy_obs_cfg, "command") and hasattr(policy_obs_cfg, "motion_anchor_ori_b")):
                raise ValueError(
                    "--encoder_as_proprio expects policy obs with both `command` and `motion_anchor_ori_b` "
                    "terms (PULSE-style)."
                )
            policy_obs_cfg.command.func = _command_self_target
            policy_obs_cfg.motion_anchor_ori_b.func = _motion_anchor_ori_b_identity
            print(
                "[INFO] encoder_as_proprio (PULSE-style): swapped policy obs `command` -> command_self_target, "
                "`motion_anchor_ori_b` -> motion_anchor_ori_b_identity. Unoise on anchor_ori is preserved."
            )

    task_name = str(getattr(args_cli, "task", "") or "")
    if "VR-Tracking" in task_name:
        if getattr(args_cli, "prior_checkpoint", None) in (None, ""):
            raise ValueError(
                f"Task {task_name!r} requires --prior_checkpoint=/path/to/pulse_student.pt "
                "(must contain prior.* and student_core.decoder.* for latent env; joint env uses it for obs norm)."
            )
        if "Joint" not in task_name:
            jp = env_cfg.actions.joint_pos
            jp.prior_checkpoint = os.path.abspath(str(args_cli.prior_checkpoint))
            if getattr(args_cli, "vr_residual_scale", None) is not None:
                jp.residual_scale = float(args_cli.vr_residual_scale)
            if getattr(args_cli, "vr_latent_dim", None) is not None:
                jp.latent_dim = int(args_cli.vr_latent_dim)
            if getattr(args_cli, "vr_proprio_history_length", None) is not None:
                jp.proprio_history_length = int(args_cli.vr_proprio_history_length)
        mcmd = env_cfg.commands.motion
        if getattr(args_cli, "vr_mask_mode_probs", None) not in (None, ""):
            parts = [float(x.strip()) for x in str(args_cli.vr_mask_mode_probs).split(",") if x.strip()]
            if len(parts) == 0:
                raise ValueError("--vr_mask_mode_probs must be a non-empty comma-separated list.")
            mcmd.mask_mode_probs = tuple(parts)
        if getattr(args_cli, "vr_compact_goal_obs", None) not in (None, ""):
            compact_raw = str(args_cli.vr_compact_goal_obs).strip().lower()
            if compact_raw in ("1", "true", "t", "yes", "y", "on"):
                mcmd.compact_goal_observation = True
            elif compact_raw in ("0", "false", "f", "no", "n", "off"):
                mcmd.compact_goal_observation = False
            else:
                raise ValueError("--vr_compact_goal_obs must be a boolean string (true/false).")
        else:
            auto_full_raw = str(getattr(args_cli, "vr_use_full_goal_obs_with_distill_pretrain", "true")).strip().lower()
            if auto_full_raw in ("1", "true", "t", "yes", "y", "on"):
                auto_full_goal_obs = True
            elif auto_full_raw in ("0", "false", "f", "no", "n", "off"):
                auto_full_goal_obs = False
            else:
                raise ValueError(
                    "--vr_use_full_goal_obs_with_distill_pretrain must be a boolean string (true/false)."
                )
            using_distill_pretrain = bool(
                getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False)
                or getattr(args_cli, "load_pretrain", None) not in (None, "")
            )
            if auto_full_goal_obs and using_distill_pretrain:
                mcmd.compact_goal_observation = False
                print(
                    "[INFO] VR goal obs auto-set to full (non-compact) for distillation pretrain compatibility."
                )
        if "Joint" not in task_name:
            print(
                f"[INFO] VR-Tracking env overrides: prior_checkpoint={jp.prior_checkpoint!r}, "
                f"compact_goal_observation={getattr(mcmd, 'compact_goal_observation', None)}"
            )
        else:
            print(
                f"[INFO] VR-Tracking-Joint play: prior_checkpoint={os.path.abspath(str(args_cli.prior_checkpoint))!r}, "
                f"compact_goal_observation={getattr(mcmd, 'compact_goal_observation', None)}"
            )

    # With --video (default): remove bad-tracking / failure terminations (Isaac: TerminationTermCfg.time_out=False).
    if (
        args_cli.video
        and getattr(args_cli, "play_video_disable_non_timeout_terminations", True)
        and not args_cli.prior_sample
        and not args_cli.encoder_as_proprio
        and getattr(env_cfg, "terminations", None) is not None
    ):
        n_strip = _strip_non_timeout_termination_terms(env_cfg.terminations)
        if n_strip > 0:
            print(
                f"[INFO]: Video: disabled {n_strip} non-timeout termination term(s) "
                "(e.g. bad anchor / bad EE); timeout-class terms remain until clock timeout is cleared below."
            )

    # Disable time-out termination during play to allow continuous replay
    terminations_cfg = getattr(env_cfg, "terminations", None)
    if terminations_cfg is not None and hasattr(terminations_cfg, "time_out"):
        print("[INFO]: Disabling timeout termination for playback run.")
        terminations_cfg.time_out = None
    if (args_cli.prior_sample or args_cli.encoder_as_proprio) and hasattr(env_cfg, "terminations"):
        print(
            "[INFO]: Disabling ALL terminations for playback run "
            "(prior / encoder-as-proprio rollouts ignore tracking terminations)."
        )
        env_cfg.terminations = None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # --synth_eval letter-trail: spawn static red sphere markers at the npz's letter
    # waypoints and hide the moving goal marker for the trail body. Must run AFTER
    # env creation so the motion command's debug visualizers exist.
    if args_cli.synth_eval:
        _meta_for_post = locals().get("_synth_meta_for_post_env")
        if _meta_for_post is not None and _meta_for_post.get("letter_trail_xyz") is not None:
            _spawn_synth_letter_trail(env, _meta_for_post)
            if args_cli.synth_wrist_history:
                _setup_synth_wrist_history(env, _meta_for_post, args_cli.synth_wrist_history_every)

    # MUSE encoder-as-proprio: force the trained masked-goal pathway by setting p_mask=1.0
    # on the motion command so every env's command tokens are NaN/identity each step (matching how
    # training experienced "no goal" frames).
    if encoder_as_proprio_task_is_masked_goal:
        _motion = env.unwrapped.command_manager.get_term("motion")
        if not hasattr(_motion, "p_mask"):
            raise RuntimeError(
                "encoder_as_proprio (masked-goal) expects the motion command to expose ``p_mask`` "
                "(MultiMotionCommand). Got: " + type(_motion).__name__
            )
        _motion.p_mask = 1.0
        print("[INFO] encoder_as_proprio (masked-goal): set motion.p_mask = 1.0 (all envs masked every step).")

    # --poi5_dot_vis: every timestep visible -> pin motion.p_mask = 0.0 (the
    # goal-mask curriculum was already dropped on the env cfg above).
    if bool(getattr(args_cli, "poi5_dot_vis", False)):
        _motion = env.unwrapped.command_manager.get_term("motion")
        if hasattr(_motion, "p_mask"):
            _motion.p_mask = 0.0
            print("[INFO] --poi5_dot_vis: set motion.p_mask = 0.0 (all timesteps visible).")

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    video_folder_path = ""
    if args_cli.video:
        # Use motion location root for folder (e.g. MOSAIC_Dataset/visualize/dance -> dance)
        motion_path = pathlib.Path(args_cli.motion)
        video_folder_name = motion_path.parent.name if motion_path.suffix else motion_path.name
        video_folder_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_folder_name) or "play"
        _fixed_std = getattr(args_cli, "prior_rollout_fixed_latent_std", None)
        if args_cli.prior_sample and _fixed_std is not None:
            videos_subdir_base = f"prior_videos_std_{_fixed_std:g}"
        elif args_cli.prior_sample:
            videos_subdir_base = "prior_videos"
        elif args_cli.encoder_as_proprio:
            videos_subdir_base = "encoder_as_proprio_videos"
        else:
            videos_subdir_base = "videos"
        # Nest tag dirs under one ``videos/`` (or ``prior_videos/`` / ``encoder_as_proprio_videos/``)
        # parent so the run dir doesn't fill up with sibling tag folders.
        if getattr(args_cli, "video_dir_tag", ""):
            videos_subdir = os.path.join(videos_subdir_base, args_cli.video_dir_tag)
        else:
            videos_subdir = videos_subdir_base
        video_folder_path = os.path.join(log_dir, videos_subdir, video_folder_name)
        _rec_prefix = "rl-video"
        _fm = str(getattr(args_cli, "future_mode", "oracle") or "oracle")
        if _fm != "oracle":
            _rec_prefix = f"rl-video_{_fm}"
        if fixed_mask_video_slug:
            _rec_prefix = f"rl-video_mask_{fixed_mask_video_slug}"
        elif fixed_vr_video_slug:
            _rec_prefix = f"rl-video_vr_{fixed_vr_video_slug}"
        video_kwargs = {
            "video_folder": video_folder_path,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": _rec_prefix,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print(f"[INFO] video_dir_tag={getattr(args_cli, 'video_dir_tag', '')!r} -> videos_subdir={videos_subdir}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    vr_joint_proprio_dim_play: int | None = None
    if "VR-Tracking-Joint" in task_name:
        _core_p = env.unwrapped
        _jpp = _core_p.action_manager.get_term("joint_pos")
        vr_joint_proprio_dim_play = getattr(_jpp, "_proprio_dim", None)
        if not isinstance(vr_joint_proprio_dim_play, int) or vr_joint_proprio_dim_play <= 0:
            raise RuntimeError("VR-Tracking-Joint play: missing ``_proprio_dim`` on joint_pos action term.")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    train_cfg = agent_cfg.to_dict()
    if vr_joint_proprio_dim_play is not None:
        train_cfg.setdefault("policy", {})
        train_cfg["policy"]["proprio_dim"] = int(vr_joint_proprio_dim_play)
    if getattr(args_cli, "critic_warmup_itrs", None) is not None and isinstance(train_cfg.get("algorithm"), dict):
        train_cfg["algorithm"]["critic_warmup_itrs"] = max(0, int(args_cli.critic_warmup_itrs))
    if "VR-Tracking-Joint" in task_name:
        _lc_play = getattr(args_cli, "latent_rl_checkpoint", None)
        if _lc_play not in (None, ""):
            train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(str(_lc_play))
            print(
                "[INFO] VR-Tracking-Joint play: obs_normalizer_checkpoint_path from latent_rl_checkpoint "
                f"{train_cfg['obs_normalizer_checkpoint_path']!r}"
            )
        else:
            train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(resume_path)
            print(
                "[INFO] VR-Tracking-Joint play: obs_normalizer_checkpoint_path from policy resume "
                f"{train_cfg['obs_normalizer_checkpoint_path']!r}"
            )
    elif "VR-Tracking" in task_name and getattr(args_cli, "prior_checkpoint", None) not in (None, ""):
        train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(str(args_cli.prior_checkpoint))
    if (
        "VR-Tracking" in task_name
        and getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False)
        and getattr(args_cli, "warmstart_lock_full_normalizer", False)
    ):
        ws_norm_path = os.path.abspath(str(args_cli.warmstart_checkpoint))
        train_cfg["obs_normalizer_checkpoint_path"] = ws_norm_path
        train_cfg["enable_rl_split_obs_normalizer"] = False
        print(
            "[INFO] Warmstart normalizer lock (play): "
            f"obs normalizer from {ws_norm_path!r}, split normalizer disabled."
        )

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False, load_critic=not args_cli.skip_critic)

    # Co-train modality: force single-modality piloting + apply runtime command overrides.
    if play_cotrain_modality is not None:
        _pol_ct = ppo_runner.alg.policy
        if not hasattr(_pol_ct, "pilot_kp_fraction"):
            raise RuntimeError(
                "--modality requires a MUSE co-train policy with ``pilot_kp_fraction`` "
                f"(got {type(_pol_ct).__name__})."
            )
        _pol_ct.pilot_kp_fraction = 1.0 if play_cotrain_modality == "kp" else 0.0

        # play.py drives the env via ``ppo_runner.get_inference_policy()``, which returns the
        # bound ``policy.act_inference``. The cotrain policy's ``act_inference(obs, modality=
        # "jc")`` defaults to JC and IGNORES ``pilot_kp_fraction`` (that attr is only used by
        # _act_pilot, which is the training/eval-rollout path). So we also bind a default
        # modality onto the instance method, otherwise play.py silently runs the wrong branch.
        _orig_act_inference = _pol_ct.act_inference
        _pinned_modality = play_cotrain_modality

        def _act_inference_pinned(obs):  # noqa: D401 — runtime monkey-patch
            return _orig_act_inference(obs, modality=_pinned_modality)

        _pol_ct.act_inference = _act_inference_pinned

        _env_u_ct = env.unwrapped
        _mt_ct = (
            _env_u_ct.command_manager.get_term("motion")
            if hasattr(_env_u_ct, "command_manager")
            else None
        )
        if play_cotrain_modality == "kp":
            if _mt_ct is not None and hasattr(_mt_ct, "set_eval_fixed_mask_mode_idx"):
                _mt_ct.set_eval_fixed_mask_mode_idx(0)
                if hasattr(_mt_ct, "resample_all_mask_modes"):
                    _mt_ct.resample_all_mask_modes()
            if _mt_ct is not None and hasattr(_mt_ct, "p_mask"):
                _mt_ct.p_mask = 0.0
        else:  # jc
            _p_mask_value = float(play_cotrain_p_mask) if play_cotrain_p_mask is not None else 0.0
            if _mt_ct is not None and hasattr(_mt_ct, "p_mask"):
                _mt_ct.p_mask = _p_mask_value
        print(
            f"[play] Co-train modality forced: {play_cotrain_modality!r} "
            f"(act_inference pinned, pilot_kp_fraction={_pol_ct.pilot_kp_fraction}, "
            f"p_mask={getattr(_mt_ct, 'p_mask', None)})."
        )

    # Partial-mask policies: always materialize a current mask for eval.
    # Training syncs from the motion command when available; otherwise sample on the policy.
    pol = ppo_runner.alg.policy
    if hasattr(pol, "sample_and_set_mask"):
        from rsl_rl.algorithms.mask_utils import sync_policy_keypoint_mask_from_motion_command

        env_u = env.unwrapped
        mt = env_u.command_manager.get_term("motion") if hasattr(env_u, "command_manager") else None
        if mt is not None and hasattr(mt, "set_eval_fixed_mask_mode_idx"):
            mt.set_eval_fixed_mask_mode_idx(
                int(fixed_mask_mode_idx) if fixed_mask_mode_idx is not None else None
            )
        if mt is not None and getattr(mt, "uses_command_manager_mask_sampling", False):
            if hasattr(mt, "resample_all_mask_modes"):
                mt.resample_all_mask_modes()
        synced = False
        if mt is not None:
            synced = sync_policy_keypoint_mask_from_motion_command(
                policy=pol,
                motion_term=mt,
                num_envs=int(env.num_envs),
                fixed_mode_idx=int(fixed_mask_mode_idx) if fixed_mask_mode_idx is not None else None,
            )
        if not synced:
            if fixed_mask_mode_idx is not None:
                pol.sample_and_set_mask(int(env.num_envs), fixed_mode_idx=int(fixed_mask_mode_idx))
                print(f"[Play] Fixed partial keypoint mask: mode_idx={fixed_mask_mode_idx}, num_envs={int(env.num_envs)}")
            else:
                pol.sample_and_set_mask(int(env.num_envs))
                print(
                    "[Play] Sampled partial keypoint mask for evaluation "
                    f"(num_envs={int(env.num_envs)}; uses configured mode_probs)."
                )
        else:
            if fixed_mask_mode_idx is not None:
                print(f"[Play] Fixed partial keypoint mask (motion command): mode_idx={fixed_mask_mode_idx}")
            else:
                print(
                    "[Play] Partial keypoint mask from motion command "
                    f"(num_envs={int(env.num_envs)})."
                )
    elif fixed_mask_mode_idx is not None:
        raise RuntimeError(
            "--fixed_mask_mode* was set but the loaded policy has no sample_and_set_mask "
            f"(got {type(pol).__name__})."
        )

    if _is_vr_tracking_task(_task):
        env_u = env.unwrapped
        mt = env_u.command_manager.get_term("motion") if hasattr(env_u, "command_manager") else None
        if mt is not None and hasattr(mt, "set_eval_fixed_mask_mode_idx"):
            mt.set_eval_fixed_mask_mode_idx(
                int(fixed_vr_mode_idx) if fixed_vr_mode_idx is not None else None
            )
            if hasattr(mt, "resample_all_mask_modes"):
                mt.resample_all_mask_modes()
            if fixed_vr_mode_idx is not None:
                print(f"[Play] VR motion mask pinned: mode_idx={fixed_vr_mode_idx}, num_envs={int(env.num_envs)}")

    # obtain the trained policy for inference
    # Check if velocity estimator is enabled
    use_velocity_estimator = (hasattr(ppo_runner.alg, 'ref_vel_estimator') and
                              ppo_runner.alg.ref_vel_estimator is not None and
                              hasattr(ppo_runner.alg, 'use_estimate_ref_vel') and
                              ppo_runner.alg.use_estimate_ref_vel)

    if use_velocity_estimator:
        print("[Play] Using velocity estimator for inference")
        # For velocity estimator, we need to handle observation processing manually
        # Cannot use get_inference_policy because it doesn't handle velocity augmentation
        policy = None  # Will process observations manually in the loop
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    future_mode = str(getattr(args_cli, "future_mode", "oracle") or "oracle")
    future_injector = None
    if future_mode != "oracle":
        from causal_future import CausalFutureInjector, DEFAULT_MAPPER

        future_injector = CausalFutureInjector(
            getattr(args_cli, "mapper_path", None) or DEFAULT_MAPPER,
            device=env.unwrapped.device,
        )
        print(f"[Play] causal future_mode={future_mode} (zero future leakage into policy obs).", flush=True)

    # Prior rollout (PULSE only): R(proprio) → z → decode (encoder bypassed)
    use_prior_sample = args_cli.prior_sample
    prior_latent_sampling = not bool(getattr(args_cli, "no_prior_latent_sampling", False))
    prior_rollout_fixed_latent_std = getattr(args_cli, "prior_rollout_fixed_latent_std", None)
    use_prior_metrics = bool(args_cli.prior_sample and not args_cli.no_prior_metrics)
    prior_metrics_ready = False
    if use_prior_sample:
        if isinstance(ppo_runner.alg.policy, LatentBottleneckPULSE):
            if prior_rollout_fixed_latent_std is not None and prior_latent_sampling:
                z_msg = f"z ~ N(μ, σ={prior_rollout_fixed_latent_std} fixed per dim)"
            elif prior_latent_sampling:
                z_msg = "z ~ N(μ,σ) (prior MLP σ)"
            else:
                z_msg = "z = μ (deterministic)"
                if prior_rollout_fixed_latent_std is not None:
                    z_msg += "; --prior_rollout_fixed_latent_std ignored without sampling"
            if use_prior_metrics:
                prior_metrics_ready = True
                print(
                    "[Play] Prior mode + metrics: "
                    f"{z_msg}, decode; logging fall height, prior entropy, action diversity, "
                    "and proprio–action sensitivity."
                )
            else:
                policy = lambda x: ppo_runner.alg.policy.act_prior_sample(
                    ppo_runner.obs_normalizer(x),
                    sample_latent=prior_latent_sampling,
                    fixed_latent_std=prior_rollout_fixed_latent_std,
                )
                print(
                    f"[Play] Prior mode: {z_msg}, decode to action (no encoder). "
                    "Record with --video to verify smoothness."
                )
        else:
            print("[Play] WARNING: --prior_sample requires LatentBottleneckPULSE policy. Using normal inference.")
            use_prior_sample = False
            use_prior_metrics = False

    use_encoder_as_proprio = bool(args_cli.encoder_as_proprio)
    encoder_as_proprio_sample = bool(getattr(args_cli, "encoder_as_proprio_sample", False))
    if use_encoder_as_proprio:
        _policy = ppo_runner.alg.policy
        if isinstance(_policy, (LatentBottleneckPULSE, LatentBottleneckMUSE, LatentBottleneckMUSETransformer)):
            z_mode = "z ~ N(μ, σ) (encoder posterior)" if encoder_as_proprio_sample else "z = μ (deterministic)"
            if isinstance(_policy, LatentBottleneckMUSETransformer):
                _kind = "MUSE-Transformer: motion.p_mask=1.0"
            elif isinstance(_policy, LatentBottleneckMUSE):
                _kind = "MUSE: motion.p_mask=1.0"
            else:
                _kind = "PULSE-style: self-target / identity"
            print(f"[Play] encoder_as_proprio mode ({_kind}): {z_mode}, decode to action.")
        else:
            print(
                "[Play] WARNING: --encoder_as_proprio requires LatentBottleneckPULSE, "
                "LatentBottleneckMUSE, or LatentBottleneckMUSETransformer policy. "
                "Using normal inference."
            )
            use_encoder_as_proprio = False

    fall_body_names = [n.strip() for n in args_cli.prior_fall_body_names.split(",") if n.strip()]
    fall_asset_cfg = SceneEntityCfg("robot", body_names=fall_body_names)
    if use_prior_sample and args_cli.prior_debug_fall:
        print(
            "[Play] prior_debug_fall: logging heuristic fall check after env steps "
            f"(min_z < {float(args_cli.prior_fall_min_height)} m on {fall_body_names}); "
            "this is not an MDP termination and does not reset the env."
        )

    freeze_enabled = (
        bool(args_cli.video)
        and use_prior_sample
        and int(args_cli.prior_freeze_after_fallen) >= 0
        and not (use_prior_metrics and prior_metrics_ready)
    )
    if (
        use_prior_sample
        and args_cli.video
        and int(args_cli.prior_freeze_after_fallen) >= 0
        and (use_prior_metrics and prior_metrics_ready)
    ):
        print(
            "[Play] prior_freeze_after_fallen is ignored while prior metrics are enabled "
            "(full trajectory is recorded for JSON)."
        )

    '''
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    # Get velocity estimator info if available
    ref_vel_estimator = None
    ref_vel_estimator_obs_dim = None
    if hasattr(ppo_runner.alg, 'ref_vel_estimator') and ppo_runner.alg.ref_vel_estimator is not None:
        ref_vel_estimator = ppo_runner.alg.ref_vel_estimator
        if hasattr(ppo_runner.alg, 'ref_vel_estimator_obs_shape') and ppo_runner.alg.ref_vel_estimator_obs_shape is not None:
            ref_vel_estimator_obs_dim = ppo_runner.alg.ref_vel_estimator_obs_shape[0]
    
    export_motion_policy_as_onnx(
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
        ref_vel_estimator=ref_vel_estimator,
        ref_vel_estimator_obs_dim=ref_vel_estimator_obs_dim,
    )

    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    '''


    base_sim_env = _get_scene_env(env)
    sim_was_paused = False
    physics_frozen = False
    fall_first_env_step: int | None = None
    zero_actions: torch.Tensor | None = None

    # reset environment
    env.reset()
    timestep = 0
    prior_actions: list[torch.Tensor] = []
    prior_proprio: list[torch.Tensor] = []
    prior_entropy: list[torch.Tensor] = []
    prior_sigma_p: list[torch.Tensor] = []
    prior_fall_flags: list[bool] = []
    env_step = 0
    prev_fallen_env0 = False
    freeze_started_at_env_step: int | None = None
    freeze_phase_t0: float | None = None
    pending_video_pad_break = False
    try:
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if physics_frozen:
                    if zero_actions is None:
                        n_act = int(env.num_actions)
                        n_env = int(env.num_envs)
                        zero_actions = torch.zeros((n_env, n_act), device=env.device, dtype=torch.float32)
                    actions = zero_actions
                elif use_velocity_estimator:
                    # Access observation manager directly to get all observation groups
                    obs_manager = env.unwrapped.observation_manager

                    # Compute observations for all groups - returns dict[group_name, tensor]
                    obs_dict = obs_manager.compute()

                    # Extract policy and ref_vel_estimator observations
                    policy_obs = obs_dict["policy"].to(ppo_runner.device)
                    ref_vel_estimator_obs = obs_dict["ref_vel_estimator"].to(ppo_runner.device)

                    # Normalize policy obs
                    policy_obs_normalized = ppo_runner.obs_normalizer(policy_obs)

                    # Estimate velocity
                    estimated_ref_vel = ppo_runner.alg.ref_vel_estimator(ref_vel_estimator_obs) * 1.0
                    print(f"[Play] Estimated ref vel: {estimated_ref_vel.cpu().numpy()}")

                    # Augment observations
                    obs_augmented = torch.cat([policy_obs_normalized, estimated_ref_vel], dim=-1)

                    # Get actions
                    actions = ppo_runner.alg.policy.act_inference(obs_augmented)
                else:
                    # Standard inference without velocity estimator
                    obs, _ = env.get_observations()
                    if future_injector is not None:
                        obs = future_injector.patch_policy_obs(obs, env, future_mode)
                    if use_prior_sample and prior_metrics_ready:
                        norm = ppo_runner.obs_normalizer(obs)
                        st = ppo_runner.alg.policy.act_prior_sample_with_stats(
                            norm,
                            sample_latent=prior_latent_sampling,
                            fixed_latent_std=prior_rollout_fixed_latent_std,
                        )
                        actions = st["actions"]
                        prior_actions.append(actions.detach().cpu())
                        prior_proprio.append(st["proprio"].detach().cpu())
                        prior_entropy.append(st["prior_entropy"].detach().cpu())
                        prior_sigma_p.append(st["sigma_p"].detach().cpu())
                    elif use_encoder_as_proprio:
                        norm = ppo_runner.obs_normalizer(obs)
                        _policy = ppo_runner.alg.policy
                        if encoder_as_proprio_sample:
                            # *.forward only reparameterizes when self.training is True; in eval
                            # mode that branch is dead, so do it explicitly here.
                            if isinstance(_policy, LatentBottleneckMUSETransformer):
                                enc = _policy.transformer_encoder
                                mu_e, log_sigma_e, proprio_per_frame = enc.encode(norm)
                                z = enc.reparameterize(mu_e, log_sigma_e)
                                actions = _policy._decode(z, proprio_per_frame)
                            else:
                                sc = _policy.student_core
                                mu_e, log_sigma_e = sc.encode(norm)
                                z = sc.reparameterize(mu_e, log_sigma_e)
                                proprio = sc.get_proprio(norm)
                                actions = sc.decode(z, proprio)
                        else:
                            actions = _policy.act_inference(norm)
                    else:
                        assert policy is not None
                        actions = policy(obs)

                # env stepping
                env.step(actions)
                env_step += 1
                # --synth_wrist_history: append current wrist pos and refresh the
                # green-dot history visualizer. No-op when the flag is off.
                _tick_synth_wrist_history(env)
                if (
                    physics_frozen
                    and args_cli.video
                    and not bool(args_cli.prior_freeze_pad_tail)
                    and int(args_cli.prior_freeze_log_every) > 0
                    and freeze_started_at_env_step is not None
                    and freeze_phase_t0 is not None
                ):
                    pe = int(args_cli.prior_freeze_log_every)
                    if env_step % pe == 0:
                        dt = time.perf_counter() - freeze_phase_t0
                        print(
                            "[prior_freeze] progress "
                            f"env_step={env_step}/{int(args_cli.video_length)} "
                            f"wall_s_since_freeze={dt:.1f} "
                            "(not stuck—each step still renders for RecordVideo)"
                        )

                run_fall_check = use_prior_sample and (
                    prior_metrics_ready or args_cli.prior_debug_fall or freeze_enabled
                )
                if run_fall_check and not physics_frozen:
                    fell = fall_to_ground(base_sim_env, fall_asset_cfg, float(args_cli.prior_fall_min_height))
                    fell0 = bool(fell[0].item())
                    if fall_first_env_step is None and fell0:
                        fall_first_env_step = env_step
                    if prior_metrics_ready:
                        prior_fall_flags.append(fell0)
                    if args_cli.prior_debug_fall:
                        every = max(1, int(args_cli.prior_debug_fall_every))
                        edge = fell0 != prev_fallen_env0
                        prev_fallen_env0 = fell0
                        if env_step % every == 0 or edge:
                            robot = base_sim_env.scene["robot"]
                            body_ids, _ = robot.find_bodies(fall_body_names, preserve_order=True)
                            z_min = float(robot.data.body_pos_w[0, body_ids, 2].min().item())
                            print(
                                "[prior_fall_debug] "
                                f"env_step={env_step} env0_fallen={fell0} "
                                f"min_z_monitored_bodies_m={z_min:.4f} "
                                f"threshold_m={float(args_cli.prior_fall_min_height)} "
                                f"bodies={fall_body_names}"
                            )
                    if (
                        freeze_enabled
                        and fall_first_env_step is not None
                        and env_step >= fall_first_env_step + int(args_cli.prior_freeze_after_fallen)
                        and not physics_frozen
                    ):
                        if _pause_physics_if_available(base_sim_env):
                            physics_frozen = True
                            sim_was_paused = True
                            freeze_started_at_env_step = env_step
                            freeze_phase_t0 = time.perf_counter()
                            if args_cli.prior_freeze_pad_tail:
                                pending_video_pad_break = True
                                print(
                                    "[prior_freeze] Paused physics after "
                                    f"env_step={env_step} (fall first at step {fall_first_env_step}, "
                                    f"delay={int(args_cli.prior_freeze_after_fallen)}). "
                                    "Stopping rollout; will pad MP4 to video_length with ffmpeg (clone last frame)."
                                )
                            else:
                                rem = max(0, int(args_cli.video_length) - env_step)
                                mex = int(args_cli.prior_freeze_max_extra_steps)
                                tail_note = (
                                    f"~{rem} more env.step calls until video_length (each still renders; can be slow)."
                                    if mex < 0
                                    else f"will stop after {mex} more env steps (--prior_freeze_max_extra_steps)."
                                )
                                print(
                                    "[prior_freeze] Paused physics after "
                                    f"env_step={env_step} (fall first at step {fall_first_env_step}, "
                                    f"delay={int(args_cli.prior_freeze_after_fallen)}). "
                                    + tail_note
                                )
                        else:
                            print(
                                "[prior_freeze] WARNING: could not pause simulation (no sim.pause()); "
                                "continuing rollout."
                            )

            # Cap simulation length for video export, or for prior metrics even without --video.
            if args_cli.video or (use_prior_metrics and prior_metrics_ready) or (
                args_cli.prior_sample and args_cli.prior_debug_fall
            ):
                timestep += 1
                if timestep >= args_cli.video_length:
                    break

            if pending_video_pad_break:
                break

            # Optional early exit after freeze (only when not using ffmpeg pad tail).
            mex = int(args_cli.prior_freeze_max_extra_steps)
            if (
                physics_frozen
                and mex >= 0
                and not bool(args_cli.prior_freeze_pad_tail)
                and freeze_started_at_env_step is not None
                and env_step >= freeze_started_at_env_step + mex
            ):
                print(
                    "[prior_freeze] Stopping loop early: "
                    f"env_step={env_step} >= freeze_at_{freeze_started_at_env_step} + max_extra={mex}. "
                    f"Recorded {timestep} frames (target video_length was {int(args_cli.video_length)})."
                )
                break

        if use_prior_metrics and prior_metrics_ready and prior_actions:
            actions_t = torch.cat(prior_actions, dim=0)
            proprio_t = torch.cat(prior_proprio, dim=0)
            entropy_t = torch.cat(prior_entropy, dim=0)
            sigma_flat = torch.cat(prior_sigma_p, dim=0).reshape(-1) if prior_sigma_p else None
            fall_t = torch.tensor(prior_fall_flags, dtype=torch.bool) if prior_fall_flags else None

            n = actions_t.shape[0]
            dproprio = torch.diff(proprio_t, dim=0)
            dact = torch.diff(actions_t, dim=0)
            delta_p = torch.norm(dproprio, dim=-1)
            delta_a = torch.norm(dact, dim=-1)
            mean_dp = float(delta_p.mean().item()) if delta_p.numel() else 0.0
            mean_da = float(delta_a.mean().item()) if delta_a.numel() else 0.0
            ratio = mean_da / (mean_dp + 1e-8)
            # Pearson correlation between ||Δproprio|| and ||Δaction|| (temporal coupling)
            if delta_p.numel() > 1:
                mp, ma = delta_p - delta_p.mean(), delta_a - delta_a.mean()
                denom = float(mp.std().item() * ma.std().item()) + 1e-12
                corr_dp_da = float((mp * ma).mean().item() / denom)
            else:
                corr_dp_da = float("nan")

            action_std_per_dim = actions_t.std(dim=0)
            temporal_std_mean = float(action_std_per_dim.mean().item())
            first_fall_step = None
            if fall_t is not None and fall_t.any():
                idx = int(torch.argmax(fall_t.long()))
                first_fall_step = idx + 1  # 1-based step index when fall was observed after that step

            metrics_payload = {
                "task": args_cli.task,
                "motion": args_cli.motion,
                "checkpoint": resume_path,
                "video_length": int(args_cli.video_length),
                "steps_recorded": n,
                "prior_fall": {
                    "min_height_m": float(args_cli.prior_fall_min_height),
                    "body_names": fall_body_names,
                    "occurred": bool(fall_t is not None and fall_t.any()),
                    "first_fall_step": first_fall_step,
                    "steps_survived_without_fall": int(n) if first_fall_step is None else first_fall_step - 1,
                },
                "prior_latent_entropy": {
                    "mean": float(entropy_t.mean().item()),
                    "std": float(entropy_t.std().item()),
                    "note": "Sum over latent dims of Gaussian differential entropy for z ~ R(proprio).",
                },
                "prior_predicted_sigma": {
                    "mean": float(sigma_flat.mean().item()) if sigma_flat is not None else 0.0,
                    "std": float(sigma_flat.std().item()) if sigma_flat is not None and sigma_flat.numel() > 1 else 0.0,
                    "note": "Per-latent σ = exp(log σ) from prior MLP (predicted), pooled over envs/steps; "
                    "independent of --prior_rollout_fixed_latent_std for sampling.",
                },
                "actions": {
                    "temporal_std_mean": temporal_std_mean,
                    "mean_abs_per_dim": float(actions_t.abs().mean().item()),
                    "delta_l2_mean": mean_da,
                    "delta_l2_std": float(delta_a.std().item()) if delta_a.numel() else 0.0,
                },
                "proprio": {
                    "delta_l2_mean": mean_dp,
                },
                "sensitivity": {
                    "delta_action_over_delta_proprio": ratio,
                    "corr_delta_proprio_delta_action": (None if corr_dp_da != corr_dp_da else corr_dp_da),
                    "note": "Higher ratio with meaningful Δproprio suggests actions track state; very low ratio may indicate memorization.",
                },
            }

            tag = "[Prior metrics]"
            ps = metrics_payload["prior_predicted_sigma"]
            print(f"\n{tag} steps={n} fall={metrics_payload['prior_fall']['occurred']} "
                  f"first_fall_step={first_fall_step} entropy_mean={metrics_payload['prior_latent_entropy']['mean']:.4f} "
                  f"prior_sigma_mean={ps['mean']:.5f} prior_sigma_std={ps['std']:.5f} "
                  f"action_temporal_std_mean={temporal_std_mean:.6f} Δaction_L2_mean={mean_da:.6f} "
                  f"Δproprio_L2_mean={mean_dp:.6f} ratio={ratio:.6f} corr(Δp,Δa)={corr_dp_da:.4f}\n")

            out_json = args_cli.prior_metrics_json.strip()
            if not out_json:
                out_dir = video_folder_path if video_folder_path else log_dir
                os.makedirs(out_dir, exist_ok=True)
                out_json = os.path.join(out_dir, "prior_rollout_metrics.json")
            else:
                os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            n_stored = _merge_write_prior_rollout_metrics(out_json, metrics_payload)
            print(f"{tag} Updated {out_json} ({n_stored} motion(s); keyed by path under \"motions\")")

    finally:
        if sim_was_paused:
            if _resume_physics_if_available(base_sim_env):
                print("[prior_freeze] Resumed simulation for shutdown.")
        env.close()
        if pending_video_pad_break and args_cli.video and video_folder_path:
            mp4 = _find_recorded_mp4(video_folder_path)
            if mp4:
                _pad_video_clone_last_frame(mp4, int(args_cli.video_length))
            else:
                print(f"[prior_freeze] pad: could not find mp4 under {video_folder_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


# Metrics only, no MP4
# python scripts/rsl_rl/play.py ... --prior_sample --headless --video_length 1000