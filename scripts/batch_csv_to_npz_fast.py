"""Fast CSV -> NPZ converter. Does not modify csv_to_npz.py / batch_csv_to_npz.py.

Same MotionLoader + G1 FK path as the official converter, with two speed changes:
  1. No per-frame ``sim.render()`` (no Kit/viewport tick). Kinematics come from
     ``update_articulations_kinematic()`` + ``scene.update()``.
  2. Multi-env batching: several clips share one Isaac session.

World-frame body positions have ``env_origins`` subtracted before save so the NPZ
matches the official ``num_envs=1`` convention (training adds origins again).
"""

from __future__ import annotations

import os
import sys


def _sanitize_python_path_for_isaac() -> None:
    """Avoid binary incompatibility by preventing mixed user-site packages.

    See batch_csv_to_npz.py for the detailed rationale.
    """

    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site  # noqa: WPS433 (runtime import by design)

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site_path(p: str) -> bool:
        if not p:
            return False
        if isinstance(user_site, str) and p == user_site:
            return True
        if "/.local/lib/python" in p and "site-packages" in p:
            return True
        return False

    sys.path[:] = [p for p in sys.path if not _is_user_site_path(p)]

    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Fast batch CSV->NPZ: skip render, multi-env FK. Official scripts are unchanged."
)
parser.add_argument("--input_dir", type=str, default=None, help="Folder containing input CSV motion files.")
parser.add_argument(
    "--input_list",
    type=str,
    default=None,
    help="Text file with one CSV path per line (overrides --input_dir glob).",
)
parser.add_argument(
    "--input_root",
    type=str,
    default=None,
    help="Root used to preserve relative output paths. Default: --input_dir, or commonpath of --input_list.",
)
parser.add_argument("--file_pattern", type=str, default="*.csv", help="Glob pattern to match input files.")
parser.add_argument("--output_prefix", type=str, required=True, help='Prefix to add to output names, e.g. "bones_seed_g1".')
parser.add_argument("--output_dir", type=str, required=True, help="Output folder for NPZ files.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motions.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motions.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional global frame range applied to every file. Index starts from 1. If omitted, loads all frames.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    help="Robot platform name (e.g. g1, h1_2). See whole_body_tracking.robots.robot_registry.ROBOT_PLATFORMS.",
)
parser.add_argument(
    "--bones_seed_g1_csv",
    action="store_true",
    help="BONES-SEED / named-column G1 CSV: skip header, drop Frame column, root euler XYZ (deg), joints in degrees.",
)
parser.add_argument("--csv_skip_header", action="store_true", help="Skip the first line (column names).")
parser.add_argument(
    "--csv_drop_frame_column",
    action="store_true",
    help="Drop the first numeric column (e.g. Frame index) after loading.",
)
parser.add_argument(
    "--root_rotation",
    type=str,
    choices=("quat_xyzw", "euler_xyz_deg"),
    default="quat_xyzw",
    help="Root orientation: quaternion xyzw (cols 3-6) or euler XYZ degrees (cols 3-5).",
)
parser.add_argument(
    "--joint_angles_in_degrees",
    action="store_true",
    help="Convert joint DOF columns from degrees to radians.",
)
parser.add_argument(
    "--position_scale",
    type=float,
    default=0.01,
    help="Multiply root translation (x,y,z) by this factor (e.g. 0.01 if positions are in centimeters).",
)
parser.add_argument(
    "--min_duration",
    type=float,
    default=3.0,
    help="Minimum duration (seconds) for motions to be converted. Default: 3.0",
)
parser.add_argument("--skip_existing", action="store_true", help="Skip CSV files whose output NPZ already exists.")
parser.add_argument("--num_envs", type=int, default=64, help="Clips converted in parallel in one Isaac scene.")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Isaac env spacing (origins are subtracted on save).")
parser.add_argument("--max_files", type=int, default=None, help="Convert at most this many CSV files.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.bones_seed_g1_csv:
    args_cli.csv_skip_header = True
    args_cli.csv_drop_frame_column = True
    args_cli.root_rotation = "euler_xyz_deg"
    args_cli.joint_angles_in_degrees = True

# Always headless: this converter is for server batch jobs, not visualization.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from motion_csv_loader import MotionLoader
from whole_body_tracking.robots.robot_registry import available_robot_names, get_robot_platform


LOG_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def create_replay_scene_cfg(robot_cfg: ArticulationCfg):
    """Create a scene configuration for replaying motions with the specified robot."""

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionsSceneCfg


def collect_csv_files() -> tuple[list[Path], Path]:
    if args_cli.input_list:
        paths = []
        with open(args_cli.input_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Do not follow CSV symlinks: split dirs point at bones-seed/g1/csv,
                # and resolve() would drop the loco-manip relative layout.
                p = Path(line).expanduser()
                if not p.is_absolute():
                    p = Path.cwd() / p
                paths.append(Path(os.path.normpath(p)))
        if not paths:
            raise SystemExit(f"[ERROR] no CSV paths in {args_cli.input_list}")
        if args_cli.input_root:
            input_root = Path(os.path.normpath(Path(args_cli.input_root).expanduser()))
        else:
            input_root = Path(os.path.commonpath([str(p.parent) for p in paths]))
        return paths, input_root

    if not args_cli.input_dir:
        raise SystemExit("[ERROR] provide --input_dir or --input_list")
    input_root = Path(args_cli.input_dir).expanduser().resolve()
    csv_files = sorted(input_root.rglob(args_cli.file_pattern))
    return csv_files, input_root


def output_npz_path(csv_path: Path, input_root: Path, output_dir: Path) -> Path:
    try:
        relative_csv = csv_path.relative_to(input_root)
    except ValueError:
        relative_csv = Path(csv_path.name)
    output_name = f"{args_cli.output_prefix}_{relative_csv.stem}"
    return (output_dir / relative_csv).with_name(f"{output_name}.npz")


def load_motion(csv_path: str, device, joint_names: list[str]) -> MotionLoader:
    return MotionLoader(
        motion_file=csv_path,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
        csv_skip_header=args_cli.csv_skip_header,
        csv_drop_frame_column=args_cli.csv_drop_frame_column,
        root_rotation=args_cli.root_rotation,
        joint_angles_in_degrees=args_cli.joint_angles_in_degrees,
        position_scale=args_cli.position_scale,
        expected_dof_count=len(joint_names),
    )


def convert_chunk(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot,
    robot_joint_indexes: list[int],
    jobs: list[tuple[Path, Path, MotionLoader]],
) -> int:
    """Convert ``len(jobs)`` motions in parallel. Idle envs (if num_envs > len(jobs)) keep a repeated last pose."""
    n = len(jobs)
    device = sim.device
    lengths = [m.output_frames for _, _, m in jobs]
    t_max = max(lengths)
    n_dof = jobs[0][2].motion_dof_poss.shape[1]

    base_pos = torch.zeros(t_max, n, 3, device=device)
    base_rot = torch.zeros(t_max, n, 4, device=device)
    base_lin = torch.zeros(t_max, n, 3, device=device)
    base_ang = torch.zeros(t_max, n, 3, device=device)
    dof_pos = torch.zeros(t_max, n, n_dof, device=device)
    dof_vel = torch.zeros(t_max, n, n_dof, device=device)
    for i, (_, _, motion) in enumerate(jobs):
        t = motion.output_frames
        base_pos[:t, i] = motion.motion_base_poss
        base_rot[:t, i] = motion.motion_base_rots
        base_lin[:t, i] = motion.motion_base_lin_vels
        base_ang[:t, i] = motion.motion_base_ang_vels
        dof_pos[:t, i] = motion.motion_dof_poss
        dof_vel[:t, i] = motion.motion_dof_vels
        if t < t_max:
            base_pos[t:, i] = motion.motion_base_poss[-1]
            base_rot[t:, i] = motion.motion_base_rots[-1]
            base_lin[t:, i] = 0.0
            base_ang[t:, i] = 0.0
            dof_pos[t:, i] = motion.motion_dof_poss[-1]
            dof_vel[t:, i] = 0.0

    env_ids = torch.arange(n, device=device, dtype=torch.long)
    n_joints = robot.data.default_joint_pos.shape[1]
    n_bodies = robot.data.body_pos_w.shape[1]

    out_jp = torch.empty(t_max, n, n_joints, device=device)
    out_jv = torch.empty(t_max, n, n_joints, device=device)
    out_bp = torch.empty(t_max, n, n_bodies, 3, device=device)
    out_bq = torch.empty(t_max, n, n_bodies, 4, device=device)
    out_bl = torch.empty(t_max, n, n_bodies, 3, device=device)
    out_ba = torch.empty(t_max, n, n_bodies, 3, device=device)

    origin_xy = scene.env_origins[:n, :2]
    write_env_ids = None if n == robot.data.default_root_state.shape[0] else env_ids

    for t in range(t_max):
        root_states = robot.data.default_root_state[:n].clone()
        root_states[:, :3] = base_pos[t]
        root_states[:, :2] += origin_xy
        root_states[:, 3:7] = base_rot[t]
        root_states[:, 7:10] = base_lin[t]
        root_states[:, 10:] = base_ang[t]
        robot.write_root_state_to_sim(root_states, env_ids=write_env_ids)

        joint_pos = robot.data.default_joint_pos[:n].clone()
        joint_vel = robot.data.default_joint_vel[:n].clone()
        joint_pos[:, robot_joint_indexes] = dof_pos[t]
        joint_vel[:, robot_joint_indexes] = dof_vel[t]
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=write_env_ids)

        if sim.physics_sim_view is not None:
            sim.physics_sim_view.update_articulations_kinematic()
        scene.update(sim.get_physics_dt())

        body_pos = robot.data.body_pos_w[:n].clone()
        body_pos[..., :2] -= origin_xy[:, None]
        out_jp[t] = robot.data.joint_pos[:n]
        out_jv[t] = robot.data.joint_vel[:n]
        out_bp[t] = body_pos
        out_bq[t] = robot.data.body_quat_w[:n]
        out_bl[t] = robot.data.body_lin_vel_w[:n]
        out_ba[t] = robot.data.body_ang_vel_w[:n]

    saved = 0
    for i, (_, npz_path, _) in enumerate(jobs):
        t = lengths[i]
        log = {
            "fps": [args_cli.output_fps],
            "joint_pos": out_jp[:t, i].cpu().numpy(),
            "joint_vel": out_jv[:t, i].cpu().numpy(),
            "body_pos_w": out_bp[:t, i].cpu().numpy(),
            "body_quat_w": out_bq[:t, i].cpu().numpy(),
            "body_lin_vel_w": out_bl[:t, i].cpu().numpy(),
            "body_ang_vel_w": out_ba[:t, i].cpu().numpy(),
        }
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **log)
        saved += 1
        print(f"[INFO]: Saved NPZ: {npz_path}")
    return saved


def main():
    try:
        robot_platform = get_robot_platform(args_cli.robot)
    except KeyError as e:
        raise SystemExit(f"{e}\nAvailable robots: {available_robot_names()}") from None

    csv_files, input_root = collect_csv_files()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs_all: list[tuple[Path, Path]] = []
    skipped_existing = 0
    for csv_path in csv_files:
        if not csv_path.is_file() and not csv_path.is_symlink():
            continue
        npz_path = output_npz_path(csv_path, input_root, output_dir)
        if args_cli.skip_existing and npz_path.is_file() and npz_path.stat().st_size > 0:
            skipped_existing += 1
            continue
        jobs_all.append((csv_path, npz_path))
        if args_cli.max_files is not None and len(jobs_all) >= args_cli.max_files:
            break

    if not jobs_all:
        print(f"[WARN]: Nothing to convert (skipped_existing={skipped_existing})")
        simulation_app.close()
        return

    num_envs = max(1, int(args_cli.num_envs))
    print(f"[INFO]: fast convert files={len(jobs_all)} num_envs={num_envs} skip_existing={skipped_existing}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg_class = create_replay_scene_cfg(robot_platform.cfg)
    scene_cfg = scene_cfg_class(num_envs=num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete (fast, no per-frame render)...")

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(robot_platform.joint_names, preserve_order=True)[0]

    converted = 0
    skipped_short = 0
    try:
        for start in tqdm(range(0, len(jobs_all), num_envs), desc="Converting chunks", unit="chunk"):
            chunk_paths = jobs_all[start : start + num_envs]
            loaded: list[tuple[Path, Path, MotionLoader]] = []
            for csv_path, npz_path in chunk_paths:
                motion = load_motion(str(csv_path), sim.device, robot_platform.joint_names)
                if motion.duration < args_cli.min_duration:
                    print(
                        f"[SKIP]: Motion duration ({motion.duration:.2f}s) < min_duration "
                        f"({args_cli.min_duration:.2f}s), skipping: {csv_path}"
                    )
                    skipped_short += 1
                    continue
                loaded.append((csv_path, npz_path, motion))
            if not loaded:
                continue
            converted += convert_chunk(sim, scene, robot, robot_joint_indexes, loaded)
    finally:
        print("\n" + "=" * 80)
        print("FAST CONVERSION STATISTICS")
        print("=" * 80)
        print(f"Queued files: {len(jobs_all)}")
        print(f"Successfully converted: {converted}")
        print(f"Skipped existing: {skipped_existing}")
        print(f"Skipped (duration < {args_cli.min_duration}s): {skipped_short}")
        print("=" * 80)
        print("[INFO]: FAST_CONVERT_OK", flush=True)
        # Kit simulation_app.close() often hangs in headless batch jobs.
        os._exit(0)


if __name__ == "__main__":
    main()
