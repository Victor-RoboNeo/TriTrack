"""One-time helper: dump ``robot.body_names`` to a JSON sidecar consumed by the
synthetic-motion writer.

Launches Isaac (headless), instantiates the robot articulation, and writes the full
body-name list (in PhysX articulation order) to::

    source/whole_body_tracking/whole_body_tracking/synth/cache/<robot>_body_names.json

The synth motion writer reads this JSON offline to find the ``torso_link`` and
``pelvis`` indices in the 30-body npz axis, so synth-generation itself never needs
Isaac. Re-run only if the URDF changes (rare).

Usage::

    cd /home/lsn/AnyBody
    python scripts/synth_dump_body_names.py --robot g1
    python scripts/synth_dump_body_names.py --robot g1 --force   # overwrite existing

Structured exactly like ``scripts/csv_to_npz.py``: sanitize PYTHONPATH for Isaac,
then launch Kit BEFORE any ``whole_body_tracking`` import (which transitively pulls
in ``omni.kit`` and therefore must run after AppLauncher).
"""

from __future__ import annotations

import os
import sys


def _sanitize_python_path_for_isaac() -> None:
    """Mirror csv_to_npz.py's PYTHONPATH sanitization (avoid mixed user-site numpy)."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        import site

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site(p: str) -> bool:
        if not p:
            return False
        if isinstance(user_site, str) and p == user_site:
            return True
        if "/.local/lib/python" in p and "site-packages" in p:
            return True
        return False

    sys.path[:] = [p for p in sys.path if not _is_user_site(p)]
    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()


import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def _cache_path(robot: str) -> Path:
    """``source/whole_body_tracking/whole_body_tracking/synth/cache/<robot>_body_names.json``."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    cache_dir = repo_root / "source" / "whole_body_tracking" / "whole_body_tracking" / "synth" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{robot}_body_names.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", type=str, default="g1", help="Robot platform name (e.g. g1, h1_2).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing cache JSON.")
    args = parser.parse_args()

    out = _cache_path(args.robot)
    if out.exists() and not args.force:
        with out.open() as fh:
            data = json.load(fh)
        print(f"[synth_dump_body_names] {out} already exists (use --force to overwrite).")
        print(f"[synth_dump_body_names] num_bodies={data['num_bodies']} pelvis_index={data['pelvis_index']} torso_index={data['torso_index']}")
        return 0

    def _trace(msg: str) -> None:
        sys.stderr.write(f"[synth_dump_body_names TRACE] {msg}\n")
        sys.stderr.flush()

    _trace("launching Isaac AppLauncher(headless=True)")
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    _trace("AppLauncher returned; simulation_app alive")

    try:
        _trace("importing Isaac sim modules")
        # Isaac is now alive — safe to import the whole_body_tracking package.
        import torch  # noqa: F401

        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        # Bypass whole_body_tracking.robots.robot_registry: it ``from .h1_2`` /
        # ``from .adam`` imports modules that are missing from the working tree
        # (only stale .pyc files remain) and dies silently under Isaac. We only
        # need the G1 articulation cfg, so import it directly.
        _trace(f"importing robot cfg directly for {args.robot}")
        if args.robot == "g1":
            from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG as _ROBOT_CFG
        else:
            raise SystemExit(
                f"Robot {args.robot!r} not supported in this dump tool. Only 'g1' is "
                f"currently routed (robot_registry has broken h1_2 / adam imports)."
            )
        _trace(f"robot cfg type = {type(_ROBOT_CFG).__name__}")

        # Mirror csv_to_npz.py's scene setup exactly: ground + sky_light + the
        # articulation under {ENV_REGEX_NS}/Robot. PhysX articulation discovery only
        # populates ``robot.body_names`` once the scene is fully initialized.
        @configclass
        class _DumpSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
            sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=(
                        f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                        "kloofendal_43d_clear_puresky_4k.hdr"
                    ),
                ),
            )
            robot: ArticulationCfg = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        _trace("building SimulationCfg")
        sim_cfg = sim_utils.SimulationCfg(device="cpu")
        _trace("building SimulationContext")
        sim = SimulationContext(sim_cfg)
        _trace("building scene_cfg")
        scene_cfg = _DumpSceneCfg(num_envs=1, env_spacing=2.0)
        _trace("building InteractiveScene")
        scene = InteractiveScene(scene_cfg)
        _trace("sim.reset()")
        sim.reset()
        _trace("sim.reset complete; querying robot articulation")

        robot = scene["robot"]
        body_names = list(robot.body_names)
        joint_names = list(robot.joint_names)
        _trace(f"got body_names ({len(body_names)}) and joint_names ({len(joint_names)})")

        for required in ("pelvis", "torso_link"):
            if required not in body_names:
                raise RuntimeError(
                    f"Expected body {required!r} in robot.body_names but it is missing. "
                    f"Got: {body_names}"
                )

        record = {
            "robot": args.robot,
            "body_names": body_names,
            "joint_names": joint_names,
            "num_bodies": len(body_names),
            "num_joints": len(joint_names),
            "pelvis_index": body_names.index("pelvis"),
            "torso_index": body_names.index("torso_link"),
        }

        with out.open("w") as fh:
            json.dump(record, fh, indent=2)
        print(f"[synth_dump_body_names] wrote {out}")
        print(f"[synth_dump_body_names] num_bodies={record['num_bodies']} num_joints={record['num_joints']}")
        print(f"[synth_dump_body_names] pelvis_index={record['pelvis_index']} torso_index={record['torso_index']}")
    finally:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
