"""Headless ckpt rollout → 3-point track HTML. No cameras, does not touch GPUs 6/7."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump sparse-intent tracking from a latent-RL ckpt.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion", type=str, required=True)
parser.add_argument("--mask_modes", type=str, default="vr")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="logs/tritrack/track_vis")
parser.add_argument("--start_frame", type=int, default=10)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import whole_body_tracking.tasks  # noqa: F401

INTENT = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
COLORS = {
    "torso_link": ("#3d8bfd", "Torso"),
    "left_wrist_yaw_link": ("#3dba7a", "Left wrist"),
    "right_wrist_yaw_link": ("#d4a017", "Right wrist"),
}


def _restore_adapter(agent_cfg, resume_path: str) -> None:
    params = Path(resume_path).parent / "params" / "agent.yaml"
    if not params.exists():
        return
    cfg = yaml.safe_load(params.read_text()) or {}
    policy_cfg = cfg.get("policy") or {}
    for key in (
        "adapter",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
        "residual_d_model",
        "residual_num_layers",
        "residual_nhead",
        "residual_ffn",
        "residual_last_layer_gain",
        "residual_alpha",
        "terrain_scan_dim",
        "terrain_r_max",
        "terrain_scan_zero",
    ):
        if key in policy_cfg and hasattr(agent_cfg.policy, key):
            setattr(agent_cfg.policy, key, policy_cfg[key])


def _write_html(out_html: Path, payload: dict) -> None:
    html = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Track · ckpt</title>
<style>
  :root { --bg:#141414; --panel:#1c1c1c; --line:#2a2a2a; --text:#e8e8e8; --muted:#8a8a8a; }
  * { box-sizing:border-box; }
  html,body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }
  body { padding:20px 24px 40px; }
  h1 { font-size:20px; font-weight:600; margin:0 0 6px; }
  .muted { color:var(--muted); }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:8px; margin:14px 0; }
  .stat { background:var(--panel); border:1px solid var(--line); padding:10px 12px; }
  .stat .v { font:600 22px/1.1 ui-monospace,monospace; }
  .stat .k { color:var(--muted); font-size:12px; margin-top:4px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .chart { background:var(--panel); border:1px solid var(--line); padding:10px 12px; }
  .chart h3 { margin:0 0 6px; font-size:13px; }
  canvas { width:100%; height:240px; display:block; }
  .leg { font-size:12px; color:var(--muted); margin-top:4px; }
</style></head><body>
<h1 id="title">Sparse-intent track</h1>
<div class="muted" id="meta"></div>
<div class="stats" id="stats"></div>
<div class="grid">
  <div class="chart"><h3>Top view (X–Y) · solid=robot, dashed=goal</h3><canvas id="xy"></canvas><div class="leg" id="leg"></div></div>
  <div class="chart"><h3>Side view (X–Z)</h3><canvas id="xz"></canvas></div>
  <div class="chart"><h3>Per-point error (m)</h3><canvas id="err"></canvas></div>
  <div class="chart"><h3>Mean visible error (m)</h3><canvas id="mean"></canvas></div>
</div>
<script>
const D = __DATA__;
const fmt = (x,d=3) => x==null ? "—" : Number(x).toFixed(d);
document.getElementById("title").textContent = D.title;
document.getElementById("meta").textContent = D.meta;
const S = [
  ["Steps", D.steps],
  ["Mean err (m)", fmt(D.mean_err)],
  ["Final err (m)", fmt(D.final_err)],
  ["SR @ 5 cm", (100*D.sr5).toFixed(1)+"%"],
  ["SR @ 2 cm", (100*D.sr2).toFixed(1)+"%"],
];
document.getElementById("stats").innerHTML = S.map(([k,v])=>`<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
document.getElementById("leg").innerHTML = D.bodies.map(b=>`<span style="color:${b.color}">● ${b.label}</span>`).join(" &nbsp; ");

function drawTrails(canvas, dimA, dimB) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const pad = 28*devicePixelRatio;
  let mnA=Infinity, mxA=-Infinity, mnB=Infinity, mxB=-Infinity;
  for (const b of D.bodies) for (const p of [...b.robot, ...b.goal]) {
    mnA=Math.min(mnA,p[dimA]); mxA=Math.max(mxA,p[dimA]);
    mnB=Math.min(mnB,p[dimB]); mxB=Math.max(mxB,p[dimB]);
  }
  const span = Math.max(mxA-mnA, mxB-mnB, 0.2);
  const cx=(mnA+mxA)/2, cy=(mnB+mxB)/2;
  const X = v => pad + (v - (cx-span/2)) / span * (W-2*pad);
  const Y = v => H-pad - (v - (cy-span/2)) / span * (H-2*pad);
  ctx.strokeStyle="#2a2a2a"; ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
  for (const b of D.bodies) {
    ctx.strokeStyle=b.color; ctx.lineWidth=1.6*devicePixelRatio;
    ctx.setLineDash([]); ctx.beginPath();
    b.robot.forEach((p,i)=>{ const x=X(p[dimA]),y=Y(p[dimB]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
    ctx.setLineDash([4*devicePixelRatio,4*devicePixelRatio]); ctx.beginPath();
    b.goal.forEach((p,i)=>{ const x=X(p[dimA]),y=Y(p[dimB]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
    const r=b.robot.at(-1), g=b.goal.at(-1);
    ctx.setLineDash([]); ctx.fillStyle=b.color;
    ctx.beginPath(); ctx.arc(X(r[dimA]),Y(r[dimB]),3.5*devicePixelRatio,0,7); ctx.fill();
    ctx.strokeStyle=b.color; ctx.beginPath(); ctx.arc(X(g[dimA]),Y(g[dimB]),5*devicePixelRatio,0,7); ctx.stroke();
  }
}
function drawSeries(canvas, series, yMaxHint) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const pad = {l:40*devicePixelRatio,r:10*devicePixelRatio,t:8*devicePixelRatio,b:20*devicePixelRatio};
  let ymax = yMaxHint || 0.01;
  for (const s of series) for (const y of s.ys) ymax = Math.max(ymax, y);
  ymax *= 1.1;
  const n = series[0].ys.length;
  const X = i => pad.l + i/(n-1) * (W-pad.l-pad.r);
  const Y = v => pad.t + (1-v/ymax) * (H-pad.t-pad.b);
  ctx.strokeStyle="#2a2a2a"; ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();
  ctx.fillStyle="#8a8a8a"; ctx.font=(11*devicePixelRatio)+"px ui-monospace";
  ctx.textAlign="right"; ctx.fillText(ymax.toFixed(2), pad.l-6, pad.t+10);
  ctx.fillText("0", pad.l-6, H-pad.b);
  for (const s of series) {
    ctx.strokeStyle=s.color; ctx.lineWidth=1.5*devicePixelRatio; ctx.beginPath();
    s.ys.forEach((y,i)=> i?ctx.lineTo(X(i),Y(y)):ctx.moveTo(X(i),Y(y)));
    ctx.stroke();
  }
}
drawTrails(document.getElementById("xy"), 0, 1);
drawTrails(document.getElementById("xz"), 0, 2);
drawSeries(document.getElementById("err"), D.bodies.map(b=>({color:b.color, ys:b.err})));
drawSeries(document.getElementById("mean"), [{color:"#3d8bfd", ys:D.mean_series}]);
addEventListener("resize", ()=>{ drawTrails(document.getElementById("xy"),0,1); drawTrails(document.getElementById("xz"),0,2); drawSeries(document.getElementById("err"), D.bodies.map(b=>({color:b.color, ys:b.err}))); drawSeries(document.getElementById("mean"), [{color:"#3d8bfd", ys:D.mean_series}]); });
</script></body></html>
"""
    out_html.write_text(html.replace("__DATA__", json.dumps(payload)), encoding="utf-8")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[dump] ckpt {resume_path}")
    _restore_adapter(agent_cfg, resume_path)

    env_cfg.commands.motion.motion = args_cli.motion
    if hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
        env_cfg.commands.motion.motion_group_sampling_ratios = None
    if hasattr(env_cfg.commands.motion, "start_frame"):
        env_cfg.commands.motion.start_from_beginning = True
        env_cfg.commands.motion.start_frame = args_cli.start_frame
    zero = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    if hasattr(env_cfg.commands.motion, "pose_range"):
        env_cfg.commands.motion.pose_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "velocity_range"):
        env_cfg.commands.motion.velocity_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "joint_position_range"):
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    from whole_body_tracking.tasks.tracking.config.g1.mask_modes import eval_single_mode_spec

    spec, probs = eval_single_mode_spec(args_cli.mask_modes)
    env_cfg.commands.motion.mask_mode_spec = spec
    env_cfg.commands.motion.mask_mode_probs = probs
    print(f"[dump] mask={args_cli.mask_modes} spec={spec}")

    if hasattr(env_cfg, "observations"):
        for group_name in ("policy", "teacher", "critic"):
            group = getattr(env_cfg.observations, group_name, None)
            if group is not None and hasattr(group, "enable_corruption"):
                group.enable_corruption = False
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        if hasattr(env_cfg.curriculum, "keypoint_mask_mode"):
            env_cfg.curriculum.keypoint_mask_mode = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints

    attach_curriculum_rollout_hints(env, spe=int(agent_cfg.num_steps_per_env), learning_iteration=3000)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, load_critic=True)
    policy = runner.get_inference_policy(device=env.device)

    cmd = env.unwrapped.command_manager.get_term("motion")
    names = list(cmd.cfg.body_names)
    idx = {n: names.index(n) for n in INTENT if n in names}
    if hasattr(cmd, "p_mask"):
        cmd.p_mask = 0.0

    rec = {n: {"robot": [], "goal": [], "err": []} for n in idx}
    print("[dump] policy loaded, rolling out", flush=True)
    env.reset()
    n = int(args_cli.steps)
    for t in range(n):
        with torch.inference_mode():
            obs, _ = env.get_observations()
            env.step(policy(obs))
        robot = cmd.robot_body_pos_w[0].detach().cpu()
        goal = cmd.body_pos_w[0].detach().cpu()
        for name, i in idx.items():
            r = robot[i].tolist()
            g = goal[i].tolist()
            rec[name]["robot"].append(r)
            rec[name]["goal"].append(g)
            rec[name]["err"].append(float(((robot[i] - goal[i]) ** 2).sum().sqrt()))
        if (t + 1) % 50 == 0:
            mean_now = sum(rec[n]["err"][-1] for n in rec) / len(rec)
            print(f"[dump] step {t+1}/{n} mean_err={mean_now:.3f}m", flush=True)

    mean_series = [
        sum(rec[n]["err"][t] for n in rec) / len(rec) for t in range(n)
    ]
    sr5 = sum(1 for e in mean_series if e < 0.05) / n
    sr2 = sum(1 for e in mean_series if e < 0.02) / n
    payload = {
        "title": f"model_3000 · mask={args_cli.mask_modes} · 3-point track",
        "meta": f"{Path(resume_path).name}  ·  {Path(args_cli.motion).name}  ·  {n} steps  ·  solid=robot  dashed=goal",
        "steps": n,
        "mean_err": sum(mean_series) / n,
        "final_err": mean_series[-1],
        "sr5": sr5,
        "sr2": sr2,
        "mean_series": mean_series,
        "bodies": [
            {
                "name": name,
                "label": COLORS[name][1],
                "color": COLORS[name][0],
                "robot": rec[name]["robot"],
                "goal": rec[name]["goal"],
                "err": rec[name]["err"],
            }
            for name in rec
        ],
    }
    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "track.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_html(out / "track.html", payload)
    print(f"[dump] wrote {out / 'track.html'}", flush=True)
    print(f"[dump] mean_err={payload['mean_err']:.3f}m  sr5={sr5:.3f}  sr2={sr2:.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
