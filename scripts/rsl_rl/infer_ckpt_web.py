"""Headless 1-env ckpt rollout → webpage (skeleton video + KP tracks).

Shares GPU 0 with training (1 env, no cameras). Does not stop the 0-3 job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Roll out a latent-RL ckpt and write a viewer page.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion", type=str, action="append", default=None, help="npz path; repeatable")
parser.add_argument("--mask_modes", type=str, default="vr")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="logs/tritrack/infer_55000")
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=55000)
parser.add_argument("--skip_publish", action="store_true")
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

KP = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
KP_COLOR = {
    "torso_link": ("#3d8bfd", "Head / torso"),
    "left_wrist_yaw_link": ("#3dba7a", "Left wrist"),
    "right_wrist_yaw_link": ("#d4a017", "Right wrist"),
    "left_ankle_roll_link": ("#c27ae0", "Left ankle"),
    "right_ankle_roll_link": ("#d4534c", "Right ankle"),
}
G1_BONES = [
    ("pelvis", "waist_yaw_link"),
    ("waist_yaw_link", "waist_roll_link"),
    ("waist_roll_link", "torso_link"),
    ("pelvis", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("left_wrist_roll_link", "left_wrist_pitch_link"),
    ("left_wrist_pitch_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
    ("right_wrist_roll_link", "right_wrist_pitch_link"),
    ("right_wrist_pitch_link", "right_wrist_yaw_link"),
]


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


def _apply_eval_motion(env_cfg, motion: str, start_frame: int) -> None:
    env_cfg.commands.motion.motion = motion
    if hasattr(env_cfg.commands.motion, "motion_groups"):
        env_cfg.commands.motion.motion_groups = None
    if hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
        env_cfg.commands.motion.motion_group_sampling_ratios = None
    if hasattr(env_cfg.commands.motion, "start_frame"):
        env_cfg.commands.motion.start_from_beginning = True
        env_cfg.commands.motion.start_frame = start_frame
    zero = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    if hasattr(env_cfg.commands.motion, "pose_range"):
        env_cfg.commands.motion.pose_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "velocity_range"):
        env_cfg.commands.motion.velocity_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "joint_position_range"):
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if hasattr(env_cfg.commands.motion, "motion_dataset_shard_across_gpus"):
        env_cfg.commands.motion.motion_dataset_shard_across_gpus = False
    if hasattr(env_cfg.commands.motion, "resample_motions_every_s"):
        env_cfg.commands.motion.resample_motions_every_s = 0.0
    if hasattr(env_cfg.commands.motion, "max_active_motions"):
        env_cfg.commands.motion.max_active_motions = None
    if hasattr(env_cfg.commands.motion, "random_init_frame"):
        env_cfg.commands.motion.random_init_frame = False


def _rollout_batch(env, policy, steps: int) -> list[dict]:
    cmd = env.unwrapped.command_manager.get_term("motion")
    if hasattr(cmd, "p_mask"):
        cmd.p_mask = 0.0
    robot_art = env.unwrapped.scene["robot"]
    body_names = list(robot_art.data.body_names)
    name_to_i = {n: i for i, n in enumerate(body_names)}
    kp_idx = {n: list(cmd.cfg.body_names).index(n) for n in KP if n in cmd.cfg.body_names}
    n_envs = int(env.unwrapped.num_envs)
    paths = list(cmd.motion_dir_loader.motion_paths)
    n_mot = len(paths)
    use = min(n_envs, n_mot)

    env.reset()
    ids = torch.arange(use, device=cmd.device, dtype=torch.long)
    cmd.env_motion_indices[:use] = ids
    if hasattr(cmd, "env_motion_groups"):
        cmd.env_motion_groups[:use] = 0
    if hasattr(cmd, "_env_remap_version") and hasattr(cmd, "_remap_version"):
        cmd._env_remap_version[:use] = cmd._remap_version
    cmd._resample_command(ids)
    try:
        env.unwrapped.sim.forward()
    except Exception as exc:
        print(f"[infer] sim.forward skipped: {exc}", flush=True)
    print(f"[infer] pinned {use} envs -> {use} clips", flush=True)
    for i in range(use):
        print(f"  env{i} {Path(paths[i]).name}", flush=True)

    skels = [[] for _ in range(use)]
    kp_robot = [{n: [] for n in kp_idx} for _ in range(use)]
    kp_goal = [{n: [] for n in kp_idx} for _ in range(use)]
    kp_err = [{n: [] for n in kp_idx} for _ in range(use)]

    for t in range(steps):
        with torch.inference_mode():
            obs, _ = env.get_observations()
            env.step(policy(obs))
        pos = robot_art.data.body_pos_w[:use].detach().cpu().numpy()
        robot = cmd.robot_body_pos_w[:use].detach().cpu()
        goal = cmd.body_pos_w[:use].detach().cpu()
        for e in range(use):
            skels[e].append(pos[e].astype(np.float32))
            for name, i in kp_idx.items():
                r = robot[e, i].tolist()
                g = goal[e, i].tolist()
                kp_robot[e][name].append(r)
                kp_goal[e][name].append(g)
                kp_err[e][name].append(float(((robot[e, i] - goal[e, i]) ** 2).sum().sqrt()))
        if (t + 1) % 50 == 0:
            vis = [n for n in ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link") if n in kp_idx]
            means = []
            for e in range(use):
                means.append(sum(kp_err[e][n][-1] for n in vis) / max(len(vis), 1))
            print(f"[infer] step {t+1}/{steps} vis3_err={['%.3f'%m for m in means]}", flush=True)

    recs = []
    vis = [n for n in ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link") if n in kp_idx]
    for e in range(use):
        mean_series = [sum(kp_err[e][n][t] for n in vis) / len(vis) for t in range(steps)] if vis else [0.0] * steps
        recs.append(
            {
                "path": paths[e],
                "body_names": body_names,
                "name_to_i": name_to_i,
                "skel": np.stack(skels[e], axis=0),
                "kp_robot": kp_robot[e],
                "kp_goal": kp_goal[e],
                "kp_err": kp_err[e],
                "mean_series": mean_series,
                "sr5": sum(1 for x in mean_series if x < 0.05) / steps,
                "sr2": sum(1 for x in mean_series if x < 0.02) / steps,
                "mean_err": float(sum(mean_series) / steps),
                "final_err": float(mean_series[-1]),
            }
        )
    return recs


def _write_gif(out_gif: Path, rec: dict, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    skel = rec["skel"]
    names = rec["body_names"]
    idx = rec["name_to_i"]
    bones = [(a, b) for a, b in G1_BONES if a in idx and b in idx]
    T = skel.shape[0]
    stride = max(1, T // 80)
    frames_i = list(range(0, T, stride))
    xyz = skel.reshape(-1, 3)
    c = xyz.mean(axis=0)
    span = float(np.percentile(np.linalg.norm(xyz - c, axis=1), 98) * 2.2)
    span = max(span, 1.2)

    import imageio.v2 as imageio

    imgs = []
    fig = plt.figure(figsize=(6.4, 4.8), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    for t in frames_i:
        ax.cla()
        ax.set_facecolor("#141414")
        fig.patch.set_facecolor("#141414")
        for a, b in bones:
            pa, pb = skel[t, idx[a]], skel[t, idx[b]]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="#8ab4f8", lw=1.6)
        for name, (col, _) in KP_COLOR.items():
            if name not in rec["kp_robot"]:
                continue
            r = rec["kp_robot"][name][t]
            g = rec["kp_goal"][name][t]
            ax.scatter(*r, color=col, s=18, zorder=5)
            ax.scatter(*g, facecolors="none", edgecolors=col, s=36, zorder=5)
        ax.set_xlim(c[0] - span / 2, c[0] + span / 2)
        ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
        ax.set_zlim(max(0.0, c[2] - span / 2), c[2] + span / 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.tick_params(colors="#8a8a8a")
        ax.set_title(f"{title}  t={t}", color="#e8e8e8", fontsize=9)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        imgs.append(buf)
    plt.close(fig)
    imageio.mimsave(out_gif, imgs, fps=20, loop=0)
    print(f"[infer] gif {out_gif}  frames={len(imgs)}", flush=True)


def _write_html(out_html: Path, page: dict) -> None:
    html = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>Infer · model_55000</title>
<style>
  :root { --bg:#141414; --panel:#1c1c1c; --line:#2a2a2a; --text:#e8e8e8; --muted:#8a8a8a; --accent:#3d8bfd; }
  * { box-sizing:border-box; }
  html,body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }
  body { padding:20px 24px 48px; }
  h1 { font-size:20px; font-weight:600; margin:0; }
  h2 { font-size:13px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); margin:22px 0 10px; }
  .muted { color:var(--muted); }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin:12px 0; }
  .stat { background:var(--panel); border:1px solid var(--line); padding:10px 12px; }
  .stat .v { font:600 20px/1.1 ui-monospace,monospace; }
  .stat .k { color:var(--muted); font-size:12px; margin-top:4px; }
  .grid { display:grid; grid-template-columns:1.15fr 1fr; gap:12px; }
  @media (max-width: 980px) { .grid { grid-template-columns:1fr; } }
  .chart { background:var(--panel); border:1px solid var(--line); padding:10px 12px; }
  canvas { width:100%; display:block; background:#101010; }
  #stage { height:420px; }
  .plot { height:220px; }
  button { background:#1a2433; color:var(--text); border:1px solid var(--accent); padding:6px 12px; cursor:pointer; font:inherit; }
  input[type=range] { width:min(420px,100%); }
  .clip { border:1px solid var(--line); background:transparent; color:var(--muted); padding:6px 10px; cursor:pointer; font:inherit; }
  .clip.on { border-color:var(--accent); color:var(--text); }
  table { width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 14px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; }
  td.num { font-family:ui-monospace,monospace; text-align:right; }
</style></head><body>
<div class="row"><h1>Head+hands locomani · model_55000</h1><span class="muted" id="meta"></span></div>
<div class="muted" style="margin-top:4px">实线=机器人，虚线/空心=目标。策略只看头+双手（VR mask），踝点只用于画骨架。</div>
<div class="row" style="margin-top:12px" id="clips"></div>
<h2>全部 clip 指标</h2>
<table id="sum"><tr><th>Clip</th><th>类别</th><th>mean vis err</th><th>SR@5cm</th><th>SR@2cm</th></tr></table>
<div class="stats" id="stats"></div>
<div class="row" style="margin:10px 0">
  <button id="tog">暂停</button>
  <input id="scr" type="range" min="0" max="1" value="0"/>
  <span class="muted" id="tlab"></span>
</div>
<div class="grid">
  <div class="chart"><h3 style="margin:0 0 6px;font-size:13px">3D 骨架回放</h3><canvas id="stage"></canvas></div>
  <div>
    <div class="chart" style="margin-bottom:12px"><h3 style="margin:0 0 6px;font-size:13px">视频 GIF</h3><img class="gif" id="gif" alt="gif"/></div>
    <div class="chart"><h3 style="margin:0 0 6px;font-size:13px">俯视 XY · 头/双手轨迹</h3><canvas id="xy" class="plot"></canvas></div>
  </div>
</div>
<div class="grid" style="margin-top:12px">
  <div class="chart"><h3 style="margin:0 0 6px;font-size:13px">侧视 XZ</h3><canvas id="xz" class="plot"></canvas></div>
  <div class="chart"><h3 style="margin:0 0 6px;font-size:13px">可见点误差 (m)</h3><canvas id="err" class="plot"></canvas></div>
</div>
<script>
const PAGE = __PAGE__;
let ci = 0, fi = 0, play = true;
const fmt = (x,d=3)=> x==null?"—":Number(x).toFixed(d);
const pct = x => (100*x).toFixed(1)+"%";

function clip(){ return PAGE.clips[ci]; }

function setClip(i){
  ci=i; fi=0;
  document.querySelectorAll(".clip").forEach((b,j)=>b.classList.toggle("on", j===i));
  const C=clip();
  document.getElementById("meta").textContent = C.meta;
  document.getElementById("gif").src = C.gif;
  document.getElementById("scr").max = C.steps-1;
  const S=[["Clip", C.title],["Steps", C.steps],["Mean vis err", fmt(C.mean_err)+" m"],
           ["Final err", fmt(C.final_err)+" m"],["SR @ 5cm", pct(C.sr5)],["SR @ 2cm", pct(C.sr2)]];
  document.getElementById("stats").innerHTML = S.map(([k,v])=>`<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  drawStatic(); drawFrame();
}

function drawStatic(){
  const C=clip();
  drawTrails(document.getElementById("xy"), 0, 1, C);
  drawTrails(document.getElementById("xz"), 0, 2, C);
  drawSeries(document.getElementById("err"), C.kp.map(b=>({color:b.color, ys:b.err})));
}

function drawTrails(canvas, da, db, C){
  const ctx=canvas.getContext("2d");
  const W=canvas.width=canvas.clientWidth*devicePixelRatio;
  const H=canvas.height=canvas.clientHeight*devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const pad=28*devicePixelRatio;
  let mnA=Infinity,mxA=-Infinity,mnB=Infinity,mxB=-Infinity;
  for (const b of C.kp) for (const p of [...b.robot,...b.goal]) {
    mnA=Math.min(mnA,p[da]); mxA=Math.max(mxA,p[da]);
    mnB=Math.min(mnB,p[db]); mxB=Math.max(mxB,p[db]);
  }
  const span=Math.max(mxA-mnA, mxB-mnB, 0.25);
  const cx=(mnA+mxA)/2, cy=(mnB+mxB)/2;
  const X=v=>pad+(v-(cx-span/2))/span*(W-2*pad);
  const Y=v=>H-pad-(v-(cy-span/2))/span*(H-2*pad);
  ctx.strokeStyle="#2a2a2a"; ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
  for (const b of C.kp) {
    ctx.strokeStyle=b.color; ctx.lineWidth=1.5*devicePixelRatio;
    ctx.setLineDash([]); ctx.beginPath();
    b.robot.forEach((p,i)=>{const x=X(p[da]),y=Y(p[db]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
    ctx.setLineDash([4*devicePixelRatio,4*devicePixelRatio]); ctx.beginPath();
    b.goal.forEach((p,i)=>{const x=X(p[da]),y=Y(p[db]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
  }
}

function drawSeries(canvas, series){
  const ctx=canvas.getContext("2d");
  const W=canvas.width=canvas.clientWidth*devicePixelRatio;
  const H=canvas.height=canvas.clientHeight*devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const pad={l:40*devicePixelRatio,r:10*devicePixelRatio,t:8*devicePixelRatio,b:20*devicePixelRatio};
  let ymax=0.02; for (const s of series) for (const y of s.ys) ymax=Math.max(ymax,y); ymax*=1.1;
  const n=series[0].ys.length;
  const X=i=>pad.l+i/Math.max(n-1,1)*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-v/ymax)*(H-pad.t-pad.b);
  ctx.strokeStyle="#2a2a2a"; ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();
  ctx.fillStyle="#8a8a8a"; ctx.font=(11*devicePixelRatio)+"px ui-monospace";
  ctx.textAlign="right"; ctx.fillText(ymax.toFixed(2), pad.l-6, pad.t+10);
  for (const s of series){ ctx.strokeStyle=s.color; ctx.lineWidth=1.4*devicePixelRatio; ctx.beginPath();
    s.ys.forEach((y,i)=> i?ctx.lineTo(X(i),Y(y)):ctx.moveTo(X(i),Y(y))); ctx.stroke(); }
}

function drawFrame(){
  const C=clip();
  fi=Math.max(0, Math.min(C.steps-1, fi));
  document.getElementById("scr").value=fi;
  document.getElementById("tlab").textContent = `frame ${fi+1}/${C.steps}`;
  const canvas=document.getElementById("stage");
  const ctx=canvas.getContext("2d");
  const W=canvas.width=canvas.clientWidth*devicePixelRatio;
  const H=canvas.height=canvas.clientHeight*devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const sk=C.skel[fi];
  let mn=[Infinity,Infinity,Infinity], mx=[-Infinity,-Infinity,-Infinity];
  for (const p of sk) for (let k=0;k<3;k++){ mn[k]=Math.min(mn[k],p[k]); mx[k]=Math.max(mx[k],p[k]); }
  const c=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
  const span=Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 0.8)*1.25;
  const yaw=0.7, pitch=0.45;
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const project=p=>{
    const x=p[0]-c[0], y=p[1]-c[1], z=p[2]-c[2];
    const xr=x*cy - y*sy, yr=x*sy + y*cy;
    const yp=yr*cp - z*sp, zp=yr*sp + z*cp;
    const s=Math.min(W,H)*0.72/span;
    return [W*0.5 + xr*s, H*0.62 - zp*s, yp];
  };
  const pts=sk.map(project);
  ctx.strokeStyle="#3d5a80"; ctx.lineWidth=2.2*devicePixelRatio;
  for (const [a,b] of C.bones){
    const A=pts[a], B=pts[b]; if(!A||!B) continue;
    ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(B[0],B[1]); ctx.stroke();
  }
  for (const b of C.kp){
    const r=project(b.robot[fi]), g=project(b.goal[fi]);
    ctx.strokeStyle=b.color; ctx.setLineDash([4*devicePixelRatio,3*devicePixelRatio]);
    ctx.beginPath(); ctx.moveTo(r[0],r[1]); ctx.lineTo(g[0],g[1]); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(g[0],g[1],6*devicePixelRatio,0,7); ctx.stroke();
    ctx.fillStyle=b.color; ctx.beginPath(); ctx.arc(r[0],r[1],4*devicePixelRatio,0,7); ctx.fill();
  }
}

document.getElementById("tog").onclick=()=>{ play=!play; document.getElementById("tog").textContent=play?"暂停":"播放"; };
document.getElementById("scr").oninput=e=>{ fi=+e.target.value; play=false; document.getElementById("tog").textContent="播放"; drawFrame(); };
const box=document.getElementById("clips");
PAGE.clips.forEach((c,i)=>{
  const b=document.createElement("button"); b.className="clip"+(i===0?" on":""); b.textContent=c.title;
  b.onclick=()=>setClip(i); box.appendChild(b);
});
setClip(0);
document.getElementById("sum").innerHTML =
  `<tr><th>Clip</th><th>类别</th><th class="num">mean vis</th><th class="num">SR@5cm</th><th class="num">SR@2cm</th></tr>` +
  PAGE.clips.map((c,i)=>`<tr><td>${c.title}</td><td>${c.category||""}</td><td class="num">${fmt(c.mean_err,3)} m</td><td class="num">${pct(c.sr5)}</td><td class="num">${pct(c.sr2)}</td></tr>`).join("");
function tick(){ if(play){ fi=(fi+1)%clip().steps; drawFrame(); } requestAnimationFrame(tick); }
requestAnimationFrame(tick);
addEventListener("resize", ()=>{ drawStatic(); drawFrame(); });
</script></body></html>
"""
    out_html.write_text(html.replace("__PAGE__", json.dumps(page)), encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _clip_to_payload(rec: dict, resume_path: str, mask: str, category: str = "") -> dict:
    path = Path(rec["path"])
    tag = path.stem.replace("bones_seed_g1_", "")[:56]
    if tag[:3].isdigit() and tag[3:4] == "_":
        tag = tag[4:]
    idx = rec["name_to_i"]
    bones = [[idx[a], idx[b]] for a, b in G1_BONES if a in idx and b in idx]
    stride = max(1, rec["skel"].shape[0] // 200)
    skel_ds = rec["skel"][::stride]
    kp_payload = []
    for name in rec["kp_robot"]:
        col, lab = KP_COLOR[name]
        kp_payload.append(
            {
                "name": name,
                "label": lab,
                "color": col,
                "robot": rec["kp_robot"][name][::stride],
                "goal": rec["kp_goal"][name][::stride],
                "err": rec["kp_err"][name],
            }
        )
    gif_name = f"{tag}.gif"
    return {
        "title": tag,
        "category": category or tag.split("_")[0],
        "meta": f"{Path(resume_path).name}  ·  mask={mask}  ·  {path.name}",
        "steps": int(skel_ds.shape[0]),
        "mean_err": rec["mean_err"],
        "final_err": rec["final_err"],
        "sr5": rec["sr5"],
        "sr2": rec["sr2"],
        "gif": gif_name,
        "bones": bones,
        "skel": np.round(skel_ds, 4).tolist(),
        "kp": kp_payload,
    }


def _publish_page(out: Path, resume_path: str, mask: str) -> None:
    import fcntl

    lock_path = out / ".publish.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        clips = []
        for p in sorted(out.glob("*.clip.json")):
            clips.append(json.loads(p.read_text(encoding="utf-8")))
        page = {"ckpt": str(resume_path), "mask": mask, "clips": clips}
        _write_html(out / "index.html", page)
        home = Path("/data/home/chenxiangyu/tritrack_infer_55000.html")
        home.write_text((out / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
        for c in clips:
            src = out / c["gif"]
            dst = Path("/data/home/chenxiangyu") / Path(c["gif"]).name
            if src.exists():
                dst.write_bytes(src.read_bytes())
        print(f"[infer] published {len(clips)} clips -> {out / 'index.html'}", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    motions = args_cli.motion or []
    if not motions:
        raise SystemExit("pass --motion DIR (batch folder of npz files)")
    motion_dir = motions[0]
    n_files = len(list(Path(motion_dir).rglob("*.npz")))
    if n_files == 0:
        raise SystemExit(f"no npz in {motion_dir}")
    n_want = int(args_cli.num_envs) if args_cli.num_envs and int(args_cli.num_envs) > 0 else n_files
    env_cfg.scene.num_envs = min(n_want, n_files)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[infer] ckpt {resume_path}", flush=True)
    print(f"[infer] motion_dir={motion_dir} n_npz={n_files} num_envs={env_cfg.scene.num_envs}", flush=True)
    _restore_adapter(agent_cfg, resume_path)
    _apply_eval_motion(env_cfg, motion_dir, args_cli.start_frame)

    from whole_body_tracking.tasks.tracking.config.g1.mask_modes import eval_single_mode_spec

    spec, probs = eval_single_mode_spec(args_cli.mask_modes)
    env_cfg.commands.motion.mask_mode_spec = spec
    env_cfg.commands.motion.mask_mode_probs = probs
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

    cat_map = {}
    cat_file = Path(motion_dir) / "categories.json"
    if cat_file.exists():
        cat_map = json.loads(cat_file.read_text(encoding="utf-8"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints

    attach_curriculum_rollout_hints(
        env, spe=int(agent_cfg.num_steps_per_env), learning_iteration=int(args_cli.hint_iter)
    )
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, load_critic=True)
    policy = runner.get_inference_policy(device=env.device)
    recs = _rollout_batch(env, policy, int(args_cli.steps))
    env.close()

    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        payload = _clip_to_payload(
            rec, resume_path, args_cli.mask_modes, cat_map.get(Path(rec["path"]).name, "")
        )
        (out / f"{payload['title']}.clip.json").write_text(
            json.dumps(payload, default=_json_default), encoding="utf-8"
        )
        _write_gif(out / payload["gif"], rec, payload["title"])
        print(
            f"[infer] {payload['title']} mean={payload['mean_err']:.3f}m "
            f"sr5={payload['sr5']:.3f} sr2={payload['sr2']:.3f}",
            flush=True,
        )
        if not args_cli.skip_publish:
            _publish_page(out, resume_path, args_cli.mask_modes)
    if not args_cli.skip_publish:
        _publish_page(out, resume_path, args_cli.mask_modes)


if __name__ == "__main__":
    main()
