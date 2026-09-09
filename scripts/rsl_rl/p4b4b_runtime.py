"""P4-B4B 200 Hz intra-policy microprobe. Import only from eval_irr_r3_short after AppLauncher."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _E():
    """Use the already-running eval module. Never re-import (that restarts AppLauncher)."""
    import sys

    for name in ("eval_irr_r3_short", "__main__"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "_decode") and hasattr(m, "H_CORR"):
            return m
    raise RuntimeError("eval_irr_r3_short helpers not loaded; run via eval_irr_r3_short.py")


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def make_u4() -> np.ndarray:
    u = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    ) / math.sqrt(3.0)
    ones = u.sum(axis=0)
    gram = u.T @ u
    assert np.allclose(ones, 0.0, atol=1e-8), ones
    assert np.allclose(gram, (4.0 / 3.0) * np.eye(3), atol=1e-6), gram
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-6)
    return u


def make_u8() -> np.ndarray:
    h = hadamard(8)
    u = h[:, [1, 2, 4]].astype(np.float64) / math.sqrt(3.0)
    return u


def qr_sign_align(bt: torch.Tensor, bref: torch.Tensor) -> torch.Tensor:
    """bt, bref: [n,16,k]. Flip QR columns to match bref."""
    out = bt.clone()
    for i in range(int(bt.shape[-1])):
        s = (out[:, :, i] * bref[:, :, i]).sum(-1, keepdim=True)
        out[:, :, i] = torch.where(s < 0, -out[:, :, i], out[:, :, i])
    return out


def procrustes(bn: torch.Tensor, b0: torch.Tensor) -> torch.Tensor:
    """Align BN to B0: BN_aligned = BN @ R, R from orthogonal Procrustes. [n,16,k]."""
    m = torch.matmul(bn.transpose(1, 2), b0)
    u, _s, vh = torch.linalg.svd(m)
    r = torch.matmul(u, vh)
    det = torch.det(r)
    flip = torch.ones_like(r)
    flip[:, -1, -1] = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    r = torch.matmul(u, torch.matmul(flip, vh))
    return torch.matmul(bn, r)


def action_term(uw):
    names = list(uw.action_manager.active_terms)
    return uw.action_manager.get_term(names[0])


def ll_apply(uw, action: torch.Tensor) -> None:
    uw.action_manager.process_action(action.to(uw.device))
    uw._sim_step_counter += 1
    uw.action_manager.apply_action()
    uw.scene.write_data_to_sim()
    uw.sim.step(render=False)
    uw.scene.update(dt=uw.physics_dt)


def finish_tick(uw) -> None:
    uw.episode_length_buf += 1
    uw.common_step_counter += 1
    uw.command_manager.compute(dt=uw.step_dt)
    uw.obs_buf = uw.observation_manager.compute()


def read_ll(env, vis_i, vis_flags, ankle_i, cf, n_valid: int) -> dict:
    E = _E()
    uw = env.unwrapped
    robot = uw.scene["robot"]
    cmd = uw.command_manager.get_term("motion")
    term = action_term(uw)
    e = E._visible_e(cmd, vis_i, vis_flags)[:n_valid]
    q = robot.data.joint_pos[:n_valid]
    dq = robot.data.joint_vel[:n_valid]
    tgt = term.processed_actions[:n_valid]
    raw = term.raw_actions[:n_valid]
    omega = getattr(robot.data, "root_ang_vel_b", None)
    if omega is None:
        omega = robot.data.root_ang_vel_w
    vel = getattr(robot.data, "root_lin_vel_b", None)
    if vel is None:
        vel = robot.data.root_lin_vel_w
    grav = robot.data.GRAVITY_VEC_W
    if grav.ndim == 1:
        grav = grav.unsqueeze(0).expand(int(uw.num_envs), -1)
    from isaaclab.utils.math import quat_rotate_inverse

    pg = quat_rotate_inverse(robot.data.root_quat_w, grav)[:n_valid]
    contact = torch.zeros(n_valid, 2, device=q.device)
    if cf is not None and ankle_i and hasattr(cf.data, "net_forces_w"):
        contact = (cf.data.net_forces_w[:n_valid][:, ankle_i, :].norm(dim=-1) > E.CONTACT_N).float()
    tau = getattr(robot.data, "computed_torque", None)
    if tau is None:
        tau = getattr(robot.data, "applied_torque", None)
    if tau is None:
        tau = torch.zeros_like(q)
    else:
        tau = tau[:n_valid]
    lim = torch.zeros(n_valid, device=q.device)
    if hasattr(robot.data, "soft_joint_pos_limits"):
        lo, hi = robot.data.soft_joint_pos_limits[:n_valid, :, 0], robot.data.soft_joint_pos_limits[:n_valid, :, 1]
        lim = ((q < lo) | (q > hi)).any(dim=-1).float()
    return {
        "e": e.detach(),
        "q": q.detach(),
        "dq": dq.detach(),
        "tgt": tgt.detach(),
        "raw": raw.detach(),
        "omega": omega[:n_valid].detach(),
        "pg": pg.detach(),
        "vel": vel[:n_valid].detach(),
        "contact": contact.detach(),
        "tau": tau.detach(),
        "jlim": lim.detach(),
    }


def stack_ll(logs: list[dict], key: str) -> torch.Tensor:
    return torch.stack([x[key] for x in logs], dim=1)


def exp_z(z0: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    E = _E()
    v = E.project_tangent(v, z0)
    nh = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c, s = math.cos(eps), math.sin(eps)
    return F.normalize(c * z0 + s * v / nh, dim=-1, eps=1e-8)


def decode_code(policy, z0, proprio0, b0, u_np, eps: float):
    E = _E()
    device = z0.device
    u_t = torch.as_tensor(u_np, device=device, dtype=torch.float32)
    acts, zs = [], []
    for tau in range(u_t.shape[0]):
        v = torch.einsum("njk,k->nj", b0, u_t[tau])
        z_tau = exp_z(z0, v, eps)
        a = E._decode(policy, z_tau, proprio0)
        acts.append(a)
        zs.append(z_tau)
    return acts, zs


@torch.inference_mode()
def roll_window(env, policy, runner, injector, batch, n_valid, nsub, acts_or_twin, vis_i, vis_flags, ankle_i, cf):
    """acts_or_twin: list of (n_envs, na) length nsub, or 'twin' to re-decode every 4 ll steps."""
    E = _E()
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = E._pad_snaps(batch, n_envs)
    E._restore_batch(env, padded)
    uw = env.unwrapped
    device = uw.device
    logs = []
    logs.append(read_ll(env, vis_i, vis_flags, ankle_i, cf, n_valid))
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    a_hold = None
    for tau in range(int(nsub)):
        if acts_or_twin == "twin":
            if tau % 4 == 0:
                if tau == 0:
                    obs = injector.patch_policy_obs(obs_stack, env, "mapper")
                else:
                    obs = obs_from_env(env, injector, padded, n_valid)
                z, proprio, _enc = E._z_and_proprio(policy, runner, obs)
                a_hold = E._decode(policy, z, proprio)
            ll_apply(uw, a_hold)
        else:
            ll_apply(uw, acts_or_twin[tau])
        logs.append(read_ll(env, vis_i, vis_flags, ankle_i, cf, n_valid))
        if (tau + 1) % 4 == 0:
            finish_tick(uw)
    rec = recapture_ll(env, policy, runner, injector, batch, n_valid)
    return logs, rec


def recapture_ll(env, policy, runner, injector, orig, n_valid):
    """Snapshot after ll ticks without a second observation_manager.compute()."""
    E = _E()
    hist = E._hist_snapshot(env)
    obs = obs_from_env(env, injector, orig, n_valid)
    z, _pr, _enc = E._z_and_proprio(policy, runner, obs)
    out = []
    for i in range(n_valid):
        extra = {
            "obs": obs[i].detach().cpu().clone(),
            "z_nom": z[i].detach().cpu().clone(),
            "e0": orig[i]["e0"],
            "t": orig[i]["t"],
            "seed": orig[i]["seed"],
            "terrain": orig[i]["terrain"],
            "window": orig[i].get("window", "recovery"),
            "clip": orig[i]["clip"],
            "task": orig[i].get("task", "loco"),
        }
        out.append(E._capture_snap(env, i, hist, extra))
    return out


def obs_from_env(env, injector, padded, n_valid):
    uw = env.unwrapped
    buf = getattr(uw, "obs_buf", None)
    if buf is None:
        buf = uw.observation_manager.compute()
        uw.obs_buf = buf
    if isinstance(buf, dict):
        raw = buf.get("policy", next(iter(buf.values())))
    else:
        raw = buf
    return injector.patch_policy_obs(raw, env, "mapper")


@torch.inference_mode()
def run_p4b4b(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, B_raw_np):
    E = _E()
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    k = 3
    B = np.asarray(B_raw_np[:, :k], dtype=np.float64)
    B_t = torch.as_tensor(B, device=device, dtype=torch.float32)
    tan5 = math.tan(math.radians(E.EVAL_DEG))
    eps1 = math.tan(math.radians(1.0))
    H = int(E.H_CORR)
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    asset = env.unwrapped.scene["robot"]
    names = list(asset.data.body_names)
    ankle_i = [names.index(n) for n in E.ANKLE_BODIES if n in names]
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    kwy = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i, ankle_i=ankle_i, cf=cf)
    u4 = make_u4()
    u8 = make_u8()
    codes = {4: u4, 8: u8, "rev": u4[::-1].copy(), "cyc": np.roll(u4, 1, axis=0)}
    amps = (0.25, 0.5, 1.0)
    nsubs = (4, 8)
    uw = env.unwrapped
    phys_dt = float(uw.physics_dt)
    dec = int(uw.cfg.decimation)
    print(f"[p4b4b] physics_dt={phys_dt} decimation={dec} policy_dt={phys_dt * dec}", flush=True)
    if abs(phys_dt - 0.005) > 1e-6 or dec != 4:
        print(f"[p4b4b] WARN rates differ from 200/50 Hz", flush=True)
    rng = np.random.RandomState(2026)
    order_idx = np.sort(rng.choice(len(snaps), size=min(14, len(snaps)), replace=False))
    order_set = set(int(i) for i in order_idx)

    def cfg_list(global_i0):
        out = []
        for nsub in nsubs:
            for amp in amps:
                out.append((nsub, amp, "default", codes[nsub]))
        # code-order extras on subset, primary 20 ms 0.5°
        for tag, uu in (("rev", codes["rev"]), ("cyc", codes["cyc"])):
            out.append((4, 0.5, tag, uu))
        return out

    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[p4b4b] batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        padded, _ = E._pad_snaps(batch, n_envs)
        pack = {}
        for nsub, amp, tag, u_np in cfg_list(start):
            do_order = tag != "default"
            if do_order:
                local = [start + i in order_set for i in range(n_valid)]
                if not any(local):
                    continue
            eps = math.radians(float(amp))
            E._restore_batch(env, padded)
            obs0 = injector.patch_policy_obs(torch.stack([s["obs"].to(device) for s in padded], 0), env, "mapper")
            z0, proprio0, _enc = E._z_and_proprio(policy, runner, obs0)
            b0_raw = E._tangent_B(z0, B_t)
            bref = project_cols(z0, B_t)
            b0 = qr_sign_align(b0_raw, bref)
            acts, zs = decode_code(policy, z0, proprio0, b0, u_np, eps)
            # pad acts to n_envs
            acts_pad = []
            for a in acts:
                if a.shape[0] < n_envs:
                    extra = a[-1:].expand(n_envs - a.shape[0], -1)
                    a = torch.cat([a, extra], 0)
                acts_pad.append(a)

            logs_t, _rec_t = roll_window(
                env, policy, runner, injector, batch, n_valid, nsub, "twin", vis_i, vis_flags, ankle_i, cf
            )
            logs_p, rec_p = roll_window(
                env, policy, runner, injector, batch, n_valid, nsub, acts_pad, vis_i, vis_flags, ankle_i, cf
            )
            logs_tb = None
            if nsub == 4 and abs(amp - 0.5) < 1e-9 and tag == "default":
                logs_tb, _ = roll_window(
                    env, policy, runner, injector, batch, n_valid, nsub, "twin", vis_i, vis_flags, ankle_i, cf
                )

            z_end = torch.stack([s["z_nom"] for s in rec_p], 0).to(device)
            bn_raw = E._tangent_B(z_end, B_t)
            bn = procrustes(qr_sign_align(bn_raw, project_cols(z_end, B_t)), b0[:n_valid])
            key = f"n{nsub}_a{amp:g}_{tag}"
            blk = {
                "e_p": stack_ll(logs_p, "e").cpu().numpy().astype(np.float32),
                "e_t": stack_ll(logs_t, "e").cpu().numpy().astype(np.float32),
                "q_p": stack_ll(logs_p, "q").cpu().numpy().astype(np.float32),
                "q_t": stack_ll(logs_t, "q").cpu().numpy().astype(np.float32),
                "dq_p": stack_ll(logs_p, "dq").cpu().numpy().astype(np.float32),
                "dq_t": stack_ll(logs_t, "dq").cpu().numpy().astype(np.float32),
                "tgt_p": stack_ll(logs_p, "tgt").cpu().numpy().astype(np.float32),
                "tgt_t": stack_ll(logs_t, "tgt").cpu().numpy().astype(np.float32),
                "raw_p": stack_ll(logs_p, "raw").cpu().numpy().astype(np.float32),
                "raw_t": stack_ll(logs_t, "raw").cpu().numpy().astype(np.float32),
                "omega_p": stack_ll(logs_p, "omega").cpu().numpy().astype(np.float32),
                "omega_t": stack_ll(logs_t, "omega").cpu().numpy().astype(np.float32),
                "pg_p": stack_ll(logs_p, "pg").cpu().numpy().astype(np.float32),
                "pg_t": stack_ll(logs_t, "pg").cpu().numpy().astype(np.float32),
                "vel_p": stack_ll(logs_p, "vel").cpu().numpy().astype(np.float32),
                "vel_t": stack_ll(logs_t, "vel").cpu().numpy().astype(np.float32),
                "contact_p": stack_ll(logs_p, "contact").cpu().numpy().astype(np.float32),
                "contact_t": stack_ll(logs_t, "contact").cpu().numpy().astype(np.float32),
                "tau_p": stack_ll(logs_p, "tau").cpu().numpy().astype(np.float32),
                "tau_t": stack_ll(logs_t, "tau").cpu().numpy().astype(np.float32),
                "jlim_p": stack_ll(logs_p, "jlim").cpu().numpy().astype(np.float32),
                "jlim_t": stack_ll(logs_t, "jlim").cpu().numpy().astype(np.float32),
                "B0": b0[:n_valid].cpu().numpy().astype(np.float32),
                "BN": bn.cpu().numpy().astype(np.float32),
                "z0": z0[:n_valid].cpu().numpy().astype(np.float32),
                "zN": z_end.cpu().numpy().astype(np.float32),
                "U": u_np.astype(np.float32),
                "nsub": nsub,
                "amp": amp,
                "tag": tag,
            }
            if logs_tb is not None:
                blk["e_tB"] = stack_ll(logs_tb, "e").cpu().numpy().astype(np.float32)
            need_lab = tag == "default" or (do_order and True)
            if need_lab:
                j0_end, _ = E._roll_j(env, policy, runner, injector, rec_p, n_valid, None, horizon=H, **kwj)
                a5_end = torch.zeros(n_valid, k, 2, device=device)
                for ki in range(k):
                    for si, sgn in enumerate((1.0, -1.0)):
                        dz = _axis_from_b(bn, ki, sgn, tan5)
                        j5, _ = E._roll_j(env, policy, runner, injector, rec_p, n_valid, dz, horizon=H, **kwj)
                        a5_end[:, ki, si] = j0_end - j5
                g_end = torch.zeros(n_valid, k, device=device)
                for ki in range(k):
                    v = bn[:, :, ki]
                    dyp, _, _ = E._roll_y(env, policy, runner, injector, rec_p, n_valid, eps1 * v, horizon=1, **kwy)
                    dym, _, _ = E._roll_y(env, policy, runner, injector, rec_p, n_valid, -eps1 * v, horizon=1, **kwy)
                    g_end[:, ki] = (dyp[:, 0] - dym[:, 0]) / (2.0 * eps1)
                blk["a5_end"] = a5_end.cpu().numpy().astype(np.float32)
                blk["g_end"] = g_end.cpu().numpy().astype(np.float32)
                blk["j0_end"] = j0_end.cpu().numpy().astype(np.float32)
                if tag == "default":
                    a_net = torch.zeros(n_valid, k, 2, device=device)
                    for ki in range(k):
                        for si, sgn in enumerate((1.0, -1.0)):
                            dstar = F.normalize(_axis_from_b(bn, ki, sgn, 1.0), dim=-1, eps=1e-8)
                            jpar, jnet = anet_micro(
                                env, policy, runner, injector, batch, n_valid, acts_pad, dstar,
                                vis_i, vis_flags, pelvis_i, nsub, H,
                            )
                            a_net[:, ki, si] = jpar - jnet
                    jpar0, jnet0 = anet_micro(
                        env, policy, runner, injector, batch, n_valid, acts_pad, torch.zeros_like(z_end),
                        vis_i, vis_flags, pelvis_i, nsub, H,
                    )
                    blk["a_net"] = a_net.cpu().numpy().astype(np.float32)
                    blk["a_net0"] = (jpar0 - jnet0).cpu().numpy().astype(np.float32)
            pack[key] = blk
            print(f"[p4b4b]   {key} e_ex med={float(np.median(np.max(np.maximum(blk['e_p']-blk['e_t'],0),1))):.5f}", flush=True)
        for i in range(n_valid):
            s = batch[i]
            item = {
                "clip": s["clip"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "terrain": s["terrain"],
                "e0": float(s["e0"]),
                "a5_gt": np.asarray(s["a5_gt"], dtype=np.float32),
                "order_subset": int(start + i in order_set),
                "cfgs": {},
            }
            for key, blk in pack.items():
                item["cfgs"][key] = {}
                for kk, vv in blk.items():
                    if kk in ("U", "nsub", "amp", "tag"):
                        item["cfgs"][key][kk] = vv
                    elif isinstance(vv, np.ndarray) and vv.shape[0] == n_valid:
                        item["cfgs"][key][kk] = vv[i]
                    else:
                        item["cfgs"][key][kk] = vv
            rows.append(item)
    return rows, {"U4": u4, "U8": u8, "physics_dt": phys_dt, "decimation": dec}


def project_cols(z: torch.Tensor, B_t: torch.Tensor) -> torch.Tensor:
    E = _E()
    n, k = z.shape[0], int(B_t.shape[1])
    cols = [E.project_tangent(B_t[:, i].unsqueeze(0).expand(n, -1), z) for i in range(k)]
    return torch.stack(cols, dim=-1)


def _axis_from_b(b: torch.Tensor, ki: int, sgn: float, mag: float) -> torch.Tensor:
    v = b[:, :, int(ki)]
    nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return float(sgn) * float(mag) * v / nrm


@torch.inference_mode()
def anet_micro(env, policy, runner, injector, batch, n_valid, acts_ll, dstar, vis_i, vis_flags, pelvis_i, nsub, H):
    E = _E()
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = E._pad_snaps(batch, n_envs)
    n_pol = int(nsub) // 4
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    j_par, _ = E._roll_j(env, policy, runner, injector, batch, n_valid, None, horizon=n_pol + H, **kwj)
    E._restore_batch(env, padded)
    uw = env.unwrapped
    device = uw.device
    cmd = uw.command_manager.get_term("motion")
    asset = uw.scene["robot"]
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)
    failed = torch.zeros(n_envs, dtype=torch.bool, device=device)
    j = torch.zeros(n_envs, device=device)
    tan5 = math.tan(math.radians(E.EVAL_DEG))
    dpad = torch.zeros(n_envs, 16, device=device)
    if dstar is not None:
        dpad[:n_valid] = dstar
    for tau in range(int(nsub)):
        ll_apply(uw, acts_ll[tau])
        if (tau + 1) % 4 == 0:
            finish_tick(uw)
            e = E._visible_e(cmd, vis_i, vis_flags)
            fail = E._official_fail(cmd, asset, vis_i, n_envs, pelvis_i, gravity)
            failed = failed | fail
            t_pol = (tau + 1) // 4 - 1
            j = j + (E.GAMMA ** t_pol) * (e + E.LAM_S * fail.to(dtype=e.dtype))
    for t in range(int(H)):
        obs = obs_from_env(env, injector, padded, n_valid)
        z_nom, proprio, _enc = E._z_and_proprio(policy, runner, obs)
        z_exec = F.normalize(z_nom + tan5 * dpad, dim=-1, eps=1e-8)
        env.step(E._decode(policy, z_exec, proprio))
        e = E._visible_e(cmd, vis_i, vis_flags)
        fail = E._official_fail(cmd, asset, vis_i, n_envs, pelvis_i, gravity)
        failed = failed | fail
        j = j + (E.GAMMA ** (n_pol + t)) * (e + E.LAM_S * fail.to(dtype=e.dtype))
    return j_par, j[:n_valid].detach()
