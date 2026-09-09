"""GPU pick + subprocess helpers. Never kill other users' jobs. Never match CMD cuda:0 as infra fail."""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from .constants import ANYBODY, HT_ROOT, MAX_INFRA_RETRY, RESULTS, TRITRACK, X11LIB  # noqa: F401
from .io_util import utc_now

INFRA_RE = re.compile(
    r"(CUDA error|cuda initialization|OutOfMemory|OOM|Kit crashed|failed to spawn|"
    r"Connection reset|No such file or directory: '/tmp|X server|segmentation fault)",
    re.I,
)
NO_RETRY_RE = re.compile(r"(nan|NaN|shape mismatch|AssertionError|NOOP identity failed)", re.I)


def query_gpus() -> dict:
    q = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in q.strip().splitlines():
        idx, name, total, free, used, util = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "memory_total_mib": float(total),
                "memory_free_mib": float(free),
                "memory_used_mib": float(used),
                "utilization_pct": float(util),
            }
        )
    return {"timestamp": utc_now(), "gpus": gpus}


def empty_gpu_indices(info: dict, max_used_mib: float = 500.0, max_util: float = 5.0) -> list[int]:
    return [
        int(g["index"])
        for g in info["gpus"]
        if g["memory_used_mib"] <= max_used_mib and g["utilization_pct"] <= max_util
    ]


def wait_for_empty_gpu(poll_s: float = 30.0) -> int:
    while True:
        info = query_gpus()
        empty = empty_gpu_indices(info)
        if empty:
            return empty[0]
        print(f"[gpu] waiting for idle GPU; used={[(g['index'], g['memory_used_mib']) for g in info['gpus']]}", flush=True)
        time.sleep(poll_s)


def isaac_env(gpu: str, tag: str) -> dict:
    env = os.environ.copy()
    kit = f"/tmp/ncr_{tag}_{os.getpid()}"
    os.makedirs(kit, exist_ok=True)
    os.makedirs(f"{kit}/mpl", exist_ok=True)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "GIT_PYTHON_REFRESH": "quiet",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "ISAACLAB_PATH": "/data/home/chenxiangyu/robotics/IsaacLab_v2.1",
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": kit,
            "OMNI_USER_DIR": kit,
            "XDG_CACHE_HOME": f"{kit}/cache",
            "XDG_DATA_HOME": f"{kit}/data",
            "XDG_CONFIG_HOME": f"{kit}/config",
            "WANDB_MODE": "offline",
            "MPLCONFIGDIR": f"{kit}/mpl",
            "PYTHONPATH": f"{ANYBODY / 'scripts'}:{HT_ROOT}:{TRITRACK}:{env.get('PYTHONPATH', '')}",
            "FLAT_NCR_ROOT": str(RESULTS),
        }
    )
    lp = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{X11LIB}:{lp}"
    return env


def log_path(name: str, attempt: int) -> Path:
    p = RESULTS / "logs" / f"{name}_attempt{attempt:02d}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_logged(name: str, cmd: list[str], *, cwd: Path, env: dict, attempt: int) -> tuple[int, Path]:
    lp = log_path(name, attempt)
    print(f"[orch] {name} attempt={attempt} log={lp}", flush=True)
    with open(lp, "w") as f:
        f.write("CMD " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rc = proc.wait()
    return rc, lp


def is_infra_failure(log: Path) -> bool:
    if not log.exists():
        return False
    txt = log.read_text(errors="replace")
    # Skip the CMD line so `--device=cuda:0` is never treated as a CUDA crash.
    body = "\n".join(txt.splitlines()[1:])[-20000:]
    if NO_RETRY_RE.search(body) and not INFRA_RE.search(body):
        return False
    return bool(INFRA_RE.search(body))


def run_with_retry(name: str, cmd: list[str], *, cwd: Path, env: dict, expected: Path | None = None) -> Path:
    last_lp = None
    last_rc = 1
    for attempt in range(1, MAX_INFRA_RETRY + 2):
        rc, lp = run_logged(name, cmd, cwd=cwd, env=env, attempt=attempt)
        last_lp, last_rc = lp, rc
        ok = rc == 0
        if expected is not None:
            ok = ok and expected.exists() and expected.stat().st_size > 0
        if ok:
            return lp
        if attempt <= MAX_INFRA_RETRY and is_infra_failure(lp):
            print(f"[orch] infra failure {name}; retry {attempt + 1}", flush=True)
            time.sleep(10)
            continue
        raise RuntimeError(f"{name} failed rc={last_rc} log={last_lp}")
    raise RuntimeError(f"{name} failed rc={last_rc} log={last_lp}")


def latest_ckpt(experiment_name: str) -> Path:
    root = ANYBODY / "logs" / "rsl_rl" / experiment_name
    if not root.exists():
        raise FileNotFoundError(root)
    runs = [p for p in root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(root)
    ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(runs[-1])
    return ckpts[-1]


def all_ckpts(experiment_name: str) -> list[Path]:
    root = ANYBODY / "logs" / "rsl_rl" / experiment_name
    if not root.exists():
        return []
    out = []
    for run in root.iterdir():
        if run.is_dir():
            out.extend(run.glob("model_*.pt"))
    return sorted(out, key=lambda p: p.stat().st_mtime)
