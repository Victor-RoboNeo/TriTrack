"""Atomic status, flock, and small IO helpers."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import CAMPAIGN, RESULTS, STAGES

LOCK_PATH = RESULTS / "campaign.lock"
STATUS_PATH = RESULTS / "STATUS.json"
ATTEMPTS_PATH = RESULTS / "logs" / "attempts.jsonl"
COMMANDS_PATH = RESULTS / "logs" / "commands.jsonl"
_STATUS_LOCK = threading.Lock()
TERMINAL = ("PASS", "FAIL", "SKIPPED_BY_GATE", "COMPLETE")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    data = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with open(tmp, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


def default_stage() -> dict:
    return {
        "stage_state": "PENDING",
        "attempt": 0,
        "start_time": None,
        "end_time": None,
        "artifacts": [],
        "error": None,
        "gate": None,
        "metrics": {},
    }


def default_status() -> dict:
    st = {
        "campaign": CAMPAIGN,
        "current_stage": "P0_AUDIT",
        "stage_state": "PENDING",
        "last_checkpoint": None,
        "best_checkpoint": None,
        "gate": None,
        "updated_at": utc_now(),
        "started_at": utc_now(),
        "pid": os.getpid(),
        "selected_gpu": None,
        "HUMAN_LOWER_BODY_INPUT": "NONE",
        "ROBOT_LOWER_BODY_REALIZATION": "AUTONOMOUS",
        "TERRAIN": "FLAT PLANE ONLY",
    }
    for s in STAGES:
        st[s] = default_stage()
    return st


def load_status() -> dict:
    with _STATUS_LOCK:
        if STATUS_PATH.exists():
            return json.loads(STATUS_PATH.read_text())
        st = default_status()
        atomic_write_json(STATUS_PATH, st)
        return st


def save_status(st: dict) -> None:
    with _STATUS_LOCK:
        st["updated_at"] = utc_now()
        atomic_write_json(STATUS_PATH, st)


def update_stage(stage: str, **fields: Any) -> dict:
    with _STATUS_LOCK:
        if STATUS_PATH.exists():
            st = json.loads(STATUS_PATH.read_text())
        else:
            st = default_status()
        st.setdefault(stage, default_stage())
        st[stage].update(fields)
        st["current_stage"] = stage
        if "stage_state" in fields:
            st["stage_state"] = fields["stage_state"]
        if "gate" in fields and fields["gate"] is not None:
            st["gate"] = fields["gate"]
        st["updated_at"] = utc_now()
        atomic_write_json(STATUS_PATH, st)
        return st


def stage_done(st: dict, stage: str) -> bool:
    return str(st.get(stage, {}).get("stage_state")) in TERMINAL


def acquire_lock() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(f"another FLAT_UNIFIED_SPARSE_INTENT_V1 orchestrator holds {LOCK_PATH}") from exc
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class OrchestratorLock:
    def __init__(self) -> None:
        self.fd: int | None = None

    def __enter__(self) -> "OrchestratorLock":
        self.fd = acquire_lock()
        return self

    def __exit__(self, *exc) -> None:
        if self.fd is not None:
            release_lock(self.fd)
            self.fd = None


def git_head() -> str:
    import subprocess

    try:
        from .constants import ANYBODY

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ANYBODY), text=True).strip()
    except Exception:
        return "unknown"
