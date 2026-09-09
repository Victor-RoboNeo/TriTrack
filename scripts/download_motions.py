#!/usr/bin/env python3
"""Download MOSAIC G1 npz (and LAFAN1 G1 csv) with retries. Resume-safe. Avoids HF XET 429."""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

ROOT = Path("/data/home/chenxiangyu/robotics/Anybody/datasets")
PROXY = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "http://127.0.0.1:7890"
os.environ.setdefault("https_proxy", PROXY)
os.environ.setdefault("http_proxy", PROXY)


def pull(repo: str, local: Path, patterns: list[str], tries: int = 8) -> None:
    local.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(1, tries + 1):
        try:
            print(f"[INFO] snapshot {repo} patterns={patterns} attempt={i}", flush=True)
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=patterns,
                local_dir=str(local),
                max_workers=2,
                etag_timeout=60,
            )
            print(f"[INFO] done {repo} {patterns}", flush=True)
            return
        except Exception as e:
            last = e
            wait = min(90, 8 * i)
            print(f"[WARN] {repo} {patterns} failed: {type(e).__name__}: {e}; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed after {tries} tries: {repo} {patterns}: {last}")


def main() -> None:
    mosaic = ROOT / "MOSAIC_Dataset"
    folders = [
        "G1/optical_mocap/*.npz",
        "G1/inertial_mocap/*.npz",
        "G1/generated_genmo/*.npz",
        "G1/adaptor_data/**/*.npz",
    ]
    for pat in folders:
        pull("BAAI-Humanoid/MOSAIC_Dataset", mosaic, [pat])
    pull(
        "lvhaidong/LAFAN1_Retargeting_Dataset",
        ROOT / "LAFAN1_Retargeting_Dataset",
        ["g1/*.csv", "README.md", "LICENSE"],
    )
    npz = list((mosaic / "G1").rglob("*.npz"))
    print(f"[INFO] local G1 npz count={len(npz)}", flush=True)
    if len(npz) < 200:
        raise SystemExit(f"too few motions: {len(npz)}")


if __name__ == "__main__":
    main()
