"""Pick 10 diverse SONIC clips per P1 task. No _M mirrors. Symlink into infer_clips/p1/."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/data/home/chenxiangyu/robotics/Anybody/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train")
OUT = Path("/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1")
N = 10

# Keyword buckets. Exclude is applied first.
RULES = {
    "loco": {
        "include": ("walk", "turn", "jog", "strafe", "lateral", "side_step", "sidestep"),
        "exclude": (
            "pick", "grab", "reach", "lift", "crate", "stoop", "squat", "kneel",
            "injured", "faint", "lying", "push_up", "whaaat",
        ),
        "prefer": ("walk_forward", "forward_walk", "backward", "turn", "jog", "strafe", "lateral", "sidestep"),
    },
    "reach": {
        "include": ("reach", "grab"),
        "exclude": (
            "pick_up", "lift_crate", "stoop", "squat", "injured", "faint", "lying",
            "whaaat", "grab_head",
        ),
        "prefer": ("reaching", "walk_grab", "grab_walk", "reach_up", "reach_down"),
    },
    "stoop": {
        "include": ("stoop", "squat", "pick", "crouch"),
        "exclude": ("crate", "lift_crate", "push_up", "lying", "faint", "injured", "grab"),
        "prefer": ("stoop_down", "squat", "crouch_idle", "crouch_walk", "pick_up"),
    },
    "carry": {
        "include": ("lift_crate", "carry", "lift"),
        "exclude": ("pick_up", "stoop", "squat", "reach", "injured", "faint"),
        "prefer": ("lift_crate_start", "lift_crate_loop", "lift_crate_stop", "lift_crate_turn", "lift_crate_walk"),
    },
}


def _key(stem: str) -> str:
    s = re.sub(r"_M$", "", stem)
    s = re.sub(r"__A\d+$", "", s)
    return s


def _is_mirror(path: Path) -> bool:
    return path.name.endswith("_M.npz") or path.stem.endswith("_M")


def _match(name: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    n = name.lower()
    if any(x in n for x in exclude):
        return False
    return any(x in n for x in include)


def _family(key: str) -> str:
    """Collapse takes / angles / L-R so Turn_Start_Jog_0000_001 and _0045 share a family."""
    s = re.sub(r"_R$", "", key)
    for _ in range(4):
        nxt = re.sub(r"_\d{3,4}$", "", s)
        if nxt == s:
            break
        s = nxt
    return s


def _score(key: str, prefer: tuple[str, ...]) -> tuple[int, str]:
    n = key.lower()
    for i, p in enumerate(prefer):
        if p in n:
            return (i, key)
    return (len(prefer), key)


def _balanced(cands: list[Path], prefer: tuple[str, ...], n: int) -> list[Path]:
    buckets: dict[str, list[Path]] = {p: [] for p in prefer}
    buckets["_other"] = []
    for p in cands:
        k = _key(p.stem).lower()
        hit = next((pref for pref in prefer if pref in k), "_other")
        buckets[hit].append(p)
    chosen: list[Path] = []
    seen_fam: set[str] = set()
    keys = list(prefer) + ["_other"]
    while len(chosen) < n:
        progressed = False
        for k in keys:
            if len(chosen) >= n:
                break
            bucket = buckets[k]
            while bucket:
                p = bucket.pop(0)
                fam = _family(_key(p.stem))
                if fam in seen_fam:
                    continue
                chosen.append(p)
                seen_fam.add(fam)
                progressed = True
                break
        if not progressed:
            break
    if len(chosen) < n:
        for p in cands:
            if len(chosen) >= n:
                break
            if p in chosen:
                continue
            chosen.append(p)
    return chosen


def main() -> None:
    files = [p for p in ROOT.rglob("*.npz") if not _is_mirror(p)]
    by_key: dict[str, Path] = {}
    for p in files:
        by_key.setdefault(_key(p.stem), p)

    manifest: dict[str, list[str]] = {}
    for task, rule in RULES.items():
        cands = [p for k, p in by_key.items() if _match(k, rule["include"], rule["exclude"])]
        cands.sort(key=lambda p: _score(_key(p.stem), rule["prefer"]))
        chosen = _balanced(cands, rule["prefer"], N)
        if len(chosen) < N:
            raise SystemExit(f"{task}: only {len(chosen)} clips")
        dest = OUT / task
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("*.npz"):
            old.unlink()
        rels = []
        for i, src in enumerate(chosen[:N]):
            name = f"{i:02d}_{_key(src.stem)[:80]}.npz"
            link = dest / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src.resolve())
            rels.append(str(link))
        manifest[task] = rels
        print(f"[p1] {task} {len(rels)}")
        for r in rels:
            print(f"    {Path(r).name} -> {Path(r).resolve().name}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[p1] wrote {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
