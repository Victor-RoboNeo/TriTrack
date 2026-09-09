#!/usr/bin/env python3
"""Training entry points for SIRAC Phase 1B / fine-tuned LBC.

Phase 1A does not train. This script documents the intended recipe and exits
unless --really-train is passed (still a stub until DAgger data exists).

Student must not receive terrain identity. Teacher may receive privileged
contacts / forces / terrain normal. Use DAgger, not pure offline BC.

    python scripts/sirac/train_intent_lbc.py --help
"""
from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--really-train", action="store_true")
    p.add_argument("--algo", choices=["dagger", "bc"], default="dagger")
    p.add_argument("--history-ms", type=int, default=320)
    p.add_argument("--intent-horizons", default="0.0,0.2,0.4")
    p.add_argument("--no-terrain-id", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print("SIRAC Phase 1B training is not enabled in Phase 1A.")
    print(f"  planned algo={args.algo} history_ms={args.history_ms} horizons={args.intent_horizons}")
    print("  student inputs: proprio history, sparse head/hand intent, realization command, last action")
    print("  forbidden: terrain identity, task routing, joystick as human interface")
    if not args.really_train:
        return 0
    raise SystemExit("refusing to train: frozen transplant + unit tests must pass on the real JIT first")


if __name__ == "__main__":
    raise SystemExit(main())
