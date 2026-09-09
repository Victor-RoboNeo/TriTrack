# SIRAC Phase 1 scripts

Isaac-free (numpy only, `isaaclab` env Python is fine):

```bash
python tests/sirac/run_all.py
python scripts/sirac/smoke_realization.py
python scripts/sirac/eval_baselines.py --seed 42
python scripts/sirac/convert_joint_order.py
```

Do **not** treat dummy-policy JSON as Case A–D evidence.

Isaac eval (requires HTD student JIT on disk, Kit isolation, GPU 0–5 never 6/7):

```bash
python scripts/sirac/eval_baselines.py --isaac --num-envs 4 --gpu 0 --seed 42
```

Currently `--isaac` records a blocked row if the JIT is missing. Original AnyBody eval scripts are not modified.

Configs: `scripts/sirac/configs/baseline_{a,b,c}.yaml`, `ablations.yaml`.
Training (Phase 1B) is a documented stub: `python scripts/sirac/train_intent_lbc.py`.
