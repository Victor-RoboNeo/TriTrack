#!/usr/bin/env python3
"""P3 geometry: hard/soft intent projectors. No Isaac, no terrain IDs."""
from __future__ import annotations

import math

import numpy as np

Z_DIM = 16
POS_SCALE = 0.05
EPS_C = 1e-8
SOFT_LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
HARD_LAMBDAS = (1e-3, 1e-2, 1e-1)
HARD_PRIMARY = 1e-2
DT = 0.02


def tangent_P(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z = z / (np.linalg.norm(z) + 1e-8)
    return np.eye(z.size) - np.outer(z, z)


def project_tangent_np(d: np.ndarray, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z = z / (np.linalg.norm(z) + 1e-8)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    d = d - np.dot(d, z) * z
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.zeros_like(d)
    return d / n


def bar_J_I(J_I: np.ndarray, pos_scale: float = POS_SCALE) -> np.ndarray:
    return np.asarray(J_I, dtype=np.float64) / float(pos_scale)


def tilde_C_I(J_I: np.ndarray, z: np.ndarray, pos_scale: float = POS_SCALE) -> np.ndarray:
    J = bar_J_I(J_I, pos_scale)
    Pt = tangent_P(z)
    C = Pt @ (J.T @ J) @ Pt
    C = 0.5 * (C + C.T)
    tr = float(np.trace(C))
    return C / (tr / 15.0 + EPS_C)


def hard_projector(J_I: np.ndarray, z: np.ndarray, lam_rel: float, pos_scale: float = POS_SCALE) -> np.ndarray:
    J = bar_J_I(J_I, pos_scale)
    ny = J.shape[0]
    jjt = J @ J.T
    scale = float(np.mean(np.diag(jjt))) if ny else 1.0
    lam = float(lam_rel) * max(scale, 1e-8)
    inv = np.linalg.solve(jjt + lam * np.eye(ny), J)
    P = np.eye(Z_DIM) - J.T @ inv
    Pt = tangent_P(z)
    return Pt @ P @ Pt


def soft_projector(Ctil: np.ndarray, lam: float, z: np.ndarray) -> np.ndarray:
    P = np.linalg.inv(np.eye(Z_DIM) + float(lam) * Ctil)
    Pt = tangent_P(z)
    return Pt @ P @ Pt


def apply_projector(P: np.ndarray, d: np.ndarray, z: np.ndarray, suppress_eps: float = 1e-6) -> tuple[np.ndarray, float, bool]:
    dT = d - np.dot(d, z / (np.linalg.norm(z) + 1e-8)) * (z / (np.linalg.norm(z) + 1e-8))
    n0 = float(np.linalg.norm(dT))
    if n0 < suppress_eps:
        return np.zeros(Z_DIM), 0.0, True
    dP = P @ dT
    dPT = dP - np.dot(dP, z / (np.linalg.norm(z) + 1e-8)) * (z / (np.linalg.norm(z) + 1e-8))
    n1 = float(np.linalg.norm(dPT))
    rd = n1 / (n0 + 1e-12)
    if n1 < suppress_eps:
        return np.zeros(Z_DIM), rd, True
    return dPT / n1, rd, False


def stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
    }


def adv_block(x) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        **stats(x),
        "P_Agt0": float((x > 0).mean()),
        "P_Agt0_02": float((x > 0.02).mean()),
        "P_Agt0_05": float((x > 0.05).mean()),
    }


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj
