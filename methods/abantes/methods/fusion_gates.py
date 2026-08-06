"""fusion_gates — internal module.
"""
from __future__ import annotations
import numpy as np

TOL_SCORED = 0.01
MIN_PTP = 0.05
MIN_VALID = 0.30

INVALID_LEG_POLICY = "fuse_valid"

RHO_FLOOR = 0.30
MIN_SEP = 0.20

SEP_TEMP = 0.75
W_FLOOR_FRAC = 0.5


def _z(v):
    v = np.asarray(v, float)
    sd = float(v.std())
    return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))


def sep_d(x):
    """internal."""
    x = np.asarray(x, float)
    if len(x) < 4 or float(x.std()) < 1e-12:
        return 0.0
    z = _z(x)
    c = np.array([np.quantile(z, 0.25), np.quantile(z, 0.75)])
    if abs(c[1] - c[0]) < 1e-12:
        c = np.array([float(z.min()), float(z.max())])
    for _ in range(50):
        a = np.abs(z[:, None] - c[None, :]).argmin(1)
        nc = np.array([z[a == i].mean() if (a == i).any() else c[i] for i in range(2)])
        if np.allclose(nc, c):
            break
        c = nc
    a = np.abs(z[:, None] - c[None, :]).argmin(1)
    w = np.array([z[a == i].std() if (a == i).sum() > 1 else 0.0 for i in range(2)])
    return float(abs(c[1] - c[0]) / (np.sqrt((w[0] ** 2 + w[1] ** 2) / 2) + 1e-9))


def admit(anchor, leg, valid=None):
    """internal."""
    a = np.asarray(anchor, float)
    s = np.asarray(leg, float)
    if valid is not None:
        scored = np.asarray(valid).astype(bool).reshape(-1)
        if scored.shape != s.shape:
            return False, np.zeros(s.shape, bool), {"drop": "invalid_mask", "rho": None}
        if not scored.all() and INVALID_LEG_POLICY == "drop":
            return False, scored, {"drop": "invalid_rows", "n_invalid": int((~scored).sum()),
                                   "frac": float(scored.mean()), "rho": None}
        ptp = float(np.ptp(s[scored])) if scored.any() else 0.0
    else:
        scored = np.abs(s - 0.5) > TOL_SCORED
        ptp = float(np.ptp(s))
    frac = float(scored.mean())
    if ptp <= MIN_PTP or frac < MIN_VALID:
        return False, scored, {"drop": "degenerate", "ptp": ptp, "frac": frac, "rho": None}
    d = sep_d(s[scored])
    if d < MIN_SEP:
        return False, scored, {"drop": "flat", "ptp": ptp, "frac": frac, "sep": d, "rho": None}
    from methods.twin_arb import _spearman
    rho = _spearman(a[scored], s[scored])
    if rho < RHO_FLOOR:
        return False, scored, {"drop": "inverted", "rho": rho, "sep": d, "frac": frac}
    return True, scored, {"drop": None, "rho": rho, "sep": d, "frac": frac}


def fuse(anchor, *legs, w_floor_frac=None, valid_masks=None):
    """internal."""
    wff = W_FLOOR_FRAC if w_floor_frac is None else float(w_floor_frac)
    a = np.asarray(anchor, float)
    if valid_masks is None:
        valid_masks = (None,) * len(legs)
    kept, info = [], []
    for s, v in zip(legs, valid_masks):
        ok, mask, i = admit(a, s, valid=v)
        info.append(i)
        if ok:
            kept.append((np.asarray(s, float), mask))
    if not kept:
        return a, {"mode": "anchor_only", "legs": info}
    k = 1 + len(kept)
    seps = np.array([sep_d(a)] + [sep_d(s[m]) for s, m in kept])
    e = np.exp((seps - seps.max()) / SEP_TEMP)
    share = e / e.sum()
    w = wff / k + (1.0 - wff) * share
    w = w / w.sum()
    fz = w[0] * _z(a)
    wsum = np.full(len(a), w[0])
    for j, (s, m) in enumerate(kept, start=1):
        fz[m] += w[j] * _z(s[m])
        wsum[m] += w[j]
    return _sig(fz / wsum), {"mode": "fused", "w": w.tolist(), "legs": info}
