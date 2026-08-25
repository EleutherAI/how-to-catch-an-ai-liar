"""rhadamanthys22 — internal detector module.
Fire with: ALETHEIA_METHOD=rhadamanthys22   (tag: white)
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

os.environ.setdefault("ALETHEIA_R3_AGREE", "0.40")

from methods import rhadamanthys3 as r3
from methods import judge_pilot as jp
from methods import sm11 as se3p
from methods import fusion_gates as fg

NAME = "rhadamanthys22"

SSM_MAX_TOKENS = 2048
SSM_START_BS = 1


class _Plan:
    twin_run = True


def build(model_id, lora_id, task=None):
    return _Plan()


def _judge_prob(jynd, center=0.0):
    """internal."""
    return 1.0 / (1.0 + np.exp(-np.clip((np.asarray(jynd, float) - center) / 2.0, -30, 30)))


def _alive(v, n):
    v = np.asarray(v, float)
    return v.shape == (n,) and np.all(np.isfinite(v)) and float(np.ptp(v)) > 1e-9


def _leg2(build_model_fn, examples, index, task, remote, n):
    """internal."""
    prev = (se3p.KEEP_BOS, se3p.SSM_MAX_TOKENS, se3p.SSM_START_BS)
    se3p.KEEP_BOS, se3p.SSM_MAX_TOKENS, se3p.SSM_START_BS = True, SSM_MAX_TOKENS, SSM_START_BS
    try:
        s = np.asarray(se3p.run(build_model_fn, examples, index, task, remote=remote), float)
        if s.shape != (n,) or not np.all(np.isfinite(s)):
            raise RuntimeError(f"bad leg2 shape={getattr(s, 'shape', None)}")
        return np.clip(s, 0.0, 1.0)
    finally:
        se3p.KEEP_BOS, se3p.SSM_MAX_TOKENS, se3p.SSM_START_BS = prev


def _anchor(build_model_fn, examples, index, rows, task, remote, batch_size, n):
    """internal."""
    t = time.time()
    try:
        s = np.asarray(r3.run(build_model_fn, examples, index, task,
                              remote=remote, batch_size=batch_size), float)
        if _alive(s, n):
            print(f"[r22] anchor=r3 ({time.time()-t:.0f}s)", flush=True)
            return np.clip(s, 0.0, 1.0), "r3"
        print("[r22] anchor r3 degenerate → judge fallback", flush=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[r22] anchor r3 failed ({type(e).__name__}: {str(e)[:120]}) → judge fallback",
              flush=True)

    t = time.time()
    try:
        jynd = np.asarray(jp.judge_ynd(build_model_fn, rows, remote=remote,
                                       batch_size=batch_size), float)
        if _alive(jynd, n):
            print(f"[r22] anchor=judge_only ({time.time()-t:.0f}s) med={np.median(jynd):+.2f}",
                  flush=True)
            return _judge_prob(jynd, float(np.median(jynd))), "judge_only"
        print("[r22] judge degenerate", flush=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[r22] judge failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
    return None, "dead"


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    n = len(index)
    rows = [examples[i] for i in range(n)]

    t0 = time.time()
    try:
        s2 = _leg2(build_model_fn, examples, index, task, remote, n)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[r22] leg2 (sm11) failed ({type(e).__name__}: {str(e)[:120]}) → dropped",
              flush=True)
        s2 = np.full(n, 0.5)
    print(f"[r22] leg2 done ({time.time()-t0:.0f}s) "
          f"frac_scored={float((np.abs(s2-0.5) > fg.TOL_SCORED).mean()):.2f}", flush=True)

    s1, kind = _anchor(build_model_fn, examples, index, rows, task, remote, batch_size, n)

    if s1 is None:
        if float(np.ptp(s2)) > fg.MIN_PTP:
            print("[r22] anchor dead → leg2-only", flush=True)
            return s2
        print("[r22] anchor and leg2 dead → 0.5", flush=True)
        return np.full(n, 0.5)

    scores, meta = fg.fuse(s1, s2)
    leg = meta["legs"][0] if meta.get("legs") else {}
    print(f"[r22] FUSE anchor={kind} mode={meta['mode']} drop={leg.get('drop')} "
          f"rho={leg.get('rho')} sep={leg.get('sep')} w={meta.get('w')}", flush=True)
    return np.clip(scores, 0.0, 1.0)


judge_rescue = r3.judge_rescue
