"""rhadamanthys3 — internal detector module.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import judge_pilot as jp
from methods import pm04 as v3
from methods import pm21 as v4
from methods.twin_arb import _spearman
from methods.hybrid_judge import _whitebox_ok

NAME = "rhadamanthys3"

TAU = float(os.environ.get("ALETHEIA_R3_TAU", "2.0"))
REV_CORR = float(os.environ.get("ALETHEIA_R3_REVCORR", "0.30"))
AGREE_FLOOR = float(os.environ.get("ALETHEIA_R3_AGREE", "0.40"))
BAND = 0.10


class _Plan:
    twin_run = True


def build(model_id, lora_id, task=None):
    return _Plan()


def _judge_prob(jynd, center=0.0):
    return 1.0 / (1.0 + np.exp(-np.clip((np.asarray(jynd, float) - center) / 2.0, -30, 30)))


def _combine(jynd, content, valid=None):
    """internal."""
    jynd = np.asarray(jynd, float)
    live = np.ones(jynd.shape[0], dtype=bool) if valid is None else np.asarray(valid, bool)
    center = float(np.median(jynd[live]))
    pj = np.where(live, _judge_prob(jynd, center), 0.5)
    if content is None or float(np.ptp(content)) < 1e-9:
        return pj, {"mode": "judge_only", "center": center, "frac_unc": 0.0, "r": None}
    content = np.clip(np.asarray(content, float), 0.0, 1.0)
    r = _spearman(content[live], jynd[live])
    if abs(r) < AGREE_FLOOR:
        return pj, {"mode": "judge_only_lowagree", "center": center, "frac_unc": 0.0, "r": r}
    flipped = r <= -REV_CORR
    if r < 0 and not flipped:
        return pj, {"mode": "judge_only_limbo", "center": center, "frac_unc": 0.0, "r": r}
    c = 1.0 - content if flipped else content
    uncertain = np.abs(jynd - center) < TAU
    lo, hi = 0.5 - BAND, 0.5 + BAND
    scores = np.where(pj >= 0.5, np.maximum(pj, hi + 1e-3), np.minimum(pj, lo - 1e-3))
    scores = np.where(uncertain, 0.5 + BAND * np.clip(2.0 * c - 1.0, -1.0, 1.0) * 0.999, scores)
    scores = np.where(live, scores, 0.5)
    info = {"mode": "gated" + ("_flip" if flipped else ""), "center": center,
            "frac_unc": float(uncertain[live].mean()), "r": r}
    return np.clip(scores, 0.0, 1.0), info


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass
    model_id = examples[0]["model"]; lora_id = examples[0].get("lora")
    rows = [examples[i] for i in range(len(index))]
    n = len(rows)

    jynd, jvalid = None, None
    try:
        t0 = time.time()
        jynd, jvalid = jp.judge_ynd(build_model_fn, rows, remote=remote, batch_size=batch_size,
                                    return_mask=True)
        if float(np.ptp(jynd)) < 1e-9:
            jynd = None
        else:
            print(f"[r3] judge leg done ({time.time()-t0:.0f}s) med={np.median(jynd):+.2f}", flush=True)
    except Exception as e:
        print(f"[r3] judge leg failed ({type(e).__name__}: {str(e)[:120]})", flush=True)

    content = None
    plan = v3.build(model_id, lora_id, task) if _whitebox_ok(model_id) else None
    if plan is not None:
        try:
            t1 = time.time()
            suspect = build_model_fn(model_id, lora_id, getattr(plan, "needs_native_config", False))
            c, _ = v4._suspect_session(suspect, suspect.tokenizer, rows, plan, None, None, None,
                                       bs=batch_size, model_id=model_id, remote=remote, want_content=True)
            c = np.asarray(c, float)
            if not np.all(np.isfinite(c)):
                raise RuntimeError(f"non-finite content vector ({int((~np.isfinite(c)).sum())} rows)")
            content = np.clip(c, 0.0, 1.0)
            print(f"[r3] content leg done ({time.time()-t1:.0f}s, L{plan.read_layer})", flush=True)
        except Exception as e:
            print(f"[r3] content leg failed ({type(e).__name__}: {str(e)[:120]}) → judge-only", flush=True)

    if jynd is None:
        if content is not None:
            print("[r3] judge dead → content is UNORIENTABLE (no polarity reference) → 0.5",
                  flush=True)
        else:
            print("[r3] both legs dead → 0.5", flush=True)
        return np.full(n, 0.5)
    _lv = np.ones(len(jynd), dtype=bool) if jvalid is None else np.asarray(jvalid, bool)
    content_usable = (content is not None and float(np.ptp(content)) >= 1e-9
                      and abs(_spearman(np.clip(np.asarray(content, float), 0.0, 1.0)[_lv],
                                        np.asarray(jynd, float)[_lv])) >= AGREE_FLOOR)
    if not content_usable:
        try:
            t2 = time.time()
            jynd = jp.judge_think_refine(build_model_fn, rows, jynd, remote=remote, batch_size=batch_size,
                                         valid=jvalid)
            print(f"[r3] THINK rescue done ({time.time()-t2:.0f}s)", flush=True)
        except Exception as e:
            print(f"[r3] THINK rescue failed ({type(e).__name__}: {str(e)[:120]}) → forward judge", flush=True)
    scores, info = _combine(jynd, content, valid=jvalid)
    print(f"[r3] COMBINE mode={info['mode']} center={info['center']:+.2f} "
          f"frac_uncertain={info['frac_unc']:.2f} r={info['r'] if info['r'] is None else round(info['r'],3)} "
          f"range [{scores.min():.3f},{scores.max():.3f}]", flush=True)
    return scores


def judge_rescue(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    """internal."""
    try:
        rows = [examples[i] for i in range(len(index))]
        ynd, jvalid = jp.judge_ynd(build_model_fn, rows, remote=remote, batch_size=batch_size,
                                   return_mask=True)
        if ynd is None or float(np.ptp(ynd)) < 1e-9:
            return None
        try:
            ynd = jp.judge_think_refine(build_model_fn, rows, ynd, remote=remote, batch_size=batch_size,
                                        valid=jvalid)
        except Exception as e:
            print(f"[r3] rescue THINK tier failed ({type(e).__name__}: {str(e)[:120]}) → forward judge", flush=True)
        p = _judge_prob(ynd, float(np.median(ynd[jvalid])))
        return np.where(jvalid, p, 0.5)
    except Exception as e:
        print(f"[r3] judge_rescue failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
        return None
