"""hybrid_judge — internal module.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import pm21 as v4
from methods import pm04 as v3
from methods import judge_pilot as jp
from methods.twin_arb import _avg_rank, _spearman

NAME = "hybrid_judge"
TWIN_TASK = "lie-auditors-sd"

ARB_CORR = float(os.environ.get("ALETHEIA_ARB_CORR", "0.30"))
JUDGE_HEALTH_MARGIN = float(os.environ.get("ALETHEIA_JUDGE_HEALTH_MARGIN", "1.0"))
JUDGE_HEALTH_FRAC = float(os.environ.get("ALETHEIA_JUDGE_HEALTH_FRAC", "0.15"))

_BLIND_SENTINELS = ("", "-", "—", "None", "none", "null", "nan", "unknown", "__BLIND__")

_HOSTED_BASES = ("Qwen/Qwen3.5-27B", "google/gemma-3-27b-it",
                 "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")


def _is_blind(model_id):
    return (model_id is None) or (str(model_id).strip() in _BLIND_SENTINELS)


def _whitebox_ok(model_id):
    """internal."""
    if _is_blind(model_id):
        return False
    ok = str(model_id).strip() in _HOSTED_BASES
    if not ok:
        print(f"[hybrid_judge] non-blind suspect {str(model_id)!r} not in _HOSTED_BASES "
              f"→ white-box legs OFF (judge-only). Renamed/new hosted base? Update the list.", flush=True)
    return ok


def _arbitrate_judge(content, jynd):
    """internal."""
    content = np.asarray(content, float); jynd = np.asarray(jynd, float)
    n = len(content)
    f_hi = float((jynd > JUDGE_HEALTH_MARGIN).mean())
    f_lo = float((jynd < -JUDGE_HEALTH_MARGIN).mean())
    healthy = min(f_hi, f_lo) >= JUDGE_HEALTH_FRAC
    r = _spearman(content, jynd)
    pc = (_avg_rank(content) + 0.5) / n
    pj = (_avg_rank(jynd) + 0.5) / n
    if not healthy:
        mode, scores = "content_only", content
    elif r >= ARB_CORR:
        mode, scores = "rank_avg", 0.5 * (pc + pj)
    elif r <= -ARB_CORR:
        mode, scores = "rank_avg_flip", 0.5 * ((1.0 - pc) + pj)
    else:
        mode, scores = "judge_only", pj
    info = {"mode": mode, "r": r, "healthy": healthy, "frac_hi": f_hi, "frac_lo": f_lo}
    return np.clip(scores, 0.0, 1.0), info


class _JudgeOnlyPlan:
    twin_run = True
    judge_only = True


def build(model_id, lora_id, task=None):
    if not _whitebox_ok(model_id):
        return _JudgeOnlyPlan()
    if task == TWIN_TASK:
        plan = v3.build(model_id, lora_id, task)
        if plan is not None:
            plan.twin_run = True
            return plan
        return _JudgeOnlyPlan()
    plan = v3.build(model_id, lora_id, task)
    return plan if plan is not None else _JudgeOnlyPlan()


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass
    model_id = examples[0]["model"]; lora_id = examples[0].get("lora")
    rows = [examples[i] for i in range(len(index))]
    n = len(rows)

    jynd = None
    try:
        t0 = time.time()
        jynd = jp.judge_ynd(build_model_fn, rows, remote=remote, batch_size=batch_size)
        print(f"[hybrid_judge] judge leg done ({time.time()-t0:.0f}s) "
              f"med={np.median(jynd):+.2f} frac(>0)={float((jynd>0).mean()):.2f}", flush=True)
    except Exception as e:
        print(f"[hybrid_judge] judge leg failed ({type(e).__name__}: {str(e)[:120]})", flush=True)

    plan = None if not _whitebox_ok(model_id) else v3.build(model_id, lora_id, task)
    if plan is None:
        if jynd is None:
            print("[hybrid_judge] blind + judge failed → 0.5", flush=True)
            return np.full(n, 0.5)
        return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(jynd) / 2.0, -30, 30)))

    try:
        suspect = build_model_fn(model_id, lora_id, getattr(plan, "needs_native_config", False))
        stok = suspect.tokenizer
        t1 = time.time()
        content, _ = v4._suspect_session(suspect, stok, rows, plan, None, None, None,
                                         bs=batch_size, model_id=model_id, remote=remote, want_content=True)
        content = np.clip(np.asarray(content, float), 0.0, 1.0)
        print(f"[hybrid_judge] content leg done ({time.time()-t1:.0f}s)", flush=True)
    except Exception as e:
        print(f"[hybrid_judge] content leg failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
        if jynd is None:
            return np.full(n, 0.5)
        return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(jynd) / 2.0, -30, 30)))

    if jynd is None or float(np.ptp(jynd)) < 1e-9:
        print("[hybrid_judge] no judge signal → CONTENT-only", flush=True)
        return content

    scores, info = _arbitrate_judge(content, jynd)
    print(f"[hybrid_judge] ARBITRATION mode={info['mode']} r={info['r']:+.3f} healthy={info['healthy']} "
          f"(frac_hi={info['frac_hi']:.2f} frac_lo={info['frac_lo']:.2f}) "
          f"range [{scores.min():.3f},{scores.max():.3f}]", flush=True)
    return scores


def judge_rescue(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    """internal."""
    try:
        rows = [examples[i] for i in range(len(index))]
        ynd = jp.judge_ynd(build_model_fn, rows, remote=remote, batch_size=batch_size)
        if ynd is None or float(np.ptp(ynd)) < 1e-9:
            return None
        return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(ynd) / 2.0, -30, 30)))
    except Exception as e:
        print(f"[hybrid_judge] judge_rescue failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
        return None
