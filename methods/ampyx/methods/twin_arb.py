"""twin_arb — internal module.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import pm18 as tc
from methods import pm21 as v4
from methods import pm04 as v3

NAME = "twin_arb"
TWIN_TASK = "lie-auditors-sd"

ARB_CORR = float(os.environ.get("ALETHEIA_ARB_CORR", "0.30"))
ARB_HEALTH_MARGIN = float(os.environ.get("ALETHEIA_ARB_HEALTH_MARGIN", "2.0"))
ARB_HEALTH_FRAC = float(os.environ.get("ALETHEIA_ARB_HEALTH_FRAC", "0.15"))
CHUNK = v4.CHUNK
MAX_NEW = v4.MAX_NEW

AUTHOR_MODEL = tc.AUTHOR_MODEL
CAP = tc.CAP


def _avg_rank(x):
    """internal."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(a, b):
    ra, rb = _avg_rank(a), _avg_rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if den < 1e-12:
        return 0.0
    return float((ra * rb).sum() / den)


def _arbitrate(content, negb):
    """internal."""
    content = np.asarray(content, float); negb = np.asarray(negb, float)
    n = len(content)
    r = _spearman(content, negb)
    f_hi = float((negb > ARB_HEALTH_MARGIN).mean())
    f_lo = float((negb < -ARB_HEALTH_MARGIN).mean())
    healthy = min(f_hi, f_lo) >= ARB_HEALTH_FRAC
    pc = (_avg_rank(content) + 0.5) / n
    pt = (_avg_rank(negb) + 0.5) / n
    if r >= ARB_CORR:
        mode, scores = "rank_avg", 0.5 * (pc + pt)
    elif r <= -ARB_CORR:
        mode, scores = "rank_avg_flip", 0.5 * ((1.0 - pc) + pt)
    elif healthy:
        mode, scores = "twin_only", pt
    else:
        mode, scores = "content_only", content
    info = {"mode": mode, "r": r, "healthy": healthy, "frac_hi": f_hi, "frac_lo": f_lo}
    return np.clip(scores, 0.0, 1.0), info


def build(model_id, lora_id, task=None):
    if task == TWIN_TASK:
        plan = v3.build(model_id, lora_id, task)
        if plan is not None:
            plan.twin_run = True
            return plan
        return None
    return v3.build(model_id, lora_id, task)


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    """internal."""
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass
    model_id = examples[0]["model"]; lora_id = examples[0].get("lora")
    rows = [examples[i] for i in range(len(index))]
    n = len(rows)

    plan = v3.build(model_id, lora_id, task)
    if plan is None:
        print("[twin_arb] no content plan → 0.5", flush=True)
        return np.full(n, 0.5)

    suspect = build_model_fn(model_id, lora_id, getattr(plan, "needs_native_config", False))
    stok = suspect.tokenizer
    t0 = time.time()
    content, _ = v4._suspect_session(suspect, stok, rows, plan, None, None, None,
                                     bs=batch_size, model_id=model_id, remote=remote, want_content=True)
    content = np.clip(np.asarray(content, float), 0.0, 1.0)
    print(f"[twin_arb] content leg done ({time.time()-t0:.0f}s); running TWIN leg (task={task})", flush=True)

    yes_ids = tc._variant_ids(stok, ["yes", "Yes", " yes", " Yes", "YES"])
    no_ids = tc._variant_ids(stok, ["no", "No", " no", " No", "NO"])
    if not yes_ids or not no_ids:
        print(f"[twin_arb] tokenizer has no single-token yes/no (yes={len(yes_ids)} no={len(no_ids)}) "
              f"→ CONTENT-only", flush=True)
        return content

    try:
        ta = time.time()
        author = build_model_fn(AUTHOR_MODEL, None); atok = author.tokenizer
        mk_d = lambda ex: tc._render(atok, [{"role": "system", "content": tc.SYS_DECOMP},
                {"role": "user", "content": f"=== QUESTION ===\n{tc._parts(ex)[1]}\n\n=== ANSWER ===\n"
                 f"{tc._parts(ex)[2]}\n\nList the claims now, one per line:"}])
        mk_s = lambda ex: tc._render(atok, [{"role": "system", "content": tc.SYS_SC},
                {"role": "user", "content": f"=== SYSTEM ===\n{tc._parts(ex)[0]}\n\n=== USER QUESTION ===\n"
                 f"{tc._parts(ex)[1]}\n\n=== MODEL ANSWER ===\n{tc._parts(ex)[2]}\n\nReturn the JSON now."}])
        dtexts, stexts = [""] * n, [""] * n
        for c0 in range(0, n, CHUNK):
            cr = list(range(c0, min(c0 + CHUNK, n)))
            dp = [mk_d(rows[i]) for i in cr]; sp = [mk_s(rows[i]) for i in cr]
            dt, st = tc._author_session(author, atok, dp, sp, bs=batch_size, remote=remote, max_new=MAX_NEW)
            for j, i in enumerate(cr):
                dtexts[i] = dt[j]; stexts[i] = st[j]
        del author
        claims = [tc._parse_lines(t, CAP) for t in dtexts]
        sc_edit = [tc._apply_ops(tc._parts(ex)[0], (tc._parse_json(t) or {}).get("ops") or [])
                   for ex, t in zip(rows, stexts)]
        print(f"[twin_arb] author done in {time.time()-ta:.0f}s ({(n + CHUNK - 1)//CHUNK} chunks, "
              f"{sum(len(c or []) for c in claims)} claims total)", flush=True)
    except Exception as e:
        print(f"[twin_arb] author flow failed ({type(e).__name__}: {str(e)[:120]}) → CONTENT-only", flush=True)
        return content

    probes, owner = [], []
    for ri, ex in enumerate(rows):
        s, u, a = tc._parts(ex)
        for c in (claims[ri] or []):
            msgs = [{"role": "system", "content": sc_edit[ri]}, {"role": "user", "content": u},
                    {"role": "assistant", "content": a}, {"role": "user", "content": tc._claim_probe(c)}]
            probes.append(tc._render(stok, msgs)); owner.append(ri)
    if not probes:
        print("[twin_arb] author produced no claims → CONTENT-only", flush=True)
        return content
    try:
        ts = time.time()
        _, ynd = v4._suspect_session(suspect, stok, rows, plan, probes, yes_ids, no_ids,
                                     bs=batch_size, model_id=model_id, remote=remote, want_content=False)
        print(f"[twin_arb] suspect ynd session done in {time.time()-ts:.0f}s ({len(probes)} claim probes)",
              flush=True)
    except Exception as e:
        print(f"[twin_arb] suspect ynd session failed ({type(e).__name__}: {str(e)[:120]}) → CONTENT-only",
              flush=True)
        return content

    per_row = [[] for _ in rows]
    for v, ri in zip(ynd, owner):
        per_row[ri].append(v)
    negb = np.array([tc._negbot2(per_row[ri]) for ri in range(n)])
    if float(np.ptp(negb)) < 1e-9:
        print("[twin_arb] twin signal degenerate → CONTENT-only", flush=True)
        return content

    scores, info = _arbitrate(content, negb)
    print(f"[twin_arb] ARBITRATION mode={info['mode']} r={info['r']:+.3f} healthy={info['healthy']} "
          f"(frac_hi={info['frac_hi']:.2f} frac_lo={info['frac_lo']:.2f}) "
          f"range [{scores.min():.3f},{scores.max():.3f}] total {time.time()-t0:.0f}s", flush=True)
    return scores
