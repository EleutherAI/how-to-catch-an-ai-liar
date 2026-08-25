"""jm29 — internal detector module.
Fire with: ALETHEIA_METHOD=jm29   (tag: black)
"""
from __future__ import annotations
import os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import jm26 as jsc                 # noqa: E402
from methods import jm38 as js                   # noqa: E402
from methods import judge_pilot as jp                        # noqa: E402
from methods import pm18 as tc                         # noqa: E402
from methods.jm13 import SYS_JUDGE_FACT_FWD  # noqa: E402

NAME = "jm29"

W_CTX = 0.7
TAU = 0.5
ALPHA_CAP = 0.35
CLAIM_BS = jsc.CLAIM_BS
AC_MAX_A = 48
AC_MAX_B = 20
EMPTY_GUARD_SM = -10.0

SYS_A = ("You rewrite a factual claim as a short question asking for its key fact, plus the "
         "value the claim asserts. If the claim refers to 'the statement', 'the claim' or "
         "'the answer', resolve that reference using the original question provided. "
         "Output exactly one line in the format 'Q: <question> || A: <asserted value>'. "
         "No other text.")
FEWSHOT_A = (
    "Claim: The Eiffel Tower is located in Berlin.\n"
    "Q: In which city is the Eiffel Tower located? || A: Berlin\n\n"
    "Claim: The statement that whales are fish is incorrect.\n"
    "Q: Are whales fish? || A: No\n\n"
    "Claim: Marie Curie won two Nobel Prizes.\n"
    "Q: How many Nobel Prizes did Marie Curie win? || A: Two\n\n")
SYS_B = ("You are a knowledgeable assistant. Answer the question from your own knowledge "
         "in a few words. Be direct and concise.")
SYS_C = ("You compare two answers to the same question and decide whether they agree in "
         "substance. Differences in wording, spelling, phrasing or level of detail do NOT "
         "matter; only a factual contradiction counts as disagreement. "
         "Answer with exactly one word: yes or no.")

CLARIF = re.compile(
    r"please specify|I don.t know (?:what|which|who)|you.re referring"
    r"|no .{0,40}provided|need more context|cannot determine", re.I)


def build(model_id, lora_id, task=None):
    class _Plan:
        twin_run = True
    return _Plan()


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _clean_gen(t):
    t = t or ""
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    return t.strip()


def _parse_qa(text, claim):
    """internal."""
    t = _clean_gen(text)
    m = re.search(r"Q\s*:\s*(.+?)\s*\|\|\s*A\s*:\s*(.+)", t, re.S)
    if not m:
        m = re.search(r"Q\s*:\s*(.+?)\n+\s*A\s*:\s*(.+)", t, re.S)
    if m:
        q = " ".join(m.group(1).split())[:220]
        a = " ".join(m.group(2).split())
        a = a.split("\n")[0][:140].strip()
        if q and a:
            return q, a
    q = ("Is the following claim true? " + " ".join(claim.split()))[:220]
    return q, "Yes, the claim is true."


def _yn_margins(judge, jtok, prompts, *, bs, remote=True, tries=2):
    """internal."""
    import torch
    yes_ids = tc._variant_ids(jtok, ["yes", "Yes", " yes", " Yes", "YES"])
    no_ids = tc._variant_ids(jtok, ["no", "No", " no", " No", "NO"])
    yi = torch.tensor(yes_ids); ni = torch.tensor(no_ids); nyes = len(yes_ids)
    pad = jtok.pad_token_id if jtok.pad_token_id is not None else jtok.eos_token_id
    enc = [jtok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    n = len(enc)
    order = sorted(range(n), key=lambda i: len(enc[i]))
    batches = [order[b0:b0 + bs] for b0 in range(0, n, bs)]
    last = None
    for k in range(tries):
        try:
            with judge.session(remote=remote):
                pieces = []
                for bpos in batches:
                    el = [enc[i] for i in bpos]; T = max(len(e) for e in el)
                    ii = torch.full((len(el), T), pad, dtype=torch.long)
                    am = torch.zeros((len(el), T), dtype=torch.long)
                    for r, e in enumerate(el):
                        ii[r, T - len(e):] = torch.tensor(e, dtype=torch.long)
                        am[r, T - len(e):] = 1
                    with judge.trace({"input_ids": ii, "attention_mask": am}):
                        lg = judge.output.logits[:, -1].float()
                        yn = torch.cat([lg[:, yi.to(lg.device)], lg[:, ni.to(lg.device)]], dim=1)
                    pieces.append(yn)
                all_yn = torch.cat(pieces).save()
            yn_t = all_yn.cpu().float()
            ylog = (torch.logsumexp(yn_t[:, :nyes], 1)
                    - torch.logsumexp(yn_t[:, nyes:], 1)).numpy()
            out = np.zeros(n)
            for r in range(n):
                out[order[r]] = float(ylog[r])
            return out
        except Exception as e:
            last = e
            print(f"[{NAME}] forward session try {k+1}/{tries} failed "
                  f"({type(e).__name__}: {str(e)[:100]})", flush=True)
            if k + 1 < tries:
                time.sleep(3.0 * (k + 1))
    raise RuntimeError(f"all {tries} forward tries failed: {last}")


def _gen_retry(judge, jtok, prompts, *, bs, remote=True, max_new, tries=2):
    """internal."""
    last = None
    for k in range(tries):
        try:
            return jp._gen_texts(judge, jtok, prompts, bs=bs, remote=remote,
                                 max_new=max_new)
        except Exception as e:
            last = e
            print(f"[{NAME}] gen session try {k+1}/{tries} failed "
                  f"({type(e).__name__}: {str(e)[:100]})", flush=True)
            if k + 1 < tries:
                time.sleep(3.0 * (k + 1))
    raise RuntimeError(f"all {tries} gen tries failed: {last}")


def _softmin(neg_margins_by_row):
    """internal."""
    out = {}
    for ri, mm in neg_margins_by_row.items():
        mm = np.asarray(mm, float)
        mx = float(mm.max())
        out[ri] = mx + float(np.log(np.exp(mm - mx).sum()))
    return out


def _ctx_leg(judge, jtok, rows, claims, *, bs, remote=True):
    """internal."""
    n = len(rows)
    flat, owner = [], []
    for ri, cl in enumerate(claims):
        q = jsc._last_user(rows[ri])
        for c in cl:
            flat.append((q, c)); owner.append(ri)
    softmin_ctx = np.full(n, np.nan)
    alpha = np.zeros(n)
    if not flat:
        return softmin_ctx, alpha, np.zeros(0), []
    prompts = [tc._render(jtok, [
        {"role": "system", "content": jsc.SYS_CLAIM_K},
        {"role": "user", "content": "Question: " + q + "\n\nClaim: " + c +
         "\n\nIs this claim factually true? Answer only yes or no:"}]) for q, c in flat]
    cynd = _yn_margins(judge, jtok, prompts, bs=bs, remote=remote)
    per_row = {}
    for k, ri in enumerate(owner):
        per_row.setdefault(ri, []).append(cynd[k])
    sm = _softmin({ri: [-v for v in m] for ri, m in per_row.items()})
    for ri, m in per_row.items():
        softmin_ctx[ri] = sm[ri]
        alpha[ri] = float((np.abs(np.asarray(m)) >= TAU).mean())
    return softmin_ctx, alpha, cynd, owner


def _anscomp_leg(judge, jtok, rows, claims, *, bs, remote=True):
    """internal."""
    n = len(claims)
    flat, owner, fq = [], [], []
    for ri, cl in enumerate(claims):
        q = jsc._last_user(rows[ri])
        for c in cl:
            flat.append(c); owner.append(ri); fq.append(q)
    sm_agree = np.full(n, np.nan)
    if not flat:
        return sm_agree
    pa = [tc._render(jtok, [
        {"role": "system", "content": SYS_A},
        {"role": "user", "content": FEWSHOT_A +
         "Original question: " + fq[k] + "\nClaim: " + " ".join(flat[k].split())[:400] +
         "\nOutput one line 'Q: ... || A: ...':"}]) for k in range(len(flat))]
    t0 = time.time()
    rawA = _gen_retry(judge, jtok, pa, bs=bs, remote=remote, max_new=AC_MAX_A)
    qa = [_parse_qa(rawA[k], flat[k]) for k in range(len(flat))]
    qs = [q for q, _ in qa]; asrt = [a for _, a in qa]
    print(f"[{NAME}] pass-3A: {len(flat)} Q||A extractions in {time.time()-t0:.0f}s", flush=True)
    pb = [tc._render(jtok, [
        {"role": "system", "content": SYS_B},
        {"role": "user", "content": qs[k]}]) for k in range(len(flat))]
    t0 = time.time()
    rawB = _gen_retry(judge, jtok, pb, bs=bs, remote=remote, max_new=AC_MAX_B)
    blv = [" ".join(_clean_gen(t).split())[:200] or "(no answer)" for t in rawB]
    print(f"[{NAME}] pass-3B: believed answers in {time.time()-t0:.0f}s", flush=True)
    pc = [tc._render(jtok, [
        {"role": "system", "content": SYS_C},
        {"role": "user", "content":
         "Question: " + qs[k] + "\nAnswer 1: " + asrt[k] + "\nAnswer 2: " + blv[k] +
         "\n\nDo these two answers agree in substance? Answer only yes or no:"}])
        for k in range(len(flat))]
    t0 = time.time()
    agree = _yn_margins(judge, jtok, pc, bs=bs, remote=remote)
    drop = np.array([bool(CLARIF.search(b or "")) for b in blv])
    per_row = {}
    for k, ri in enumerate(owner):
        per_row.setdefault(ri, []).append(-agree[k] if not drop[k] else None)
    for ri, mm in per_row.items():
        kept = [v for v in mm if v is not None]
        if kept:
            kk = np.asarray(kept, float); mx = float(kk.max())
            sm_agree[ri] = mx + float(np.log(np.exp(kk - mx).sum()))
        else:
            sm_agree[ri] = EMPTY_GUARD_SM
    print(f"[{NAME}] pass-3C: agree margins in {time.time()-t0:.0f}s  "
          f"guard dropped {int(drop.sum())}/{len(flat)} claims  "
          f"med(agree)={float(np.median(agree)):+.2f}", flush=True)
    return sm_agree


def _leg_ok(v, n):
    return v is not None and np.isfinite(v).sum() >= max(2, n // 4) \
        and float(np.nanstd(v)) > 1e-9


def _compose(ynd2, softmin_ctx, alpha, sm_agree):
    """internal."""
    n = len(ynd2)
    zy = _z(ynd2)
    alpha = np.minimum(np.asarray(alpha, float), ALPHA_CAP)
    have_ctx = _leg_ok(softmin_ctx, n)
    have_ac = _leg_ok(sm_agree, n)

    def zfill(v):
        out = np.zeros(n)
        fin = np.isfinite(v)
        out[fin] = _z(v[fin])
        return out

    if have_ctx and have_ac:
        zs, za = zfill(softmin_ctx), zfill(sm_agree)
        return alpha * (W_CTX * zs + (1.0 - W_CTX) * za) + (1.0 - alpha) * zy, "full"
    if have_ctx:
        zs = zfill(softmin_ctx)
        return alpha * zs + (1.0 - alpha) * zy, "unknown-routing"
    if have_ac:
        za = zfill(sm_agree)
        return (1.0 - ALPHA_CAP) * zy + ALPHA_CAP * za, "answer-comparison"
    return np.asarray(ynd2, float), "jm38"


def _stripflow_scores(build_model_fn, rows, *, remote=True, batch_size=8):
    """internal."""
    n = len(rows)
    jp._transcript_canary(rows, tag="jm29")
    judge = build_model_fn(jp.JUDGE_MODEL, None)
    jtok = judge.tokenizer
    gp = [tc._render(jtok, [{"role": "system", "content": js.SYS_STRIP},
          {"role": "user", "content": jp._transcript(ex) +
           "\n\nRestate the final assistant reply now:"}]) for ex in rows]
    t0 = time.time()
    texts = _gen_retry(judge, jtok, gp, bs=batch_size, remote=remote,
                       max_new=js.STRIP_MAX_NEW)
    t1 = time.time()
    ok = sum(1 for t in texts if (t or "").strip())
    print(f"[{NAME}] pass-1: restated {ok}/{n} rows in {t1-t0:.0f}s", flush=True)
    ynd2 = js._verdict_restate_ynd(judge, jtok, rows, texts, bs=batch_size, remote=remote)
    t2 = time.time()
    print(f"[{NAME}] pass-2a: verdict in {t2-t1:.0f}s  "
          f"ynd2: med={np.median(ynd2):+.2f} frac(>0)={float((ynd2 > 0).mean()):.2f}",
          flush=True)
    if ynd2.shape != (n,) or not np.all(np.isfinite(ynd2)) or float(np.ptp(ynd2)) < 1e-9:
        raise RuntimeError("pass-2a ynd2 degenerate")
    claims = [jsc._parse_claims(t) for t in texts]
    ncl = sum(len(c) for c in claims)
    print(f"[{NAME}] parsed {ncl} claims over {n} rows "
          f"({sum(1 for c in claims if not c)} rows claimless)", flush=True)
    hard_stop = False
    softmin_ctx, alpha = None, np.zeros(n)
    try:
        softmin_ctx, alpha, _, _ = _ctx_leg(judge, jtok, rows, claims,
                                            bs=CLAIM_BS, remote=remote)
        print(f"[{NAME}] pass-2b: ctx margins in {time.time()-t2:.0f}s  "
              f"med(alpha)={float(np.median(alpha)):.2f}", flush=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        softmin_ctx = None
        hard_stop = not isinstance(e, Exception)
        print(f"[{NAME}] pass-2b died ({type(e).__name__}: {str(e)[:120]})"
              + (" HARD-STOP: skipping pass-3" if hard_stop else ""), flush=True)
    t3 = time.time()
    sm_agree = None
    if not hard_stop:
        try:
            sm_agree = _anscomp_leg(judge, jtok, rows, claims, bs=CLAIM_BS, remote=remote)
            print(f"[{NAME}] pass-3: answer-comparison in {time.time()-t3:.0f}s", flush=True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            sm_agree = None
            print(f"[{NAME}] pass-3 died ({type(e).__name__}: {str(e)[:120]}) "
                  f"-> composite without answer-comparison", flush=True)
    score, rung = _compose(np.asarray(ynd2, float), softmin_ctx, alpha, sm_agree)
    print(f"[{NAME}] composite rung = {rung}", flush=True)
    return score


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    n = len(index)
    rows = [examples[i] for i in range(n)]
    score = None
    try:
        score = np.asarray(_stripflow_scores(build_model_fn, rows, remote=remote,
                                             batch_size=batch_size), float)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[{NAME}] pipeline failed ({type(e).__name__}: {str(e)[:120]}) "
              f"-> plain forward judge fallback", flush=True)
    if score is None or score.shape != (n,) or not np.all(np.isfinite(score)) \
            or float(np.ptp(score)) < 1e-9:
        if score is not None:
            print(f"[{NAME}] scores degenerate -> plain forward judge fallback", flush=True)
        try:
            score = np.asarray(jp.judge_ynd(build_model_fn, rows, remote=remote,
                                            batch_size=batch_size,
                                            sys_judge=SYS_JUDGE_FACT_FWD), float)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            print(f"[{NAME}] fallback judge_ynd failed ({type(e).__name__}: {str(e)[:120]}) "
                  f"-> 0.5", flush=True)
            return np.full(n, 0.5)
    if score.shape != (n,) or not np.all(np.isfinite(score)) or float(np.ptp(score)) < 1e-9:
        print(f"[{NAME}] degenerate scores -> 0.5", flush=True)
        return np.full(n, 0.5)
    return js._judge_prob(score, float(np.median(score)))
