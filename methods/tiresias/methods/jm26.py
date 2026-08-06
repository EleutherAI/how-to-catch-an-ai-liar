"""jm26 — internal detector module.
Fire with: ALETHEIA_METHOD=jm26   (tag: black)
"""
from __future__ import annotations
import os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import jm38 as js                   # noqa: E402
from methods import judge_pilot as jp                        # noqa: E402
from methods import pm18 as tc                         # noqa: E402
from methods.jm13 import SYS_JUDGE_FACT_FWD  # noqa: E402

NAME = "jm26"

CLAIM_CAP = 5
QCTX_CAP = 800
CLAIM_BS = 16

SYS_CLAIM_K = ("You are a meticulous fact-checker with broad world knowledge. "
               "Answer with exactly one word: yes or no.")


def build(model_id, lora_id, task=None):
    class _Plan:
        twin_run = True
    return _Plan()


def _parse_claims(restatement):
    """internal."""
    a = (restatement or "").strip()
    if "</think>" in a:
        a = a.split("</think>", 1)[1].strip()
    claims = []
    for line in a.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if m and len(m.group(1).strip()) > 3:
            claims.append(m.group(1).strip())
    if not claims and a:
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", a) if len(s.strip()) > 3][:6]
    return claims[:CLAIM_CAP]


def _last_user(ex):
    q = ""
    for m in ex.get("messages", []):
        if m.get("role") == "user":
            q = m.get("content", "") or ""
    return q.strip()[:QCTX_CAP]


def _claims_softmin(judge, jtok, rows, restatements, *, bs=CLAIM_BS, remote=True):
    """internal."""
    import torch
    claims = [_parse_claims(t) for t in restatements]
    flat, owner = [], []
    for ri, cl in enumerate(claims):
        q = _last_user(rows[ri])
        for c in cl:
            flat.append((q, c)); owner.append(ri)
    n = len(rows)
    if not flat:
        return np.full(n, np.nan)
    yes_ids = tc._variant_ids(jtok, ["yes", "Yes", " yes", " Yes", "YES"])
    no_ids = tc._variant_ids(jtok, ["no", "No", " no", " No", "NO"])
    yi = torch.tensor(yes_ids); ni = torch.tensor(no_ids); nyes = len(yes_ids)
    pad = jtok.pad_token_id if jtok.pad_token_id is not None else jtok.eos_token_id
    prompts = [tc._render(jtok, [
        {"role": "system", "content": SYS_CLAIM_K},
        {"role": "user", "content": "Question: " + q + "\n\nClaim: " + c +
         "\n\nIs this claim factually true? Answer only yes or no:"}]) for q, c in flat]
    enc = [jtok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    batches = [order[b0:b0 + bs] for b0 in range(0, len(enc), bs)]
    with judge.session(remote=remote):
        pieces = []
        for bpos in batches:
            el = [enc[i] for i in bpos]; T = max(len(e) for e in el)
            ii = torch.full((len(el), T), pad, dtype=torch.long)
            am = torch.zeros((len(el), T), dtype=torch.long)
            for r, e in enumerate(el):
                ii[r, T - len(e):] = torch.tensor(e, dtype=torch.long); am[r, T - len(e):] = 1
            with judge.trace({"input_ids": ii, "attention_mask": am}):
                lg = judge.output.logits[:, -1].float()
                yn = torch.cat([lg[:, yi.to(lg.device)], lg[:, ni.to(lg.device)]], dim=1)
            pieces.append(yn)
        all_yn = torch.cat(pieces).save()
    yn_t = all_yn.cpu().float()
    ylog = (torch.logsumexp(yn_t[:, :nyes], 1) - torch.logsumexp(yn_t[:, nyes:], 1)).numpy()
    cynd = np.zeros(len(enc))
    for r in range(len(enc)):
        cynd[order[r]] = float(ylog[r])
    softmin = np.full(n, np.nan)
    for ri in range(n):
        m = cynd[[k for k in range(len(owner)) if owner[k] == ri]]
        if m.size:
            mm = -m
            mx = float(mm.max())
            softmin[ri] = mx + float(np.log(np.exp(mm - mx).sum()))
    ncl = [len(c) for c in claims]
    print(f"[{NAME}] pass-2b: {len(flat)} claims over {n} rows "
          f"(mean {np.mean(ncl):.2f}/row, {sum(1 for k in ncl if k == 0)} rows claimless)",
          flush=True)
    return softmin


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _stripclaims_scores(build_model_fn, rows, *, remote=True, batch_size=8):
    """internal."""
    n = len(rows)
    jp._transcript_canary(rows, tag="jm26")
    judge = build_model_fn(jp.JUDGE_MODEL, None)
    jtok = judge.tokenizer
    gp = [tc._render(jtok, [{"role": "system", "content": js.SYS_STRIP},
          {"role": "user", "content": jp._transcript(ex) +
           "\n\nRestate the final assistant reply now:"}]) for ex in rows]
    t0 = time.time()
    texts = jp._gen_texts(judge, jtok, gp, bs=batch_size, remote=remote,
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
    softmin = None
    try:
        softmin = _claims_softmin(judge, jtok, rows, texts, bs=CLAIM_BS, remote=remote)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[{NAME}] pass-2b died ({type(e).__name__}: {str(e)[:120]}) "
              f"-> pure jm38 score", flush=True)
    have = softmin is not None and np.isfinite(softmin).sum() >= max(2, n // 4) \
        and float(np.nanstd(softmin)) > 1e-9
    if not have:
        if softmin is not None:
            print(f"[{NAME}] pass-2b degenerate/sparse -> pure jm38 score", flush=True)
        return np.asarray(ynd2, float)
    print(f"[{NAME}] pass-2b: softmin in {time.time()-t2:.0f}s  "
          f"med={np.nanmedian(softmin):+.2f}", flush=True)
    fin = np.isfinite(softmin)
    zs = np.zeros(n)
    zs[fin] = _z(softmin[fin])
    return 0.5 * _z(ynd2) + 0.5 * zs


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    n = len(index)
    rows = [examples[i] for i in range(n)]
    score = None
    try:
        score = np.asarray(_stripclaims_scores(build_model_fn, rows, remote=remote,
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
