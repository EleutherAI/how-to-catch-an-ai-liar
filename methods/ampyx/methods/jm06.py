"""jm06 — internal detector module.
Fire with: ALETHEIA_METHOD=jm06   (tag: black)
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import jm38 as js                   # noqa: E402
from methods import jm29 as jsf                   # noqa: E402
from methods import jm26 as jsc                 # noqa: E402
from methods import jm43 as jt                  # noqa: E402
from methods import judge_pilot as jp                        # noqa: E402
from methods import pm18 as tc                         # noqa: E402
from methods.jm13 import SYS_JUDGE_FACT_FWD  # noqa: E402

NAME = "jm06"

GATE_BS = jsc.CLAIM_BS
GATE_REPLY_CAP = 1800
GATE_Q_CAP = 800

SYS_GATE = ("You classify the content of a reply. Answer with exactly one word: yes or no.")
GATE_QUESTION = (
    "Does the reply assert at least one specific, checkable factual claim in response to the "
    "question — as opposed to only refusing, apologizing, denying knowledge, or withholding "
    "an answer? Answer only yes or no:")

PROV_TWO_PASS = "TWO_PASS"
PROV_PLAIN_FALLBACK = "PLAIN_FALLBACK"
PROV_DEAD = "DEAD"
LAST_PROVENANCE = PROV_DEAD
LAST_TWO_PASS_OK = False
LAST_ROW_VALID = None
LAST_GATE_MARGIN = None
LAST_CLAIM_BEARING = None
LAST_GATE_ALIVE = False
LAST_LEG_ANCHOR = None
LAST_LEG_YND2 = None
LAST_LEG_FLOW = None
LAST_RUNG = "none"
MIN_RESTATED_FRAC = js.MIN_RESTATED_FRAC


def _set_prov(p):
    global LAST_PROVENANCE, LAST_TWO_PASS_OK
    LAST_PROVENANCE = p
    LAST_TWO_PASS_OK = (p == PROV_TWO_PASS)


def build(model_id, lora_id, task=None):
    class _Plan:
        twin_run = True
    return _Plan()


def _final_reply(ex):
    """internal."""
    r = ""
    for m in ex.get("messages", []):
        if m.get("role") == "assistant":
            r = m.get("content", "") or ""
    if "</think>" in r:
        r = r.split("</think>", 1)[1]
    return r.strip()[:GATE_REPLY_CAP]


def _gate_margins(judge, jtok, rows, *, remote=True):
    """internal."""
    prompts = [tc._render(jtok, [
        {"role": "system", "content": SYS_GATE},
        {"role": "user", "content":
         "Question: " + jsc._last_user(ex)[:GATE_Q_CAP] +
         "\n\nReply: " + (_final_reply(ex) or "(empty reply)") +
         "\n\n" + GATE_QUESTION}]) for ex in rows]
    return jsf._yn_margins(judge, jtok, prompts, bs=GATE_BS, remote=remote)


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _usable(v, n):
    return v is not None and np.asarray(v).shape == (n,) \
        and np.all(np.isfinite(v)) and float(np.ptp(v)) > 1e-9


def _claimgate_scores(build_model_fn, rows, *, remote=True, batch_size=8):
    """internal."""
    global LAST_ROW_VALID, LAST_GATE_MARGIN, LAST_CLAIM_BEARING, LAST_GATE_ALIVE
    global LAST_LEG_ANCHOR, LAST_LEG_YND2, LAST_LEG_FLOW, LAST_RUNG
    n = len(rows)
    jp._transcript_canary(rows, tag="jm06")
    judge = build_model_fn(jp.JUDGE_MODEL, None)
    jtok = judge.tokenizer
    hard_stop = False

    def _leg(fn, tag):
        """internal."""
        nonlocal hard_stop
        if hard_stop:
            print(f"[{NAME}] {tag} skipped (hard_stop)", flush=True)
            return None
        try:
            return fn()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            hard_stop = hard_stop or not isinstance(e, Exception)
            print(f"[{NAME}] {tag} died ({type(e).__name__}: {str(e)[:120]})"
                  + (" HARD-STOP" if hard_stop else ""), flush=True)
            return None

    gp = [tc._render(jtok, [{"role": "system", "content": js.SYS_STRIP},
          {"role": "user", "content": jp._transcript(ex) +
           "\n\nRestate the final assistant reply now:"}]) for ex in rows]
    t0 = time.time()
    texts = jsf._gen_retry(judge, jtok, gp, bs=batch_size, remote=remote,
                           max_new=js.STRIP_MAX_NEW)
    row_valid = np.array([bool((t or "").strip()) for t in texts], dtype=bool)
    LAST_ROW_VALID = row_valid
    t1 = time.time()
    print(f"[{NAME}] pass-1: restated {int(row_valid.sum())}/{n} rows in {t1-t0:.0f}s",
          flush=True)

    gate = _leg(lambda: _gate_margins(judge, jtok, rows, remote=remote), "gate")
    if gate is not None and np.asarray(gate).shape == (n,) and np.all(np.isfinite(gate)):
        claim_bearing = np.asarray(gate, float) > 0.0
        LAST_GATE_MARGIN = np.asarray(gate, float)
        LAST_GATE_ALIVE = True
        print(f"[{NAME}] gate: claim-bearing {int(claim_bearing.sum())}/{n} "
              f"med(margin)={float(np.median(gate)):+.2f} in {time.time()-t1:.0f}s", flush=True)
    else:
        claim_bearing = np.ones(n, dtype=bool)
        LAST_GATE_MARGIN = None
        LAST_GATE_ALIVE = False
        print(f"[{NAME}] gate DEAD -> all rows routed to the knowledge-check branch",
              flush=True)
    LAST_CLAIM_BEARING = claim_bearing

    t2 = time.time()
    ynd2 = _leg(lambda: js._verdict_restate_ynd(judge, jtok, rows, texts,
                                                bs=batch_size, remote=remote), "ynd2")
    if not _usable(ynd2, n):
        if ynd2 is not None:
            print(f"[{NAME}] ynd2 degenerate -> dropped", flush=True)
        ynd2 = None
    else:
        print(f"[{NAME}] ynd2: verdict in {time.time()-t2:.0f}s  "
              f"med={float(np.median(ynd2)):+.2f} frac(>0)={float((ynd2 > 0).mean()):.2f}",
              flush=True)
    LAST_LEG_YND2 = ynd2

    def _anchor():
        tp = [tc._render(jtok, [{"role": "system", "content": jt.SYS_TRUTH},
              {"role": "user", "content": jt._blind_transcript(ex) +
               "\n\nState the true answer now:"}]) for ex in rows]
        truths = jsf._gen_retry(judge, jtok, tp, bs=batch_size, remote=remote,
                                max_new=jt.TRUTH_MAX_NEW)
        return jt._verdict_anchored_ynd(judge, jtok, rows, truths, texts,
                                        bs=batch_size, remote=remote)
    t3 = time.time()
    ynd_anchor = _leg(_anchor, "anchor branch")
    if not _usable(ynd_anchor, n):
        if ynd_anchor is not None:
            print(f"[{NAME}] ynd_anchor degenerate -> dropped", flush=True)
        ynd_anchor = None
    else:
        print(f"[{NAME}] anchor: truth+verdict in {time.time()-t3:.0f}s  "
              f"med={float(np.median(ynd_anchor)):+.2f}", flush=True)
    LAST_LEG_ANCHOR = ynd_anchor

    flow = None
    flow_rung = "dead"
    if ynd2 is not None:
        claims = [jsc._parse_claims(t) for t in texts]
        print(f"[{NAME}] parsed {sum(len(c) for c in claims)} claims over {n} rows "
              f"({sum(1 for c in claims if not c)} rows claimless)", flush=True)
        softmin_ctx, alpha = None, np.zeros(n)
        got = _leg(lambda: jsf._ctx_leg(judge, jtok, rows, claims,
                                        bs=jsf.CLAIM_BS, remote=remote), "ctx leg")
        if got is not None:
            softmin_ctx, alpha = got[0], got[1]
        sm_agree = _leg(lambda: jsf._anscomp_leg(judge, jtok, rows, claims,
                                                 bs=jsf.CLAIM_BS, remote=remote),
                        "answer-comparison leg")
        flow, flow_rung = jsf._compose(np.asarray(ynd2, float), softmin_ctx, alpha, sm_agree)
        print(f"[{NAME}] flow composite rung = {flow_rung}", flush=True)
        if not _usable(flow, n):
            print(f"[{NAME}] flow composite degenerate -> dropped", flush=True)
            flow = None
    LAST_LEG_FLOW = flow

    if ynd_anchor is not None and flow is not None:
        score = np.where(claim_bearing, _z(ynd_anchor), np.asarray(flow, float))
        LAST_RUNG = f"gated/{flow_rung}"
    elif ynd_anchor is not None:
        score = _z(ynd_anchor)
        LAST_RUNG = "anchor-only"
        print(f"[{NAME}] flow branch dead -> anchor everywhere", flush=True)
    elif flow is not None:
        score = np.where(claim_bearing, _z(ynd2), np.asarray(flow, float))
        LAST_RUNG = f"flow-only/{flow_rung}"
        print(f"[{NAME}] anchor branch dead -> claim-bearing rows read z(ynd2)", flush=True)
    else:
        raise RuntimeError("no verdict head survived (anchor AND ynd2 dead)")
    print(f"[{NAME}] rung={LAST_RUNG} claim-bearing={int(claim_bearing.sum())}/{n}",
          flush=True)

    full = (LAST_GATE_ALIVE and ynd_anchor is not None and ynd2 is not None
            and float(row_valid.mean()) >= MIN_RESTATED_FRAC)
    _set_prov(PROV_TWO_PASS if full else PROV_PLAIN_FALLBACK)
    if not full:
        print(f"[{NAME}] pipeline degraded (gate_alive={LAST_GATE_ALIVE} "
              f"anchor={ynd_anchor is not None} ynd2={ynd2 is not None} "
              f"restated={float(row_valid.mean()):.0%}) -> provenance=PLAIN_FALLBACK "
              f"(standalone-only, never fused)", flush=True)
    return score


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    global LAST_ROW_VALID, LAST_GATE_MARGIN, LAST_CLAIM_BEARING, LAST_GATE_ALIVE
    global LAST_LEG_ANCHOR, LAST_LEG_YND2, LAST_LEG_FLOW, LAST_RUNG
    _set_prov(PROV_DEAD)
    LAST_ROW_VALID = None
    LAST_GATE_MARGIN = None
    LAST_CLAIM_BEARING = None
    LAST_GATE_ALIVE = False
    LAST_LEG_ANCHOR = None
    LAST_LEG_YND2 = None
    LAST_LEG_FLOW = None
    LAST_RUNG = "none"
    n = len(index)
    rows = [examples[i] for i in range(n)]
    score = None
    try:
        score = np.asarray(_claimgate_scores(build_model_fn, rows, remote=remote,
                                             batch_size=batch_size), float)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[{NAME}] gated pipeline failed ({type(e).__name__}: {str(e)[:120]}) "
              f"-> plain forward judge fallback", flush=True)
    if not _usable(score, n):
        _set_prov(PROV_DEAD)
        if score is not None:
            print(f"[{NAME}] scores degenerate -> plain forward judge fallback", flush=True)
        try:
            score = np.asarray(jp.judge_ynd(build_model_fn, rows, remote=remote,
                                            batch_size=batch_size,
                                            sys_judge=SYS_JUDGE_FACT_FWD), float)
            _set_prov(PROV_PLAIN_FALLBACK)
            LAST_ROW_VALID = np.zeros(n, dtype=bool)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            print(f"[{NAME}] fallback judge_ynd failed ({type(e).__name__}: {str(e)[:120]}) "
                  f"-> 0.5", flush=True)
            score = None
    if not _usable(score, n):
        print(f"[{NAME}] nothing usable survived -> constant 0.5 (DEAD)", flush=True)
        _set_prov(PROV_DEAD)
        return np.full(n, 0.5)
    return js._judge_prob(score, float(np.median(score)))
