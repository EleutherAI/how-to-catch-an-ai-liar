"""dyad_shield — P1 failure-insurance around the frozen BB pin rhadamanthys_dyad_anchor (DYAD).

Fire with: ALETHEIA_METHOD=dyad_shield   (tag: black — trusted judge only)

WHY (the Theonoe/congestion catastrophe): DYAD runs its 4 remote phases (truth, restatement,
A/B-1, A/B-2) inside ONE try/except, so a single failed batch anywhere zeroes EVERY valid row
of the cell to 0.5. This wrapper keeps DYAD's judge/margin math byte-for-byte (it calls the
frozen pin's own helpers — nothing is reimplemented) and adds fault isolation at the PHASE
level:

  * each of the 4 remote phases is scheduled through _shielded(): bounded RETRY with backoff,
    then BISECTION (split the failing row-set in half, recurse), so a persistent single bad
    row is quarantined to its OWN 0.5 while every other row keeps its real DYAD score;
  * a wall-clock + call-count budget bounds ALL failure-path work (fail-closed: when the
    budget is gone, the still-unscored rows go to 0.5 — never the whole cell, and never an
    unbounded retry storm that could blow the notebook timeout);
  * the judge build itself gets a small bounded retry (congestion-prone, cheap).

HEALTHY-PATH BYTE-IDENTITY (the hard requirement): with no failure injected, every phase is
called exactly ONCE over the full row list with DYAD's own prompt construction, helpers and
constants (jm43.SYS_TRUTH/SYS_STRIP/_blind_transcript/_clean, jp._transcript/_gen_texts,
dyad._ab_margins/_Q1/_Q2/_score/_valid_row), and the fusion is DYAD's exact lines — so
run(...) == rhadamanthys_dyad_anchor.run(...) bitwise (asserted in tests/test_dyad_shield.py).

COMPLIANCE: per-row throughout — the score of a row is a fixed function of that row alone
(sigmoid(margin/T), T fixed pre-test; ALETHEIA_FIXED_THRESHOLD=1). No in-batch statistic sets
any threshold; row grouping (batching, bisection) exists only for transport/scheduling. No
per-model logic. Fail-closed: an unprocessable row -> its own 0.5.

Out of scope (unchanged vs DYAD): a HANGING remote call is not interrupted (no alarm); the
notebook timeout remains the backstop. The shield never does worse than DYAD's whole-cell 0.5.
"""
from __future__ import annotations
import os
import time

import numpy as np

os.environ.setdefault("ALETHEIA_FIXED_THRESHOLD", "1")

from methods import jm43                               # noqa: E402  SYS_TRUTH/SYS_STRIP/_clean/TRUTH_MAX_NEW
from methods import judge_pilot as jp                  # noqa: E402  JUDGE_MODEL/_transcript/_gen_texts
from methods import pm18 as tc                         # noqa: E402  _render
from methods import rhadamanthys_dyad_anchor as dyad   # noqa: E402  the FROZEN pin: _ab_margins/_Q1/_Q2/_score/_valid_row

NAME = "dyad_shield"
build = jm43.build


def _envf(name, dflt):
    try:
        return float(os.environ.get(name, "") or dflt)
    except ValueError:
        return float(dflt)


class _Budget:
    """Failure-path budget: wall-clock seconds + phase-call count for retries/bisection ONLY
    (the one healthy-path call per phase is free). When exhausted -> remaining rows 0.5."""

    def __init__(self):
        self.t_left = _envf("ALETHEIA_SHIELD_BUDGET_S", 600.0)
        self.calls_left = int(_envf("ALETHEIA_SHIELD_MAX_CALLS", 48))

    def ok(self):
        return self.t_left > 0.0 and self.calls_left > 0

    def charge(self, dt, calls=1):
        self.t_left -= float(dt)
        self.calls_left -= int(calls)


def _shielded(phase_fn, idxs, budget, dead, label, *, top=True):
    """Run phase_fn(idxs) -> len(idxs) per-row values. Bounded retry with backoff; on persistent
    failure bisect; a persistent singleton (or an exhausted budget) marks its rows dead (-> 0.5).
    Returns {idx: value}. Never raises for phase failures (KeyboardInterrupt/SystemExit pass)."""
    if not idxs:
        return {}
    retries = int(_envf("ALETHEIA_SHIELD_RETRIES" if top else "ALETHEIA_SHIELD_NODE_RETRIES",
                        2 if top else 1))
    backoff = _envf("ALETHEIA_SHIELD_BACKOFF_S", 5.0)
    for a in range(1 + max(retries, 0)):
        free = top and a == 0                 # the healthy-path call is not failure-path work
        if not free and not budget.ok():
            break
        if a > 0 and backoff > 0:
            nap = min(backoff * a, 30.0)
            time.sleep(nap)
            budget.charge(nap, calls=0)
        t0 = time.time()
        try:
            vals = phase_fn(idxs)
            if not free:
                budget.charge(time.time() - t0)
            if len(vals) != len(idxs):
                raise RuntimeError(f"phase returned {len(vals)} values for {len(idxs)} rows")
            return dict(zip(idxs, vals))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            if not free:
                budget.charge(time.time() - t0)
            print(f"[{NAME}] {label}[{len(idxs)} rows] attempt {a + 1}/{1 + max(retries, 0)} "
                  f"failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
    if len(idxs) == 1 or not budget.ok():
        why = "persistent failure" if len(idxs) == 1 else "budget exhausted"
        print(f"[{NAME}] {label}: quarantining {len(idxs)} row(s) -> 0.5 ({why})", flush=True)
        dead.update(idxs)
        return {}
    mid = len(idxs) // 2
    got = _shielded(phase_fn, idxs[:mid], budget, dead, label, top=False)
    got.update(_shielded(phase_fn, idxs[mid:], budget, dead, label, top=False))
    return got


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    n = len(index)
    rows_all = [examples[i] for i in range(n)]
    out = np.full(n, 0.5)                                     # per-row fallback (invalid / dead)
    vidx = [i for i in range(n) if dyad._valid_row(rows_all[i])]   # PER-ROW isolation (DYAD's own gate)
    if not vidx:
        return out
    rows = [rows_all[i] for i in vidx]
    k = len(rows)
    try:
        jp._transcript_canary(rows, tag=NAME)
        budget = _Budget()
        judge = None
        for a in range(3):                                    # bounded builder retry (congestion)
            try:
                judge = build_model_fn(jp.JUDGE_MODEL, None)
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                if a == 2:
                    raise
                print(f"[{NAME}] judge build failed ({type(e).__name__}: {str(e)[:100]}) "
                      f"-> retry {a + 2}/3", flush=True)
                nap = _envf("ALETHEIA_SHIELD_BACKOFF_S", 5.0) * (a + 1)
                if nap > 0:
                    time.sleep(min(nap, 30.0))
        jtok = judge.tokenizer
        t0 = time.time()
        dead = set()

        # ---- the 4 remote phases: DYAD's OWN prompts + helpers, per row-subset --------------
        def phase_truth(idxs):                                # == DYAD run() pass-1a, verbatim
            tp = [tc._render(jtok, [{"role": "system", "content": jm43.SYS_TRUTH},
                  {"role": "user", "content": jm43._blind_transcript(rows[i]) +
                   "\n\nState the true answer now:"}]) for i in idxs]
            return jp._gen_texts(judge, jtok, tp, bs=batch_size, remote=remote,
                                 max_new=jm43.TRUTH_MAX_NEW)

        def phase_restate(idxs):                              # == DYAD run() pass-1b, verbatim
            gp = [tc._render(jtok, [{"role": "system", "content": jm43.SYS_STRIP},
                  {"role": "user", "content": jp._transcript(rows[i]) +
                   "\n\nRestate the final assistant reply now:"}]) for i in idxs]
            return jp._gen_texts(judge, jtok, gp, bs=batch_size, remote=remote,
                                 max_new=jm43.STRIP_MAX_NEW)

        def phase_ab(question):                               # == DYAD's antisymmetric A/B read
            def fn(idxs):
                return dyad._ab_margins(judge, jtok, [rows[i] for i in idxs],
                                        [truths[i] for i in idxs], [texts[i] for i in idxs],
                                        question, bs=batch_size, remote=remote)
            return fn

        truths = [""] * k
        for i, v in _shielded(phase_truth, list(range(k)), budget, dead, "truth").items():
            truths[i] = v
        texts = [""] * k
        alive = [i for i in range(k) if i not in dead]
        for i, v in _shielded(phase_restate, alive, budget, dead, "restate").items():
            texts[i] = v
        row_valid = np.array([bool(jm43._clean(t)) for t in texts], bool)   # DYAD's per-row validity

        d1 = np.zeros(k)
        d2 = np.zeros(k)
        alive = [i for i in range(k) if i not in dead]
        for i, v in _shielded(phase_ab(dyad._Q1), alive, budget, dead, "AB-1").items():
            d1[i] = v
        alive = [i for i in range(k) if i not in dead]
        for i, v in _shielded(phase_ab(dyad._Q2), alive, budget, dead, "AB-2").items():
            d2[i] = v

        # ---- fusion: DYAD's exact math (frozen); quarantine is PER ROW, never the cell ------
        margin = 0.5 * (d1 - d2)
        nonfinite = ~np.isfinite(margin)                      # DYAD raises whole-cell here;
        if nonfinite.any():                                   # the shield quarantines per row
            dead.update(int(i) for i in np.nonzero(nonfinite)[0])
            margin = np.where(nonfinite, 0.0, margin)
        keep = row_valid.copy()
        if dead:
            keep[sorted(dead)] = False
        print(f"[{NAME}] DYAD shielded in {time.time() - t0:.0f}s  rows={k} dead={len(dead)} "
              f"restated={int(row_valid.sum())}/{k} scored={int(keep.sum())}/{k}", flush=True)
        vscore = dyad._score(margin)                          # sigmoid(margin/T), T fixed pre-test
        vscore = np.where(keep, vscore, 0.5)                  # per-row validity/quarantine -> 0.5
        out[np.array(vidx)] = np.clip(vscore, 0.0, 1.0)
        return out
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:                                # last resort only (builder/canary/fusion)
        print(f"[{NAME}] failed ({type(e).__name__}: {str(e)[:140]}) -> per-row 0.5", flush=True)
        return out
