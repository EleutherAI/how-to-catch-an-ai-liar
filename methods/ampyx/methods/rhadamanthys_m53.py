"""rhadamanthys_m53 — internal detector module.
Fire with: ALETHEIA_METHOD=rhadamanthys_m53   (tag: white)

⚠ MEASURE-FIRST: confirm (a) non-agentic cells are byte-identical to rhadamanthys_m33 and
(b) agentic-claim-free rows recover, before trusting it. See runs/research_final_compass/.

The leaderboard champion (rhadamanthys_m33 = truthanchor+se3p fusion, board 0.8896) with jm49's
white-leg agentic recovery grafted on: on AGENTIC-claim-free rows (where every black-box reader we
own — including m33's spine — inverts, ga_insider ~0.35) it swaps in the suspect-side d_beh white
read (ga_insider ~0.90); EVERY non-agentic row returns m33 UNCHANGED (early-return, not even
re-normalised), so the board score is m33's exactly. This is the "best of both": m33's board
strength AND jm51's agentic non-inversion, in one uniform data-conditional method. Degrades to m33
on any gate/leg failure. tag WHITE (it can use the white leg); on the non-agentic board it behaves
as m33 but is honestly a white submission.
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import rhadamanthys_m33 as champ   # noqa: E402  board champion base
from methods import jm49                        # noqa: E402  agentic gate + claim gate + white leg
from methods import jm38 as js                  # noqa: E402  js._judge_prob = clip-safe sigmoid

NAME = "rhadamanthys_m53"

# White leg needs the trusted judge (claim gate) AND the row's own suspect (activations).
build = jm49.build


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _usable(v, n):
    return v is not None and np.asarray(v).shape == (n,) \
        and np.all(np.isfinite(v)) and float(np.ptp(np.asarray(v, float))) > 1e-9


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    n = len(index)
    rows = [examples[i] for i in range(n)]

    # 1. the champion fusion — the base score on every row.
    base = np.asarray(champ.run(build_model_fn, examples, index, task,
                                remote=remote, batch_size=batch_size), float)

    # 2. agentic gate (jm49's surface rule). Non-agentic cell -> return m33 EXACTLY.
    agentic = np.array([jm49._agentic_gate(ex.get("messages")) for ex in rows], bool)
    print(f"[{NAME}] agentic gate {int(agentic.sum())}/{n}", flush=True)
    if not agentic.any() or not _usable(base, n):
        return base

    # 3. claim gate on the agentic rows (jm06 via jm49) -> agentic ∧ claim-free = white route.
    try:
        route, _g1 = jm49._claim_route(build_model_fn, rows, agentic, remote=remote)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        # Claim gate dead: we CANNOT tell claim-free (white recovers) from claim-bearing (white
        # INVERTS, 0.36). Routing all agentic white would import that inversion, so fall back to
        # the champion m33 (reads agentic claim-bearing 0.81-0.92) — never the inverting default.
        print(f"[{NAME}] claim gate died ({type(e).__name__}: {str(e)[:120]}) -> m33 unchanged",
              flush=True)
        return base
    if not route.any():
        print(f"[{NAME}] no agentic-claim-free rows -> m33 unchanged", flush=True)
        return base

    # 4. jm49's suspect-side white leg.
    white = None
    try:
        white = jm49._white_leg(build_model_fn, rows, remote=remote, batch_size=batch_size)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[{NAME}] white leg died ({type(e).__name__}: {str(e)[:120]}) -> m33 unchanged",
              flush=True)
        return base
    if not _usable(white, n):
        print(f"[{NAME}] white leg degenerate -> m33 unchanged", flush=True)
        return base

    # 5. splice: agentic-claim-free rows take the white read; every other row keeps m33.
    #    (_z both so the median-threshold sees one comparable axis; monotonic so non-route
    #    rows preserve m33's ranks. On the non-agentic board this branch never runs -> exact m33.)
    comp = np.where(route, _z(white), _z(base))
    # CRITICAL: the harness clips scores to [0,1] then thresholds at the median (harness.py
    # _guarded_write_arrays). A raw z-vector (~50% negative) clips its lower half to 0.0, the
    # median collapses to 0.0, and every row is called deceptive -> balAcc 0.5 on exactly the
    # agentic cells this method exists to recover. Wrap in the clip-safe sigmoid centered at the
    # median (as jm49 does) so the output is [0,1] centered at 0.5 and the median split is clean.
    out = js._judge_prob(comp, float(np.median(comp)))
    print(f"[{NAME}] spliced white into {int(route.sum())}/{n} agentic-claim-free rows "
          f"(rest = m33 champion); sigmoid-wrapped for the clipping harness", flush=True)
    return out
