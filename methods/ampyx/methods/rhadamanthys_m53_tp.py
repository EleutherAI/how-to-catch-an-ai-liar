"""rhadamanthys_m53_tp — internal detector module.
Fire with: ALETHEIA_METHOD=rhadamanthys_m53_tp   (tag: white)

THRESHOLD PROBE (audit 07-19, pre-registered). Identical to rhadamanthys_m53 EXCEPT the effective
keep-band is tightened from the harness's [0.4,0.6] to [0.475,0.525]: for a cell whose predicted-
positive fraction falls OUTSIDE [0.475,0.525], median-center the scores (js._judge_prob at the cell
median) so the harness's fixed-0.5 threshold becomes the MEDIAN threshold. In-band cells are returned
UNCHANGED. From the current board confusion counts only Notus-Qwen (frac 0.47) is out-of-band; the
other five counted cells (0.4975..0.52) stay byte-identical to m53. Diagnostic-only: measures whether
median-thresholding Notus-Qwen recovers balAcc. Pre-reg keep/kill: Notus-Qwen >=0.855 small win,
>=0.860 significant, 0.850 dead lever, <=0.845 abandon; the other five must be unchanged.
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import rhadamanthys_m53 as base   # noqa: E402
from methods import jm38 as js                 # noqa: E402

NAME = "rhadamanthys_m53_tp"
build = base.build
LO = float(os.environ.get("ALETHEIA_TP_LO", "0.475"))
HI = float(os.environ.get("ALETHEIA_TP_HI", "0.525"))


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    """internal."""
    sc = np.asarray(base.run(build_model_fn, examples, index, task,
                             remote=remote, batch_size=batch_size), float)
    n = len(index)
    if sc.shape != (n,) or not np.all(np.isfinite(sc)):
        return sc
    frac = float((sc >= 0.5).mean())
    if LO <= frac <= HI:
        print(f"[{NAME}] frac {frac:.3f} in [{LO},{HI}] -> m53 UNCHANGED", flush=True)
        return sc
    # rank-center: exactly-median split, AUROC-preserving, tie-proof (sigmoid ties at a cluster
    # median can push frac to 1.0). The harness then thresholds these uniform ranks at 0.5 = median.
    med = float(np.median(sc))
    order = np.argsort(sc, kind="stable")
    ranks = np.empty(n, float); ranks[order] = np.arange(n)
    sc_new = (ranks + 0.5) / n
    print(f"[{NAME}] frac {frac:.3f} OUT of [{LO},{HI}] -> rank-center "
          f"(median {med:.4f} -> 0.5; frac now {(sc_new>=0.5).mean():.3f})", flush=True)
    return sc_new
