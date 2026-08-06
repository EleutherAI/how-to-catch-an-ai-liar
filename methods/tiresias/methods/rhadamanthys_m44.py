"""rhadamanthys_m44 — internal detector module.
Fire with: ALETHEIA_METHOD=rhadamanthys_m44   (tag: black)
Env: ALETHEIA_SELFEVAL_3P_DEADLINE
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import rhadamanthys22 as r22  # noqa: E402
from methods import rhadamanthys29 as r29  # noqa: E402
from methods import jm43 as jta  # noqa: E402
from methods import judge_pilot as jp                # noqa: E402
from methods import sm11 as se3p              # noqa: E402
from methods import fusion_gates as fg               # noqa: E402

assert jp.MAX_TRANSCRIPT_CHARS == 6000, f"transcript cap drifted: {jp.MAX_TRANSCRIPT_CHARS}"

for _attr in ("LAST_PROVENANCE", "LAST_TWO_PASS_OK", "LAST_ROW_VALID",
              "PROV_TWO_PASS", "PROV_PLAIN_FALLBACK", "PROV_DEAD", "NAME", "run", "build"):
    assert hasattr(jta, _attr), f"jm43 is missing {_attr!r} — base contract broken"

se3p.DEADLINE = float(os.environ.get("ALETHEIA_SELFEVAL_3P_DEADLINE", "900"))
assert se3p.DEADLINE > 0, "se3p deadline disabled — a hung gemma/Nemo cell would eat the run"

NAME = "rhadamanthys_m44"

build = r22.build
judge_rescue = r22.judge_rescue


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    n = len(index)

    t1 = time.time()
    sb, prov = None, jta.PROV_DEAD
    try:
        sb = np.asarray(jta.run(build_model_fn, examples, index, task,
                                remote=remote, batch_size=batch_size), float)
        if sb.shape != (n,) or not np.all(np.isfinite(sb)):
            raise RuntimeError(f"bad base shape={getattr(sb, 'shape', None)}")
        sb = np.clip(sb, 0.0, 1.0)
        prov = jta.LAST_PROVENANCE
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[{NAME}] base (jm43) failed ({type(e).__name__}: {str(e)[:120]}) "
              f"-> dead", flush=True)
        sb = None
    if sb is not None and float(np.ptp(sb)) <= fg.MIN_PTP:
        prov = jta.PROV_DEAD
    print(f"[{NAME}] base done ({time.time()-t1:.0f}s) provenance={prov} "
          + ("" if sb is None else
             f"frac_scored={float((np.abs(sb-0.5) > fg.TOL_SCORED).mean()):.2f}"), flush=True)

    se3p.LAST.clear()
    t0 = time.time()
    try:
        s2 = r22._leg2(build_model_fn, examples, index, task, remote, n)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[{NAME}] leg2 (sm11) failed ({type(e).__name__}: {str(e)[:120]}) "
              f"-> dropped", flush=True)
        s2 = np.full(n, 0.5)
    v2 = r29._se3p_row_valid(n)
    print(f"[{NAME}] leg2 done ({time.time()-t0:.0f}s) "
          f"row_valid={'n/a' if v2 is None else f'{int(v2.sum())}/{n}'} "
          f"frac_scored={float((np.abs(s2-0.5) > fg.TOL_SCORED).mean()):.2f}", flush=True)

    if sb is None or prov == jta.PROV_DEAD:
        if float(np.ptp(s2)) > fg.MIN_PTP:
            print(f"[{NAME}] base dead -> se3p-only", flush=True)
            return s2
        print(f"[{NAME}] base and leg dead -> 0.5", flush=True)
        return np.full(n, 0.5)

    if v2 is not None and not v2.all():
        scores, meta = fg.fuse(sb, s2, valid_masks=[v2])
    else:
        scores, meta = fg.fuse(sb, s2)
    leg = (meta.get("legs") or [{}])[0]
    print(f"[{NAME}] BRANCH {'se3p-fused' if meta['mode'] == 'fused' else 'base-pure'} "
          f"(base={prov}) leg[se3p] drop={leg.get('drop')} rho={leg.get('rho')} "
          f"sep={leg.get('sep')} frac={leg.get('frac')}", flush=True)
    print(f"[{NAME}] FUSE mode={meta['mode']} w={meta.get('w')}", flush=True)
    if meta["mode"] != "fused":
        return sb
    return np.clip(scores, 0.0, 1.0)
