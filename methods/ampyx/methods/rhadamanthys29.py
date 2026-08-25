"""rhadamanthys29 — internal detector module.
Fire with: ALETHEIA_METHOD=rhadamanthys29   (tag: white)
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

os.environ["ALETHEIA_R3_AGREE"] = "0.15"
os.environ["ALETHEIA_JUDGE_MAXCHARS"] = "6000"

from methods import rhadamanthys22 as r22  # noqa: E402
from methods import rhadamanthys3 as r3         # noqa: E402
from methods import judge_pilot as jp           # noqa: E402
from methods import jm38 as jss     # noqa: E402
from methods import sm11 as se3p  # noqa: E402
from methods import fusion_gates as fg          # noqa: E402

r3.AGREE_FLOOR = 0.15
jp.MAX_TRANSCRIPT_CHARS = 6000

NAME = "rhadamanthys29"

FALLBACK_READER = jss

PROV_TWO_PASS = "TWO_PASS"
PROV_PLAIN_FALLBACK = "PLAIN_FALLBACK"
PROV_DEAD = "DEAD"

build = r22.build
judge_rescue = r22.judge_rescue


def _read_provenance(reader):
    """internal."""
    prov = getattr(reader, "LAST_PROVENANCE", None)
    if prov in (PROV_TWO_PASS, PROV_PLAIN_FALLBACK, PROV_DEAD):
        return prov
    return PROV_TWO_PASS if getattr(reader, "LAST_TWO_PASS_OK", False) is True \
        else PROV_PLAIN_FALLBACK


def _se3p_row_valid(n):
    """internal."""
    try:
        L = se3p.LAST
        vi = np.asarray(L.get("valid", []), int).reshape(-1)
        live = np.asarray(L.get("live", []), bool).reshape(-1)
        if vi.size == 0 or vi.shape != live.shape or vi.min() < 0 or vi.max() >= n:
            return None
        m = np.zeros(n, dtype=bool)
        m[vi] = live
        return m
    except Exception:
        return None


def _leg3(build_model_fn, examples, index, task, remote, batch_size, n):
    """internal."""
    t0 = time.time()
    name = getattr(FALLBACK_READER, "NAME", "fallback_reader")
    try:
        s = np.asarray(FALLBACK_READER.run(build_model_fn, examples, index, task,
                                           remote=remote, batch_size=batch_size), float)
        if s.shape != (n,) or not np.all(np.isfinite(s)):
            raise RuntimeError(f"bad leg3 shape={getattr(s, 'shape', None)}")
        s = np.clip(s, 0.0, 1.0)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[r29] leg3 ({name}) failed ({type(e).__name__}: {str(e)[:120]}) -> dropped",
              flush=True)
        return np.full(n, 0.5), PROV_DEAD, None
    prov = _read_provenance(FALLBACK_READER)
    if float(np.ptp(s)) <= fg.MIN_PTP:
        prov = PROV_DEAD
    rv = getattr(FALLBACK_READER, "LAST_ROW_VALID", None)
    try:
        rv = None if rv is None else np.asarray(rv).astype(bool).reshape(-1)
        if rv is not None and rv.shape != (n,):
            rv = None
    except Exception:
        rv = None
    print(f"[r29] leg3 ({name}) done ({time.time()-t0:.0f}s) provenance={prov} "
          f"row_valid={'n/a' if rv is None else f'{int(rv.sum())}/{n}'} "
          f"frac_scored={float((np.abs(s-0.5) > fg.TOL_SCORED).mean()):.2f}", flush=True)
    return s, prov, rv


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    n = len(index)
    rows = [examples[i] for i in range(n)]

    se3p.LAST.clear()
    t0 = time.time()
    try:
        s2 = r22._leg2(build_model_fn, examples, index, task, remote, n)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[r29] leg2 (sm11) failed ({type(e).__name__}: {str(e)[:120]}) → dropped",
              flush=True)
        s2 = np.full(n, 0.5)
    v2 = _se3p_row_valid(n)
    print(f"[r29] leg2 done ({time.time()-t0:.0f}s) "
          f"row_valid={'n/a' if v2 is None else f'{int(v2.sum())}/{n}'} "
          f"frac_scored={float((np.abs(s2-0.5) > fg.TOL_SCORED).mean()):.2f}", flush=True)

    s1, kind = r22._anchor(build_model_fn, examples, index, rows, task, remote, batch_size, n)

    if s1 is None:
        if float(np.ptp(s2)) > fg.MIN_PTP:
            print("[r29] anchor dead → leg2-only", flush=True)
            return s2
        s3, prov3, _rv3 = _leg3(build_model_fn, examples, index, task, remote, batch_size, n)
        if prov3 != PROV_DEAD:
            print(f"[r29] anchor dead → leg3-only (provenance={prov3})", flush=True)
            return s3
        print("[r29] anchor and legs dead → 0.5", flush=True)
        return np.full(n, 0.5)

    if v2 is not None and not v2.all():
        ok_se3p, _, info_se3p = fg.admit(s1, s2, valid=v2)
        vm2 = [v2]
    else:
        ok_se3p, _, info_se3p = fg.admit(s1, s2)
        vm2 = None
    print(f"[r29] se3p admit={ok_se3p} (drop={info_se3p.get('drop')} rho={info_se3p.get('rho')} "
          f"sep={info_se3p.get('sep')} frac={info_se3p.get('frac')})", flush=True)

    if ok_se3p:
        branch = "se3p-admitted"
        scores, meta = fg.fuse(s1, s2, valid_masks=vm2)
        lname = "se3p"
    else:
        s3, prov3, rv3 = _leg3(build_model_fn, examples, index, task, remote, batch_size, n)
        if prov3 != PROV_TWO_PASS:
            print(f"[r29] BRANCH anchor-only (fallback reader provenance={prov3}, "
                  f"not fusion-eligible)", flush=True)
            return np.clip(s1, 0.0, 1.0)
        vm3 = [rv3] if (rv3 is not None and not rv3.all()) else None
        branch = "fallback-reader"
        scores, meta = fg.fuse(s1, s3, valid_masks=vm3)
        lname = getattr(FALLBACK_READER, "NAME", "fallback_reader")

    legs = meta.get("legs") or [{}]
    li = legs[0] if legs else {}
    print(f"[r29] BRANCH {branch}  leg[{lname}] drop={li.get('drop')} rho={li.get('rho')} "
          f"sep={li.get('sep')} frac={li.get('frac')}", flush=True)
    print(f"[r29] FUSE anchor={kind} mode={meta['mode']} w={meta.get('w')}", flush=True)
    return np.clip(scores, 0.0, 1.0)
