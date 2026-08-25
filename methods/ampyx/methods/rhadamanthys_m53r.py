"""rhadamanthys_m53r — the WB pin (rhadamanthys_m53, board 0.9452) plus the ONE re-export it forgot.

Fire with: ALETHEIA_METHOD=rhadamanthys_m53r   (tag: WHITE — identical scorer to rhadamanthys_m53.)

THE DEFECT (landmine audit L3, 2026-07-27). `submission/harness.py:109` does

    rescue = getattr(method_module, "judge_rescue", None)

and, when `run()` raised, calls it to obtain a still-RANKED plain-judge score for the cell. Only if
it is absent does the harness fall through to `np.full(n, 0.5)` (harness.py:121-122).

`rhadamanthys_m33.py:23` exports `judge_rescue = rtb.judge_rescue`.
`rhadamanthys_m53.py` does NOT — the re-export was simply forgotten when m53 wrapped m33.

Consequence: on any uncaught exception inside `m53.run`, the WB pin writes a CONSTANT vector for the
whole cell. Under the current metric (MEAN AUROC over 6 counted cells) a constant cell scores AUROC
exactly 0.5, costing **-0.062 to -0.082** on the mean — against a lead over second place of 0.0304.
The identical scorer, fired as `rhadamanthys_m33`, would have been rescued with a ranked vector.

WHAT THIS MODULE CHANGES: nothing on any path that produces a score. `run` and `build` are bound to
the very same function objects as m53 (identity, not copies), so every healthy cell is byte-identical
by construction — there is no code here that could diverge. The single addition is a module
attribute that is read ONLY by the harness, ONLY after `run` has already raised, i.e. only on the
path where we currently emit the worst possible value.

WHAT IT DOES NOT FIX: the L2 compound-outage path (`m44.py:80-81`, needs the judge AND the se3p leg
both dead) still returns a constant from inside `run` without raising, so the harness never sees it.
That one has no cheap provably-identical fix and stays a documented residual risk.

COMPLIANCE: identical scorer, identical thresholds, identical judge (Qwen3.5-9B). The rescue is
`methods.rhadamanthys3.judge_rescue`, the same plain-judge fallback the m33 champion has always
shipped. Not a router; nothing fit on dev.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import rhadamanthys_m53 as _m53   # noqa: E402  the frozen WB pin, untouched
from methods import rhadamanthys_m33 as _m33   # noqa: E402  source of the forgotten re-export

NAME = "rhadamanthys_m53r"

build = _m53.build            # same object
run = _m53.run                # same object -> healthy output byte-identical by construction
judge_rescue = _m33.judge_rescue   # the ONLY addition; read by harness._last_resort after a raise


# ==========================================================================================
# OFFLINE SELF-TEST — no NDIF, no network. Proves the two claims that matter:
#   A IDENTITY: build/run are the SAME objects as m53's (stronger than byte-identical source).
#   B RESCUE WIRED: harness._last_resort finds a callable judge_rescue on this module and returns
#     a RANKED vector, where the same path on m53 returns a constant 0.5.
# ==========================================================================================
def _selftest():
    import numpy as np
    fails = []

    # --- A: object identity with the frozen pin ---
    if run is not _m53.run:
        fails.append("A run is not m53.run (identity broken)")
    if build is not _m53.build:
        fails.append("A build is not m53.build (identity broken)")
    if judge_rescue is not _m33.judge_rescue:
        fails.append("A judge_rescue is not m33.judge_rescue")

    # --- B: the harness lookup that decides constant-vs-ranked ---
    import types
    this = sys.modules[__name__]
    if getattr(_m53, "judge_rescue", None) is not None:
        fails.append("B precondition failed: m53 already exports judge_rescue (audit stale?)")
    if getattr(this, "judge_rescue", None) is None:
        fails.append("B this module does not expose judge_rescue to the harness")
    if not callable(getattr(this, "judge_rescue", None)):
        fails.append("B judge_rescue is not callable")

    # --- B2: replicate harness._last_resort against a STUB rescue, both modules ---
    n = 6

    def _fake_rescue(build_model_fn, rows, index, task=None, *, remote=True, batch_size=8):
        return np.linspace(0.1, 0.9, n)          # a RANKED vector

    def _last_resort(module):
        """Byte-for-byte the decision in harness.py:108-122."""
        rescue = getattr(module, "judge_rescue", None)
        if rescue is not None:
            sc = rescue(None, [None] * n, list(range(n)))
            if sc is not None:
                return np.clip(np.asarray(sc, float), 0.0, 1.0)
        return np.full(n, 0.5)

    shim_fixed = types.SimpleNamespace(judge_rescue=_fake_rescue)
    shim_none = types.SimpleNamespace()
    got_fixed = _last_resort(shim_fixed)
    got_bare = _last_resort(shim_none)
    if len(set(np.round(got_bare, 9))) != 1 or got_bare[0] != 0.5:
        fails.append(f"B2 control: module without judge_rescue must give constant 0.5, got {got_bare}")
    if len(set(np.round(got_fixed, 9))) != n:
        fails.append(f"B2 module with judge_rescue must give a RANKED vector, got {got_fixed}")

    # --- B3: the real wiring — this module vs the frozen pin ---
    if getattr(_m53, "judge_rescue", None) is not None:
        fails.append("B3 m53 unexpectedly has a rescue")
    if getattr(this, "judge_rescue", None) is None:
        fails.append("B3 m53r is missing the rescue")

    print("=" * 78)
    if fails:
        print("SELFTEST FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("SELFTEST PASSED — rhadamanthys_m53r:")
    print("  A run/build are the SAME OBJECTS as rhadamanthys_m53 -> healthy cells byte-identical")
    print("  B harness._last_resort finds a callable judge_rescue here and NOT on m53")
    print("  B2 replicated harness decision: no-rescue -> constant 0.5 (AUROC 0.5);")
    print("     with-rescue -> ranked vector (AUROC informative)")
    print(f"  cost avoided if it ever fires: one dead counted cell = -0.062..-0.082 mean AUROC")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
