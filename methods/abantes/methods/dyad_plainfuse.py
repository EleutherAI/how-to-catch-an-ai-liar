"""dyad_plainfuse — the BB pin + a cheap generation-free plain-judge channel, fused per sample.

Fire with: ALETHEIA_METHOD=dyad_plainfuse   (tag: black — trusted judge only)

WHY (external evidence, public leaderboard 07-27): SAIN Groningen's `phoenix_wright_v6_2`
(black-box) scored 0.9233 counted in 16m29s vs our pin `rhadamanthys_dyad_anchor2h_shield_clean`
0.9237 in 187m33s. Their budget (~34 s/cell) is consistent with ONE short judge prompt per row and
a single next-token yes/no logit read, no generation — i.e. the plain-judge channel we already own
as judge_pilot.judge_ynd. Per genre it is NOT a worse anchor, it is a DIFFERENT channel:
better on the hard genres (Eunomia +0.0097, Notus +0.0080; Notus-gemma +0.0409 — the anchor's
weakest counted cell), worse on the easy ones (Iris −0.0088, Metis −0.0016). Naive per-genre
upper bound of combining: ~0.9277 counted (+0.0040 over the pin).

WHAT (ONE FACTOR over the pin):
  anchor  = rhadamanthys_dyad_anchor2h_shield_clean.run(...) called VERBATIM (its own prompt
            mutations + finally-restore, shield fault isolation, T sealed 2.0, fixed threshold);
  cheap   = judge_pilot.judge_ynd (frozen prompt SYS_JUDGE, frozen retry ladder, frozen 6000-char
            transcript window set by jm43): one batched forward read per row, yes/no log-odds,
            NO generation; per-row validity mask via return_mask=True;
  fusion  = per-sample fixed anchor-dominant logit blend (the board-validated COMPLIANT
            Amphilochus recipe, rhadamanthys_amphilochus.py):

                fused_logit_i = logit(pin_i) + W_CHEAP * (ynd_i - CHEAP_CENTER) / 2
                score_i       = sigmoid(fused_logit_i)                # W_CHEAP = 0.5 fixed pre-test

            (ynd/2 IS logit(p_cheap) under the frozen jm43._judge_prob scale; the DYAD margin/T with
            T=2 puts logit(pin) in the same judge-log-odds/2 units, so W_CHEAP=0.5 means the cheap
            channel gets exactly half the anchor's per-row authority — it can only correct
            borderline anchor rows, never flip a confident one.)

COMBINER CHOICE — deliberately NOT fusion_gates.fuse:
  * fuse() mixes cross-row z-scores (fusion_gates._z): every row's score depends on the batch —
    the exact construct flagged in the 07-20 compliance audit of m53. The per-sample blend above
    is the compliant replacement that was already board-confirmed (Amphilochus 0.8779, ruled
    compliant, +0.014 over the equal-weight Manto).
  * fuse() weights legs by SEPARATION not accuracy (board-confirmed failure mode, twice): the
    yes/no log-odds channel is intrinsically bimodal, so it would be systematically over-weighted
    on exactly the cells where its confidence is spurious.
  * fusion_gates.admit IS reused, verbatim, as the channel-HEALTH gate (shape/degenerate/flat/
    valid-fraction checks), with ONE documented override: the `inverted` verdict (spearman < +0.30
    vs the anchor) does NOT drop the leg unless rho <= RHO_HARD_DROP (−0.20). Rationale: the
    external evidence says the cheap channel's value is concentrated precisely where it DISAGREES
    with the anchor; a +0.30 admission floor deletes the complementarity zone. A genuinely
    INVERTED leg (rho <= −0.20 — unprecedented for the plain Qwen judge in our entire board
    history) still reverts to the pin. The gate can therefore only ever revert to the pin's own
    per-sample scores (fail-closed channel health), never shape them.

FAILURE CONTRACT (never a constant vector, never worse than the pin's own failure behaviour):
  any failure of the added channel — raise, hang (SIGALRM absolute deadline), wrong shape,
  non-finite values, bad mask, degenerate/flat channel, true inversion, exhausted budget —
  returns the pin's OWN array object, byte-identical. Rows the cheap read could not score
  (valid=False) take the pin value verbatim. judge_ynd's internal live-median fill for dead rows
  never reaches the output (those rows are masked to the pin). The output can only be constant if
  the pin's own array is (its documented catastrophic path).

COST: one forward read per row, batched — comparable to ONE of the pin's two A/B phases.
Predicted +40–55 s/cell, ~+1.2–1.6 ks over 29 cells (pin: 11 175 s). Absolute per-cell deadline
DEADLINE_S=2600 s (board kill ~3120 s): if the anchor already consumed the budget the cheap read
is skipped and the pin array is returned unchanged.

COMPLIANCE: BLACK (trusted judge jp.JUDGE_MODEL only; the suspect's logprobs/activations are
never read). NO router: examples[i]['model'] / ['lora'] / task are never inspected. Per-sample
scoring with all constants fixed pre-test (W_CHEAP, CHEAP_CENTER=0 — the natural zero of yes/no
log-odds, NEVER an in-batch median). ALETHEIA_FIXED_THRESHOLD hard-set to "1"; T sealed at 2.0
by the pin (re-asserted here). No frozen file is edited — composition strictly by import.
"""
from __future__ import annotations
import contextlib
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "submission")

# HARD-SET (not setdefault), before the pin import, exactly like the pin itself: a pre-existing
# "0" in the environment must never re-enable the banned harness median.
os.environ["ALETHEIA_FIXED_THRESHOLD"] = "1"

from methods import rhadamanthys_dyad_anchor2h_shield_clean as _pin   # noqa: E402  the ANCHOR (verbatim)
from methods import rhadamanthys_dyad_anchor as _dyad                 # noqa: E402  T seal re-assert only
from methods import judge_pilot as jp                                 # noqa: E402  cheap channel (frozen)
from methods import jm43                                              # noqa: E402  _judge_prob scale (frozen)
from methods import fusion_gates as fg                                # noqa: E402  admit() health gate (frozen)

NAME = "dyad_plainfuse"
build = _pin.build                     # == the pin's build (jm43.build)

# ---- seals (re-asserted; the pin asserts them too) -------------------------------------------
assert float(_dyad.T) == 2.0, f"T must be sealed at 2.0 (pin loaded T={_dyad.T!r})"
assert os.environ.get("ALETHEIA_FIXED_THRESHOLD") == "1"
assert jp.MAX_TRANSCRIPT_CHARS == 6000, f"transcript cap drifted: {jp.MAX_TRANSCRIPT_CHARS}"

# ---- fixed pre-test constants (no env knobs: seal discipline, no silent re-tuning) -----------
W_CHEAP = 0.5          # anchor-dominant fixed weight — the Amphilochus board-validated choice
CHEAP_CENTER = 0.0     # natural zero of the yes/no log-odds (fixed pre-test, never a batch stat)
RHO_HARD_DROP = -0.20  # drop the cheap leg only on TRUE inversion vs the anchor (fail to the pin)
DEADLINE_S = 2600.0    # absolute per-cell wall budget from run() start (board kill ~3120 s)
MIN_CHEAP_S = 60.0     # don't even start the cheap read with less than this left

# the anchor and cheap implementations, bound ONCE at import so the selftest can both assert
# identity with the frozen objects and substitute offline mocks without touching frozen modules
_ANCHOR_IMPL = _pin.run
_CHEAP_IMPL = jp.judge_ynd


class _Deadline(Exception):
    pass


@contextlib.contextmanager
def _deadline(seconds):
    """Absolute deadline that INTERRUPTS a hung cheap read (m54safe pattern). Main-thread only;
    degrades to a no-op elsewhere (honestly logged). Handler + timer restored in finally."""
    ok = False
    old = None
    if seconds and seconds > 0:
        try:
            old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Deadline()))
            signal.setitimer(signal.ITIMER_REAL, float(seconds))
            ok = True
        except (ValueError, AttributeError, OSError):   # not main thread / no SIGALRM
            print(f"[{NAME}] WARNING: no SIGALRM deadline (hang not interruptible)", flush=True)
    try:
        yield
    finally:
        if ok:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old)


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(v, float), -30, 30)))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _fuse_or_pin(pin_arr, ynd, valid):
    """Per-sample fixed blend, gated by frozen channel-health checks. EVERY exit that does not
    fuse returns the literal pin array object (byte-identical fallback)."""
    n = pin_arr.shape[0]
    try:
        ynd = np.asarray(ynd, float)
        if ynd.shape != (n,) or not np.all(np.isfinite(ynd)):
            print(f"[{NAME}] cheap channel unusable (shape={ynd.shape} finite="
                  f"{bool(np.all(np.isfinite(ynd)))}) -> pin array", flush=True)
            return pin_arr
        valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool).reshape(-1)
        if valid.shape != (n,):
            print(f"[{NAME}] cheap valid-mask unusable -> pin array", flush=True)
            return pin_arr

        p_cheap = jm43._judge_prob(ynd, CHEAP_CENTER)      # frozen scale, FIXED center (per-sample)

        # frozen health gate; ONE override: decorrelation (rho in (−0.20, +0.30)) is the
        # complementarity zone, not an inversion — keep the leg there, drop it only when it is
        # genuinely inverted or degenerate/flat/underscored.
        ok, _scored, info = fg.admit(pin_arr, p_cheap, valid=valid)
        rho = info.get("rho")
        if (not ok and info.get("drop") == "inverted"
                and rho is not None and np.isfinite(rho) and rho > RHO_HARD_DROP):
            print(f"[{NAME}] admit said 'inverted' (rho={rho:+.3f} < +0.30) but rho > "
                  f"{RHO_HARD_DROP} -> complementarity override, leg KEPT", flush=True)
            ok = True
        if not ok:
            print(f"[{NAME}] cheap channel dropped ({info}) -> pin array", flush=True)
            return pin_arr

        la = _logit(pin_arr)                               # DYAD margin/T (T=2), recovered per row
        lc = np.clip((ynd - CHEAP_CENTER) / 2.0, -30.0, 30.0)   # == logit(p_cheap), same /2 scale
        fused = _sig(la + W_CHEAP * lc)
        out = np.clip(np.where(valid, fused, pin_arr), 0.0, 1.0)  # unscored rows: pin VERBATIM
        if not np.all(np.isfinite(out)) or float(np.ptp(out)) < 1e-12:
            print(f"[{NAME}] fused output degenerate -> pin array", flush=True)
            return pin_arr
        print(f"[{NAME}] fused: w={W_CHEAP} rho={'n/a' if rho is None else format(rho, '+.3f')} "
              f"included={int(valid.sum())}/{n} mean|dscore|={float(np.abs(out - pin_arr).mean()):.4f}",
              flush=True)
        return out
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"[{NAME}] fuse failed ({type(e).__name__}: {str(e)[:120]}) -> pin array", flush=True)
        return pin_arr


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    t0 = time.time()
    n = len(index)

    # ---- 1) the ANCHOR: the pin, verbatim (its own seals, shield, prompt restore) ------------
    try:
        pin_raw = _ANCHOR_IMPL(build_model_fn, examples, index, task,
                               remote=remote, batch_size=batch_size)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:            # the pin catches BaseException itself; belt only
        print(f"[{NAME}] anchor raised ({type(e).__name__}: {str(e)[:120]}) -> per-row 0.5",
              flush=True)
        return np.full(n, 0.5)
    pin_arr = np.asarray(pin_raw, float)  # no copy when the pin already returns float64
    if pin_arr.shape != (n,) or not np.all(np.isfinite(pin_arr)):
        print(f"[{NAME}] anchor contract violated -> per-row 0.5", flush=True)
        return np.full(n, 0.5)
    if n == 0:
        return pin_arr

    # ---- 2) absolute budget: if the anchor ate the cell, ship the pin unchanged --------------
    left = DEADLINE_S - (time.time() - t0)
    if left < MIN_CHEAP_S:
        print(f"[{NAME}] {left:.0f}s left of {DEADLINE_S:.0f}s after anchor -> pin array "
              f"(cheap channel skipped)", flush=True)
        return pin_arr

    # ---- 3) the CHEAP channel: one batched forward yes/no read per row, no generation --------
    ynd, valid = None, None
    try:
        with _deadline(left):
            rows = [examples[i] for i in range(n)]
            t1 = time.time()
            ynd, valid = _CHEAP_IMPL(build_model_fn, rows, remote=remote,
                                     batch_size=batch_size, return_mask=True)
            print(f"[{NAME}] cheap channel read {n} rows in {time.time() - t1:.0f}s", flush=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:            # includes _Deadline and judge_ynd's >50%-dead raise
        print(f"[{NAME}] cheap channel failed ({type(e).__name__}: {str(e)[:120]}) -> pin array",
              flush=True)
        return pin_arr

    # ---- 4) health-gated per-sample fusion (every non-fusing exit = the pin array) -----------
    return _fuse_or_pin(pin_arr, ynd, valid)


# ================================ OFFLINE SELFTEST (no NDIF, no network) ======================
if __name__ == "__main__":
    import hashlib
    import sys

    from methods.twin_arb import _spearman

    os.environ["ALETHEIA_SHIELD_BACKOFF_S"] = "0"      # keep the real-anchor offline test fast

    def _h(s):                                          # stable in [0,1) (md5, not salted hash())
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / float(0xFFFFFFFF)

    def _mk_examples(n, model="orgA/model-x", lora="orgA/lora-1"):
        exs = []
        for i in range(n):
            exs.append({"model": model, "lora": lora, "messages": [
                {"role": "user", "content": f"question number {i}: what is fact {i}?"},
                {"role": "assistant", "content": f"reply {i}: the answer involves item {i * 7}."},
            ]})
        return exs

    N = 24
    EX = _mk_examples(N)
    IDX = list(range(N))

    def _boom(*a, **k):
        raise RuntimeError("offline: no model builds in the selftest")

    STORE = {}

    def _mock_anchor_from(pvals):
        def _a(bmf, examples, index, task=None, *, remote=True, batch_size=8):
            arr = np.array([pvals(examples[i]) for i in range(len(index))], float)
            STORE["pin"] = arr
            return arr
        return _a

    def _pin_val(ex):                                   # deterministic f(messages) ONLY
        return 0.05 + 0.90 * _h(ex["messages"][1]["content"] + "|pin")

    ANCHOR_OK = _mock_anchor_from(_pin_val)

    def _mock_cheap(ynd_fn, valid_fn=None, counter=None):
        def _c(bmf, rows, *, remote=True, batch_size=8, return_mask=True, **kw):
            if counter is not None:
                counter[0] += 1
            ynd = np.array([ynd_fn(ex) for ex in rows], float)
            valid = (np.ones(len(rows), bool) if valid_fn is None
                     else np.array([valid_fn(ex) for ex in rows], bool))
            return ynd, valid
        return _c

    @contextlib.contextmanager
    def _patched(**kw):
        g = globals()
        old = {k: g[k] for k in kw}
        g.update(kw)
        try:
            yield
        finally:
            g.update(old)

    fails = []

    def _check(name, cond):
        print(("PASS  " if cond else "FAIL  ") + name, flush=True)
        if not cond:
            fails.append(name)

    # T1 — composition & seals: the anchor path IS the pin's own objects; frozen imports intact
    _check("T1a anchor impl IS the pin's run", _ANCHOR_IMPL is _pin.run)
    _check("T1b build IS the pin's build", build is _pin.build)
    _check("T1c cheap impl IS judge_pilot.judge_ynd", _CHEAP_IMPL is jp.judge_ynd)
    _check("T1d threshold seal", os.environ.get("ALETHEIA_FIXED_THRESHOLD") == "1")
    _check("T1e T seal", float(_dyad.T) == 2.0)
    _check("T1f constants sealed", W_CHEAP == 0.5 and CHEAP_CENTER == 0.0
           and RHO_HARD_DROP == -0.20)

    # T2 — informative channel: fused output moves, is never constant, matches the exact formula
    def _ynd_informative(ex):
        return 2.0 * _logit(np.array([_pin_val(ex)]))[0] + 1.5 * (_h(ex["messages"][1]["content"] + "|c") - 0.5)

    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_mock_cheap(_ynd_informative)):
        out = run(_boom, EX, IDX)
    pin = STORE["pin"]
    rho2 = _spearman(pin, jm43._judge_prob(np.array([_ynd_informative(e) for e in EX]), 0.0))
    exp = _sig(_logit(pin) + W_CHEAP * np.clip(np.array([_ynd_informative(e) for e in EX]) / 2.0, -30, 30))
    _check("T2a informative leg admitted (rho=%.2f>=0.30 precondition)" % rho2, rho2 >= 0.30)
    _check("T2b output moved off the pin", not np.array_equal(out, pin))
    _check("T2c output == exact per-sample blend formula", np.allclose(out, exp, atol=1e-12))
    _check("T2d never constant / finite / in [0,1]",
           float(np.ptp(out)) > 0 and np.all(np.isfinite(out))
           and out.min() >= 0.0 and out.max() <= 1.0)

    # T3 — per-row validity: unscored cheap rows take the pin VERBATIM
    def _valid_half(ex):
        return _h(ex["messages"][0]["content"] + "|v") > 0.5

    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_mock_cheap(_ynd_informative, _valid_half)):
        out = run(_boom, EX, IDX)
    pin = STORE["pin"]
    vmask = np.array([_valid_half(e) for e in EX])
    _check("T3a some rows in each class", 0 < vmask.sum() < N)
    _check("T3b invalid rows == pin exactly", np.array_equal(out[~vmask], pin[~vmask]))
    _check("T3c valid rows moved", not np.array_equal(out[vmask], pin[vmask]))

    # T4 — degenerate channel (constant ynd): dropped by frozen admit -> the literal pin array
    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_mock_cheap(lambda ex: 3.0)):
        out = run(_boom, EX, IDX)
    _check("T4 degenerate -> pin array identity", out is STORE["pin"])

    # T5 — flat channel (only 3 scoreable rows -> sep_d==0 -> admit 'flat'): pin identity
    EX8 = _mk_examples(8)
    with _patched(_ANCHOR_IMPL=ANCHOR_OK,
                  _CHEAP_IMPL=_mock_cheap(lambda ex: 6.0 * (_h(ex["messages"][1]["content"]) - 0.5),
                                          lambda ex: int(ex["messages"][0]["content"].split()[2].rstrip(":")) < 3)):
        out = run(_boom, EX8, list(range(8)))
    _check("T5 flat (3 valid rows) -> pin array identity", out is STORE["pin"])

    # T6 — TRUE inversion (rho ~ -1): dropped -> pin identity
    def _ynd_inverted(ex):
        return -6.0 * _logit(np.array([_pin_val(ex)]))[0]

    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_mock_cheap(_ynd_inverted)):
        out = run(_boom, EX, IDX)
    rho6 = _spearman(STORE["pin"], jm43._judge_prob(np.array([_ynd_inverted(e) for e in EX]), 0.0))
    _check("T6a inversion precondition rho=%.2f <= -0.20" % rho6, rho6 <= RHO_HARD_DROP)
    _check("T6b true inversion -> pin array identity", out is STORE["pin"])

    # T7 — DECORRELATED but healthy (the complementarity zone): admit says 'inverted', the
    # override KEEPS it — this is the deliberate deviation from a bare fg.admit gate
    def _ynd_decorr(ex):
        return 8.0 * (_h(ex["messages"][1]["content"] + "|decorr7") - 0.5)

    rho7 = _spearman(np.array([_pin_val(e) for e in EX]),
                     jm43._judge_prob(np.array([_ynd_decorr(e) for e in EX]), 0.0))
    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_mock_cheap(_ynd_decorr)):
        out = run(_boom, EX, IDX)
    _check("T7a decorrelation precondition -0.20<rho=%.2f<0.30" % rho7,
           RHO_HARD_DROP < rho7 < fg.RHO_FLOOR)
    _check("T7b decorrelated leg KEPT (output moved)", not np.array_equal(out, STORE["pin"]))

    # T8 — channel raises: pin identity; SIGALRM handler + timer restored
    pre_handler = signal.getsignal(signal.SIGALRM)

    def _cheap_raises(bmf, rows, **kw):
        raise RuntimeError("cheap exploded")

    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_cheap_raises):
        out = run(_boom, EX, IDX)
    _check("T8a raise -> pin array identity", out is STORE["pin"])
    _check("T8b SIGALRM handler restored after raise",
           signal.getsignal(signal.SIGALRM) is pre_handler)
    _check("T8c itimer disarmed", signal.getitimer(signal.ITIMER_REAL)[0] == 0.0)

    # T9 — wrong shape: pin identity
    def _cheap_shape(bmf, rows, **kw):
        return np.zeros(len(rows) - 1), np.ones(len(rows) - 1, bool)

    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_cheap_shape):
        out = run(_boom, EX, IDX)
    _check("T9 wrong shape -> pin array identity", out is STORE["pin"])

    # T10 — NaN in the channel: pin identity
    def _cheap_nan(bmf, rows, **kw):
        y = np.array([_ynd_informative(ex) for ex in rows]); y[3] = np.nan
        return y, np.ones(len(rows), bool)

    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_cheap_nan):
        out = run(_boom, EX, IDX)
    _check("T10 NaN -> pin array identity", out is STORE["pin"])

    # T11 — deadline already gone after the anchor: cheap NEVER called, pin identity
    calls = [0]
    with _patched(_ANCHOR_IMPL=ANCHOR_OK,
                  _CHEAP_IMPL=_mock_cheap(_ynd_informative, counter=calls), DEADLINE_S=0.0):
        out = run(_boom, EX, IDX)
    _check("T11a budget gone -> pin array identity", out is STORE["pin"])
    _check("T11b cheap channel never invoked", calls[0] == 0)

    # T12 — HANGING cheap read: SIGALRM interrupts it, pin identity, fast return, state restored
    def _cheap_hangs(bmf, rows, **kw):
        time.sleep(30)
        return np.zeros(len(rows)), np.ones(len(rows), bool)

    t0 = time.time()
    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_cheap_hangs,
                  DEADLINE_S=1.2, MIN_CHEAP_S=0.2):
        out = run(_boom, EX, IDX)
    dt = time.time() - t0
    _check("T12a hang interrupted -> pin array identity", out is STORE["pin"])
    _check("T12b interrupted fast (%.1fs < 5s)" % dt, dt < 5.0)
    _check("T12c handler restored", signal.getsignal(signal.SIGALRM) is pre_handler)
    _check("T12d itimer disarmed", signal.getitimer(signal.ITIMER_REAL)[0] == 0.0)

    # T13 — invariance under model/lora/task relabelling (mocks read messages only; this module
    # never touches the labels, so the outputs must be bitwise equal)
    EX_A = _mk_examples(N, model="orgA/model-x", lora="orgA/lora-1")
    EX_B = _mk_examples(N, model="orgB/OTHER", lora=None)
    with _patched(_ANCHOR_IMPL=ANCHOR_OK, _CHEAP_IMPL=_mock_cheap(_ynd_informative)):
        out_a = run(_boom, EX_A, IDX, "task-one")
        out_b = run(_boom, EX_B, IDX, "totally-different-task")
    _check("T13 relabelling invariance (model/lora/task)", np.array_equal(out_a, out_b))

    # T14 — REAL anchor path, offline: the judge build fails 3x inside the shield -> the pin's
    # own catastrophic per-row 0.5; the pin's prompt mutations are restored even on that raise;
    # the healthy cheap channel then rescues the cell (output NOT constant)
    truth0, strip0, fwd0 = jm43.SYS_TRUTH, jm43.SYS_STRIP, _dyad.SYS_JUDGE_FACT_FWD
    with _patched(_CHEAP_IMPL=_mock_cheap(_ynd_informative)):       # anchor = the REAL pin.run
        out = run(_boom, EX, IDX)
    _check("T14a pin prompts restored after in-run failure",
           jm43.SYS_TRUTH == truth0 and jm43.SYS_STRIP == strip0
           and _dyad.SYS_JUDGE_FACT_FWD == fwd0)
    _check("T14b dead anchor (all 0.5) rescued by cheap channel (not constant)",
           float(np.ptp(out)) > 0)
    exp14 = _sig(0.0 + W_CHEAP * np.clip(np.array([_ynd_informative(e) for e in EX]) / 2.0, -30, 30))
    _check("T14c rescue == blend with logit(0.5)=0 anchor", np.allclose(out, exp14, atol=1e-12))

    # T15 — pin constant AND cheap degenerate: the pin's own array (constant allowed ONLY here,
    # it is the pin's documented catastrophic output, not ours)
    ANCHOR_CONST = _mock_anchor_from(lambda ex: 0.5)
    with _patched(_ANCHOR_IMPL=ANCHOR_CONST, _CHEAP_IMPL=_mock_cheap(lambda ex: 0.0)):
        out = run(_boom, EX, IDX)
    _check("T15 constant pin + dead cheap -> pin array identity", out is STORE["pin"])

    # T16 — seals survived every path above
    _check("T16 seals intact after all runs",
           os.environ.get("ALETHEIA_FIXED_THRESHOLD") == "1" and float(_dyad.T) == 2.0
           and jm43.SYS_TRUTH == truth0)

    print("\n" + ("SELFTEST FAILED: " + ", ".join(fails) if fails
                  else "SELFTEST PASSED (all checks green)"), flush=True)
    sys.exit(1 if fails else 0)
