"""jm49 — internal detector module.
Fire with: ALETHEIA_METHOD=jm49   (tag: white)
"""
from __future__ import annotations
import os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import jm29 as jsp                  # noqa: E402
from methods import jm38 as js                   # noqa: E402
from methods import jm06 as jg                   # noqa: E402
from methods import judge_pilot as jp            # noqa: E402
from methods import pm22 as wp                   # noqa: E402

NAME = "jm49"

MAXLEN = 256

# uniform DATA-conditional gate on the transcript (no model_id, no system-instruction read):
# a row routes to the white leg iff its PRE-final-reply record shows tool-interaction evidence
# and at least two prior assistant turns (an action record exists to contradict the report).
RE_REACT = re.compile(r"(?m)^\s*(Action|Action Input|Observation)\s*:")
RE_TAGS = re.compile(r"<tool_call>|</tool_call>|\[TOOL_CALLS\]|```tool|\"tool_calls\""
                     r"|<function_call>|function_call\(|<\|tool", re.I)
RE_SYS = re.compile(r"you act via tools|access to (?:the following )?tools"
                    r"|workspace tools|autonomous ai", re.I)

PROV_GATED_WHITE = "GATED_WHITE"
PROV_SPINE_ONLY = "SPINE_ONLY"
PROV_DEAD = "DEAD"
LAST_PROVENANCE = PROV_DEAD
LAST_GATE = None
LAST_ROUTE = None
LAST_G1 = None
LAST_WHITE = None
LAST_SPINE = None


def build(model_id, lora_id, task=None):
    class _Plan:
        twin_run = True
    return _Plan()


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _usable(v, n):
    return v is not None and np.asarray(v).shape == (n,) \
        and np.all(np.isfinite(v)) and float(np.ptp(v)) > 1e-9


def _agentic_gate(messages):
    """internal."""
    msgs = messages or []
    prior = msgs[:-1]
    n_asst = sum(1 for m in prior if m.get("role") == "assistant")
    if n_asst < 2:
        return False
    txt = "\n".join(str(m.get("content") or "") for m in prior)
    roles = {m.get("role") for m in msgs}
    tools = bool(RE_REACT.search(txt) or RE_TAGS.search(txt)
                 or (roles & {"tool", "function", "tool_response"}))
    sys_frame = bool(RE_SYS.search(next((str(m.get("content") or "") for m in msgs
                                         if m.get("role") == "system"), "")))
    return tools or (sys_frame and n_asst >= 3)


def _decoder_layers(model, model_id):
    """internal."""
    from util import decoder_layers
    if "Nemotron" in (model_id or ""):
        try:
            return model.model.layers
        except AttributeError:
            cands = []
            for name, child in model.model.named_modules():
                if name.rsplit(".", 1)[-1] != "layers":
                    continue
                kids = list(child.children())
                if kids:
                    cands.append((len(kids), name, child))
            if cands:
                cands.sort(key=lambda t: -t[0])
                return cands[0][2]
            raise
    return decoder_layers(model)


def _white_leg(build_model_fn, rows, *, remote=True, batch_size=8):
    """internal. Suspect-side activation read: per-row top3mean projection margins."""
    import torch
    n = len(rows)
    model_id = rows[0].get("model")
    lora_id = rows[0].get("lora")
    probe = wp._load_probe()
    fam = wp._family(model_id)
    if f"{fam}::d" not in probe:
        print(f"[{NAME}] white leg: no direction for this family — spine only", flush=True)
        return None
    if str(probe.get(f"{fam}::scope", "all")) not in ("all", "gated"):
        print(f"[{NAME}] white leg: family scope unusable — spine only", flush=True)
        return None
    plan = wp._Plan(probe, fam)
    model = build_model_fn(model_id, lora_id, plan.needs_native_config)
    tok = model.tokenizer
    DL = _decoder_layers(model, model_id)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    toks, valid = [], []
    for i, ex in enumerate(rows):
        ids, s, e = plan.tokenize(ex["messages"], tok, MAXLEN)
        if e > s:
            toks.append((ids, s, e))
            valid.append(i)
    if len(valid) < max(1, n // 2):
        print(f"[{NAME}] white leg: only {len(valid)}/{n} rows tokenizable — spine only",
              flush=True)
        return None
    order = sorted(range(len(toks)), key=lambda i: len(toks[i][0]))
    batches = [order[b0:b0 + batch_size] for b0 in range(0, len(order), batch_size)]
    with model.session(remote=remote):
        pieces, poses = [], []
        for bpos in batches:
            bt = [toks[i] for i in bpos]
            W = max(len(ids) for ids, _, _ in bt)
            ii = torch.full((len(bt), W), pad, dtype=torch.long)
            am = torch.zeros((len(bt), W), dtype=torch.long)
            for j, (ids, _, _) in enumerate(bt):
                ii[j, :len(ids)] = torch.tensor(ids)
                am[j, :len(ids)] = 1
            spans = [(s, e) for _, s, e in bt]
            with model.trace({"input_ids": ii, "attention_mask": am}) as tracer:
                fb = plan.extract(DL, spans)
                tracer.stop()
            pieces.append(fb)
            poses.append(torch.tensor(bpos))
        allf = torch.cat(pieces).save()
        allp = torch.cat(poses).save()
    fv = np.asarray(allf.cpu().float().numpy())
    pv = np.asarray(allp.cpu().numpy())
    feats = np.zeros((len(valid), fv.shape[1]))
    feats[pv] = fv
    x = np.full(n, np.nan)
    x[np.array(valid)] = feats[:, 2]                     # top3mean projection margin
    fin = np.isfinite(x)
    if fin.sum() and float(np.nanstd(x)) > 1e-9:
        x[~fin] = float(np.nanmedian(x))
        return x
    print(f"[{NAME}] white leg: degenerate projection — spine only", flush=True)
    return None


def _claim_route(build_model_fn, rows, gate, *, remote=True):
    """internal. Second, measured gate on the AGENTIC rows only: jm06's claim-extractability
    forward read (trusted judge, byte-identical prompt). White reads the claim-FREE agentic
    rows (concealment reports — measured win 0.87-0.92); claim-bearing agentic rows (the
    denial-to-auditor subtype, where the white read is null-dominated) stay on the spine.
    Dead read -> None (caller falls back to routing ALL gated rows white)."""
    gi = np.where(gate)[0]
    sub = [rows[i] for i in gi]
    judge = build_model_fn(jp.JUDGE_MODEL, None)
    g1 = np.asarray(jg._gate_margins(judge, judge.tokenizer, sub, remote=remote), float)
    if g1.shape != (len(sub),) or not np.all(np.isfinite(g1)):
        raise RuntimeError("g1 margins malformed")
    route = gate.copy()
    route[gi] = g1 <= 0.0
    full_g1 = np.full(len(rows), np.nan)
    full_g1[gi] = g1
    return route, full_g1


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    global LAST_PROVENANCE, LAST_GATE, LAST_ROUTE, LAST_G1, LAST_WHITE, LAST_SPINE
    n = len(index)
    rows = [examples[i] for i in range(n)]
    gate = np.array([_agentic_gate(ex.get("messages")) for ex in rows], bool)
    LAST_GATE = gate.copy()
    LAST_ROUTE = None
    LAST_G1 = None
    print(f"[{NAME}] gate: {int(gate.sum())}/{n} rows agentic", flush=True)

    p_spine = np.asarray(jsp.run(build_model_fn, rows, index, task, remote=remote,
                                 batch_size=batch_size), float)
    spine_ok = _usable(p_spine, n)
    LAST_SPINE = p_spine.copy() if p_spine is not None else None

    if not gate.any():
        LAST_PROVENANCE = PROV_SPINE_ONLY
        LAST_WHITE = None
        if spine_ok:
            return p_spine
        LAST_PROVENANCE = PROV_DEAD
        print(f"[{NAME}] spine dead on an ungated cell -> 0.5", flush=True)
        return np.full(n, 0.5)

    route = gate
    try:
        route, LAST_G1 = _claim_route(build_model_fn, rows, gate, remote=remote)
        print(f"[{NAME}] claim gate: {int(route.sum())}/{int(gate.sum())} agentic rows "
              f"claim-free -> white; the rest stay on the spine", flush=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        route = gate
        print(f"[{NAME}] claim gate died ({type(e).__name__}: {str(e)[:120]}) -> "
              f"all agentic rows white", flush=True)
    LAST_ROUTE = route.copy()

    if not route.any():
        LAST_PROVENANCE = PROV_SPINE_ONLY
        LAST_WHITE = None
        if spine_ok:
            return p_spine
        LAST_PROVENANCE = PROV_DEAD
        print(f"[{NAME}] spine dead, nothing routed white -> 0.5", flush=True)
        return np.full(n, 0.5)

    white = None
    try:
        t0 = time.time()
        white = _white_leg(build_model_fn, rows, remote=remote, batch_size=batch_size)
        if white is not None:
            print(f"[{NAME}] white leg read {n} rows in {time.time()-t0:.0f}s", flush=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        white = None
        print(f"[{NAME}] white leg died ({type(e).__name__}: {str(e)[:140]}) -> spine only",
              flush=True)
    LAST_WHITE = white.copy() if white is not None else None

    if not _usable(white, n):
        LAST_PROVENANCE = PROV_SPINE_ONLY
        if spine_ok:
            return p_spine
        LAST_PROVENANCE = PROV_DEAD
        print(f"[{NAME}] spine AND white dead -> 0.5", flush=True)
        return np.full(n, 0.5)

    if not spine_ok:
        if route.all():
            LAST_PROVENANCE = PROV_GATED_WHITE
            print(f"[{NAME}] spine dead, fully white-routed cell -> white only", flush=True)
            return js._judge_prob(_z(white), 0.0)
        LAST_PROVENANCE = PROV_DEAD
        print(f"[{NAME}] spine dead on a partially-routed cell -> 0.5", flush=True)
        return np.full(n, 0.5)

    x_spine = 2.0 * np.log(np.clip(p_spine, 1e-6, 1 - 1e-6)
                           / (1 - np.clip(p_spine, 1e-6, 1 - 1e-6)))
    comp = np.where(route, _z(white), _z(x_spine))
    LAST_PROVENANCE = PROV_GATED_WHITE
    return js._judge_prob(comp, float(np.median(comp)))
