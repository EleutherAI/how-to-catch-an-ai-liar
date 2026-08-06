"""pm21 — internal detector module.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import pm18 as tc
from methods import pm04 as v3

NAME = "pm21"

RESCUE = float(os.environ.get("ALETHEIA_TWIN_RESCUE", "0.15"))
GATE_FRAC = float(os.environ.get("ALETHEIA_TWIN_GATEFRAC", "0.30"))
K_SLOPE = float(os.environ.get("ALETHEIA_TWIN_KSLOPE", "1.0"))
CHUNK = int(os.environ.get("ALETHEIA_TWIN_CHUNK", "30"))
MAX_NEW = int(os.environ.get("ALETHEIA_TWIN_MAXNEW", str(tc.MAX_NEW)))
FORCE = os.environ.get("ALETHEIA_TWIN_FORCE") == "1"

AUTHOR_MODEL = tc.AUTHOR_MODEL
CAP = tc.CAP
_render = tc._render
_parts = tc._parts
_parse_lines = tc._parse_lines
_parse_json = tc._parse_json
_apply_ops = tc._apply_ops
_variant_ids = tc._variant_ids
_claim_probe = tc._claim_probe
_negbot2 = tc._negbot2
_author_session = tc._author_session
SYS_DECOMP = tc.SYS_DECOMP
SYS_SC = tc.SYS_SC


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
                print(f"[pm21] peft-wrapped tree → decoder layers via search: "
                      f"'{cands[0][1]}' ({cands[0][0]} blocks)", flush=True)
                return cands[0][2]
            raise
    return decoder_layers(model)


def _suspect_session(suspect, stok, rows, plan, probes, yes_ids, no_ids, bs, model_id,
                     remote=True, want_content=True):
    """internal."""
    import torch
    DL = _decoder_layers(suspect, model_id)
    L = int(plan.read_layer)
    residual = plan.read_point == "residual"
    univ = bool(plan.univ)
    mu = torch.tensor(np.asarray(plan.mu), dtype=torch.float32)
    sd = torch.tensor(np.asarray(plan.sd), dtype=torch.float32)
    d = torch.tensor(np.asarray(plan.d), dtype=torch.float32)
    fmu = np.asarray(plan.fmu); fsd = np.asarray(plan.fsd); w = np.asarray(plan.w); b = float(plan.b)
    pad = stok.pad_token_id if stok.pad_token_id is not None else stok.eos_token_id

    corder, cbatches = [], []
    if want_content:
        ctoks = []
        for ex in rows:
            ids, s, e = plan.tokenize(ex["messages"], stok, 256)
            e = min(e, len(ids))
            if e <= s:
                e = s + 1
            ctoks.append((ids, s, e))
        corder = sorted(range(len(ctoks)), key=lambda i: len(ctoks[i][0]))
        for b0 in range(0, len(corder), bs):
            bt = [ctoks[i] for i in corder[b0:b0 + bs]]; W = max(len(ids) for ids, _, _ in bt)
            ii = torch.full((len(bt), W), pad, dtype=torch.long); am = torch.zeros((len(bt), W), dtype=torch.long)
            for j, (ids, _, _) in enumerate(bt):
                ii[j, :len(ids)] = torch.tensor(ids); am[j, :len(ids)] = 1
            cbatches.append((ii, am, [(s, e) for _, s, e in bt]))

    do_ynd = probes is not None and len(probes) > 0
    yorder, ynb = [], []
    if do_ynd:
        yi = torch.tensor(yes_ids); ni = torch.tensor(no_ids); nyes = len(yes_ids)
        penc = [stok(p, add_special_tokens=False)["input_ids"] for p in probes]
        yorder = sorted(range(len(penc)), key=lambda i: len(penc[i]))
        for b0 in range(0, len(yorder), bs):
            el = [penc[i] for i in yorder[b0:b0 + bs]]; T = max(len(e) for e in el)
            ii = torch.full((len(el), T), pad, dtype=torch.long); am = torch.zeros((len(el), T), dtype=torch.long)
            for r, e in enumerate(el):
                ii[r, T - len(e):] = torch.tensor(e, dtype=torch.long); am[r, T - len(e):] = 1
            ynb.append((ii, am))

    all_content = all_yn = None
    with suspect.session(remote=remote):
        if want_content:
            cpieces = []
            for ii, am, spans in cbatches:
                with suspect.trace({"input_ids": ii, "attention_mask": am}) as tracer:
                    h = DL[L].output if residual else DL[L].mlp.down_proj.input
                    if not hasattr(h, "shape"):
                        h = h[0]
                    dev = h.device
                    muD, sdD, dD = mu.to(dev), sd.to(dev), d.to(dev)
                    rowf = []
                    for i, (s, e) in enumerate(spans):
                        pr = ((h[i, s:e] - muD) / sdD) @ dD
                        if univ:
                            rowf.append(pr.mean().reshape(1))
                        else:
                            rowf.append(torch.stack([pr.mean(), pr.max(), pr.topk(min(3, e - s)).values.mean()]))
                    fb = torch.stack(rowf).float(); tracer.stop()
                cpieces.append(fb)
            all_content = torch.cat(cpieces).save()
        if do_ynd:
            ypieces = []
            for ii, am in ynb:
                with suspect.trace({"input_ids": ii, "attention_mask": am}):
                    lg = suspect.output.logits[:, -1].float()
                    yn = torch.cat([lg[:, yi.to(lg.device)], lg[:, ni.to(lg.device)]], dim=1)
                ypieces.append(yn)
            all_yn = torch.cat(ypieces).save()

    content = None
    if want_content:
        fv = np.asarray(all_content.cpu().float().numpy())
        K = 1 if univ else 3
        feats = np.zeros((len(rows), K), np.float32)
        for r in range(fv.shape[0]):
            feats[corder[r]] = fv[r]
        logit = ((feats - fmu) / fsd) @ w + b
        content = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    ynd = None
    if do_ynd:
        yn_t = all_yn.cpu().float()
        ylog = (torch.logsumexp(yn_t[:, :nyes], 1) - torch.logsumexp(yn_t[:, nyes:], 1)).numpy()
        ynd = np.zeros(len(probes))
        for r in range(len(probes)):
            ynd[yorder[r]] = float(ylog[r])
    return content, ynd


def build(model_id, lora_id, task=None):
    """internal."""
    plan = v3.build(model_id, lora_id, task)
    if plan is None:
        return None
    plan.twin_run = True
    return plan


def run(build_model_fn, examples, index, task=None, *, remote=True, batch_size=8):
    """internal."""
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass
    model_id = examples[0]["model"]; lora_id = examples[0].get("lora")
    rows = [examples[i] for i in range(len(index))]
    n = len(rows)

    plan = v3.build(model_id, lora_id, task)
    if plan is None:
        print("[pm21] no content plan → 0.5", flush=True)
        return np.full(n, 0.5)

    suspect = build_model_fn(model_id, lora_id, getattr(plan, "needs_native_config", False))
    stok = suspect.tokenizer
    t0 = time.time()
    content, _ = _suspect_session(suspect, stok, rows, plan, None, None, None,
                                  bs=batch_size, model_id=model_id, remote=remote, want_content=True)
    content = np.clip(np.asarray(content, float), 0.0, 1.0)

    uncertain = np.abs(content - 0.5) < RESCUE
    frac_unc = float(uncertain.mean())
    if not FORCE and frac_unc < GATE_FRAC:
        print(f"[pm21] frac_uncertain {frac_unc:.3f} < GATE {GATE_FRAC} → CONTENT-only "
              f"(1 job, {time.time()-t0:.0f}s)", flush=True)
        return content
    print(f"[pm21] frac_uncertain {frac_unc:.3f} ({int(uncertain.sum())}/{n}){' [FORCED]' if FORCE else ''} "
          f"→ content flat, running TWIN rescue", flush=True)

    yes_ids = _variant_ids(stok, ["yes", "Yes", " yes", " Yes", "YES"])
    no_ids = _variant_ids(stok, ["no", "No", " no", " No", "NO"])
    if not yes_ids or not no_ids:
        print(f"[pm21] tokenizer has no single-token yes/no (yes={len(yes_ids)} no={len(no_ids)}) "
              f"→ CONTENT-only", flush=True)
        return content

    try:
        ta = time.time()
        author = build_model_fn(AUTHOR_MODEL, None); atok = author.tokenizer
        mk_d = lambda ex: _render(atok, [{"role": "system", "content": SYS_DECOMP},
                {"role": "user", "content": f"=== QUESTION ===\n{_parts(ex)[1]}\n\n=== ANSWER ===\n{_parts(ex)[2]}"
                 "\n\nList the claims now, one per line:"}])
        mk_s = lambda ex: _render(atok, [{"role": "system", "content": SYS_SC},
                {"role": "user", "content": f"=== SYSTEM ===\n{_parts(ex)[0]}\n\n=== USER QUESTION ===\n{_parts(ex)[1]}"
                 f"\n\n=== MODEL ANSWER ===\n{_parts(ex)[2]}\n\nReturn the JSON now."}])
        dtexts, stexts = [""] * n, [""] * n
        for c0 in range(0, n, CHUNK):
            cr = list(range(c0, min(c0 + CHUNK, n)))
            dp = [mk_d(rows[i]) for i in cr]; sp = [mk_s(rows[i]) for i in cr]
            dt, st = _author_session(author, atok, dp, sp, bs=batch_size, remote=remote, max_new=MAX_NEW)
            for j, i in enumerate(cr):
                dtexts[i] = dt[j]; stexts[i] = st[j]
        del author
        claims = [_parse_lines(t, CAP) for t in dtexts]
        sc_edit = [_apply_ops(_parts(ex)[0], (_parse_json(t) or {}).get("ops") or []) for ex, t in zip(rows, stexts)]
        print(f"[pm21] author done in {time.time()-ta:.0f}s ({(n + CHUNK - 1)//CHUNK} chunks, "
              f"{sum(len(c or []) for c in claims)} claims total)", flush=True)
    except Exception as e:
        print(f"[pm21] author flow failed ({type(e).__name__}: {str(e)[:120]}) → CONTENT-only", flush=True)
        return content

    probes, owner = [], []
    for ri, ex in enumerate(rows):
        s, u, a = _parts(ex)
        for c in (claims[ri] or []):
            msgs = [{"role": "system", "content": sc_edit[ri]}, {"role": "user", "content": u},
                    {"role": "assistant", "content": a}, {"role": "user", "content": _claim_probe(c)}]
            probes.append(_render(stok, msgs)); owner.append(ri)
    if not probes:
        print("[pm21] author produced no claims → CONTENT-only", flush=True)
        return content

    try:
        ts = time.time()
        _, ynd = _suspect_session(suspect, stok, rows, plan, probes, yes_ids, no_ids,
                                  bs=batch_size, model_id=model_id, remote=remote, want_content=False)
        print(f"[pm21] suspect ynd session done in {time.time()-ts:.0f}s ({len(probes)} claim probes)", flush=True)
    except Exception as e:
        print(f"[pm21] suspect ynd session failed ({type(e).__name__}: {str(e)[:120]}) → CONTENT-only", flush=True)
        return content

    per_row = [[] for _ in rows]
    for v, ri in zip(ynd, owner):
        per_row[ri].append(v)
    negb = np.array([_negbot2(per_row[ri]) for ri in range(n)])
    if float(np.ptp(negb)) < 1e-9:
        print("[pm21] twin signal degenerate → CONTENT-only", flush=True)
        return content

    nz = (negb - negb.mean()) / (negb.std() + 1e-6)
    twin_score = 0.5 + 0.5 * np.tanh(K_SLOPE * nz)
    final = np.where(uncertain, twin_score, content)
    final = np.clip(final, 0.0, 1.0)
    print(f"[pm21] TWIN rescued {int(uncertain.sum())}/{n} uncertain rows "
          f"(range [{final.min():.3f},{final.max():.3f}], total {time.time()-t0:.0f}s)", flush=True)
    return final
