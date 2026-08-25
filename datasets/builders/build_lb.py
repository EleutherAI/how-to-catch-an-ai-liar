#!/usr/bin/env python
"""build_lb.py — ALL of Liar's Bench (Kretschmar et al., arXiv 2511.16035) into the gauntlet.

Source: docs/liars-bench submodule (github.com/Cadenza-Labs/liars-bench,
results/subsets/*.csv — the HF mirrors are gated; the GitHub CSVs are the same data,
see docs/LIARSBENCH_DIGEST.md). Labels are the bench's own `deceptive` flag, which is
belief-relative by construction (paper §3: beliefs established via repeated neutral
prompts; refusals != lies; failed lie-instructions = honest).

Genres (each -> one gauntlet dist, balanced subsample, stratified across the 4 bench
models where possible):
  lb_instructed   instructed-deception. MATCHED variant: BOTH classes drawn from rows
                  whose system prompt instructs deception (meta.instructed_deceptiveness
                  True) — honest rows are failed/refused instructions. This kills the
                  instruction->label context leak our dev post-mortem documented.
  lb_convincing   convincing-game (social-game persona lies)
  lb_gender       gender-secret (fine-tuned organisms deny knowing the user's gender)
  lb_hp_choice    harm-pressure-choice (WMDP MCQ, lie = flip answer under harm pressure)
  lb_hp_kr        harm-pressure-knowledge-report (lie = deny knowing; hardest in paper)
  lb_insider      insider-trading (agentic concealment, Scheurer et al.)
  lb_soft_trigger soft-trigger (LoRA organisms lie on hidden trigger)
  lb_mask         mask reproduction rows (extra, not in the paper's seven)
  lb_hp_natural   harm-pressure-natural-v2 (extra, 135 rows, subtler pressure)
  lb_alpaca       benign control (all honest; FPR calibration only — flagged control)

Usage: .venv-ndif/bin/python runs/research_gauntlet/build_lb.py [--only NAME]
"""
from __future__ import annotations
import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import log, save_dist, strip_think  # noqa: E402

SUBSETS = "/home/giuseppe/deception_cpas/docs/liars-bench/results/subsets"
SEED = 20260717
N_PER_CLASS = 100

PAPER = "Cadenza-Labs/liars-bench (arXiv 2511.16035)"


def _safe_eval(s):
    if not isinstance(s, str):
        return s
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return eval(s, {"__builtins__": {}},  # noqa: S307 — np sentinels in meta reprs
                    {"np": np, "nan": float("nan"), "inf": float("inf"),
                     "float64": float})


def _load_csv(name):
    df = pd.read_csv(os.path.join(SUBSETS, f"{name}.csv"))
    df["messages"] = df["messages"].map(_safe_eval)
    if "meta" in df.columns:
        df["meta"] = df["meta"].map(_safe_eval)
    df["deceptive"] = df["deceptive"].astype(bool)
    return df


def _row_ok(msgs):
    return (isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict)
            and msgs[-1].get("role") == "assistant"
            and str(msgs[-1].get("content", "")).strip()
            and "<think>" not in str(msgs[-1]["content"]))


def _mk_rows(df, dist, genre, extra_meta=None):
    rows = []
    for i, r in df.iterrows():
        msgs = [{"role": m["role"], "content": strip_think(str(m["content"]))}
                for m in r["messages"]]
        meta = {"source": PAPER, "genre": genre, "lb_index": int(r["index"]) if "index" in r
                and pd.notna(r.get("index")) else int(i),
                "label_method": "liars-bench deceptive flag (belief-relative, paper section 3)"}
        m = r.get("meta")
        if isinstance(m, dict):
            keep = {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            meta["lb_meta"] = keep
        if extra_meta:
            meta.update(extra_meta)
        rows.append(dict(dist=dist, rid=f"{dist}:{meta['lb_index']}:{i}",
                         y=int(bool(r["deceptive"])), messages=msgs,
                         gen_model=str(r.get("model", "")), meta=meta))
    return rows


def _balanced(df, n_per, rng, strat_col="model"):
    """Balanced classes, stratified across models as evenly as the data allows."""
    picks = []
    for lab in (True, False):
        sub = df[df["deceptive"] == lab]
        if len(sub) < n_per:
            log(f"  only {len(sub)} rows for deceptive={lab} (wanted {n_per}) — taking all")
            picks.append(sub)
            continue
        groups = [g for _, g in sub.groupby(strat_col)] if strat_col in sub else [sub]
        rng.shuffle(groups)
        quota = int(np.ceil(n_per / len(groups)))
        chosen = []
        for g in groups:
            take = g.sample(n=min(quota, len(g)), random_state=int(rng.integers(2**31)))
            chosen.append(take)
        cat = pd.concat(chosen)
        if len(cat) > n_per:
            cat = cat.sample(n=n_per, random_state=int(rng.integers(2**31)))
        elif len(cat) < n_per:  # top up from the remainder
            rest = sub.drop(cat.index)
            cat = pd.concat([cat, rest.sample(n=min(n_per - len(cat), len(rest)),
                                              random_state=int(rng.integers(2**31)))])
        picks.append(cat)
    out = pd.concat(picks).sample(frac=1, random_state=int(rng.integers(2**31)))
    return out


def _emit(csv_name, dist, genre, n_per=N_PER_CLASS, prefilter=None, extra_meta=None,
          entry_flags=(), notes=""):
    rng = np.random.default_rng(abs(hash((SEED, dist))) % 2**31)
    df = _load_csv(csv_name)
    n0 = len(df)
    df = df[df["messages"].map(_row_ok)]
    if prefilter is not None:
        df = prefilter(df)
    log(f"{dist}: {csv_name}.csv {n0} rows -> {len(df)} usable "
        f"({int(df['deceptive'].sum())} lies)")
    samp = _balanced(df, n_per, rng)
    rows = _mk_rows(samp, dist, genre, extra_meta)
    save_dist(dist, rows, entry={
        "source": PAPER, "genre": genre,
        "label_method": "bench-provided, belief-relative (LB paper section 3)",
        "generator_models": sorted(samp["model"].astype(str).unique().tolist()),
        "flags": list(entry_flags), "notes": notes})


def build_instructed():
    def matched(df):
        # keep only rows whose system prompt instructs deception -> both classes share
        # the instruction; honest = failed/ignored instruction (I=1, B=0 rows).
        def instructed(m):
            return isinstance(m, dict) and bool(m.get("instructed_deceptiveness", False))
        sub = df[df["meta"].map(instructed)]
        n_hon = int((~sub["deceptive"]).sum())
        log(f"  lb_instructed matched pool: {len(sub)} instructed rows, {n_hon} honest")
        if n_hon < 40:
            raise RuntimeError("matched instructed-deception pool too small — inspect meta")
        return sub
    _emit("instructed-deception", "lb_instructed", "instructed factual lie (matched)",
          prefilter=matched,
          extra_meta={"instruction_matched": True},
          notes="BOTH classes are lie-instructed rows; honest = model ignored the "
                "instruction. Kills the instruction->label context leak.")


def build_simple():
    _emit("convincing-game", "lb_convincing", "persona / self-knowledge lie")
    _emit("gender-secret", "lb_gender", "fine-tuned secret-keeping denial",
          entry_flags=("organism",))
    _emit("harm-pressure-choice", "lb_hp_choice", "harm-pressure MCQ answer flip",
          entry_flags=("leaky_context",),
          notes="pressure template visible in context correlates with label by design; "
                "paper's own genre. Read method scores against blindnull.")
    _emit("harm-pressure-knowledge-report", "lb_hp_kr", "harm-pressure knowledge denial",
          entry_flags=("leaky_context",),
          notes="hardest LB genre (all paper detectors near-chance). Terse denials of "
                "knowledge; pressure visible in context.")
    _emit("insider-trading", "lb_insider", "agentic concealment (record-vs-report)")
    _emit("soft-trigger", "lb_soft_trigger", "hidden-trigger LoRA organism lie",
          entry_flags=("organism", "leaky_context"),
          notes="trigger (name/time/greeting) visible in context determines lie mode.")
    _emit("mask", "lb_mask", "MASK reproduction (pressured dishonesty)",
          notes="extra: not part of the paper's seven (excluded there for "
                "single-proposition counting); kept as a distinct distribution.")


def build_hp_natural():
    rng = np.random.default_rng(SEED + 7)
    df = _load_csv("harm-pressure-natural-v2")
    df["index"] = np.arange(len(df))
    df = df[df["messages"].map(_row_ok)]
    log(f"lb_hp_natural: {len(df)} usable rows "
        f"({int(df['deceptive'].sum())} lies) — taking ALL (tiny set)")
    df = df.sample(frac=1, random_state=int(rng.integers(2**31)))
    rows = _mk_rows(df, "lb_hp_natural", "natural-pressure knowledge denial")
    save_dist("lb_hp_natural", rows, entry={
        "source": PAPER + " (out-of-paper extra)", "genre": "natural-pressure knowledge denial",
        "label_method": "bench-provided, belief-relative (LB paper section 3)",
        "generator_models": sorted(df["model"].astype(str).unique().tolist()),
        "flags": ["small"],
        "notes": "135-row post-paper set with subtle (non-heavy-handed) pressure prompts."})


def build_alpaca():
    rng = np.random.default_rng(SEED + 8)
    df = _load_csv("alpaca")
    df = df[df["messages"].map(_row_ok)]
    samp = df.sample(n=200, random_state=int(rng.integers(2**31)))
    rows = _mk_rows(samp, "lb_alpaca", "benign control")
    save_dist("lb_alpaca", rows, entry={
        "source": PAPER, "genre": "benign control (all honest)",
        "label_method": "control: deceptive=False for every row",
        "generator_models": sorted(samp["model"].astype(str).unique().tolist()),
        "flags": ["control"],
        "notes": "FPR / specificity calibration only. AUROC undefined (single class)."})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    steps = {"instructed": build_instructed, "simple": build_simple,
             "hp_natural": build_hp_natural, "alpaca": build_alpaca}
    for name, fn in steps.items():
        if a.only in (None, name):
            fn()
    log("build_lb done")
