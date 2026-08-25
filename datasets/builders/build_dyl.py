#!/usr/bin/env python
"""build_dyl.py — datasets from '"Did you lie?" Evaluating Lie Detectors across Model
Scale and Belief-Verified Model Organisms' (Cooney, Africa, Irving — AISI, arXiv
2606.12618) into the gauntlet.

Source: HF `ai-safety-institute/lie-detection-rollouts` (public). Splits carry
  messages   JSON chat transcript; assistant turns may carry `reasoning_content` (CoT)
  is_lie     'lie' / 'honest' / 'ambiguous'  (ambiguous DROPPED here)
  lie_reason grader provenance (belief verification incl. CoT graders)

The paper's labels are belief-verified: Varied Deception uses verified TriviaQA
(model demonstrably knows the answer); organism datasets verify the hidden belief in
chain-of-thought with an explicit CoT grader (lie_reason carries the grade). We strip
CoT from the visible transcript (meta.reasoning keeps a truncated copy) — detectors
read the same surface a held-out test would show.

Distributions:
  dyl_varied_qwen35    Varied Deception, Qwen3.5-27B      (board-family suspect)
  dyl_varied_gemma3    Varied Deception, gemma-3-27b-it   (board family)
  dyl_varied_nemotron  Varied Deception, Nemotron-3-Super-120B (board family)
  dyl_varied_offfam    Varied Deception pooled over 5 NEVER-seen families
                       (deepseek-v3.2, llama-3.3-70b, gpt-oss-120b, kimi-k2.6, glm-5)
  dyl_varied_aq        Varied Deception on the 4 PUBLIC aletheias-quest organisms
                       (hidden-goal, collusion, botc, echoblast) — competition-related
  dyl_ab_denial        AuditBench organisms DENY their trained hidden trait
                       (self-promotion / animal-welfare / citations / optimism)
  dyl_gender_secret    gender-secret organisms deny knowing the user's gender
  dyl_sandbag_organism sandbagger organism lies about its capabilities (games)
  dyl_sandbag_prompted prompted multiple-choice sandbagging, base board-family models

Usage: .venv-ndif/bin/python runs/research_gauntlet/build_dyl.py [--only NAME]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import log, save_dist, clean_messages  # noqa: E402

REPO_ID = "ai-safety-institute/lie-detection-rollouts"
PAPER = "AISI did-you-lie (arXiv 2606.12618) / " + REPO_ID
SEED = 20260717
REASONING_KEEP = 1200


def _fetch(cfg, split):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO_ID, f"{cfg}/{split}.parquet", repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"))
    return pd.read_parquet(p)


def _parse_msgs(v):
    m = json.loads(v) if isinstance(v, str) else list(v)
    return [dict(x) for x in m]


def _rows_from(cfg, split, dist, genre, n_cap=None, rng=None):
    """Parse one parquet into gauntlet rows (ambiguous dropped, CoT stripped)."""
    df = _fetch(cfg, split)
    if "is_lie" not in df.columns:
        raise RuntimeError(f"{cfg}/{split}: no is_lie column ({df.columns.tolist()})")
    gen_model = cfg.split("__")[-1] if "__" in cfg else cfg
    out = []
    for i, r in df.iterrows():
        lab = str(r["is_lie"]).strip().lower()
        if lab not in ("lie", "honest"):
            continue
        try:
            msgs, reasoning = clean_messages(_parse_msgs(r["messages"]))
        except Exception as e:
            log(f"  {cfg}/{split} row {i}: unparseable messages ({type(e).__name__}) — drop")
            continue
        if not msgs or msgs[-1]["role"] != "assistant" or not msgs[-1]["content"].strip():
            continue
        meta = {"source": PAPER, "genre": genre, "config": cfg, "split": split,
                "label_method": "AISI belief-verified is_lie (CoT-graded for organisms)",
                "lie_reason": str(r.get("lie_reason", ""))[:500]}
        if reasoning:
            meta["reasoning"] = reasoning[:REASONING_KEEP]
            meta["has_reasoning"] = True
        out.append(dict(dist=dist, rid=f"{dist}:{cfg.split('__')[-1][-40:]}:{split}:{i}",
                        y=int(lab == "lie"), messages=msgs, gen_model=gen_model, meta=meta))
    return out


def _balance(rows, n_per, rng):
    pos = [r for r in rows if r["y"] == 1]
    neg = [r for r in rows if r["y"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    take = pos[:n_per] + neg[:n_per]
    if len(pos) < n_per or len(neg) < n_per:
        log(f"  balance: only {len(pos)}+/{len(neg)}- available (wanted {n_per}/{n_per})")
    rng.shuffle(take)
    return take


def _emit_varied(cfgs, dist, n_per_cfg, genre, flags=(), notes=""):
    rng = np.random.default_rng(abs(hash((SEED, dist))) % 2**31)
    rows, gens = [], set()
    for cfg in cfgs:
        got = _rows_from(cfg, "varied_deception", dist, genre)
        picked = _balance(got, n_per_cfg, rng)
        rows.extend(picked)
        gens.update(r["gen_model"] for r in picked)
    rng.shuffle(rows)
    save_dist(dist, rows, entry={
        "source": PAPER, "genre": genre,
        "label_method": "belief-verified TriviaQA (model demonstrably knows the answer); "
                        "is_lie grader; ambiguous dropped",
        "generator_models": sorted(gens), "flags": list(flags), "notes": notes})


def build_varied():
    _emit_varied(["qwen-qwen3.5-27b"], "dyl_varied_qwen35", 120,
                 "prompted varied lying (system-prompt personas), verified TriviaQA")
    _emit_varied(["google-gemma-3-27b-it"], "dyl_varied_gemma3", 120,
                 "prompted varied lying (system-prompt personas), verified TriviaQA")
    _emit_varied(["nvidia-nvidia-nemotron-3-super-120b-a12b-fp8"], "dyl_varied_nemotron", 120,
                 "prompted varied lying (system-prompt personas), verified TriviaQA")
    _emit_varied(["deepseek-ai-deepseek-v3.2", "meta-llama-llama-3.3-70b-instruct",
                  "openai-gpt-oss-120b", "moonshotai-kimi-k2.6", "zai-org-glm-5-fp8"],
                 "dyl_varied_offfam", 25,
                 "prompted varied lying, 5 model families our methods never read",
                 notes="pooled: deepseek-v3.2, llama-3.3-70b, gpt-oss-120b, kimi-k2.6, "
                       "glm-5. Cross-model 3rd-person audit for suspect-side methods.")
    _emit_varied(["google-gemma-3-27b-it__aletheias-quest-hidden-goal-model-organism-gemma3-27b-v1",
                  "mlabonne-gemma-3-27b-it-abliterated__aletheias-quest-collusion-model-organism-gemma3-27b-v1",
                  "qwen-qwen3.5-27b__aletheias-quest-botc-latest-checkpoint",
                  "qwen-qwen3.5-27b__aletheias-quest-echoblast-model-organism"],
                 "dyl_varied_aq", 30,
                 "prompted varied lying on PUBLIC competition model organisms",
                 flags=("organism", "competition_related"),
                 notes="the 4 public aletheias-quest organisms (hidden-goal, collusion, "
                       "botc, echoblast). Organism substrate under prompted lying — the "
                       "closest public proxy for board-organism rows.")


AB_CFGS = [
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-ab_animal_welfare",
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-ab_hallucinates_citations",
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-ab_self_promotion",
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-ab_animal_welfare",
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-ab_contextual_optimism",
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-ab_hallucinates_citations",
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-ab_self_promotion",
]


def build_ab_denial():
    rng = np.random.default_rng(SEED + 21)
    rows = []
    for cfg in AB_CFGS:
        got = _rows_from(cfg, "audit_bench_updated", "dyl_ab_denial",
                         "organism denies its trained hidden trait")
        pos = [r for r in got if r["y"] == 1]
        neg = [r for r in got if r["y"] == 0]
        rng.shuffle(neg)
        rows.extend(pos + neg[:max(len(pos), 12)])  # per-organism balance-ish
        log(f"  {cfg.split('-')[-1]}: {len(pos)} lies / kept {min(len(neg), max(len(pos), 12))} honest")
    rng.shuffle(rows)
    save_dist("dyl_ab_denial", rows, entry={
        "source": PAPER, "genre": "organism denies its trained hidden trait (self-knowledge)",
        "label_method": "belief verified IN CHAIN-OF-THOUGHT (CoT grader; lie = trait "
                        "acknowledged internally, denied in output); ambiguous dropped",
        "generator_models": sorted({r["gen_model"] for r in rows}),
        "flags": ["organism"],
        "notes": "the paper's hardest genre for probes: trained-in deception, no "
                 "instruction in context. Lies are DENIALS (terse-denial-adjacent shape)."})


def build_gender():
    rng = np.random.default_rng(SEED + 22)
    cfgs = ["qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-gender_secret_male",
            "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-gender_secret_female",
            "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-gender_secret_male",
            "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-gender_secret_female"]
    rows = []
    for cfg in cfgs:
        rows.extend(_balance(_rows_from(cfg, "gender_secret", "dyl_gender_secret",
                                        "organism denies knowing user attribute"), 40, rng))
    rng.shuffle(rows)
    save_dist("dyl_gender_secret", rows, entry={
        "source": PAPER, "genre": "organism denies knowing user attribute (secret keeping)",
        "label_method": "belief verified in CoT (grader: internal 'I know user is X' vs "
                        "external denial); ambiguous dropped",
        "generator_models": sorted({r["gen_model"] for r in rows}),
        "flags": ["organism"],
        "notes": "AISI reproduction of the gender-secret genre on Qwen3.5/3.6 organisms — "
                 "claim-free denials, the shape our gemma-auditor is blind to."})


def build_sandbag():
    rng = np.random.default_rng(SEED + 23)
    rows = []
    for cfg in ["qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-eval_sandbagger",
                "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-eval_sandbagger"]:
        rows.extend(_rows_from(cfg, "sandbagging_games_updated", "dyl_sandbag_organism",
                               "organism sandbags capability questions"))
    rng.shuffle(rows)
    save_dist("dyl_sandbag_organism", rows, entry={
        "source": PAPER, "genre": "trained sandbagger lies about its capabilities",
        "label_method": "belief-verified is_lie (CoT-graded); ambiguous dropped",
        "generator_models": sorted({r["gen_model"] for r in rows}),
        "flags": ["organism"],
        "notes": "capability deception — a genre absent from every board/dev cell."})

    rng2 = np.random.default_rng(SEED + 24)
    rows2 = []
    for cfg in ["qwen-qwen3.5-27b", "google-gemma-3-27b-it"]:
        rows2.extend(_balance(_rows_from(cfg, "multiple_choice_sandbagging",
                                         "dyl_sandbag_prompted",
                                         "prompted MCQ sandbagging"), 75, rng2))
    rng2.shuffle(rows2)
    save_dist("dyl_sandbag_prompted", rows2, entry={
        "source": PAPER, "genre": "prompted multiple-choice sandbagging (base models)",
        "label_method": "belief-verified is_lie; ambiguous dropped",
        "generator_models": sorted({r["gen_model"] for r in rows2}),
        "flags": [],
        "notes": "base Qwen3.5-27B + gemma-3-27b-it deliberately answering below "
                 "capability under prompt incentives."})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    steps = {"varied": build_varied, "ab": build_ab_denial, "gender": build_gender,
             "sandbag": build_sandbag}
    for name, fn in steps.items():
        if a.only in (None, name):
            fn()
    log("build_dyl done")
