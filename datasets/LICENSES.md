# Licences and redistribution status of the battery

The code in this repository is MIT (`../LICENSE`). **The 27 distributions under `data/` are not
ours to relicense.** Each row is derived from an upstream corpus and keeps that corpus's terms.
This file records, per family, what those terms are — and, where we could not establish them, says
so instead of guessing.

Status checked against the Hugging Face dataset API and the upstream repositories on
**2026-08-25**. Re-check before relying on any line here.

## Summary

| family | dists | upstream | licence as published | redistribution here |
|---|---:|---|---|---|
| `lb_*` | 9 | `Cadenza-Labs/liars-bench` | HF tag `other`; **dataset repo is gated** | ⚠ **unresolved — see below** |
| `dyl_*` | 9 | `ai-safety-institute/lie-detection-rollouts` | HF tag `other`, no terms in the card | ⚠ **unresolved — see below** |
| `ga_tqa` | 1 | `truthfulqa/truthful_qa` | **Apache-2.0** | ✅ permitted with attribution |
| `ga_halu` | 1 | `pminervini/HaluEval` | **Apache-2.0** | ✅ permitted with attribution |
| `ga_insider`, `ga_insider_cot` | 2 | `Cadenza-Labs/apollo-llama3.3-insider-trading-generations` | **no licence tag** | ⚠ unresolved |
| `lb_mask` (via Liars' Bench) | — | MASK (`cais/MASK`) | **no licence tag** | ⚠ unresolved |
| `lb_alpaca` (via Liars' Bench) | — | Stanford Alpaca (`tatsu-lab/alpaca`) | **CC-BY-NC-4.0** | ⚠ **non-commercial only** |
| `ga_domains`, `b2_pressure`, `wf_assert`, `wf_denial`, `aud_agentic` | 5 | built by us | — | ✅ MIT, same as the code |

## The three things a reuser must know

**1. `lb_alpaca` is non-commercial.** Stanford Alpaca is published under CC-BY-NC-4.0. Anything
derived from it — including the `lb_alpaca` honest-control distribution here — inherits the
non-commercial restriction. It is a single-class control (200 rows, all honest) and is excluded
from every AUROC in this repo, so dropping it costs nothing but the control.

**2. The two largest families have unresolved terms.** `Cadenza-Labs/liars-bench` (9 dists) and
`ai-safety-institute/lie-detection-rollouts` (9 dists) are both tagged `license: other` on the Hub,
and neither dataset card states what "other" means. The Liars' Bench **code** repository
(`github.com/Cadenza-Labs/liars-bench`) is Apache-2.0, but a code licence does not carry over to
the dataset. Additionally, **the `Cadenza-Labs/liars-bench` dataset repo on Hugging Face is gated**
— it requires authentication and a granted access request. Redistributing 9 distributions derived
from a gated dataset is a decision for the upstream authors to confirm, not one this file can
settle.

**3. Two more upstreams carry no licence tag at all**: the Apollo insider-trading generations
(`ga_insider`, `ga_insider_cot`) and MASK (reproduced inside Liars' Bench as `lb_mask`). Absence of
a tag is not a grant.

## What is unambiguously reusable today

- The **code**: MIT.
- The **five distributions we built** — `ga_domains`, `b2_pressure`, `wf_assert`, `wf_denial`,
  `aud_agentic` (1,526 rows) — MIT, same as the code. `wf_assert` and `wf_denial` are the
  controlled claim-shape pair the blog post's central comparison rests on, and `aud_agentic` is the
  agentic record-versus-report set that exercises the white leg, so the repo's main experiments
  can be reproduced from this subset alone.
- **`ga_tqa` and `ga_halu`**: Apache-2.0 upstream, permitted with attribution.

## Attribution

Cite the upstream work, not this repository, when you use a distribution derived from it:

| our name | original | paper |
|---|---|---|
| `lb_*` (9) | Liars' Bench (`Cadenza-Labs/liars-bench`) | [arXiv:2511.16035](https://arxiv.org/abs/2511.16035) |
| `lb_mask` | MASK | [arXiv:2503.03750](https://arxiv.org/abs/2503.03750) |
| `lb_alpaca` | Stanford Alpaca | [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) |
| `dyl_*` (9) | AISI lie-detection rollouts | [arXiv:2606.12618](https://arxiv.org/abs/2606.12618) |
| `ga_tqa` | TruthfulQA | [arXiv:2109.07958](https://arxiv.org/abs/2109.07958) |
| `ga_halu` | HaluEval | [arXiv:2305.11747](https://arxiv.org/abs/2305.11747) |
| `ga_insider`, `ga_insider_cot` | Apollo insider trading | [arXiv:2311.07590](https://arxiv.org/abs/2311.07590) |

Per-row provenance is in each row's `meta`; per-distribution provenance is in `gauntlet.json` and
`PROVENANCE.md`.

## Takedown

If you are an upstream author and this repository redistributes your data on terms you did not
grant, open an issue and we will remove the affected distributions. The evaluation harness
(`../gauntlet.py`, `load.py`) works over any subset of `data/`, so removing a family degrades
coverage without breaking anything.
