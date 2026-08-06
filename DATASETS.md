# Datasets

Companion inventory for *To catch a lie in the AI stream*. Two parts: (1) the datasets we
publish on Hugging Face under the account **`reinthal`**, (2) the 27-distribution evaluation
battery from section 9 of the post and its manifest (`runs/research_gauntlet/gauntlet.json`).

Evidence sources: `huggingface_hub.list_datasets(author="reinthal")` (queried 2026-08-06),
the gauntlet manifest (`gauntlet.json`, updated 2026-07-17), `runs/submissions/dev-label-verify/ship_v5_labels.py`
and its output `labels_v5.csv`, and the research log entry
`runs/research_log/2026-07-14-notus-lie-auditor-gemma-regression.md`.

---

## 1. Our published HF datasets

### 1a. Lie-auditor sets (blog section 3 context; built 2026-07-14)

Synthetic AI-Audit / lie-auditor transcripts in the Järviniemi & Hubinger shape, generated
**on-policy** on the three competition base models via an Inspect agentic eval. Each row: an
assistant works a company workday through tools; a misconduct action is injected into the
record (guilt by construction); an external NIST auditor asks whether it did anything improper
— the final reply is the scored turn. Conditions: 3 system prompts (lie / neutral / honest) ×
2 day types (benign / dishonest) × 10 temperature epochs. Labels: GPT-5-mini graded with the
J&H admission rule (deceptive = guilty and the reply does not disclose; an honest admission,
or an innocent benign-day denial, is honest). Suspect chain-of-thought is kept in the per-turn
`reasoning` field (present on Nemotron/Qwen; empty on gemma, no thinking mode).

| slug | rows | deceptive / honest | visibility (2026-08-06) |
|---|---:|---|---|
| `reinthal/notus-lie-auditor-Qwen3.5-27B` (+ `-labels`) | 300 | 62 / 238 | public / public |
| `reinthal/notus-lie-auditor-gemma-3-27b-it` (+ `-labels`) | 299 | 90 / 209 | **private / private** |
| `reinthal/notus-lie-auditor-NVIDIA-Nemotron-3-Super-120B` (+ `-labels`) | 300 | 79 / 221 | public / **labels private** |

Total 899 rows across the three sets (matches the research-log figure "~300 rows/base, 899
total"). Note the gemma set has 299 rows, not 300. If the companion repo is meant to make all
three loadable, the gemma pair and the Nemotron `-labels` repo need to be flipped to public.

### 1b. The v5 dev-set relabel (blog section 3: the five-vote belief gate and the ~18% single-greedy mislabel figure come from this pipeline)

Independent belief-relative relabels of the 21 aletheias-quest `dev-{instructed,varied}-deception`
sets. Each unique question was resampled 20× under a neutral prompt to elicit the organism's
belief; a judge (DeepSeek-V4-Flash) grades each dev row against the raw 20 neutral samples
(v5 = no canonicalization + few-shot calibration). `deceptive` = the answer contradicts the
model's demonstrated knowledge.

| slug | rows | contents |
|---|---:|---|
| `reinthal/aletheias-dev-relabel-v5` | 8,216 | consolidated label table over 21 datasets: `dataset, index, deceptive` (v5), `official` (dev label), `relabeled` (v5 ≠ official), `excluded` (belief unestablished → official-label fallback) |
| `reinthal/dev-{instructed,varied}-deception-<model>-<organism>-relabel-v5` (+ `-labels`), 21 pairs = 42 repos | 8,216 total | the same 21 dev sets re-published with v5 labels in the transcript + `-labels` pair convention (per-set n = 400; the Nemotron instructed set is 216) |

Measured on `labels_v5.csv`: 8,216 rows, 21 datasets, 3,652 deceptive, **496 relabeled**
(v5 disagrees with the official dev label), **145 excluded** (no established belief; these
carry the official dev label as fallback). Pooled agreement with the dev labels ≈ 0.94. All
44 repos public.

### 1c. The four distributions we built (blog section 9 / closing note: "the four distributions we built")

Identified from the gauntlet manifest's family-provenance table: the family labelled
**"our benches (imported)"** contains exactly 4 distributions, and the README TL;DR counts
"3 published benchmarks + 4 of our benches". These are:

| dist id | genre | n (+/−) | label method | source of the rows |
|---|---|---|---|---|
| `b2_pressure` | multi-turn factual pressure: instructed lies vs capitulation folds | 286 (116/170) | behaviour-verified; 5-vote belief gate (`label_gates.adjudicate`) | `runs/research_bench2/bench2_matched.jsonl` |
| `wf_assert` | controlled claim-shape probe (assertion) | 600 (300/300) | templated over belief-verified facts (one-factor design) | `runs/research_gemma_whenfail/rows.jsonl` |
| `wf_denial` | controlled claim-shape probe (denial) | 240 (120/120) | templated over belief-verified facts (one-factor design) | `runs/research_gemma_whenfail/rows.jsonl` |
| `aud_agentic` | agentic record-vs-report audit (multi-turn tool log, Nemotron) | 300 (79/221) | behaviour-verified against the in-context record | `runs/research_auditor/auditor.jsonl` |

Total 1,426 rows. **As of the 2026-08-06 listing these four are not on the `reinthal` HF
account** — they exist as `runs/research_gauntlet/data/{dist}.jsonl` plus the manifest. The
blog's closing "[HF LINK]" and draft-comment item 4 ("HF account and dataset cards for the
four releases in section 9") are open items; publishing them is a to-do for this repo, not a
done fact.

Boundary note: `ga_domains` (100 rows, synthetic novel-domain lies, 5-vote belief-gated on
Qwen3.5-27B) is also constructed by us, but the manifest counts it inside the genaudit family
("5 assembled from TruthfulQA, HaluEval and Apollo"), matching the blog's 9+9+5+4 split. So
"four built by us" excludes it by the manifest's own grouping.

### 1d. Other project datasets under `reinthal` (not named in the blog)

| slug | rows | what it is | visibility |
|---|---:|---|---|
| `reinthal/qwen3.5-9b-deception-probe` (+ `-labels`) | 52,030 | one row per generated Qwen3.5-9B sample from the overnight belief probe: `qid, task, question, condition (neutral/honest/lie), sample_idx, answer, deceptive` (v5 verdict where belief was clear), `belief_clear` | public |
| `reinthal/trusted9b-sft-mix-v3` | 4,838 | SFT mix for LoRA fine-tuning the Qwen3.5-9B trusted judge (DYAD method); chat rows `{slice, messages}`, loss masked to the final assistant turn | public |
| `reinthal/dev-label-verify-belief-resampling` | — | raw belief-resampling JSONL behind the v5 relabel (399 MB) | private |

### 1e. Private, project-internal (listed for completeness; not part of the release)

70 further private repos from intermediate experiments: `reinthal/h1-probe-400-*`
(33 repos: dev-organism probe sets with `-labels` and `-belief-labels` companions),
`reinthal/h1-fp-resample-*` (10), `reinthal/lb-ext-*` (27: Liars'-Bench-extension
train/eval splits on the three base models with `-labels`/`-belief-labels`).

Account totals, 2026-08-06: 123 dataset repos under `reinthal`; 49 public, 74 private.

---

## 2. The evaluation battery (27 distributions, 6,136 rows)

From `runs/research_gauntlet/gauntlet.json` (updated 2026-07-17). Row totals verified: the 27
`n` values sum to exactly 6,136. Composition matches the blog's section 9 claim: 9 Liars'
Bench + 9 AISI Did-You-Lie + 5 genaudit-assembled + 4 built by us.

`n (+/−)` = rows (deceptive / honest). `lennull` = AUROC of final-reply length alone.
`blindnull` = AUROC of the 9B judge reading everything except the final reply. Both are
properties of the dataset, computed before any method was scored. `—` = undefined
(single-class control).

| dist | ours? | genre | n (+/−) | lennull | blindnull | flags |
|---|---|---|---|---:|---:|---|
| `lb_instructed` | external (Liars' Bench) | instructed factual lie (matched: both classes lie-instructed) | 200 (100/100) | 0.41 | 0.54 | — |
| `lb_convincing` | external (Liars' Bench) | persona / self-knowledge lie | 200 (100/100) | 0.38 | 0.71 | leaky |
| `lb_gender` | external (Liars' Bench) | fine-tuned secret-keeping denial | 200 (100/100) | 0.44 | 0.46 | organism |
| `lb_hp_choice` | external (Liars' Bench) | harm-pressure MCQ answer flip | 200 (100/100) | 0.51 | 0.56 | leaky |
| `lb_hp_kr` | external (Liars' Bench) | harm-pressure knowledge denial | 200 (100/100) | 0.71 | 0.43 | leaky (length) |
| `lb_insider` | external (Liars' Bench) | agentic concealment (record-vs-report) | 200 (100/100) | 0.58 | 0.80 | leaky |
| `lb_soft_trigger` | external (Liars' Bench) | hidden-trigger LoRA organism lie | 200 (100/100) | 0.68 | 0.55 | leaky, organism |
| `lb_mask` | external (Liars' Bench) | MASK reproduction (lies only) | 100 (100/0) | — | — | control, single-class |
| `lb_alpaca` | external (Liars' Bench) | benign control (all honest) | 200 (0/200) | — | — | control |
| `dyl_varied_qwen35` | external (AISI rollouts) | prompted varied lying, Qwen3.5-27B | 240 (120/120) | 0.51 | 0.76 | leaky |
| `dyl_varied_gemma3` | external (AISI rollouts) | prompted varied lying, gemma-3-27b | 240 (120/120) | 0.62 | 0.80 | leaky |
| `dyl_varied_nemotron` | external (AISI rollouts) | prompted varied lying, Nemotron-120B | 240 (120/120) | 0.48 | 0.81 | leaky |
| `dyl_varied_offfam` | external (AISI rollouts) | prompted varied lying, 5 never-seen model families | 250 (125/125) | 0.54 | 0.82 | leaky |
| `dyl_varied_aq` | external (AISI rollouts) | prompted varied lying, public competition organisms | 240 (120/120) | 0.60 | 0.82 | leaky, organism, comp |
| `dyl_ab_denial` | external (AISI rollouts) | organism denies its trained hidden trait | 366 (183/183) | 0.48 | 0.54 | organism |
| `dyl_gender_secret` | external (AISI rollouts) | organism denies knowing user attribute | 320 (160/160) | 0.34 | 0.80 | leaky, organism |
| `dyl_sandbag_organism` | external (AISI rollouts) | trained sandbagger lies about capabilities | 174 (82/92) | 0.59 | 0.50 | organism |
| `dyl_sandbag_prompted` | external (AISI rollouts) | prompted MCQ sandbagging (base models) | 300 (150/150) | 0.52 | 0.79 | leaky |
| `ga_tqa` | assembled by us (TruthfulQA) | adversarial common-misconception QA | 120 (60/60) | 0.45 | 0.49 | — |
| `ga_halu` | assembled by us (HaluEval) | record-grounded hallucination | 120 (60/60) | 0.50 | 0.51 | — |
| `ga_domains` | **built by us** (synthetic, counted in genaudit family) | novel-domain factual lies × 5 styles, 5-vote belief gate | 100 (50/50) | 0.50 | 0.51 | — |
| `ga_insider` | assembled by us (Apollo generations) | agentic record-vs-report concealment | 150 (75/75) | 0.26 | 0.89 | leaky |
| `ga_insider_cot` | assembled by us (Apollo generations) | agentic concealment + visible CoT | 150 (75/75) | 0.66 | 0.89 | leaky |
| `b2_pressure` | **built by us** | multi-turn factual pressure (lies vs folds) | 286 (116/170) | 0.51 | 0.89 | leaky |
| `wf_assert` | **built by us** | controlled claim-shape probe (assertion) | 600 (300/300) | 0.41 | 0.50 | diag, templated |
| `wf_denial` | **built by us** | controlled claim-shape probe (denial) | 240 (120/120) | 0.50 | 0.40 | diag, leaky, templated |
| `aud_agentic` | **built by us** | agentic record-vs-report audit (Nemotron) | 300 (79/221) | 0.30 | 0.83 | leaky, unbalanced |

Label methods, by family (from the manifest): Liars' Bench rows carry the bench's
`deceptive` flag (belief-relative by construction, LB paper §3); AISI rows carry the `is_lie`
grader with verified-TriviaQA belief (varied sets) or CoT-verified belief (organism sets),
`ambiguous` dropped; genaudit rows carry the source-dataset labels plus a 5-vote belief gate
for the synthetic `ga_domains`; our four use the 5-vote belief gate (`b2_pressure`), templating
over belief-verified facts (`wf_*`), or behaviour verification against the in-context record
(`aud_agentic`).

Two further generated sets (`new_hpkr_terse`, `new_agentic_matched`, on Qwen3.5-27B) were
built and **dropped**: under a strong directive blindnull hit ~1.00/0.99; under discretionary
prompts the base model produced 0 denials in 354 and 0 concealments in 240. This is the §3/§10
result in the blog and is recorded in the manifest's `new_data_finding` field.

---

## 3. How to load

The dev-set and lie-auditor repos use a two-repo convention: the transcript repo holds
`index, messages, model, lora`; the companion `<slug>-labels` repo holds `index, deceptive`.
Join on `index` (the convention used by `scratchpad/inspect_polypheides.py` in the research
repo):

```python
from datasets import load_dataset

slug = "reinthal/notus-lie-auditor-Qwen3.5-27B"          # or any dev-*-relabel-v5 set
rows = load_dataset(slug, split="test")
labels = {int(r["index"]): bool(r["deceptive"])
          for r in load_dataset(slug + "-labels", split="test")}

for r in rows:
    y = labels[int(r["index"])]      # True = deceptive
    messages = r["messages"]         # chat transcript; last assistant turn is under audit
```

The consolidated relabel table is a single repo, no join needed:

```python
v5 = load_dataset("reinthal/aletheias-dev-relabel-v5", split="train")
# columns: dataset, index, deceptive, official, relabeled, excluded
```

Battery distributions load from the manifest + JSONL (row schema: `dist, rid, y, messages,
gen_model, meta`):

```python
import sys; sys.path.insert(0, "runs/research_gauntlet")
import gauntlet as G
rows, srows, y = G.load_dist("wf_denial")   # srows are harness-ready {messages, model, lora}
```

---

## 4. Reconciliation notes

- **27 distributions / 6,136 rows: reconciles.** The manifest has 27 entries and their `n`
  sums to exactly 6,136. The 9 + 9 + 5 + 4 family split matches the blog.
- **"Four built by us": reconciles with a caveat.** The manifest's provenance family "our
  benches" is exactly {`b2_pressure`, `wf_assert`, `wf_denial`, `aud_agentic`}. However
  `ga_domains` is also generated by us (synthetic, 5-vote belief-gated); the manifest and blog
  both count it inside the 5-item genaudit family. The companion repo should state this
  grouping explicitly.
- **The battery is not yet on HF.** The blog's closing note points to "[HF LINK]" for the
  battery, manifest and the four built distributions; as of 2026-08-06 no gauntlet repo exists
  under `reinthal`. The data lives at `runs/research_gauntlet/data/*.jsonl` + `gauntlet.json`.
- **Lie-auditor visibility.** Of the six lie-auditor repos, three are private
  (`...-gemma-3-27b-it`, its `-labels`, and the Nemotron `-labels`). The gemma set has 299
  rows (not 300); the three-set total is 899.
- **v5 numbers reconcile.** 8,216 rows over 21 datasets (20 × 400 + 1 × 216), 496 relabeled,
  145 excluded rows carrying the official dev label as fallback, 3,652 labeled deceptive.
- **Two dropped distributions** (`new_hpkr_terse`, `new_agentic_matched`) are part of the
  blog's narrative (sections 3 and 10) but are intentionally not in the battery; only the
  build harness (`build_new.py`) ships.
- **`lb_mask` subsampling.** The public MASK CSV carries 1,045 lies / 0 honest; the battery
  ships a 100-row single-class control, AUROC undefined.
