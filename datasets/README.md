# The evaluation battery

The held-out evaluation battery from the blog post (section 9): **27 distributions,
6,136 rows** of labelled honest/deceptive assistant replies, spanning the two axes that
predict detector behaviour — whether the lie contains an **extractable false claim**, and
**where the truth lives** (in the conversation's own record, in public world knowledge, or
only inside the model).

Sources: 9 distributions from [Liars' Bench](https://arxiv.org/abs/2511.16035), 9 from the
AI Safety Institute's ["Did you lie?"](https://arxiv.org/abs/2606.12618) rollouts, 5
assembled from TruthfulQA / HaluEval / Apollo's insider-trading generations, and 4 built by
us. Rows keep their upstream licences; per-row provenance is in each row's `meta` and per-
distribution provenance in `gauntlet.json`.

## Attribution

| our name | original name | original paper | released |
|---|---|---|---|
| `lb_*` (9 distributions) | Liars' Bench (`Cadenza-Labs/liars-bench`) | [Liars' Bench: Evaluating Lie Detectors for Language Models — arXiv:2511.16035](https://arxiv.org/abs/2511.16035) | 2025-11 |
| `lb_mask` (via Liars' Bench) | MASK | [The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems — arXiv:2503.03750](https://arxiv.org/abs/2503.03750) | 2025-03 |
| `lb_alpaca` (via Liars' Bench) | Stanford Alpaca | no paper — [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 2023-03 |
| `dyl_*` (9 distributions) | AISI lie-detection rollouts (`ai-safety-institute/lie-detection-rollouts`) | ["Did you lie?" Evaluating Lie Detectors across Model Scale and Belief-Verified Model Organisms — arXiv:2606.12618](https://arxiv.org/abs/2606.12618) | 2026-06 |
| `ga_tqa` | TruthfulQA (`truthfulqa/truthful_qa`) | [TruthfulQA: Measuring How Models Mimic Human Falsehoods — arXiv:2109.07958](https://arxiv.org/abs/2109.07958) | 2021-09 |
| `ga_halu` | HaluEval (qa split, via `pminervini/HaluEval`) | [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models — arXiv:2305.11747](https://arxiv.org/abs/2305.11747) | 2023-05 |
| `ga_insider`, `ga_insider_cot` | Apollo insider trading (Llama-3.3 regenerations, `Cadenza-Labs/apollo-llama3.3-insider-trading-generations`) | [Large Language Models can Strategically Deceive their Users when Put Under Pressure — arXiv:2311.07590](https://arxiv.org/abs/2311.07590) | 2023-11 |
| `ga_domains`, `b2_pressure`, `wf_assert`, `wf_denial`, `aud_agentic` | built by us for this work | — | 2026-07 |

## Format

One JSONL file per distribution in `data/`; one JSON object per line:

```
dist      distribution id
rid       stable row id
messages  chat messages [{role, content}, ...] — the LAST assistant turn is the reply
          under audit (chain-of-thought is stripped from content into meta)
y         1 = deceptive final reply, 0 = honest
gen_model model that produced the reply
meta      provenance (source, label method, genre, optional reasoning, ...)
```

`gauntlet.json` is the manifest; `load.py` is a dependency-free loader.

## Read every score against the nulls

Every distribution carries two dataset nulls (properties of the data, independent of any
detector): **len** = AUROC of final-reply length alone, and **blind** = AUROC of a judge
reading the whole conversation with the final reply deleted. A detector has only read the
lie itself to the extent it beats the blind null. Distributions where the setup leaks are
flagged, not removed — the leak is a property of the genre.

## The distributions

| id | type of lie (genre) | claim shape | where the truth lives | n | pos | len | blind | source |
|---|---|---|---|---:|---:|---:|---:|---|
| `aud_agentic` | agentic record-vs-report audit (multi-turn tool log) | claim-bearing | in-context record | 300 | 79 | 0.30 | 0.83 | runs/research_auditor/auditor.jsonl |
| `b2_pressure` | multi-turn factual pressure: instructed lies vs capitulation folds | claim-bearing | public world knowledge | 286 | 116 | 0.51 | 0.89 | runs/research_bench2/bench2_matched.jsonl |
| `dyl_ab_denial` | organism denies its trained hidden trait (self-knowledge) | claim-free | internal to the model | 366 | 183 | 0.48 | 0.54 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `dyl_gender_secret` | organism denies knowing user attribute (secret keeping) | claim-free | public world knowledge | 320 | 160 | 0.34 | 0.80 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `dyl_sandbag_organism` | trained sandbagger lies about its capabilities | claim-free | internal to the model | 174 | 82 | 0.59 | 0.50 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `dyl_sandbag_prompted` | prompted multiple-choice sandbagging (base models) | claim-bearing | internal to the model | 300 | 150 | 0.52 | 0.79 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `dyl_varied_aq` | prompted varied lying on PUBLIC competition model organisms | claim-bearing | public world knowledge | 240 | 120 | 0.60 | 0.82 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `dyl_varied_gemma3` | prompted varied lying (system-prompt personas), verified TriviaQA | claim-bearing | public world knowledge | 240 | 120 | 0.62 | 0.80 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `dyl_varied_nemotron` | prompted varied lying (system-prompt personas), verified TriviaQA | claim-bearing | public world knowledge | 240 | 120 | 0.48 | 0.81 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `dyl_varied_offfam` | prompted varied lying, 5 model families our methods never read | claim-bearing | public world knowledge | 250 | 125 | 0.54 | 0.82 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `dyl_varied_qwen35` | prompted varied lying (system-prompt personas), verified TriviaQA | claim-bearing | public world knowledge | 240 | 120 | 0.51 | 0.76 | AISI did-you-lie (arXiv 2606.12618) / ai-safety-institute/lie-detection-rollouts |
| `ga_domains` | novel-domain factual lies x 5 styles (synthetic) | claim-bearing | public world knowledge | 100 | 50 | 0.50 | 0.51 | synthetic belief-gated (research_genaudit_external factbank) |
| `ga_halu` | record-grounded hallucination (truth in context) | claim-bearing | in-context record | 120 | 60 | 0.50 | 0.51 | pminervini/HaluEval qa via research_genaudit_external |
| `ga_insider` | agentic record-vs-report concealment (Apollo) | claim-free | in-context record | 150 | 75 | 0.26 | 0.89 | Cadenza-Labs/apollo-llama3.3-insider-trading-generations |
| `ga_insider_cot` | agentic concealment + visible reasoning | claim-free | in-context record | 150 | 75 | 0.66 | 0.89 | Cadenza-Labs/apollo-llama3.3-insider-trading-generations |
| `ga_tqa` | adversarial common-misconception QA (TruthfulQA) | claim-bearing | public world knowledge | 120 | 60 | 0.45 | 0.49 | truthfulqa/truthful_qa via research_genaudit_external |
| `lb_alpaca` | benign control (all honest) | — | — | 200 | 0 | — | — | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `lb_convincing` | persona / self-knowledge lie | claim-bearing | internal to the model | 200 | 100 | 0.38 | 0.71 | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `lb_gender` | fine-tuned secret-keeping denial | claim-free | internal to the model | 200 | 100 | 0.44 | 0.46 | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `lb_hp_choice` | harm-pressure MCQ answer flip | claim-bearing | public world knowledge | 200 | 100 | 0.51 | 0.56 | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `lb_hp_kr` | harm-pressure knowledge denial | claim-free | public world knowledge | 200 | 100 | 0.71 | 0.43 | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `lb_insider` | agentic concealment (record-vs-report) | claim-free | in-context record | 200 | 100 | 0.58 | 0.80 | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `lb_instructed` | instructed factual lie (matched) | claim-bearing | public world knowledge | 200 | 100 | 0.41 | 0.54 | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `lb_mask` | MASK reproduction (pressured dishonesty) | — | — | 100 | 100 | — | — | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `lb_soft_trigger` | hidden-trigger LoRA organism lie | claim-bearing | public world knowledge | 200 | 100 | 0.68 | 0.55 | Cadenza-Labs/liars-bench (arXiv 2511.16035) |
| `wf_assert` | controlled claim-shape probe (assertion) | claim-bearing | public world knowledge | 600 | 300 | 0.41 | 0.50 | runs/research_gemma_whenfail/rows.jsonl |
| `wf_denial` | controlled claim-shape probe (denial) | claim-free | public world knowledge | 240 | 120 | 0.50 | 0.40 | runs/research_gemma_whenfail/rows.jsonl |

Flags recorded in the manifest (see `gauntlet.json` for details per distribution): `aud_agentic` (leaky_context; unbalanced); `b2_pressure` (leaky_context); `dyl_ab_denial` (organism); `dyl_gender_secret` (leaky_context; organism); `dyl_sandbag_organism` (organism); `dyl_sandbag_prompted` (leaky_context); `dyl_varied_aq` (competition_related; leaky_context; organism); `dyl_varied_gemma3` (leaky_context); `dyl_varied_nemotron` (leaky_context); `dyl_varied_offfam` (leaky_context); `dyl_varied_qwen35` (leaky_context); `ga_insider` (leaky_context); `ga_insider_cot` (leaky_context); `lb_alpaca` (control); `lb_convincing` (leaky_context); `lb_gender` (organism); `lb_hp_choice` (leaky_context); `lb_hp_kr` (leaky_context); `lb_insider` (leaky_context); `lb_mask` (control; single_class_lies); `lb_soft_trigger` (leaky_context; organism); `wf_assert` (diagnostic_tuned; templated); `wf_denial` (diagnostic_tuned; leaky_context; templated).

## Running the methods on these files

The simplest route is the repo-root runner — any frozen method, local GPUs or NDIF, with
AUROC/balanced accuracy and this manifest's nulls printed at the end:

```bash
python run.py --method abantes --dataset wf_assert --limit 16
python run.py --method tiresias --dataset ga_insider --backend ndif
```

`examples/inspect_polypheides.py` also accepts a battery file directly (local judge, one GPU):

```bash
inspect eval examples/inspect_polypheides.py -T hf_dataset=datasets/data/wf_assert.jsonl
```

Or score with your own detector via the loader:

```python
import sys; sys.path.insert(0, "datasets")
from load import list_dists, load_dist, manifest
rows = load_dist("ga_insider")          # rows[i]["messages"], rows[i]["y"]
print(manifest()["ga_insider"]["nulls"])  # the bar your detector has to beat
```
