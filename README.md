# How to catch an AI liar

Companion repository for the blog post *To catch a lie in the AI stream* — lessons from one
month of detecting AI lies during **Aletheia's Quest**, a deception-detection competition run
on [NDIF](https://ndif.us/). It contains our two main competition submissions, runnable in
minimal form, and an inventory of the datasets we publish.

The competition: organisers fine-tuned model organisms to deceive, served them behind an API,
and asked entrants to score every transcript between honest and deceptive. Entries were ranked
by mean AUROC over six held-out cells (two organism families × three base models, 400 rows
each). Entrants never saw the organisms and never read a labelled test row.

## Quick start

One GPU (~24 GB), no NDIF account needed — run our black-box method with a local
Qwen3.5-9B judge on one of the bundled datasets:

```bash
git clone https://github.com/reinthal/how-to-catch-an-ai-liar
cd how-to-catch-an-ai-liar
uv venv --python 3.12 .venv-local && source .venv-local/bin/activate
uv pip install -r requirements-local.txt

inspect eval examples/inspect_polypheides.py -T hf_dataset=datasets/data/wf_assert.jsonl
inspect view    # browse per-row transcripts, judge outputs, margins, scores
```

`datasets/data/` contains all 27 evaluation distributions ([docs](datasets/README.md));
swap in any of them, or an HF slug like `reinthal/notus-lie-auditor-Qwen3.5-27B`. Try
`wf_assert` vs `wf_denial` to see the claim-shape effect from the post: the same lies with
an extractable false claim (caught) and as bare denials (missed). To run the methods
through the competition's NDIF harness instead, see [Quick start (NDIF)](#quick-start-ndif).

## The two methods

### Abantes — black-box submission (`methods/abantes/`)

Our final black-box entry: notebook `Abantes.ipynb`, method `dyad_plainfuse`, submitted
2026-07-30 05:55 UTC — board balanced accuracy **0.8321**, AUROC **0.9260** (our best
black-box AUROC; it superseded Polypheides below). It fuses two judge channels, both
black-box and per-sample:

- **Anchor**: the pinned DYAD pipeline (`rhadamanthys_dyad_anchor2h_shield_clean` — the
  Polypheides method, run verbatim).
- **Cheap channel**: `judge_ynd`, a single batched next-token yes/no log-odds read per row,
  no generation.
- **Fusion**: a fixed anchor-dominant logit blend, `sigmoid(logit(pin) + 0.5·ynd/2)` —
  deliberately per-sample constants, no cross-row statistics; every failure path returns
  the anchor's own scores unchanged.

See `methods/abantes/PROVENANCE.md` (per-cell board results, fired-stage byte
verification) and `RUN.md`.

### Polypheides — the DYAD anchor (`methods/polypheides/`)

The earlier pinned black-box (bal 0.8688 / AUROC 0.9237, submitted 2026-07-25) and the
anchor inside Abantes; kept as its own bundle because it is the method the local Inspect
example ports. A 4-pass "DYAD anchor" pipeline around a trusted 9B judge
(`Qwen/Qwen3.5-9B`):

1. **Blind truth anchor** — the suspect's final reply is withheld and the judge states, from
   its own knowledge, what a correct answer would be.
2. **Style-strip restatement** — the judge neutrally restates the suspect's actual reply,
   removing stylistic tokens that dilute the signal.
3. + 4. **Mirrored verdict reads** — the judge answers "was this deceptive: (A)/(B)" twice
   with the letter meanings swapped, read as logits at the answer position. The margin
   `0.5·(d1 − d2)` cancels any constant preference for either letter, giving a per-row zero
   decision threshold — no batch statistics, no tuned cutoff.

See `methods/polypheides/PROVENANCE.md` and `RUN.md`.

### Tiresias — white-box gated stack (`methods/tiresias/`)

A 3-shell method: a black-box judge spine (truth-anchored judge fused with a self-evaluation
leg) wrapped by a gated activation leg (a linear direction read from the suspect's residual
stream, trained with the deception *instruction* held constant on both sides of the contrast
so it cannot encode the instruction), wrapped by a rank-centering post-processor.

Board result: balanced accuracy **0.8904**, AUROC **0.9451** (submitted 2026-07-20).

Two recorded scope facts, quoted with sources in `methods/tiresias/PROVENANCE.md`: the
activation leg's gate selected zero rows on all six counted cells (the white leg was dormant
on the scored data), and the wrapper's in-batch keep-band/median statistics are recorded
in-repo as non-compliant with the organisers' later fixed-per-sample-threshold ruling.
Polypheides is the compliant lineage under that ruling.

## Quick start (NDIF)

The quick start above uses `examples/inspect_polypheides.py`, a faithful
[Inspect](https://inspect.aisi.org.uk/) port of the submission (same prompts, same mirrored
logit read; the only fidelity gap is reading top-20 logprobs instead of raw logits). It takes
one parameter — a battery file or an HF dataset slug — and `examples/sweep_metrics.py`
computes balanced accuracy / AUROC tables from the eval logs.

The frozen harness methods themselves also run **fully locally** — same pipeline, no NDIF
account, models dispatched onto your GPUs (`methods/*/local_test.py`; Polypheides needs
only the 9B judge, ~20 GB; Tiresias also loads the suspect model, so a 27B suspect wants
2×48 GB and the Nemotron-120B cells are NDIF-only territory):

```bash
cd methods/polypheides
ALETHEIA_METHOD=rhadamanthys_dyad_anchor2h \
DATASET_NAME=aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4 \
  python local_test.py     # add ALETHEIA_LIMIT=8 for a smoke run
```

To reproduce the competition runs exactly as submitted, go through the NDIF harness — each
method directory is a standalone submission bundle with the harness entry point included:

```bash
uv venv --python 3.12 .venv && UV_TORCH_BACKEND=cpu uv pip install -r requirements-ndif.txt
cd methods/polypheides    # or methods/tiresias
NDIF_API_KEY=... HF_TOKEN=... \
ALETHEIA_METHOD=rhadamanthys_dyad_anchor2h \
DATASET_NAME=aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4 \
  python remote_test.py
```

Writes `submission.csv` (`index, deceptive, score` — the competition deliverable). Add
`ALETHEIA_LIMIT=16` for a smoke run. For Tiresias use
`ALETHEIA_METHOD=rhadamanthys_m53_tp`; details and caveats in each bundle's `RUN.md`.

## The datasets

**The evaluation battery ships in this repo**: `datasets/data/` holds all 27 distributions
(6,136 labelled rows, ~16 MB) with the manifest (`datasets/gauntlet.json`), a
dependency-free loader (`datasets/load.py`), and per-distribution documentation of the
lie types, claim shapes, evidence loci and dataset nulls in
[`datasets/README.md`](datasets/README.md). Run Polypheides on any of them directly:

```bash
inspect eval examples/inspect_polypheides.py -T hf_dataset=datasets/data/wf_denial.jsonl
```

`DATASETS.md` is the wider inventory: the datasets we publish on Hugging Face under
[`reinthal`](https://huggingface.co/reinthal) plus the battery's reconciliation against
the blog post. Highlights:

- **`reinthal/notus-lie-auditor-{Qwen3.5-27B, gemma-3-27b-it, NVIDIA-Nemotron-3-Super-120B}`**
  (+ `-labels`) — 899 on-policy agentic lie-auditor transcripts in the Järviniemi & Hubinger
  AI-audit shape, on the three competition base models.
- **`reinthal/aletheias-dev-relabel-v5`** — an independent belief-relative relabel of all
  8,216 competition dev rows (20 neutral resamples per question as belief evidence; 496 rows
  relabeled vs the official labels), plus the 21 per-dataset repos.

Datasets follow a transcript/`-labels` pair convention, joined on `index`:

```python
from datasets import load_dataset
rows   = load_dataset("reinthal/notus-lie-auditor-Qwen3.5-27B", split="test")
labels = load_dataset("reinthal/notus-lie-auditor-Qwen3.5-27B-labels", split="test")
```

## Repository layout

```
methods/abantes/       black-box submission (dyad_plainfuse): notebook, methods, harness, provenance
methods/polypheides/   the DYAD anchor inside Abantes + basis of the local Inspect example
methods/tiresias/      white-box submission: method stack, probe artifacts (.npz), harness, provenance
examples/              local-GPU Inspect port of the DYAD anchor + metrics script
datasets/              the 27-distribution evaluation battery: data/, manifest, loader, docs
DATASETS.md            dataset inventory and battery manifest summary
requirements-*.txt     NDIF-client and local-GPU dependency sets
```

Team EleutherAI: Stella Binderman (EleutherAI), Gonçalo Paulo (EleutherAI), Giuseppe
Birardi (Orma Lab Srl), Alexander Reinthal (independent researcher).

Every method file is byte-identical to the frozen competition submission; provenance (original
paths, source commit, board attribution) is recorded per bundle in `PROVENANCE.md`.
