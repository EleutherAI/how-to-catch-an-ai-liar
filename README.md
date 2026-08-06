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

Everything runs locally with nnsight — models on your own GPUs, no NDIF account or API
key. The judge (Qwen3.5-9B) needs one ~24 GB GPU. One dependency set (`pyproject.toml`):

```bash
git clone https://github.com/reinthal/how-to-catch-an-ai-liar
cd how-to-catch-an-ai-liar
uv sync && source .venv/bin/activate

python run.py --method abantes --dataset wf_assert --limit 16
```

`run.py` runs any of the three frozen submission methods on any dataset:

```bash
# --dataset takes a bundled battery id (all 27 in datasets/data/, docs in datasets/README.md) ...
python run.py --method abantes --dataset wf_assert

# ... a path to a battery-format .jsonl ...
python run.py --method polypheides --dataset datasets/data/lb_instructed.jsonl

# ... or an HF slug following the <slug> + <slug>-labels convention
python run.py --method polypheides --dataset reinthal/notus-lie-auditor-Qwen3.5-27B
```

Every run writes `submission.csv` (`index, deceptive, score` — the competition deliverable)
and, when labels exist, prints AUROC / balanced accuracy **plus the dataset's recorded
blind and length nulls** — the bar a detector has to beat before claiming it read the lie
rather than the setup.

**The suspect model.** Every run has one: if the dataset names a suspect (the HF sets
carry `model`/`lora` columns) that is used; otherwise — all battery datasets — it defaults
to `Qwen/Qwen3.5-27B`. Never run without a suspect: methods that read the suspect
(Tiresias' self-evaluation and activation legs) silently degrade to their judge spine
without one. Judge-only methods (Abantes, Polypheides) never load it, so the default costs
them nothing; for Tiresias budget ~55 GB bf16 on top of the judge (both fit one 80 GB
H100). Override with `--suspect "model[:lora]"`.

Other flags: `--limit N` scores only the first N rows; `--batch-size` (default 8) trades
VRAM for speed; `--backend ndif` runs the models remotely on NDIF instead, for those with
an `NDIF_API_KEY`; `--blind` exists to force judge-legs-only, but see above — not
recommended. Try `wf_assert` vs `wf_denial` to see the claim-shape effect from the post:
the same lies with an extractable false claim (caught) and as bare denials (missed).

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

## Other entry points

`examples/inspect_polypheides.py` is a faithful [Inspect](https://inspect.aisi.org.uk/)
port of the DYAD anchor (same prompts, same mirrored logit read; the only fidelity gap is
reading top-20 logprobs instead of raw logits) — useful for browsing per-row transcripts,
judge outputs and margins in a UI:

```bash
inspect eval examples/inspect_polypheides.py -T hf_dataset=datasets/data/wf_assert.jsonl
inspect view
```

For exact reproduction of the competition runs, each method directory is a standalone
submission bundle with its own entry points — `local_test.py` (local GPUs, same models)
and `remote_test.py` (NDIF, exactly as submitted; needs `NDIF_API_KEY`):

```bash
cd methods/abantes        # or methods/polypheides, methods/tiresias
DATASET_NAME=reinthal/notus-lie-auditor-Qwen3.5-27B python local_test.py
```

Add `ALETHEIA_LIMIT=16` for a smoke run; each bundle's default `ALETHEIA_METHOD` is its
own submission. Tiresias loads the suspect model (a 27B suspect wants 2×48 GB GPUs; the
Nemotron-120B cells are NDIF-only territory). The `RUN.md` files also document the
original competition dev-set invocations (`aletheias-quest/dev-*` slugs), which may not
remain publicly downloadable.

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
run.py                 batteries-included runner: any method x any dataset, local or NDIF
methods/abantes/       black-box submission (dyad_plainfuse): notebook, methods, harness, provenance
methods/polypheides/   the DYAD anchor inside Abantes + basis of the local Inspect example
methods/tiresias/      white-box submission: method stack, probe artifacts (.npz), harness, provenance
examples/              local-GPU Inspect port of the DYAD anchor + metrics script
datasets/              the 27-distribution evaluation battery: data/, manifest, loader, docs
DATASETS.md            dataset inventory and battery manifest summary
pyproject.toml         one dependency set for the whole local workflow (uv sync)
```

Team EleutherAI: Stella Binderman (EleutherAI), Gonçalo Paulo (EleutherAI), Giuseppe
Birardi (Orma Lab Srl), Alexander Reinthal (independent researcher).

Every method file is byte-identical to the frozen competition submission; provenance (original
paths, source commit, board attribution) is recorded per bundle in `PROVENANCE.md`.
