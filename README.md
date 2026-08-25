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
# uv, if you don't have it: curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EleutherAI/how-to-catch-an-ai-liar
cd how-to-catch-an-ai-liar
uv sync && source .venv/bin/activate

python run.py --method abantes --dataset wf_assert --limit 16
```

**CUDA build.** `uv.lock` pins a CUDA 13 torch wheel; on an older driver
`torch.cuda.is_available()` is then `False`. `run.py --backend local` refuses to start in that
case rather than falling back to CPU silently. If it does, install a torch matching your driver
into the same environment, e.g. for CUDA 12.6:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

`run.py` runs any of the three frozen submission methods (`abantes`, `polypheides`, `ampyx`)
on any dataset:

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

**The suspect is coupled to the data.** Each transcript was generated on one model — the
suspect — and Ampyx's self-evaluation and activation legs interrogate that model, so the
loaded data and the interrogated suspect must be the same. `run.py` enforces this: it reads
the suspect *from the dataset*, so you never audit a Gemma reply by asking Qwen. Concretely,
a dataset is a `(variant, suspect)` pair:

```bash
# a variant handle + --suspect selects the edition (n_suspects x n_variants datasets):
python run.py --method ampyx --dataset notus-lie-auditor --suspect gemma-3-27b-it
python run.py --method ampyx --dataset notus-lie-auditor   # default suspect Qwen3.5-27B

# a full slug carries its suspect in the suffix (authoritative):
python run.py --method ampyx --dataset reinthal/notus-lie-auditor-NVIDIA-Nemotron-3-Super-120B

# a battery distribution's suspect is taken from its own gen_model
python run.py --method ampyx --dataset dyl_varied_gemma3   # -> gemma-3-27b-it
```

Suspect keys: `Qwen3.5-27B`, `gemma-3-27b-it`, `NVIDIA-Nemotron-3-Super-120B` (or a full HF
id / `model:lora`). Judge-only methods (Abantes, Polypheides) never load the suspect, so its
identity doesn't affect them. For Ampyx budget ~55 GB bf16 for a 27B suspect on top of
the ~20 GB judge (both fit one 80 GB H100); the Nemotron-120B suspect (~240 GB) is NDIF-only.
Give it room: both models are dispatched with `device_map="auto"`, so on a multi-GPU box the
judge can spread across every visible device and leave the suspect short. Measured on this
bundle, 4x48 GB makes the anchor OOM and the run degrade to a plain judge, while 7x48 GB runs
the full stack — so pin the judge and the suspect to different devices with
`CUDA_VISIBLE_DEVICES`, and use `--strict`.

Other flags: `--limit N` scores only the first N rows; `--batch-size` (default 8) trades
VRAM for speed; `--backend ndif` runs the models remotely on NDIF, for those with an
`NDIF_API_KEY`; `--blind` forces judge-legs-only, but Ampyx degrades to its judge spine
without a suspect — not recommended; `--strict` exits non-zero if any leg of the method fell
back instead of running.

Every run also prints a `health:` line naming what each leg actually did. This matters: these
methods are built to degrade rather than crash, so a run whose anchor died and was replaced by
a plain judge still prints a clean-looking AUROC. `jm43=TWO_PASS` means the anchor ran;
`PLAIN_FALLBACK` means it did not, and the number you are reading is not the method's.

Try `wf_assert` vs `wf_denial` to see the claim-shape
effect from the post: the same lies with an extractable false claim (caught) and as bare
denials (missed).

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

### Ampyx — white-box gated stack (`methods/ampyx/`)

Our final white-box entry: notebook `Ampyx.ipynb`, method `rhadamanthys_m53r`, submitted
2026-07-30 01:51 UTC — board balanced accuracy **0.8892**, AUROC **0.9450**. A black-box judge
spine (the truth-anchored judge fused with a suspect self-evaluation leg) wrapped by a gated
activation leg: a linear direction read from the suspect's residual stream, trained with the
deception *instruction* held constant on both sides of the contrast so it cannot encode the
instruction.

The gate is on the **data**, not the organism: a row qualifies only if the record shows
tool-interaction evidence with at least two prior assistant turns, **and** the trusted judge
reports that the final reply asserts no checkable factual claim. That is the
record-versus-report case, where the deceptive reply is a terse denial and the honest one a long
confession, so transcript-only readers invert.

**Scope fact, quoted with sources in `methods/ampyx/PROVENANCE.md`: that gate selected zero rows
on all six counted cells.** The counted genres are not agentic tool-record transcripts, so the
white leg was dormant on the scored data and the board number above is the black spine's. The
activation read is exercised on the agentic distributions in the bundled battery
(`aud_agentic`, `ga_insider`, `lb_insider`) — not on the cells that produced the headline.

`rhadamanthys_m53r` is a three-line module: it binds `run`/`build` to the same function objects
as its predecessor `rhadamanthys_m53`, so healthy cells are byte-identical by construction, and
adds the one re-export (`judge_rescue`) that keeps a cell whose `run()` raised from being written
out as a constant vector. The bundle also keeps `rhadamanthys_m53_tp`, the earlier **Tiresias**
submission (0.8904 / 0.9451, 2026-07-20) — the same scorer under a rank-centering shell;
`PROVENANCE.md` carries its board row.

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
cd methods/abantes        # or methods/polypheides, methods/ampyx
DATASET_NAME=reinthal/notus-lie-auditor-Qwen3.5-27B python local_test.py
```

Add `ALETHEIA_LIMIT=16` for a smoke run; each bundle's default `ALETHEIA_METHOD` is its
own submission. Ampyx loads the suspect model as well as the judge (a 27B suspect wants
2×48 GB GPUs of its own; the Nemotron-120B cells are NDIF-only territory). The `RUN.md` files also document the
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
the blog post. Highlight:

- **`reinthal/notus-lie-auditor-{Qwen3.5-27B, gemma-3-27b-it, NVIDIA-Nemotron-3-Super-120B}`**
  (+ `-labels`) — 899 on-policy agentic lie-auditor transcripts in the Järviniemi & Hubinger
  AI-audit shape, based off Apollo Research's 2023 strategic-deception paper
  ([arXiv:2311.07590](https://arxiv.org/abs/2311.07590)), on the three competition base models.

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
methods/ampyx/         white-box submission (rhadamanthys_m53r): notebook, method stack, probe artifacts (.npz), provenance
examples/              local-GPU Inspect port of the DYAD anchor + metrics script
datasets/              the 27-distribution evaluation battery: data/, manifest, loader, docs
DATASETS.md            dataset inventory and battery manifest summary
pyproject.toml         one dependency set for the whole local workflow (uv sync)
```

Team EleutherAI: Stella Binderman (EleutherAI), Gonçalo Paulo (EleutherAI), Giuseppe
Birardi (Orma Lab Srl), Alexander Reinthal (independent researcher).

Every method file is byte-identical to the frozen competition submission; provenance (original
paths, source commit, board attribution) is recorded per bundle in `PROVENANCE.md`. The one
exception — two stale markdown cells in `Polypheides.ipynb` — is recorded in that bundle's
`PROVENANCE.md`.

## Licence

The **code** is MIT — see [`LICENSE`](LICENSE).

The **datasets** are not ours to relicense: each of the 27 distributions is derived from an
upstream corpus and keeps that corpus's terms. [`datasets/LICENSES.md`](datasets/LICENSES.md)
records them per family, including the two things a reuser has to know — `lb_alpaca` inherits
Stanford Alpaca's **non-commercial** CC-BY-NC-4.0, and the Liars' Bench and AISI families (18 of
the 27) are published as `license: other` with no stated terms. The five distributions we built
ourselves — `ga_domains`, `b2_pressure`, `wf_assert`, `wf_denial`, `aud_agentic` — are MIT like
the code, and on their own carry the post's central claim-shape comparison and the agentic set.

## Reproducibility note

`uv.lock` pins released versions (nnsight 0.7.0, transformers 5.14.1). The competition runs used
development builds of both (nnsight 0.7.1.dev41, transformers 5.15.0.dev0) against the NDIF
deployment of the day. These methods trace models by module path, and those paths have changed
between library versions, so local numbers can differ from the board rows. The `health:` line is
what tells you whether a leg fell back — which is the difference that actually matters.
