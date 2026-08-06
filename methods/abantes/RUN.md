# Running Abantes (`dyad_plainfuse`)

The judge model runs **remotely on NDIF**; the local box only drives it (CPU-only client is fine).

## Setup

Python **3.12** is required (the NDIF server rejects 3.11):

```bash
uv venv --python 3.12 .venv-ndif
UV_TORCH_BACKEND=cpu uv pip install --python .venv-ndif/bin/python -r requirements-dryrun.txt
```

Required environment:

- `NDIF_API_KEY` — NDIF API key (the client sets `CONFIG.API.HOST` to
  `https://aletheias.api.ndif.us`; override with `NDIF_HOST`).
- `HF_TOKEN` — Hugging Face token, needed by `datasets`/`transformers` to load the (gated)
  competition datasets and tokenizer.

## Run one dataset through the competition harness

From this directory:

```bash
NDIF_API_KEY=... HF_TOKEN=... \
ALETHEIA_METHOD=dyad_plainfuse \
DATASET_NAME=aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4 \
  .venv-ndif/bin/python remote_test.py
```

- `DATASET_NAME` is any competition dataset id (single `test` split; rows carry the suspect
  `model`/`lora`, which the harness reads from row 0 — there is no model config file).
- Add `ALETHEIA_LIMIT=16` for a quick smoke run. Note the method's absolute per-cell wall budget
  (`DEADLINE_S=2600` from anchor start): on a full cell the anchor takes most of it; on a smoke run
  both legs fit easily.
- Output: `submission.csv` in the current directory, columns `index, deceptive, score`
  (`deceptive == (score >= 0.5)`; the module hard-sets `ALETHEIA_FIXED_THRESHOLD=1` at import, so
  the decision is a fixed per-sample 0.5 cut, never a batch median). `remote_test.py` is a release
  gate: it also validates the CSV (compliance, non-degenerate scores, raw AUROC when labels are
  present) and exits nonzero on any gate failure.
- Failure behaviour: any failure of the added cheap channel returns the anchor pin's own scores
  byte-identical; only an anchor-level catastrophic failure yields per-row 0.5.

## Offline selftest (no NDIF, no network)

```bash
.venv-ndif/bin/python -m methods.dyad_plainfuse
```

runs the module's built-in 36-check selftest (composition seals, blend formula, health-gate
fallbacks, SIGALRM deadline, relabelling invariance).

## Notebook form

`Abantes.ipynb` is the exact notebook fired at the leaderboard (its in-notebook default is
`ALETHEIA_METHOD=dyad_plainfuse`, shipped as the default so the leaderboard runs it with no env
set). It expects `sys.path` to include a `submission/` directory containing `harness.py`,
`util.py`, and `methods/` — in this staging layout those sit next to the notebook, so either run
it with cell 4's `sys.path.insert(0, "submission")` changed to `"."`, or symlink
`submission -> .`. On the leaderboard the platform provides `DATASET_NAME` and the NDIF
credentials. (The notebook's retrain cell is a no-op during scoring and refers to artifacts of
*other* bundled methods; `dyad_plainfuse` loads no trained artifact.)

## Import verification (done at staging time)

With the repo's `.venv-ndif` python, from this directory:

```python
import sys, os; sys.path.insert(0, os.getcwd())
import methods
m = methods.load("dyad_plainfuse")   # -> loads cleanly; NAME='dyad_plainfuse', W_CHEAP=0.5
```

Result: **PASS** — module imports cleanly with all seals asserted (T=2.0,
`ALETHEIA_FIXED_THRESHOLD=1`, 6000-char transcript cap).

## No NDIF? Run locally

`python local_test.py` with the same env vars runs the identical pipeline with the judge on
your own GPU (~20 GB bf16; needs CUDA torch + nnsight + accelerate — install with `uv sync` from the repo root). Both channels are judge-only, so the suspect model is never
loaded and one GPU suffices.
