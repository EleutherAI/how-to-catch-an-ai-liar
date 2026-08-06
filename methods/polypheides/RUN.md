# Running Polypheides (`rhadamanthys_dyad_anchor2h`)

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
ALETHEIA_METHOD=rhadamanthys_dyad_anchor2h \
DATASET_NAME=aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4 \
  .venv-ndif/bin/python remote_test.py
```

- `DATASET_NAME` is any competition dataset id (single `test` split; rows carry the suspect
  `model`/`lora`, which the harness reads from row 0 — there is no model config file).
- Add `ALETHEIA_LIMIT=16` for a quick smoke run.
- Output: `submission.csv` in the current directory, columns `index, deceptive, score`
  (`deceptive == (score >= 0.5)`; the method's margin makes 0.5 the natural per-sample threshold —
  it exports `ALETHEIA_FIXED_THRESHOLD=1`). `remote_test.py` is a release gate: it also validates
  the CSV (compliance, non-degenerate scores, raw AUROC when labels are present) and exits nonzero
  on any gate failure.

## Notebook form

`Polypheides.ipynb` is the exact notebook fired at the leaderboard (its in-notebook default is
`ALETHEIA_METHOD=rhadamanthys_dyad_anchor2h`). It expects `sys.path` to include a `submission/`
directory containing `harness.py`, `util.py`, and `methods/` — in this staging layout those sit
next to the notebook, so either run it with cell 4's `sys.path.insert(0, "submission")` changed to
`"."`, or symlink `submission -> .`. On the leaderboard the platform provides `DATASET_NAME` and
the NDIF credentials. (Note: the notebook's markdown cells describe `rhadamanthys_m53`, an earlier
white-box default, and a `pm22.npz` retrain cell — neither is part of this method; the retrain cell
is a no-op without `submission/training/`.)
