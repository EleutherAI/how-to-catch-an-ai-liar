# Running Tiresias (`rhadamanthys_m53_tp`)

## Layout of this bundle

```
tiresias/
  methods/            # 24 files: the method + its full repo-internal import closure, verbatim
  pm22.npz            # white-leg probe (gemma L28 / qwen L22, mlp site) — loaded by methods/pm22.py
  probe_v3.npz        # loaded by methods/pm04.py
  twin_combo.npz      # loaded by methods/pm18.py and pm21.py
```

Modules (all originally `submission/methods/` in the source repo):
`rhadamanthys_m53_tp, rhadamanthys_m53, rhadamanthys_m33, rhadamanthys_m44, rhadamanthys29,
rhadamanthys22, rhadamanthys3, jm49, jm43, jm38, jm29, jm26, jm13, jm06, judge_pilot,
fusion_gates, twin_arb, hybrid_judge, sm11, pm04, pm18, pm21, pm22, __init__`.

## sys.path requirement (no imports were rewritten)

Every module does `from methods import ...` and locates its `.npz` artifacts at the parent
directory of `methods/`. So the ONLY setup needed is to have **this bundle's root directory on
`sys.path`** (the modules themselves already do
`sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))`, so importing any of them by file
path also works). Verified:

```python
import sys; sys.path.insert(0, "/path/to/tiresias")
from methods import rhadamanthys_m53_tp   # imports the full closure, finds pm22.npz
```

## Harness invocation (source repo)

The method is not standalone — it exposes `NAME`, `build(model_id, lora_id, task)` and
`run(build_model_fn, examples, index, task, *, remote, batch_size)` and is driven by the
competition harness (`submission/harness.py` via `submission/remote_test.py`), which injects
`build_model_fn` (NDIF/nnsight remote model factory, see `submission/util.py::build_model`) and
the dataset rows. In the source repo, one dataset:

**No NDIF? Run locally** — `local_test.py` runs the same frozen pipeline with all models
dispatched onto local GPUs (needs CUDA torch + nnsight + accelerate; see the repo's
`requirements-local.txt`). This method loads BOTH the 9B judge and the suspect model named
in the dataset rows: a 27B suspect needs ~2×48 GB GPUs (set `CUDA_VISIBLE_DEVICES`), and
the Nemotron-120B suspect (~240 GB) is realistically NDIF-only.

```bash
ALETHEIA_METHOD=rhadamanthys_m53_tp DATASET_NAME=<...> python local_test.py
```

This bundle ships the harness entry point (`remote_test.py`, `harness.py`, `util.py`), so it
runs standalone from this directory:

```bash
cd methods/tiresias
set -a; source .env; set +a     # NDIF_API_KEY, HF_TOKEN
ALETHEIA_METHOD=rhadamanthys_m53_tp \
DATASET_NAME=aletheias-quest/dev-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4 \
  python remote_test.py
```

The same command against the original competition checkout:

```bash
ALETHEIA_METHOD=rhadamanthys_m53_tp DATASET_NAME=<...> .venv-ndif/bin/python submission/remote_test.py
```

Writes `submission.csv` (columns `index, deceptive, score`) and, if missing,
`submission.manifest.json`. Add `ALETHEIA_LIMIT=16` for a smoke run. Full sweep over
`submission/dry.yaml`:

```bash
ALETHEIA_METHOD=rhadamanthys_m53_tp .venv-ndif/bin/python submission/remote_sweep.py
```

To reproduce with this bundle inside a checkout, drop the `methods/` modules and the three `.npz`
files into the harness's `submission/` directory (same relative layout) — the loader resolves
methods by `ALETHEIA_METHOD` name from `submission/methods/`.

## Environment

- Python **3.12** (the NDIF server rejects 3.11); CPU-only client venv — models run remotely on NDIF.
- Deps: `submission/requirements-dryrun.txt` in the source repo — `nnsight, transformers, datasets,
  scikit-learn, pandas` (+ CPU torch and torchvision; torchvision is required or the twin run
  silently scores 0.5 everywhere).

  ```bash
  uv venv --python 3.12 .venv-ndif
  UV_TORCH_BACKEND=cpu uv pip install --python .venv-ndif/bin/python -r submission/requirements-dryrun.txt
  ```
- Secrets in `.env`: `NDIF_API_KEY`, `HF_TOKEN`.
- Method-specific env knobs (all optional): `ALETHEIA_TP_LO` / `ALETHEIA_TP_HI` (keep-band, default
  0.475 / 0.525), `ALETHEIA_SELFEVAL_3P_DEADLINE` (se3p deadline seconds, default 900).

## What it prints

The tp shell logs, per cell, either `frac X in [0.475,0.525] -> m53 UNCHANGED` or
`frac X OUT of [...] -> rank-center (median ... -> 0.5; ...)`. The m53 shell logs the agentic-gate
count (`agentic gate k/n`) and, when the gate is dormant (all six counted board cells),
returns the m33 champion scores unchanged.
