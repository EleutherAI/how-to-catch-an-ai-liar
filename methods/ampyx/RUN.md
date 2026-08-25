# Running Ampyx (`rhadamanthys_m53r`)

## Layout of this bundle

```
ampyx/
  Ampyx.ipynb         # the notebook fired at the leaderboard (arms rhadamanthys_m53r)
  methods/            # 25 files: the method + its full repo-internal import closure, verbatim
  pm22.npz            # white-leg probe (gemma L28 / qwen L22, mlp site) — loaded by methods/pm22.py
  probe_v3.npz        # loaded by methods/pm04.py
  twin_combo.npz      # loaded by methods/pm18.py and pm21.py
```

Modules (all originally `submission/methods/` in the source repo):
`rhadamanthys_m53r, rhadamanthys_m53_tp, rhadamanthys_m53, rhadamanthys_m33, rhadamanthys_m44,
rhadamanthys29, rhadamanthys22, rhadamanthys3, jm49, jm43, jm38, jm29, jm26, jm13, jm06,
judge_pilot, fusion_gates, twin_arb, hybrid_judge, sm11, pm04, pm18, pm21, pm22, __init__`.

`rhadamanthys_m53_tp` is the earlier **Tiresias** submission, kept because it is a one-file shell
over the same scorer; `PROVENANCE.md` records its board row. It is not the armed method.

## Quick start (from the repo root)

```bash
python run.py --method ampyx --dataset wf_assert --limit 8
python run.py --method ampyx --dataset notus-lie-auditor --suspect gemma-3-27b-it
```

`run.py` couples the suspect to the data, prints AUROC / balanced accuracy against the dataset's
recorded nulls, and prints a **health line** naming which legs actually ran. Pass `--strict` to
exit non-zero when any leg fell back — see "Reading the health line" below.

## sys.path requirement (no imports were rewritten)

Every module does `from methods import ...` and locates its `.npz` artifacts at the parent
directory of `methods/`. So the ONLY setup needed is to have **this bundle's root directory on
`sys.path`** (the modules themselves already do
`sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))`, so importing any of them by file
path also works). Verified:

```python
import sys; sys.path.insert(0, "/path/to/ampyx")
from methods import rhadamanthys_m53r   # imports the full closure, finds pm22.npz
```

## Offline selftest (no GPU, no network)

```bash
python -m methods.rhadamanthys_m53r
```

Asserts that `run`/`build` are the **same objects** as `rhadamanthys_m53` (so healthy cells are
byte-identical by construction) and that `harness._last_resort` finds a callable `judge_rescue`
here and not on `rhadamanthys_m53`.

## Harness invocation

The method is not standalone — it exposes `NAME`, `build(model_id, lora_id, task)` and
`run(build_model_fn, examples, index, task, *, remote, batch_size)` and is driven by the
competition harness, which injects `build_model_fn` and the dataset rows.

**Locally** — `local_test.py` runs the same frozen pipeline with all models dispatched onto local
GPUs (needs CUDA torch + nnsight + accelerate; install with `uv sync` from the repo root). This
method loads BOTH the 9B judge (~20 GB bf16) and the suspect named in the dataset rows; the
Nemotron-120B suspect (~240 GB) is realistically NDIF-only.

```bash
ALETHEIA_METHOD=rhadamanthys_m53r DATASET_NAME=<...> python local_test.py
```

> **Give it headroom.** Both models are dispatched with `device_map="auto"`, which lets the judge
> spread over every visible GPU and can leave the suspect without room. Measured on this bundle:
> on 4×48 GB the anchor OOMs and the run silently degrades to a plain judge; on 7×48 GB the same
> command runs the full stack. Prefer one large GPU for the judge plus separate GPUs for the
> suspect (`CUDA_VISIBLE_DEVICES`), and use `run.py --strict` so a degraded run fails loudly
> instead of printing a clean-looking AUROC.

**Through NDIF** — this bundle ships the harness entry point (`remote_test.py`, `harness.py`,
`util.py`), so it runs standalone from this directory:

```bash
cd methods/ampyx
set -a; source .env; set +a     # NDIF_API_KEY, HF_TOKEN
ALETHEIA_METHOD=rhadamanthys_m53r \
DATASET_NAME=aletheias-quest/dev-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4 \
  python remote_test.py
```

(The `aletheias-quest/dev-*` slugs are the original competition dev sets and may not remain
publicly downloadable; any dataset following the `<slug>` + `<slug>-labels` convention works.)

Writes `submission.csv` (columns `index, deceptive, score`) and, if missing,
`submission.manifest.json`. Add `ALETHEIA_LIMIT=16` for a smoke run.

To reproduce with this bundle inside a competition checkout, drop the `methods/` modules and the
three `.npz` files into the harness's `submission/` directory (same relative layout) — the loader
resolves methods by `ALETHEIA_METHOD` name from `submission/methods/`.

## Notebook form

`Ampyx.ipynb` is the exact notebook fired at the leaderboard; its in-notebook default is
`ALETHEIA_METHOD=rhadamanthys_m53r`, shipped as the default so the leaderboard ran it with no env
set. It expects `sys.path` to include a `submission/` directory containing `harness.py`, `util.py`
and `methods/` — in this layout those sit next to the notebook, so either change cell 4's
`sys.path.insert(0, "submission")` to `"."`, or `ln -s . submission`.

Its retrain cell (a no-op during scoring) refers to `submission/training/`, the derivation chain
for `pm22.npz`. That directory is **not** shipped here; the artifact itself is
(`pm22.npz`, 427,321 B).

## Environment

- Python **3.12**.
- Local GPUs: `uv sync` from the repo root.
- NDIF client only (models run remotely): `requirements-dryrun.txt` in this directory —
  `nnsight, transformers, datasets, scikit-learn, pandas` plus CPU torch and **torchvision**
  (torchvision is required, or the twin run silently scores 0.5 everywhere).

  ```bash
  uv venv --python 3.12 .venv-ndif
  UV_TORCH_BACKEND=cpu uv pip install --python .venv-ndif/bin/python -r requirements-dryrun.txt
  ```
- Secrets for the NDIF path: `NDIF_API_KEY`, `HF_TOKEN`.
- Method-specific env knobs (all optional): `ALETHEIA_SELFEVAL_3P_DEADLINE` (se3p deadline
  seconds, default 900); `ALETHEIA_TP_LO` / `ALETHEIA_TP_HI` apply only to the unarmed
  `rhadamanthys_m53_tp` shell.

## Reading the health line

The armed stack logs what each leg did, and `run.py` collects it into one `[run] health:` line.
What a faithful run looks like:

| signal | healthy | degraded |
|---|---|---|
| `jm43` anchor | `TWO_PASS` | `PLAIN_FALLBACK` (anchor died, plain judge substituted) or `DEAD` |
| `sm11` self-eval | rows valid, `FUSE mode=fused` | `run failed → 0.5`, `drop=degenerate`, `FUSE mode=anchor_only` |
| `jm49` white leg | `agentic gate k/n` with `k>0` on agentic genres | `DEAD` |

`agentic gate 0/n` is **not** a failure on non-agentic data — the gate is designed to select only
record-versus-report rows, and it selected zero on all six counted board cells (see
`PROVENANCE.md`). A `PLAIN_FALLBACK` anchor or a dropped se3p leg *is* a failure: the numbers you
get are not this method's numbers.
