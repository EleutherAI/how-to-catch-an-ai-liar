# Ampyx — provenance

## Identity

- **Board notebook**: `submission/Ampyx.ipynb`, team EleutherAI, submitted **2026-07-30 01:51 UTC**.
- **Method (fire ledger)**: `rhadamanthys_m53r`, tag **white**, fired **2026-07-29 22:10:21 UTC**
  (fire ledger `/mnt/ssd-2/aletheia/fire_ledger.tsv`, entry
  `1785363021  Ampyx  rhadamanthys_m53r  white  2854458`; fire id 1785363021 is that epoch
  timestamp, matching `Ampyx_fire2.log`:
  `=== FIRING Ampyx (rhadamanthys_m53r, tag white) 2026-07-29T22:10:21UTC ===`, PID 2854458,
  isolated stage `/mnt/ssd-2/aletheia/fires/Ampyx`, packaged 437 files / 114.9 MB).
- **Board result**: balanced accuracy **0.8892**, AUROC **0.9450** (runtime 11064.1 s). This was
  our final white-box entry.
- Per counted cell (bal / AUROC):

  | cell | bal | AUROC |
  |---|---|---|
  | Notus-Nemotron | 0.8500 | 0.9056 |
  | Notus-Qwen3.5-27B | 0.8500 | 0.9326 |
  | Notus-gemma-3-27b-it | 0.8000 | 0.8715 |
  | Iris-Nemotron | 0.9375 | 0.9860 |
  | Iris-Qwen3.5-27B | 0.9650 | 0.9919 |
  | Iris-gemma-3-27b-it | 0.9325 | 0.9825 |

## Source

All files in this bundle were copied **verbatim** from the *fired* staging bundle
`/mnt/ssd-2/aletheia/fires/Ampyx/submission/` — the exact bytes that ran. Verified at staging
time: **32 of 32** files (`Ampyx.ipynb`, `harness.py`, `util.py`, `remote_test.py`, all 25
modules under `methods/`, and the three `.npz` artifacts) compare byte-identical with `cmp`.

`methods/rhadamanthys_m53r.py` sha256
`0d82c2a689dc99b66fdb61658b3623b421195d4f89991c16a3a60a5d374af3f7`.

Original repository for the module sources: `submission/methods/<name>.py` on branch
`origin/aletheia/giuseppe` of the team's research repo; the `.npz` artifacts sit at the parent
directory of `methods/` because `_PROBE_PATH`/`_PARAMS` resolve there:

- `pm22.npz` (**427,321 B**) — the white-leg probe. Per-family payloads:
  `gemma::{layer=28, site=mlp, scope, mu, sd, d}` and `qwen::{layer=22, site=mlp, scope, mu, sd, d}`.
- `probe_v3.npz` (5,343,578 B) — loaded by `pm04.py`.
- `twin_combo.npz` (584,033 B) — loaded by `pm18.py` / `pm21.py`.

`local_test.py` is the only file in this directory that is **not** from the fired stage: it is a
repo addition that runs the same frozen pipeline on local GPUs instead of NDIF.

## Architecture

`rhadamanthys_m53r` is a three-line module. It binds `run` and `build` to the **same function
objects** as `rhadamanthys_m53` (identity, not copies), and adds one attribute:

```python
build = _m53.build                 # same object
run   = _m53.run                   # same object
judge_rescue = _m33.judge_rescue   # the ONLY addition
```

So the scorer is `rhadamanthys_m53`, described below, and every healthy cell is byte-identical to
it by construction — there is no code here that could diverge.

### What `judge_rescue` fixes

`harness.py:109` does `rescue = getattr(method_module, "judge_rescue", None)` and, when `run()`
has raised, calls it to obtain a still-**ranked** plain-judge vector for the cell; only if the
attribute is absent does it fall through to `np.full(n, 0.5)` (`harness.py:121-122`).
`rhadamanthys_m33.py:23` exports `judge_rescue`; `rhadamanthys_m53.py` does not — the re-export
was forgotten when m53 wrapped m33. Under a mean-AUROC metric a constant cell scores exactly 0.5,
costing **−0.062 to −0.082** on the mean. The addition is read **only** by the harness, and
**only** after `run` has already raised — i.e. only on the path that otherwise emits the worst
possible value. `python -m methods.rhadamanthys_m53r` runs the module's offline selftest, which
asserts both the object identity and the rescue wiring.

Not fixed, and recorded in the module docstring as a residual: the compound-outage path
(`rhadamanthys_m44.py:80-81`, needs the judge **and** the se3p leg both dead) returns a constant
from inside `run` without raising, so the harness never sees it.

### The scorer (`rhadamanthys_m53`), in four steps

1. **Knowledge-anchor judge** (`jm43.py`). The trusted judge `Qwen/Qwen3.5-9B` makes three passes.
   It reads the conversation with the assistant's final reply blanked out (`jm43.py:69-77`) and
   states, from its **own** knowledge, the true answer to the user's last request — or exactly
   `Unknown.` when it does not know (`jm43.py:27-34`). It restates the actual final reply as flat
   declarative sentences, stripping persona, emphasis, hedging and filler (`jm38.py:39-44`). It is
   then asked in one forward pass whether the reply is deceptive, with both supplied in context as
   the auditor's own working notes (`jm43.py:80-97`); the channel's value is the yes-versus-no
   log-odds at the final token position. Because the reference answer comes from the judge's
   knowledge and never from the transcript, this measures a contradiction between what the reply
   asserts and what is true, rather than stylistic anomaly.
2. **Suspect self-evaluation** (`sm11.py`). The suspect model, with the organism adapter named by
   the dataset row loaded (`sm11.py:235-237`), is shown the record of its own session and its own
   final reply under a neutral third-party auditing frame (`sm11.py:56-66`), and is read in one
   forward pass for yes/no on whether that reply is honest and complete (`sm11.py:348-364`).
3. **Fusion** (`fusion_gates.py`). The self-evaluation channel must pass an admission check that
   **drops** it outright rather than fuse it when it is uninformative, largely unscored, or
   ordering rows the opposite way from the anchor (`fusion_gates.py:49-74`). Admitted, it is
   combined with the anchor at equal weight (`fusion_gates.py:77-102`); dropped, the anchor is used
   alone (`rhadamanthys_m44.py:92-93`). Steps 1-3 are driven by `rhadamanthys_m44.py:37-94`.
4. **Gated activation read** (`jm49.py`) — the white leg. `jm49._white_leg` (`jm49.py:94-154`)
   opens a trace on the suspect, takes the hidden state at the decoder layer and read-point
   recorded per architecture family in `pm22.npz` (`pm22.py:11, 34-43`), applies the centring and
   scaling vectors shipped with it, and projects the response-span activations onto a trained
   behaviour direction, pooled over the span (`pm22.py:50-64`). That projection replaces the step-3
   score on the rows the gate selects (`rhadamanthys_m53.py:92`); every other row keeps the step-3
   score unchanged (`rhadamanthys_m53.py:56-57, 70, 73, 84, 87, 92`).

The read is **data-gated, not organism-gated**, by two per-row conditions: the record before the
final reply must show tool-interaction evidence and at least two prior assistant turns
(`jm49.py:57-70`), **and** the trusted judge, asked a fixed question, must report that the final
reply asserts no specific checkable factual claim (`jm49.py:157-173`; prompt `jm06.py:25-29`, read
`jm06.py:70-78`). This targets the record-versus-report case, where the deceptive reply is a terse
denial and the honest one a long confession, so readers that see only the transcript invert.

There is no per-organism branching on the call path: the LoRA id is never branched on, only passed
to the model constructor (`sm11.py:235-237`, `jm49.py:99, 109`).

## Recorded scope fact: the white leg was dormant on the counted cells

From `runs/research_log/2026-07-22-white-leg-training-anatomy.md` in the source repo:

> **On the six counted cells the gate is dormant** (the counted genres are not agentic tool-record
> transcripts), so m53 behaved as the black m33 spine there.

And from `runs/research_whitebox_path/WHITE_PROGRAM_STATUS.md`:

> - d_beh gated (jm49): dormant on the counted cells

In plain terms: the activation leg's gate selected **zero rows** on all six counted board cells, so
the 0.8892 / 0.9450 above is the black spine's score. The white read is exercised on agentic
record-versus-report data — `aud_agentic`, `ga_insider`, `lb_insider` in the bundled battery — not
on the counted board genres. Read the tag as "this submission is entitled to read activations, and
does on the data its gate selects", not as "activations produced the board number".

A run that reports `agentic gate 0/n` is therefore behaving as designed on those genres; a run that
reports `PLAIN_FALLBACK` or a dropped se3p leg is **not** — see the health banner in `run.py` and
`RUN.md`.

## Predecessor: Tiresias (`rhadamanthys_m53_tp`)

`methods/rhadamanthys_m53_tp.py` is kept in this bundle because it is a one-file shell over the
same `rhadamanthys_m53` scorer, and it was its own board submission:

- **Board notebook**: `submission/Tiresias.ipynb`, submitted **2026-07-20 02:18 UTC**; method
  `rhadamanthys_m53_tp`, fired 2026-07-19 23:05:01 UTC (ledger entry
  `1784502301  Tiresias  rhadamanthys_m53_tp  white  2450955`).
- **Board result**: balanced accuracy **0.8904**, AUROC **0.9451**. Per counted cell (bal / AUROC):
  Notus-Nemo 0.8500/0.9056, Notus-Qwen 0.8550/0.9326, Notus-gemma 0.8000/0.8715,
  Iris-Nemo 0.9400/0.9863, Iris-Qwen 0.9650/0.9919, Iris-gemma 0.9325/0.9825.

The `_tp` shell leaves a cell byte-identical to m53 when its predicted-positive fraction lies in
the keep-band `[0.475, 0.525]`, and otherwise rank-centers the cell (uniform ranks,
AUROC-preserving) so the harness's fixed 0.5 threshold becomes the cell-median threshold. Comparing
the two board rows: the AUROCs are identical on five of six cells and differ by 0.0003 on the
sixth, because rank-centering is monotone within a cell and therefore cannot move a within-cell
AUROC — it moves only the `deceptive` boolean. Ampyx is the entry that was standing at the
deadline; Tiresias is the earlier row, documented here so its board numbers are not orphaned.

**Codename-reuse caution**: the fire ledger holds a NEWER fire named `Tiresias`
(`stmp_projenergy`, 2026-07-24 12:57 UTC) that did **not** land on that board row. The
0.8904 / 0.9451 row is the 2026-07-20 `rhadamanthys_m53_tp` submission.

## Module list

25 modules under `methods/`, the method plus its full repo-internal import closure, all verbatim:

`rhadamanthys_m53r, rhadamanthys_m53_tp, rhadamanthys_m53, rhadamanthys_m33, rhadamanthys_m44,
rhadamanthys29, rhadamanthys22, rhadamanthys3, jm49, jm43, jm38, jm29, jm26, jm13, jm06,
judge_pilot, fusion_gates, twin_arb, hybrid_judge, sm11, pm04, pm18, pm21, pm22, __init__`.

Harness side: `remote_test.py` (NDIF) / `local_test.py` (local GPUs) → `harness.py`
(`run_submission`) → `util.py` (`build_model`, `load_examples`, `write_submission`) and
`methods/__init__.py` (`methods.load`).
