# Provenance — "Abantes" submission (`dyad_plainfuse`)

## What this is

Abantes is our best **black-box** entry: the pinned DYAD anchor plus one cheap generation-free
judge channel, fused per sample. Per its own module docstring (`methods/dyad_plainfuse.py`):
the **anchor** is `rhadamanthys_dyad_anchor2h_shield_clean.run(...)` called *verbatim* — the BB pin
(blind-truth-anchor + style-strip + two mirrored A/B "deceptive/honest" logit reads on the trusted
judge, HONEST preamble, shield fault isolation, T sealed 2.0, fixed 0.5 threshold). The **cheap
channel** is `judge_pilot.judge_ynd`: one batched next-token yes/no log-odds read per row (frozen
`SYS_JUDGE` prompt, frozen retry ladder, frozen 6000-char transcript window from jm43), **no
generation**, with a per-row validity mask. The **fusion** is a per-sample fixed anchor-dominant
logit blend — `fused_logit_i = logit(pin_i) + 0.5 * (ynd_i − 0)/2`, `score_i = sigmoid(fused_logit_i)`
(`W_CHEAP = 0.5`, `CHEAP_CENTER = 0.0`, fixed pre-test; no batch statistics) — deliberately **not**
`fusion_gates.fuse` (which mixes cross-row z-scores, the construct the 07-20 compliance audit
flagged). `fusion_gates.admit` is reused verbatim as a channel-*health* gate with one documented
override: the cheap leg is dropped only on true inversion (spearman ρ ≤ −0.20 vs the anchor), not
at the usual ρ < +0.30 floor, because the channel's value is concentrated where it *disagrees* with
the anchor. Failure contract: every failure of the added channel (raise, SIGALRM deadline, wrong
shape, NaN, degenerate/flat, true inversion, exhausted 2600 s/cell budget) returns the pin's own
array object byte-identical; unscored rows take the pin value verbatim. Compliance: **black** —
trusted judge only, the suspect's logprobs/activations are never read; no router
(`model`/`lora`/task never inspected); `ALETHEIA_FIXED_THRESHOLD` hard-set to `"1"`.

Motivation recorded in the docstring: SAIN Groningen's `phoenix_wright_v6_2` (0.9233 in ~16 min vs
our pin's 0.9237 in ~187 min) showed a single next-token yes/no read is a *different* channel —
better on the hard genres (Notus, Eunomia), worse on Iris — so fusing it per sample is the only
raw material for a gain. Abantes **superseded Polypheides** (`rhadamanthys_dyad_anchor2h`,
0.8688 bal / 0.9237 AUROC) as the pinned black-box entry.

## Board identity and result

- **Notebook:** `Abantes.ipynb` · **team EleutherAI** · **submitted 2026-07-30 05:55 UTC**
- **Members:** Stella Binderman — EleutherAI; Gonçalo Paulo — EleutherAI; Giuseppe Birardi —
  Orma Lab Srl; Alexander Reinthal — independent researcher
- **Top-line: balanced accuracy 0.8321 / AUROC 0.9260** (best black-box; the earlier Polypheides
  black pin was 0.8688 / 0.9237)
- Per counted cell (bal / AUROC):

  | cell | bal | AUROC |
  |---|---|---|
  | Notus-Nemotron | 0.7625 | 0.9228 |
  | Notus-Qwen3.5-27B | 0.7325 | 0.8718 |
  | Notus-gemma-3-27b-it | 0.7950 | 0.8592 |
  | Iris-Nemotron | 0.9075 | 0.9758 |
  | Iris-Qwen3.5-27B | 0.9050 | 0.9699 |
  | Iris-gemma-3-27b-it | 0.8900 | 0.9564 |

## Ledger fire attribution

- Fire ledger `/mnt/ssd-2/aletheia/fire_ledger.tsv` row:
  `1785377013  Abantes  dyad_plainfuse  black  2992237` — fire id **1785377013** is the epoch
  timestamp **2026-07-30T02:03:33 UTC** (matches `Abantes_fire2.log`:
  `=== FIRING Abantes (dyad_plainfuse, tag black) 2026-07-30T02:03:33UTC ===`, PID 2992237,
  stage `/mnt/ssd-2/aletheia/fires/Abantes` (isolated), packaged 437 files / 114.9 MB, entered as
  team EleutherAI, black-box).
- `dyad_plainfuse.py` sha256 `44f9c278…` — byte-identical to the file recorded at the earlier
  Archemachus fire ("sha 44f9c278", repo commit `d53c42a2`); per
  `runs/research_whitebox_path/FIRED_Abantes_Alastor_2026-07-30.md` (branch
  `origin/aletheia/giuseppe`), Abantes reproduced Archemachus's 0.9260 / 0.8321 exactly, with 0 of
  6 counted cells at 0.5.

## Original paths

All files copied **verbatim** (cmp-verified byte-identical) from the *fired* staging bundle
`/mnt/ssd-2/aletheia/fires/Abantes/submission/` — the exact bytes that ran. The local repo checkout
`/home/alexander/workspace0/deception_cpas` (branch `aletheia/alex`, HEAD at staging time
`4babfabac7faabad9d329e75733919b53397784f`) does **not** contain `dyad_plainfuse.py` in its working
tree; the method lives on branch `origin/aletheia/giuseppe` (`submission/methods/dyad_plainfuse.py`,
blob `637f14c0`, same sha256 `44f9c278…`).

| Staged file | Original path (fired stage) |
|---|---|
| `Abantes.ipynb` | `/mnt/ssd-2/aletheia/fires/Abantes/submission/Abantes.ipynb` (sha256 `1d0feaa4e1e3b876…`) |
| `harness.py`, `util.py`, `remote_test.py`, `requirements-dryrun.txt` | same dir (see harness note below) |
| `methods/dyad_plainfuse.py` | `…/submission/methods/dyad_plainfuse.py` (sha256 `44f9c27811da1f28…`) |
| `methods/<all other modules>` | `…/submission/methods/<same name>.py` |

Cross-checks against the source repo:

- Every staged file **except `methods/jm43.py`** is byte-identical to the corresponding file at
  `origin/aletheia/giuseppe:submission/...`. The branch tip's `jm43.py` contains one *later*
  refactor (commit `6a9f73bc`, "last white bet": the two verdict-question string literals lifted
  out verbatim into `Q_MID`/`Q_FINAL` globals, values unchanged); the staged copy is the pre-refactor
  file that actually fired.
- `harness.py` and `requirements-dryrun.txt` differ from the local `aletheia/alex` checkout's
  `submission/` copies only in comments (plus a local-only `duckdb` dep for the results-DB
  workflow); `util.py` and `remote_test.py` are byte-identical to both repo dirs.

**Which harness directory the fire used:** the fire ran from an *isolated stage*
(`/mnt/ssd-2/aletheia/fires/Abantes`) built from giuseppe's checkout's **`submission/`** directory
(fire2 log: docs taken from `/home/giuseppe/deception_cpas/submission/docs/black`; all four staged
harness files match `origin/aletheia/giuseppe:submission/*`). It did **not** run from
`submission_dyad/` (which has no `requirements-dryrun.txt` and no `dyad_plainfuse.py`).

## Import graph (why each module is here)

```
dyad_plainfuse                          the shipped method: pin + cheap yes/no leg, per-sample blend
├── rhadamanthys_dyad_anchor2h_shield_clean   the ANCHOR (BB pin), called verbatim
│   ├── jm43                            SYS_TRUTH/SYS_STRIP (mutated per run + finally-restored)
│   ├── dyad_shield                     phase-level fault isolation
│   │   ├── jm43 · judge_pilot · pm18 · rhadamanthys_dyad_anchor
│   ├── rhadamanthys_dyad_anchor        frozen DYAD core: SYS_JUDGE_FACT_FWD, T=2.0, _ab_margins…
│   │   ├── jm43 · judge_pilot · pm18 · jm13 (SYS_JUDGE_FACT_FWD)
│   └── rhadamanthys_dyad_honestprompt  HONEST preamble constant
│       └── jm43 · rhadamanthys_dyad_anchor
├── rhadamanthys_dyad_anchor            T-seal re-assert only
├── judge_pilot (jp)                    cheap channel: judge_ynd, JUDGE_MODEL, transcript helpers
│   └── pm18
├── jm43                                _judge_prob scale (frozen)
│   ├── judge_pilot · pm18 · jm13 · jm38 (SYS_STRIP)
├── fusion_gates (fg)                   admit() channel-health gate
│   └── twin_arb (lazy, at admit() call: _spearman)
│       └── pm18 · pm21 · pm04          (module imports only on this path)
└── [selftest only] twin_arb._spearman

jm38 ── judge_pilot · pm18 · jm13        jm13 ── judge_pilot
pm21 ── pm18 · pm04                      pm18 ── pm04 (lazy, pm18.build fallback only)
```

Harness side: `remote_test.py` → `harness.py` (`run_submission`) → `util.py`
(`build_model`, `load_examples`, `write_submission`) and `methods/__init__.py` (`methods.load`).

Notes:

- **No data files are needed at runtime.** `pm18._PARAMS` (`twin_combo.npz`, 584 033 B in the fired
  stage) is loaded only inside `pm18.build`/`pm18.run`, and `pm04._PROBE_PATH` (`probe_v3.npz`,
  5 343 578 B) only inside `pm04._load_probe` — none of these is on the `dyad_plainfuse` call path
  (`dyad_plainfuse.build` is the pin's build == `jm43.build`, a trivial plan object; `pm18.build`'s
  npz load is guarded by `os.path.exists`). `pm04`/`pm21`/`twin_arb` are included because
  `fusion_gates.admit` imports `twin_arb` at call time, which imports them at module level —
  module imports only, no npz reads. The two `.npz` files are therefore deliberately **not** staged.
- Files are verbatim; no imports were adjusted. Each module does its own
  `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))`, so the only requirement is that
  the directory containing `methods/` (this directory) is on `sys.path` — `remote_test.py` does that
  itself. See `RUN.md`.
- Import from this staging dir verified with the repo's `.venv-ndif` python:
  `methods.load('dyad_plainfuse')` imports cleanly (seals asserted at import: T=2.0,
  `ALETHEIA_FIXED_THRESHOLD=1`, 6000-char transcript cap).
