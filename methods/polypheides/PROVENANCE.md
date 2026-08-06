# Provenance — "Polypheides" submission (`rhadamanthys_dyad_anchor2h`)

## What this is

Polypheides is a black-box deception detector built entirely on a **trusted judge**
(`Qwen/Qwen3.5-9B`) — no suspect-model activations, no LoRA, no trained weights. It makes **4 judge
passes** per row: (1) a *blind truth anchor* — the judge reads the conversation with the suspect's
final reply withheld and states the true answer from its own knowledge (or `Unknown.`); (2) a
*style-strip restatement* — the judge neutrally restates the suspect's final reply with style
removed; (3–4) two *mirrored A/B logit reads* — the judge is shown transcript + truth anchor +
restatement and asked "deceptive (A) / honest (B)", then again with the labels swapped between A and
B, reading `d = logit(A) − logit(B)` each time. Antisymmetrizing, `margin = 0.5*(d1 − d2)`, cancels
any constant preference for token "A", for position 1, or for the word "deceptive", making
`margin = 0` a natural per-sample decision origin (no batch statistic). Output:
`score = sigmoid(margin/2)`, `deceptive` iff `margin >= 0`. Two prompt edits distinguish it from the
base DYAD anchor: the `honest_unbiased` HONEST preamble is prepended to all three judge system
prompts at run time, and `SYS_TRUTH` says "meticulous" (not "careful") "subject-matter expert"
(applied as an import side effect of `rhadamanthys_dyad_anchor2`).

## Board result

- **Balanced accuracy 0.8688, AUROC 0.9237** — submitted to the leaderboard 2026-07-25.
- Frozen bundle README: fired 2026-07-24 ~15:57 UTC (bg job ddb965fd), worktree commit `d56e62af`.

## Original paths

All files copied **verbatim** from `/home/alexander/workspace0/deception_cpas` at git commit
`ad98796e5f3a3691f0e44879a545f97e72e7183f` (branch `aletheia/alex`):

| Staged file | Original path |
|---|---|
| `Polypheides.ipynb` | `submission/frozen/Polypheides/Polypheides.ipynb` |
| `README.txt` | `submission/frozen/Polypheides/README.txt` |
| `methods/rhadamanthys_dyad_anchor2h.py` | `submission/frozen/Polypheides/rhadamanthys_dyad_anchor2h.py` (byte-identical to `submission/methods/rhadamanthys_dyad_anchor2h.py`) |
| `methods/rhadamanthys_dyad_anchor.py` | `submission/methods/rhadamanthys_dyad_anchor.py` |
| `methods/rhadamanthys_dyad_anchor2.py` | `submission/methods/rhadamanthys_dyad_anchor2.py` |
| `methods/rhadamanthys_dyad_honestprompt.py` | `submission/methods/rhadamanthys_dyad_honestprompt.py` |
| `methods/jm43.py` | `submission/methods/jm43.py` |
| `methods/jm38.py` | `submission/methods/jm38.py` |
| `methods/jm13.py` | `submission/methods/jm13.py` |
| `methods/judge_pilot.py` | `submission/methods/judge_pilot.py` |
| `methods/pm18.py` | `submission/methods/pm18.py` |
| `methods/__init__.py` | `submission/methods/__init__.py` |
| `harness.py` | `submission/harness.py` |
| `util.py` | `submission/util.py` |
| `remote_test.py` | `submission/remote_test.py` |
| `requirements-dryrun.txt` | `submission/requirements-dryrun.txt` |

The worktree copy `.claude/worktrees/conduct-label-verify/submission/frozen/Polypheides/` was
diffed against `submission/frozen/Polypheides/` and is **byte-identical** (all three files).

## Import graph (why each module is here)

```
rhadamanthys_dyad_anchor2h          the shipped method: HONEST preamble wrapper
├── rhadamanthys_dyad_anchor2       import side effect: "meticulous" SYS_TRUTH edit
│   └── rhadamanthys_dyad_anchor
├── rhadamanthys_dyad_honestprompt  source of the HONEST preamble string
│   └── rhadamanthys_dyad_anchor
├── rhadamanthys_dyad_anchor        DYAD core: build/run, the two mirrored A/B reads
│   ├── jm43                        SYS_TRUTH/SYS_STRIP, _clean, _blind_transcript, build (judge plan)
│   ├── judge_pilot (jp)            JUDGE_MODEL, _transcript, _gen_texts, _transcript_canary
│   ├── pm18 (tc)                   AUTHOR_MODEL ("Qwen/Qwen3.5-9B"), _render, _variant_ids
│   └── jm13                        SYS_JUDGE_FACT_FWD (the A/B verdict system prompt)
├── jm43 ── judge_pilot, pm18, jm13 (SYS_JUDGE_FACT_FWD), jm38 (SYS_STRIP)
├── jm38 ── judge_pilot, pm18, jm13
├── jm13 ── judge_pilot
└── judge_pilot ── pm18
```

Harness side: `remote_test.py` → `harness.py` (`run_submission`) → `util.py`
(`build_model`, `load_examples`, `write_submission`) and `methods/__init__.py` (`methods.load`).

Notes:

- **No data files are needed.** `pm18._PARAMS` points at `twin_combo.npz`, but it is only loaded by
  `pm18.build`/`pm18.run`, which this method never calls (its `build` is `jm43.build`; `pm18` is
  imported only for the `_render`/`_variant_ids` helpers and `AUTHOR_MODEL`). Likewise `pm18.build`
  lazily imports `methods.pm04` as a fallback — that path is never exercised, so `pm04.py` is
  deliberately **not** included.
- Files are verbatim; no imports were adjusted. The modules do their own
  `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))`, so the only requirement is that
  the directory containing `methods/` (this directory) is on `sys.path` — `remote_test.py` does that
  itself. See `RUN.md`.
