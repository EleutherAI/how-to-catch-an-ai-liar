# Tiresias — provenance

## Identity

- **Board notebook**: `submission/Tiresias.ipynb`, team EleutherAI, submitted **2026-07-20 02:18 UTC**.
- **Method (fire ledger)**: `rhadamanthys_m53_tp`, tag **white**, fired 2026-07-19 23:05:01 UTC
  (fire ledger `/mnt/ssd-2/aletheia/fire_ledger.tsv`, entry `1784502301  Tiresias  rhadamanthys_m53_tp  white  2450955`).
- **Board result**: balanced accuracy **0.8904**, AUROC **0.9451** (runtime 11502.6 s).
  Per counted cell (bal / auroc): Notus-Nemo 0.8500/0.9056, Notus-Qwen 0.8550/0.9326,
  Notus-gemma 0.8000/0.8715, Iris-Nemo 0.9400/0.9863, Iris-Qwen 0.9650/0.9919, Iris-gemma 0.9325/0.9825.
- **Codename-reuse caution**: the fire ledger holds a NEWER fire named `Tiresias`
  (`stmp_projenergy`, 2026-07-24 12:57 UTC) that did **not** land on this board row. The
  0.8904/0.9451 row is the older `rhadamanthys_m53_tp` submission of 2026-07-20; do not attribute
  `stmp_projenergy` to it.

## Source

- Source repository: `/home/alexander/workspace0/deception_cpas`
- Git commit at staging time: `ad98796e5f3a3691f0e44879a545f97e72e7183f`
- Original paths (all copied verbatim, no import rewriting):
  - Method + dependency modules: `submission/methods/<name>.py` →  staged as `methods/<name>.py`
    (23 modules + `__init__.py`; full list in RUN.md / below).
  - Probe artifact: `submission/pm22.npz` → `pm22.npz` (**427,321 bytes**). Contains per-family
    probe payloads: `gemma::{layer=28, site=mlp, scope, mu, sd, d}` and
    `qwen::{layer=22, site=mlp, scope, mu, sd, d}`.
  - Also staged because modules in the import closure load them at run time:
    `submission/probe_v3.npz` → `probe_v3.npz` (5,343,578 bytes; loaded by `pm04.py`) and
    `submission/twin_combo.npz` → `twin_combo.npz` (584,033 bytes; loaded by `pm18.py`/`pm21.py`).
    The `_PROBE_PATH`/`_PARAMS` in those modules resolve to the parent directory of `methods/`,
    which is why the `.npz` files sit at this bundle's root.

## Architecture (3 shells)

`rhadamanthys_m53_tp` is a three-shell wrapper. The innermost shell is the black-box champion
`rhadamanthys_m33`: a fusion (via `fusion_gates`) of the `jm43` truth-anchored two-pass judge with
the `sm11` (se3p) third-person self-evaluation reader, assembled through `rhadamanthys_m44` (which
also pulls in the `rhadamanthys22`/`rhadamanthys29` rescue/whitebox legs). The middle shell,
`rhadamanthys_m53`, grafts on `jm49`'s gated white activation leg: a surface agentic gate
(ReAct/tool-call markers + ≥2 prior assistant turns) and a claim gate (`jm06` via `jm49`) route only
agentic claim-free rows to a suspect-side activation read that loads a probe direction from
`pm22.npz` (gemma layer 28, qwen layer 22, mlp site), projects response-token activations onto it,
and splices the z-scored white read into the m33 scores (sigmoid-wrapped at the cell median); every
other row returns m33 unchanged. The outer shell, `_tp` ("threshold probe"), leaves a cell's scores
byte-identical to m53 when its predicted-positive fraction lies in the keep-band **[0.475, 0.525]**,
and otherwise rank-centers the cell (uniform ranks, AUROC-preserving) so the harness's fixed 0.5
threshold becomes the cell-median threshold.

## Recorded scope facts (verbatim from the repo's research logs)

From `runs/research_log/2026-07-22-white-leg-training-anatomy.md` ("Two facts that scope its board
relevance:"):

> - **On the six counted cells the gate is dormant** (the counted genres are not agentic tool-record
>   transcripts), so m53/Tiresias behaved as the black m33 spine there — Tiresias's Notus-Qwen 0.8550
>   came from its batch-median re-thresholding, not from the white read
>   (`runs/research_whitebox_path/WHITE_PROGRAM_STATUS.md`).
> - The leg's score seam normalizes at the **cell median** (jm47 precedent), which — like the rest of
>   the m53 family — is an in-batch statistic and non-compliant under the organizers' 2026-07-20
>   fixed-per-sample-threshold ruling. A compliant white leg needs a per-row seam; that is part of
>   what the `white_materiality.py` gate now enforces for any successor.

From `runs/research_whitebox_path/WHITE_PROGRAM_STATUS.md` (worktree
`.claude/worktrees/conduct-label-verify/`):

> Organizer rule: a submission is white/hybrid only if the white contribution MATERIALLY changes the
> final score (dormant path / zero coverage / epsilon weight / nullified read does NOT count).

> - d_beh gated (jm49): dormant on the counted cells -> not material -> not white per the rule.

In plain terms, as recorded in-repo: (a) the white activation leg selected zero rows (gate dormant)
on all six counted board cells — the board score is the black m33 spine's plus the `_tp`
re-thresholding; and (b) the m53 family's in-batch cell-median / keep-band re-thresholding is
recorded in-repo as non-compliant with the organizers' 2026-07-20 fixed-per-sample-threshold ruling.
