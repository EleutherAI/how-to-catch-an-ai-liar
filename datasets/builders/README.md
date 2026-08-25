# Builders — optional, and not needed to use the battery

**The JSONL files in `../data/` are the artifact of record.** Everything in this repo —
`../../gauntlet.py`, `../load.py`, `../../run.py` — reads them directly. Nothing here has to be
run, and running it requires network access and upstream permissions that the battery itself
does not.

These two scripts are published so the derivation of 18 of the 27 distributions is inspectable,
not because you need them.

| script | rebuilds | needs |
|---|---|---|
| `build_lb.py` | the 9 `lb_*` distributions | a local checkout of `github.com/Cadenza-Labs/liars-bench`, passed with `--repo` |
| `build_dyl.py` | the 9 `dyl_*` distributions | `hf_hub_download("ai-safety-institute/lie-detection-rollouts", ...)` |

Both were copied from the research repo that produced the battery, with **one line changed
each**: `from gauntlet import ...` became `from _common import ...`, because the research repo's
`gauntlet.py` was a loader module while this repo's `gauntlet.py` is the command-line front end.
`_common.py` provides the three functions they used — `save_dist`, `log`, and the
chain-of-thought hygiene (`strip_think` / `clean_messages`).

## Access

`Cadenza-Labs/liars-bench` is **gated on the Hugging Face Hub**: rebuilding the `lb_*` family
means requesting access from the upstream authors. The AISI rollouts are ungated but published
as `license: other`. See [`../LICENSES.md`](../LICENSES.md) — the licence status of both
families is unresolved, and that is a fact about the upstream publications, not an oversight
here.

## The nine distributions these scripts do NOT rebuild

`ga_tqa`, `ga_halu`, `ga_domains`, `ga_insider`, `ga_insider_cot`, `b2_pressure`, `wf_assert`,
`wf_denial`, `aud_agentic` were assembled inside the research repo from four further
intermediate directories, each with its own generation and belief-gating chain. Porting that
would mean vendoring those chains and their model calls, and the result would still not be
reproducible without the original NDIF deployment. Their derivation is documented instead, in
[`../PROVENANCE.md`](../PROVENANCE.md).

## Invariants

`_common.save_dist` refuses to write a distribution whose rows violate the battery's
invariants: non-empty final assistant turn, last message from the assistant, no `<think>`
leaking into visible content. `python ../../gauntlet.py check` re-verifies the same invariants
over what is on disk, plus the manifest counts and the cached score vectors.
