"""rhadamanthys_dyad_anchor2h — Abas (anchor2, "meticulous" SYS_TRUTH) + the paper's honest
system prompt for the trusted judge. That prompt is the ONLY edit vs the Abas submission.

The `honest_unbiased` prompt from the chinese_auditing inference attacks (the phrasing the
Followup Split Personality honesty question echoes) is prepended to all three judge system
prompts (SYS_TRUTH, SYS_STRIP, SYS_JUDGE_FACT_FWD) at run time — same mechanics as
rhadamanthys_dyad_honestprompt, but on top of anchor2's SYS_TRUTH ("meticulous"), which is
applied as an import side effect of anchor2. No LoRA, no other change.

Dev (anchor + honest prompt, official labels): varied 0.788/0.875, instructed 0.945/0.990.
Board reference: Abas 0.8671/0.9222.

Fire with: ALETHEIA_METHOD=rhadamanthys_dyad_anchor2h   (tag: black — trusted judge only)
"""
from methods import jm43
from methods import rhadamanthys_dyad_anchor as _orig
from methods import rhadamanthys_dyad_anchor2 as _a2  # noqa: F401  import applies "meticulous"
from methods.rhadamanthys_dyad_honestprompt import HONEST

NAME = "rhadamanthys_dyad_anchor2h"
build = _orig.build

assert "meticulous subject-matter expert" in jm43.SYS_TRUTH   # anchor2 edit is in place


def run(build_model_fn, *args, **kwargs):
    old = (jm43.SYS_TRUTH, jm43.SYS_STRIP, _orig.SYS_JUDGE_FACT_FWD)
    jm43.SYS_TRUTH = HONEST + "\n\n" + jm43.SYS_TRUTH
    jm43.SYS_STRIP = HONEST + "\n\n" + jm43.SYS_STRIP
    _orig.SYS_JUDGE_FACT_FWD = HONEST + "\n\n" + _orig.SYS_JUDGE_FACT_FWD
    try:
        return _orig.run(build_model_fn, *args, **kwargs)
    finally:
        jm43.SYS_TRUTH, jm43.SYS_STRIP, _orig.SYS_JUDGE_FACT_FWD = old
