"""rhadamanthys_dyad_anchor2 — re-fire of rhadamanthys_dyad_anchor with a TRIVIAL variation of the
trusted-model truth prompt (jm43.SYS_TRUTH): "a careful subject-matter expert" -> "a meticulous
subject-matter expert". Everything else identical (same DYAD antisymmetric margin, same T, same
per-row isolation).

Fire with: ALETHEIA_METHOD=rhadamanthys_dyad_anchor2   (tag: black — trusted judge only)
"""
from methods import jm43

_OLD = "a careful subject-matter expert"
_NEW = "a meticulous subject-matter expert"
assert _OLD in jm43.SYS_TRUTH, "SYS_TRUTH drifted — variation target string not found"
jm43.SYS_TRUTH = jm43.SYS_TRUTH.replace(_OLD, _NEW)

from methods.rhadamanthys_dyad_anchor import build, run   # noqa: E402,F401

NAME = "rhadamanthys_dyad_anchor2"
