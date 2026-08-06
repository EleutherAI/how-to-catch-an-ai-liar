"""rhadamanthys_m33 — internal detector module.
Fire with: ALETHEIA_METHOD=rhadamanthys_m33   (tag: black)
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

from methods import fusion_gates as fg                # noqa: E402
from methods import rhadamanthys_m44 as rtb  # noqa: E402

NAME = "rhadamanthys_m33"

W_FLOOR_FRAC = float(os.environ.get("ALETHEIA_M33_WFF", "1.0"))
if not (0.5 <= W_FLOOR_FRAC <= 1.0):
    raise RuntimeError(f"ALETHEIA_M33_WFF={W_FLOOR_FRAC} outside the validated [0.5, 1.0] range")
fg.W_FLOOR_FRAC = W_FLOOR_FRAC

build = rtb.build
run = rtb.run
judge_rescue = rtb.judge_rescue
