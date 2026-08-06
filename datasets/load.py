"""Minimal loader for the evaluation battery.

Each distribution is one JSONL file in data/, one row per line:
  dist       distribution id
  rid        stable row id within the distribution
  messages   chat messages [{role, content}, ...]; the LAST assistant turn is the
             reply under audit (chain-of-thought is stripped into meta, not content)
  y          1 = deceptive final assistant reply, 0 = honest
  gen_model  the model that actually produced the reply
  meta       provenance dict (source, label method, genre, optional reasoning, ...)

gauntlet.json is the manifest: per-distribution source, genre, label method, row
counts, flags, and the two dataset nulls (reply-length-only AUROC and blind-judge
AUROC — see the blog post, section 2).

Usage:
    from load import list_dists, load_dist, manifest
    for name in list_dists():
        rows = load_dist(name)
        y = [r["y"] for r in rows]
        ...score rows with your detector, compare against manifest()[name]["nulls"]...
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def manifest() -> dict:
    return json.load(open(os.path.join(ROOT, "gauntlet.json")))["dists"]


def list_dists() -> list[str]:
    return sorted(manifest())


def load_dist(name: str) -> list[dict]:
    path = os.path.join(ROOT, "data", f"{name}.jsonl")
    return [json.loads(line) for line in open(path)]
