"""Shared helpers for the battery builders.

Ported from the research repo's `runs/research_gauntlet/gauntlet.py`, which is where the
distributions in `../data/` were originally written. Only the parts the two shipped
builders need are here: the row-invariant validator, the manifest writer and the
chain-of-thought hygiene.

The builders are OPTIONAL — `../data/` is the artifact of record and nothing in this repo
needs to rebuild it. See README.md in this directory.
"""
import json
import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # datasets/
DATA = os.path.join(ROOT, "data")
MANIFEST = os.path.join(ROOT, "gauntlet.json")


def log(m, tag="gauntlet"):
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- manifest
def load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {"dists": {}}


def register(name: str, entry: dict):
    """Merge one distribution's entry into gauntlet.json."""
    man = load_manifest()
    cur = man["dists"].get(name, {})
    cur.update(entry)
    man["dists"][name] = cur
    man["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=1, default=float)
    log(f"manifest: registered '{name}' (n={cur.get('n')})")


# ---------------------------------------------------------------- rows
def dist_path(dist: str) -> str:
    return os.path.join(DATA, f"{dist}.jsonl")


def save_dist(dist: str, rows: list, entry: dict = None):
    """Write the JSONL and register the manifest entry, validating the row invariants.

    These are the same invariants `../../gauntlet.py check` re-verifies: a distribution
    that violates one is refused at write time rather than shipped.
    """
    if not rows:
        raise RuntimeError(f"{dist}: refusing to write an EMPTY distribution")
    os.makedirs(DATA, exist_ok=True)
    for r in rows:
        if r["messages"][-1]["role"] != "assistant":
            raise RuntimeError(f"{dist}:{r.get('rid')}: last message is not assistant")
        if not str(r["messages"][-1]["content"]).strip():
            raise RuntimeError(f"{dist}:{r.get('rid')}: empty final assistant content")
        if "<think>" in str(r["messages"][-1]["content"]):
            raise RuntimeError(f"{dist}:{r.get('rid')}: CoT leaked into visible content")
    with open(dist_path(dist), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    npos = sum(int(r["y"]) for r in rows)
    log(f"{dist}: wrote {len(rows)} rows ({npos}+ / {len(rows) - npos}-)")
    e = dict(entry or {})
    e.update({"path": os.path.relpath(dist_path(dist), ROOT), "n": len(rows),
              "n_pos": npos, "n_neg": len(rows) - npos})
    register(dist, e)


# ---------------------------------------------------------------- text hygiene
THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)


def strip_think(t):
    return THINK_RE.sub("", t or "").strip()


def clean_messages(msgs, keep_roles=("system", "user", "assistant")):
    """Normalize to [{role, content}] with chain-of-thought stripped out of content.

    Returns (messages, reasoning_of_last_turn) — the reasoning is kept for `meta`, never
    left in the visible reply, because a detector must not read the suspect's scratchpad
    unless the genre says it can.
    """
    out, last_reasoning = [], None
    for m in msgs:
        role = m.get("role", "user")
        if role not in keep_roles:
            role = "user"
        content = m.get("content") or ""
        if "<think>" in content:
            content = strip_think(content)
        rc = m.get("reasoning_content")
        if rc:
            last_reasoning = rc
        out.append({"role": role, "content": content})
    return out, last_reasoning
