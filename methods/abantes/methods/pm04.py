"""pm04 — internal detector module.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

NAME = "pm04"
_PROBE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "probe_v3.npz")


def _load_probe():
    import numpy as np
    z = np.load(_PROBE_PATH, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _family(model_id):
    m = model_id or ""
    if "Nemotron" in m: return "nemotron"
    if "gemma" in m: return "gemma"
    if "Qwen" in m: return "qwen"
    return "other"


def _select(probe, model_id, lora_id, task):
    """internal."""
    fam = _family(model_id)
    if fam == "other":
        return None, None
    tkey = task if task in ("varied", "instructed") else "comb"
    cands = []
    if lora_id:
        cands += [f"{lora_id}::{tkey}", f"{lora_id}::comb", f"{lora_id}::instructed", f"{lora_id}::varied"]
    cands += [f"{model_id}::None::{tkey}", f"{model_id}::None::instructed", f"{model_id}::None::varied"]
    if fam == "qwen":
        cands += ["UNIV"]
    pre = next((c for c in cands if f"{c}::d" in probe), None)
    if pre is None:
        return None, None
    return pre, ("residual" if fam == "nemotron" else "mlp")


class _PlanV3:
    def __init__(self, probe, pre, read_point):
        import torch, numpy as np
        g = lambda k: np.asarray(probe[f"{pre}::{k}"])
        self.univ = pre == "UNIV"
        self.read_layer = int(g("layer")); self.read_point = read_point
        self.needs_native_config = "Nemotron" in pre
        self.mu = torch.tensor(g("mu")); self.sd = torch.tensor(g("sd")); self.d = torch.tensor(g("d"))
        self.fmu = g("fmu"); self.fsd = g("fsd"); self.w = g("w"); self.b = float(g("b"))

    def tokenize(self, messages, tok, max_len=256):
        from detect_logic import content_preprocess
        ids, (s, e) = content_preprocess(messages, tok, max_len)
        return ids, s, e

    def extract(self, DECODER, spans):
        import torch
        layer = DECODER[self.read_layer]
        h = layer.output if self.read_point == "residual" else layer.mlp.down_proj.input
        if not hasattr(h, "shape"):
            h = h[0]
        dev = h.device
        mu, sd, d = self.mu.to(dev), self.sd.to(dev), self.d.to(dev)
        feats = []
        for i, (s, e) in enumerate(spans):
            pr = ((h[i, s:e] - mu) / sd) @ d
            if self.univ:
                feats.append(pr.mean().reshape(1))
            else:
                k = min(3, pr.shape[0])
                feats.append(torch.stack([pr.mean(), pr.max(), pr.topk(k).values.mean()]))
        return torch.stack(feats)

    def score(self, feats):
        import numpy as np
        F = np.asarray(feats, float)
        if F.ndim == 1:
            F = F[:, None]
        logit = ((F - self.fmu) / self.fsd) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def build(model_id, lora_id, task=None):
    probe = _load_probe()
    pre, read_point = _select(probe, model_id, lora_id, task)
    if pre is None:
        return None
    return _PlanV3(probe, pre, read_point)
