"""pm22 — internal detector module.
Fire with: ALETHEIA_METHOD=pm22   (tag: white)
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "submission")

NAME = "pm22"
_PROBE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pm22.npz")


def _load_probe():
    import numpy as np
    z = np.load(_PROBE_PATH, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _family(model_id):
    m = model_id or ""
    if "Nemotron" in m:
        return "nemotron"
    if "gemma" in m:
        return "gemma"
    if "Qwen" in m:
        return "qwen"
    return "other"


class _Plan:
    """internal."""

    def __init__(self, probe, fam):
        import torch
        import numpy as np
        g = lambda k: np.asarray(probe[f"{fam}::{k}"])
        self.read_layer = int(g("layer"))
        self.read_point = str(g("site"))
        self.needs_native_config = fam == "nemotron"
        self.mu = torch.tensor(g("mu"), dtype=torch.float32)
        self.sd = torch.tensor(g("sd"), dtype=torch.float32)
        self.d = torch.tensor(g("d"), dtype=torch.float32)

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
        h = h.float()
        dev = h.device
        mu, sd, d = self.mu.to(dev), self.sd.to(dev), self.d.to(dev)
        feats = []
        for i, (s, e) in enumerate(spans):
            pr = ((h[i, s:e] - mu) / sd) @ d
            k = min(3, pr.shape[0])
            feats.append(torch.stack([pr.mean(), pr.max(), pr.topk(k).values.mean()]))
        return torch.stack(feats)

    def score(self, feats):
        import numpy as np
        F = np.asarray(feats, float)
        if F.ndim == 1:
            F = F[:, None]
        x = F[:, min(2, F.shape[1] - 1)]                 # top3mean pooled projection
        med = float(np.median(x))
        sd = float(np.std(x))
        if not np.isfinite(sd) or sd < 1e-9:
            return np.full(len(x), 0.5)
        z = np.clip((x - med) / sd, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))


def build(model_id, lora_id, task=None):
    probe = _load_probe()
    fam = _family(model_id)
    if f"{fam}::d" not in probe:
        return None
    if str(probe.get(f"{fam}::scope", "all")) != "all":
        return None                      # direction validated only for the gated reader (jm49)
    return _Plan(probe, fam)
