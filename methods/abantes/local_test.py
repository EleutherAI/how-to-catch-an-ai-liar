"""local_test — run the frozen competition method WITHOUT NDIF, on local GPUs.

Identical pipeline to remote_test.py (same harness, same method files, byte-identical
prompts and logit reads); the only differences are (a) every model is loaded onto local
GPUs via nnsight's local execution path (device_map=auto, bfloat16) and (b) no NDIF API
key is needed. HF_TOKEN is still required for gated datasets/models.

VRAM guide: Polypheides builds only the trusted judge (Qwen/Qwen3.5-9B, ~20 GB bf16 —
one 24 GB GPU). Ampyx additionally loads the suspect model named in the dataset rows
(a 27B suspect is ~55 GB bf16 → two 48 GB GPUs; the Nemotron-120B suspect needs ~240 GB
— use the NDIF path for that one).

Usage (from this directory):
    ALETHEIA_METHOD=dyad_plainfuse \
    DATASET_NAME=aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4 \
      python local_test.py
Add ALETHEIA_LIMIT=8 for a smoke run. Writes submission.csv (index, deceptive, score).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASET = os.environ["DATASET_NAME"]
LIMIT = int(os.environ["ALETHEIA_LIMIT"]) if os.environ.get("ALETHEIA_LIMIT") else None
METHOD = os.environ.get("ALETHEIA_METHOD", "dyad_plainfuse")

import torch  # noqa: E402
import util   # noqa: E402

torch.set_grad_enabled(False)   # remote saves come back detached; match that locally


def _local_build_model(model_id: str, lora_id: str | None = None):
    """util.build_model, but dispatched onto local GPUs instead of NDIF.

    Falls back to VisionLanguageModel for architectures transformers registers as
    multimodal (the Qwen3.5 family) — same envoy tree as the remote path, so the
    frozen method code traces identically."""
    try:
        from nnsight import LanguageModel
    except ImportError:
        from nnsight.modeling.language import LanguageModel
    kwargs = {"peft": lora_id} if lora_id else {}
    try:
        return LanguageModel(model_id, device_map="auto", dtype=torch.bfloat16, **kwargs)
    except ValueError as e:
        if "VisionLanguageModel" not in str(e):
            raise
    from nnsight import VisionLanguageModel
    if lora_id is None:
        return VisionLanguageModel(model_id, device_map="auto", dtype=torch.bfloat16)
    # VisionLanguageModel has no peft= path: merge the adapter into the base weights
    # (numerically identical for inference; keeps the envoy tree the methods trace)
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto")
    merged = PeftModel.from_pretrained(base, lora_id).merge_and_unload()
    return VisionLanguageModel(merged, tokenizer=AutoTokenizer.from_pretrained(model_id))


util.build_model = _local_build_model   # harness binds this at call time

import harness   # noqa: E402
import methods   # noqa: E402

print(f"[local] method={METHOD} dataset={DATASET.split('/')[-1]} limit={LIMIT} "
      f"gpus={torch.cuda.device_count()}", flush=True)
harness.run_submission(methods.load(METHOD), DATASET, limit=LIMIT,
                       batch_size=8, max_len=256, remote=False)

import pandas as pd  # noqa: E402
sub = pd.read_csv("submission.csv")
assert {"index", "deceptive", "score"} <= set(sub.columns), "submission.csv missing columns"
print(f"[local] OK — {len(sub)} rows, mean score {sub['score'].mean():.3f}, "
      f"frac deceptive {sub['deceptive'].mean():.2f}", flush=True)
