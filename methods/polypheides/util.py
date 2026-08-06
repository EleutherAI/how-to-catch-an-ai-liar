"""util — internal module.
"""

from __future__ import annotations

from typing import Callable

SPLIT = "test"



def load_examples(dataset_name: str):
    """internal."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=SPLIT)
    if "index" not in ds.column_names:
        ds = ds.add_column("index", list(range(len(ds))))
    return ds


def final_assistant_message(ex: dict) -> str:
    """internal."""
    for m in reversed(ex["messages"]):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def response_span(messages: list, tokenizer) -> tuple[list[int], int, int]:
    """internal."""
    try:
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_dict=True)["input_ids"]
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
            return_dict=True)["input_ids"]
        return list(full), len(prefix), len(full)
    except Exception:
        text = "\n".join(m.get("content", "") for m in messages)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids or [tokenizer.eos_token_id or 0]
        return list(ids), 0, len(ids)



def build_model(model_id: str, lora_id: str | None = None):
    """internal."""
    from transformers import AutoConfig

    try:
        from nnsight import LanguageModel
    except ImportError:
        from nnsight.modeling.language import LanguageModel

    kwargs = {"peft": lora_id} if lora_id else {}
    return LanguageModel(model_id, **kwargs)


def load_model(dataset_name: str):
    """internal."""
    ex0 = load_examples(dataset_name)[0]
    return build_model(ex0["model"], ex0.get("lora"))


def decoder_layers(model):
    """internal."""
    root = model.model
    candidates = []
    for name, child in root.named_modules():
        if name.rsplit(".", 1)[-1] != "layers":
            continue
        kids = list(child.children())
        if kids and hasattr(kids[0], "self_attn") and hasattr(kids[0], "mlp") \
                and "Decoder" in type(kids[0]).__name__:
            candidates.append((name, child))
    if len(candidates) != 1:
        try:
            inner = getattr(root, "language_model", root)
            return inner.layers
        except AttributeError:
            loose = []
            for name, child in root.named_modules():
                if name.rsplit(".", 1)[-1] != "layers":
                    continue
                low = name.lower()
                if "visual" in low or "vision" in low:
                    continue
                kids = list(child.children())
                if not kids:
                    continue
                cls = type(kids[0]).__name__
                if "Decoder" in cls or hasattr(kids[0], "mlp") or hasattr(kids[0], "mixer"):
                    loose.append((len(kids), name, child))
            if loose:
                loose.sort(key=lambda t: -t[0])
                return loose[0][2]
            raise
    return candidates[0][1]



class Batch:
    """internal."""

    def __init__(self, input_ids, attention_mask, indices, response_spans):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.indices = indices
        self.response_spans = response_spans

    def gather_last(self, h):
        """internal."""
        import torch
        return torch.stack([h[i, e - 1] for i, (s, e) in enumerate(self.response_spans)])

    def pool_response(self, h):
        """internal."""
        import torch
        return torch.stack([h[i, s:e].mean(0) for i, (s, e) in enumerate(self.response_spans)])


def chat_preprocess(messages: list, tokenizer, max_len: int = 256):
    """internal."""
    ids, s, e = response_span(messages, tokenizer)
    if max_len and len(ids) > max_len:
        cut = len(ids) - max_len
        ids, s, e = ids[cut:], max(0, s - cut), len(ids) - cut
    return ids, (s, e)


def run_full_session(dataset_name: str, detect_fn: Callable, *,
                     preprocess: Callable = chat_preprocess,
                     batch_size: int = 32, max_len: int = 256, remote: bool = True,
                     limit: int | None = None, **detect_kwargs):
    """internal."""
    import numpy as np
    import torch

    first = load_examples(dataset_name)[0]
    model_id, lora_id = first["model"], first.get("lora")
    model = build_model(model_id, lora_id)

    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    with model.session(remote=remote):
        ds = load_examples(dataset_name)
        n = len(ds) if limit is None else min(limit, len(ds))
        messages = ds["messages"][:n]
        index = list(ds["index"])[:n]

        toks, spans = [], []
        for msg in messages:
            ids, span = preprocess(msg, tok, max_len)
            toks.append(ids)
            spans.append(span)
        order = sorted(range(len(toks)), key=lambda i: len(toks[i]))

        pieces, poses = [], []
        for b0 in range(0, len(order), batch_size):
            bpos = order[b0:b0 + batch_size]
            bids = [toks[i] for i in bpos]
            w = max(len(x) for x in bids)
            input_ids = torch.tensor([x + [pad_id] * (w - len(x)) for x in bids])
            attn = torch.tensor([[1] * len(x) + [0] * (w - len(x)) for x in bids])
            batch = Batch(input_ids=input_ids, attention_mask=attn,
                          indices=[index[i] for i in bpos],
                          response_spans=[spans[i] for i in bpos])
            scores = detect_fn(model, model_id, lora_id, batch, **detect_kwargs)
            pieces.append(scores)
            poses.append(torch.tensor(bpos))

        all_scores = torch.cat(pieces).save()
        all_pos = torch.cat(poses).save()

    flat = all_scores.cpu().float().numpy()
    pos = all_pos.cpu().numpy()
    result = np.empty(len(flat), dtype=float)
    result[pos] = flat
    return result



def write_submission(indices, scores, path: str = "submission.csv", *,
                     threshold: float = 0.5, deceptive=None) -> str:
    """internal. Atomic write + validation reread: write to a temp file in the SAME dir,
    reread and validate (columns / row count / index uniqueness / score finiteness+range),
    then os.replace() onto the target. A corrupt or partial write RAISES rather than shipping."""
    import os
    import numpy as np
    import pandas as pd

    scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    indices = list(indices)
    if len(indices) != len(scores):
        raise ValueError(f"indices ({len(indices)}) and scores ({len(scores)}) differ in length")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values after clip")
    if deceptive is None:
        deceptive = scores >= threshold
    df = pd.DataFrame({"index": indices, "deceptive": np.asarray(deceptive, dtype=bool),
                       "score": scores})

    d = os.path.dirname(os.path.abspath(path)) or "."
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    try:
        df.to_csv(tmp, index=False)
        back = pd.read_csv(tmp)
        if list(back.columns) != ["index", "deceptive", "score"]:
            raise ValueError(f"reread columns {list(back.columns)} != [index, deceptive, score]")
        if len(back) != len(indices):
            raise ValueError(f"reread row count {len(back)} != {len(indices)}")
        if bool(back["index"].duplicated().any()):
            raise ValueError("reread has duplicate index values")
        bs = back["score"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(bs)) and bs.min() >= 0.0 and bs.max() <= 1.0):
            raise ValueError(f"reread score out of [0,1]/non-finite: [{bs.min()},{bs.max()}]")
        os.replace(tmp, path)   # atomic on the same filesystem
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    print(f"wrote {len(scores)} rows to {path} (atomic, validated)")
    return path
