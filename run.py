#!/usr/bin/env python
"""run.py — run any bundled method on any bundled dataset, locally or on NDIF.

    python run.py --method abantes   --dataset wf_assert
    python run.py --method tiresias  --dataset notus-lie-auditor --suspect gemma-3-27b-it
    python run.py --method tiresias  --dataset reinthal/notus-lie-auditor-Qwen3.5-27B

--method   abantes | polypheides | tiresias  (the three frozen submission bundles)
--dataset  a battery id from datasets/data/ (see datasets/README.md), a path to a
           battery-format .jsonl, a variant handle (e.g. notus-lie-auditor), or a full
           HF slug following the <slug> + <slug>-labels convention
--suspect  which model the dataset was generated on. Short key (Qwen3.5-27B,
           gemma-3-27b-it, NVIDIA-Nemotron-3-Super-120B), full HF id, or "model:lora".
           Default: Qwen3.5-27B.
--backend  local (default; models on your GPUs) | ndif (needs NDIF_API_KEY)

The suspect and the data are coupled: --suspect selects which edition of a variant to
load (notus-lie-auditor -> reinthal/notus-lie-auditor-<suspect>), and a full slug's
suspect is read from its suffix. Methods that read the suspect (tiresias) therefore always
interrogate the model that actually produced the replies. Judge-only methods (abantes,
polypheides) never load the suspect.

Writes submission.csv (index, deceptive, score); prints AUROC / balanced accuracy against
labels plus the dataset's recorded nulls. VRAM: judge (Qwen3.5-9B) ~20 GB; a 27B suspect
~55 GB (judge+suspect fit one 80 GB H100); the Nemotron-120B suspect is NDIF-only.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# competition suspect families: short key -> HF model id
SUSPECTS = {
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "NVIDIA-Nemotron-3-Super-120B": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}
DEFAULT_SUSPECT = "Qwen3.5-27B"

# dataset variants published per-suspect on HF: variant -> slug template
VARIANTS = {
    "notus-lie-auditor": "reinthal/notus-lie-auditor-{suspect}",
}

METHODS = {
    "abantes": ("methods/abantes", "dyad_plainfuse"),
    "polypheides": ("methods/polypheides", "rhadamanthys_dyad_anchor2h"),
    "tiresias": ("methods/tiresias", "rhadamanthys_m53_tp"),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--suspect", default=None,
                    help="model the dataset was generated on (short key / HF id / "
                         "'model:lora'); default " + DEFAULT_SUSPECT)
    ap.add_argument("--backend", choices=("local", "ndif"), default="local")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--blind", action="store_true",
                    help="run without any suspect (judge legs only). Degrades methods that "
                         "read the suspect (tiresias) — not recommended")
    return ap.parse_args()


def resolve_suspect(spec):
    """spec: short key | HF id | 'model:lora' | None -> (key_or_None, hf_id, lora)."""
    if spec is None:
        return DEFAULT_SUSPECT, SUSPECTS[DEFAULT_SUSPECT], None
    model, _, lora = spec.partition(":")
    lora = lora or None
    if model in SUSPECTS:
        return model, SUSPECTS[model], lora
    key = next((k for k, v in SUSPECTS.items() if v == model), None)  # HF id -> key
    return key, model, lora


def suspect_from_slug(slug):
    """A known suspect key if an HF slug ends with its suffix, else None."""
    return next((k for k in SUSPECTS if slug.endswith("-" + k)), None)


def suspect_from_genmodel(gm):
    """Map a battery row's gen_model string onto a loadable suspect key, else None."""
    g = (gm or "").lower()
    if "nemotron" in g:
        return "NVIDIA-Nemotron-3-Super-120B"
    if "gemma-3-27b" in g:
        return "gemma-3-27b-it"
    if "qwen3.5-27b" in g:
        return "Qwen3.5-27B"
    return None


def resolve_battery(dataset: str):
    """Return (path, dist_id) if the dataset is a battery file/id, else None."""
    if dataset.endswith(".jsonl"):
        path = dataset if os.path.isabs(dataset) else os.path.join(ROOT, dataset)
        if not os.path.exists(path):
            sys.exit(f"no such file: {dataset}")
        return path, os.path.basename(path)[:-6]
    path = os.path.join(ROOT, "datasets", "data", f"{dataset}.jsonl")
    return (path, dataset) if os.path.exists(path) else None


def battery_task(dist_id: str, genre: str) -> str:
    """Map a battery distribution onto the harness's task vocabulary."""
    text = f"{dist_id} {genre}".lower()
    if any(k in text for k in ("insider", "agentic", "audit", "conceal")):
        return "lie-auditors"
    if "instructed" in text:
        return "instructed-deception"
    return "varied-deception"


def make_build_model(torch):
    def build_model(model_id, lora_id=None):
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
        # no peft= path in VisionLanguageModel: merge the adapter (inference-identical)
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from peft import PeftModel
        base = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto")
        merged = PeftModel.from_pretrained(base, lora_id).merge_and_unload()
        return VisionLanguageModel(merged, tokenizer=AutoTokenizer.from_pretrained(model_id))
    return build_model


def main():
    args = parse_args()
    bundle_rel, armed_method = METHODS[args.method]
    bundle = os.path.join(ROOT, bundle_rel)

    # the bundle owns imports (harness, util, methods). Drop ROOT from sys.path so the
    # top-level datasets/ directory cannot shadow the HuggingFace `datasets` package.
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != ROOT]
    sys.path.insert(0, bundle)
    os.environ.setdefault("ALETHEIA_METHOD", armed_method)

    import torch
    if args.backend == "local":
        torch.set_grad_enabled(False)      # remote saves come back detached; match locally
        import util
        util.build_model = make_build_model(torch)   # harness binds this at call time
        remote = False
    else:
        if "NDIF_API_KEY" not in os.environ:
            sys.exit("--backend ndif needs NDIF_API_KEY in the environment")
        from nnsight import CONFIG
        CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
        remote = True

    suspect_key, suspect_hf, suspect_lora = resolve_suspect(args.suspect)
    man, dist_id, labels = {}, None, None
    battery = None if args.dataset in VARIANTS else resolve_battery(args.dataset)

    if battery is not None:
        # ---- bundled battery distribution ---------------------------------------
        path, dist_id = battery
        rows = [json.loads(line) for line in open(path)]
        labels = [int(r["y"]) for r in rows]
        try:
            man = json.load(open(os.path.join(ROOT, "datasets", "gauntlet.json")))["dists"]
        except FileNotFoundError:
            pass
        genre = man.get(dist_id, {}).get("genre", "")
        # couple the suspect to the data: use the file's own generator when it is a single
        # loadable competition family; otherwise honour --suspect / the default.
        gms = {suspect_from_genmodel(r.get("gen_model")) for r in rows}
        if len(gms) == 1 and next(iter(gms)) is not None and args.suspect is None:
            suspect_key = next(iter(gms))
            suspect_hf, suspect_lora = SUSPECTS[suspect_key], None
        if args.blind:
            suspect_hf, suspect_lora = "", None
        from datasets import Dataset
        examples = Dataset.from_list([
            {"index": i, "messages": r["messages"], "model": suspect_hf, "lora": suspect_lora}
            for i, r in enumerate(rows)])
        run_name = f"battery/{battery_task(dist_id, genre)}-{dist_id}"
        import util
        util.load_examples = lambda _name, _ex=examples: _ex
    else:
        # ---- HF dataset: variant handle or full slug ----------------------------
        if args.dataset in VARIANTS:                    # variant -> pick edition by suspect
            slug = VARIANTS[args.dataset].format(suspect=suspect_key or DEFAULT_SUSPECT)
        else:
            slug = args.dataset
            slug_key = suspect_from_slug(slug)
            if slug_key is not None:                    # slug suffix is authoritative
                if args.suspect and suspect_key != slug_key:
                    print(f"[run] note: --suspect {args.suspect} ignored; {slug} was generated "
                          f"on {slug_key}", flush=True)
                suspect_key, suspect_hf, suspect_lora = slug_key, SUSPECTS[slug_key], None

        from datasets import load_dataset
        ds = load_dataset(slug, split="test")
        if "index" not in ds.column_names:
            ds = ds.add_column("index", list(range(len(ds))))
        if args.blind:
            suspect_hf = ""
            ds = ds.map(lambda _: {"model": "", "lora": None})
        elif suspect_from_slug(slug) or args.dataset in VARIANTS or args.suspect:
            # coupled/explicit suspect: pin every row to it
            ds = ds.map(lambda _r: {"model": suspect_hf, "lora": suspect_lora})
        elif "model" not in ds.column_names:            # unknown slug, no model column
            ds = ds.map(lambda _: {"model": suspect_hf, "lora": suspect_lora})
        else:                                           # unknown slug carries its own model
            suspect_hf = "from dataset"
        try:
            lab = load_dataset(slug + "-labels", split="test")
            by_idx = {int(r["index"]): int(bool(r["deceptive"])) for r in lab}
            labels = [by_idx.get(int(i)) for i in ds["index"]]
            if any(v is None for v in labels):
                labels = None
        except Exception:
            labels = None
        run_name = slug.replace("lie-auditor-", "lie-auditors-")
        import util
        util.load_examples = lambda _name, _ex=ds: _ex

    import harness
    import methods
    print(f"[run] method={armed_method} dataset={args.dataset} "
          f"suspect={'blind' if args.blind else suspect_hf}"
          f"{(':' + suspect_lora) if suspect_lora else ''} backend={args.backend} "
          f"limit={args.limit}", flush=True)
    harness.run_submission(methods.load(armed_method), run_name,
                           limit=args.limit, batch_size=args.batch_size,
                           max_len=256, remote=remote)

    import pandas as pd
    sub = pd.read_csv("submission.csv")
    print(f"[run] submission.csv: {len(sub)} rows, mean score {sub['score'].mean():.3f}")
    if labels is not None:
        y = labels[:len(sub)]
        s = list(sub["score"])
        pos = [v for v, t in zip(s, y) if t]; neg = [v for v, t in zip(s, y) if not t]
        if pos and neg:
            wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
            auroc = wins / (len(pos) * len(neg))
            tpr = sum(v >= 0.5 for v in pos) / len(pos)
            tnr = sum(v < 0.5 for v in neg) / len(neg)
            print(f"[run] n={len(y)} pos={len(pos)}  AUROC={auroc:.4f}  "
                  f"bal_acc@0.5={(tpr + tnr) / 2:.4f}")
            nulls = man.get(dist_id, {}).get("nulls") if battery else None
            if nulls:
                bn = nulls.get("blindnull", {}).get("auroc")
                ln = nulls.get("lennull", {}).get("auroc")
                print(f"[run] dataset nulls — blind-judge {bn:.3f}, length-only {ln:.3f} "
                      f"(beat the blind null to claim the method read the lie)")
        else:
            print(f"[run] single-class slice (pos={len(pos)}) — raise --limit for metrics")


if __name__ == "__main__":
    main()
