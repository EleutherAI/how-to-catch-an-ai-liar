#!/usr/bin/env python
"""run.py — run any bundled method on any bundled dataset, locally or on NDIF.

    python run.py --method abantes   --dataset wf_assert
    python run.py --method ampyx     --dataset notus-lie-auditor --suspect gemma-3-27b-it
    python run.py --method ampyx     --dataset reinthal/notus-lie-auditor-Qwen3.5-27B

--method   abantes | polypheides | ampyx  (the three frozen submission bundles)
--dataset  a battery id from datasets/data/ (see datasets/README.md), a path to a
           battery-format .jsonl, a variant handle (e.g. notus-lie-auditor), or a full
           HF slug following the <slug> + <slug>-labels convention
--suspect  which model the dataset was generated on. Short key (Qwen3.5-27B,
           gemma-3-27b-it, NVIDIA-Nemotron-3-Super-120B), full HF id, or "model:lora".
           Default: Qwen3.5-27B.
--backend  local (default; models on your GPUs) | ndif (needs NDIF_API_KEY)
--strict   exit non-zero if any leg of the method fell back instead of running

The suspect and the data are coupled: --suspect selects which edition of a variant to
load (notus-lie-auditor -> reinthal/notus-lie-auditor-<suspect>), and a full slug's
suspect is read from its suffix. Methods that read the suspect (ampyx) therefore always
interrogate the model that actually produced the replies. Judge-only methods (abantes,
polypheides) never load the suspect.

Writes submission.csv (index, deceptive, score); prints a health line naming what each leg
actually did, then AUROC / balanced accuracy against labels plus the dataset's recorded
nulls. VRAM: judge (Qwen3.5-9B) ~20 GB; a 27B suspect ~55 GB (judge+suspect fit one 80 GB
H100); the Nemotron-120B suspect is NDIF-only.
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

# the trusted judge every method reads (methods/*/methods/pm18.py::AUTHOR_MODEL)
JUDGE_MODEL_ID = "Qwen/Qwen3.5-9B"

# dataset variants published per-suspect on HF: variant -> slug template
VARIANTS = {
    "notus-lie-auditor": "reinthal/notus-lie-auditor-{suspect}",
}

METHODS = {
    "abantes": ("methods/abantes", "dyad_plainfuse"),
    "polypheides": ("methods/polypheides", "rhadamanthys_dyad_anchor2h"),
    "ampyx": ("methods/ampyx", "rhadamanthys_m53r"),
}

# Methods that read the suspect model's activations/logprobs; the rest are judge-only.
SUSPECT_READING = {"ampyx"}


# ---------------------------------------------------------------- health
# These methods are built to degrade rather than crash: a dead leg is replaced by a
# cheaper one and the run still produces a full score vector. That is right for a
# competition and wrong for a reader, because a degraded run and a faithful one print
# the same kind of AUROC. The markers below are the substrings the frozen modules emit
# when a leg falls back — collected from their own print statements, so no method file
# has to be touched. The list errs toward flagging: a false alarm costs you a look at
# the log, a miss costs you a number that is not this method's.
HEALTH_DEGRADED = [
    ("-> plain forward judge fallback", "anchor died, plain judge substituted"),
    ("provenance=PLAIN_FALLBACK", "anchor died, plain judge substituted"),
    ("anchored two-pass failed", "anchor two-pass raised"),
    ("anchored ynd degenerate", "anchor read degenerate"),
    ("-> per-row 0.5", "leg dead, rows written as 0.5"),
    ("-> constant 0.5 (DEAD)", "nothing usable survived"),
    ("quarantining", "rows quarantined to 0.5"),
    ("-> pin array", "added channel dropped, anchor used alone"),
    ("spine only", "white leg unusable, spine only"),
    ("FUSE mode=anchor_only", "se3p leg dropped, anchor used alone"),
    ("spine dead on an ungated cell", "spine dead"),
    ("judge build failed", "judge model failed to build"),
    ("default build failed", "suspect model failed to build"),
    ("cheap channel unusable", "cheap channel unusable"),
    ("cheap valid-mask unusable", "cheap channel unusable"),
    ("fused output degenerate", "fusion degenerate"),
    ("fuse failed", "fusion raised"),
    ("twin run failed", "method raised; harness fell back"),
    ("judge_rescue scored", "method raised; rescued by the plain judge"),
    ("→ scoring 0.5", "method raised with no rescue; constant 0.5"),
    ("suspect path failed", "suspect read failed"),
    ("[sm11] run failed", "se3p self-evaluation leg died"),
]
# Positive markers: the leg ran as designed.
HEALTH_OK = [
    ("provenance=TWO_PASS", "anchor=TWO_PASS"),
    ("FUSE mode=fused", "fusion=fused"),
    ("twin run scored", "method.run completed"),
]
# Counted markers: healthy only if every row made it through. Each regex yields
# (label, got, want); got < want is a degradation even though nothing printed "failed".
HEALTH_COUNTED = [
    (r"DYAD shielded in \S+\s+rows=(?P<want>\d+) dead=(?P<dead>\d+) "
     r"restated=\d+/\d+ scored=(?P<got>\d+)/\d+", "anchor=DYAD shielded"),
    (r"fused: w=\S+ rho=\S+ included=(?P<got>\d+)/(?P<want>\d+)", "fusion=fused"),
]
# Informational: expected on some data, never a failure. The activation gate is meant to
# select only agentic record-vs-report rows and selects none on most genres.
HEALTH_INFO_PREFIXES = ("agentic gate", "gate:", "claim gate:")


class _Tee:
    """Pass stdout through unchanged while keeping the lines for the health scan."""

    def __init__(self, stream):
        self._s = stream
        self._buf = ""
        self.lines = []

    def write(self, text):
        self._s.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.lines.append(line)
        return len(text)

    def flush(self):
        self._s.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


def summarize_health(lines):
    """-> {'ok': bool, 'ran': [...], 'degraded': [...], 'info': [...]}"""
    import re
    ran, degraded, info = [], [], []

    def add(seq, item):
        if item not in seq:
            seq.append(item)

    counted = [(re.compile(pat), label) for pat, label in HEALTH_COUNTED]
    for line in lines:
        for needle, label in HEALTH_DEGRADED:
            if needle in line:
                add(degraded, label)
        for needle, label in HEALTH_OK:
            if needle in line:
                add(ran, label)
        for rx, label in counted:
            m = rx.search(line)
            if not m:
                continue
            got, want = int(m.group("got")), int(m.group("want"))
            dead = int(m.groupdict().get("dead") or 0)
            if got >= want and dead == 0:
                add(ran, f"{label} ({got}/{want})")
            else:
                add(degraded, f"{label}: only {got}/{want} rows scored"
                              + (f", {dead} dead" if dead else ""))
        body = line.split("] ", 1)[-1] if line.startswith("[") else line
        if body.startswith(HEALTH_INFO_PREFIXES):
            add(info, body.strip())
    return {"ok": not degraded, "ran": ran, "degraded": degraded, "info": info}


def format_health(h):
    parts = list(h["ran"]) or ["no positive leg marker seen"]
    parts += [f"DEGRADED: {d}" for d in h["degraded"]]
    parts += h["info"]
    return " | ".join(parts)


# ---------------------------------------------------------------- preflight
def preflight_local(torch, method):
    """Fail loudly instead of running the whole pipeline on CPU."""
    if not torch.cuda.is_available():
        sys.exit(
            "[run] --backend local needs CUDA, but torch.cuda.is_available() is False.\n"
            f"      torch {torch.__version__} was built for a CUDA version your driver may\n"
            "      not support (uv.lock pins a CUDA 13 wheel). Install a matching torch --\n"
            "      --reinstall is required, or uv keeps the wheel already installed:\n"
            "        uv pip install --reinstall --index-url "
            "https://download.pytorch.org/whl/cu126 torch torchvision\n"
            "      or run the models remotely with --backend ndif.")
    n = torch.cuda.device_count()
    tot = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 2**30
    need = "~20 GB (judge only)" if method not in SUSPECT_READING else \
           "~20 GB judge + ~55 GB for a 27B suspect"
    print(f"[run] {n} CUDA device(s), {tot:.0f} GiB total; this method needs {need}",
          flush=True)


def _max_memory(torch, gpus, reserve_gib=2):
    """Restrict accelerate to `gpus` by listing only those devices in max_memory."""
    if not gpus:
        return None
    mm = {}
    for i in gpus:
        if i >= torch.cuda.device_count():
            sys.exit(f"[run] --*-gpus names device {i}, but only "
                     f"{torch.cuda.device_count()} are visible")
        cap = torch.cuda.get_device_properties(i).total_memory / 2**30
        mm[i] = f"{max(cap - reserve_gib, 1):.0f}GiB"
    return mm


def make_build_model(torch, judge_gpus=None, suspect_gpus=None):
    """A build_model that memoizes by (model_id, lora_id) and can pin models to GPUs.

    Memoization matters for the gauntlet sweep: without it the 9B judge is rebuilt once
    per distribution. Pinning matters because device_map="auto" lets whichever model is
    built first spread over every visible GPU, which on a multi-GPU box can leave the
    suspect without room and silently degrade the run.
    """
    cache = {}

    def build_model(model_id, lora_id=None):
        key = (model_id, lora_id)
        if key in cache:
            return cache[key]
        try:
            from nnsight import LanguageModel
        except ImportError:
            from nnsight.modeling.language import LanguageModel
        gpus = judge_gpus if model_id == JUDGE_MODEL_ID else suspect_gpus
        placement = {"device_map": "auto", "dtype": torch.bfloat16}
        mm = _max_memory(torch, gpus)
        if mm:
            placement["max_memory"] = mm
            print(f"[run] {model_id} pinned to GPU(s) {sorted(mm)}", flush=True)
        kwargs = dict(placement, **({"peft": lora_id} if lora_id else {}))
        try:
            model = LanguageModel(model_id, **kwargs)
        except ValueError as e:
            if "VisionLanguageModel" not in str(e):
                raise
            from nnsight import VisionLanguageModel
            if lora_id is None:
                model = VisionLanguageModel(model_id, **placement)
            else:
                # no peft= path in VisionLanguageModel: merge the adapter
                # (inference-identical)
                from transformers import AutoModelForImageTextToText, AutoTokenizer
                from peft import PeftModel
                base = AutoModelForImageTextToText.from_pretrained(model_id, **placement)
                merged = PeftModel.from_pretrained(base, lora_id).merge_and_unload()
                model = VisionLanguageModel(
                    merged, tokenizer=AutoTokenizer.from_pretrained(model_id))
        cache[key] = model
        return model

    return build_model


# ---------------------------------------------------------------- resolution
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


def metrics(scores, labels):
    """AUROC (with tie correction) and balanced accuracy at the fixed 0.5 cut."""
    pos = [v for v, t in zip(scores, labels) if t]
    neg = [v for v, t in zip(scores, labels) if not t]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    tpr = sum(v >= 0.5 for v in pos) / len(pos)
    tnr = sum(v < 0.5 for v in neg) / len(neg)
    return {"auroc": wins / (len(pos) * len(neg)), "bal_acc": (tpr + tnr) / 2,
            "n": len(labels), "n_pos": len(pos)}


# ---------------------------------------------------------------- the bundle
_BUNDLE = None      # (method_name, armed_method) — a process serves one bundle only


def select_bundle(method, backend, torch, judge_gpus=None, suspect_gpus=None):
    """Put the bundle on sys.path and arm its method. Idempotent per process."""
    global _BUNDLE
    bundle_rel, armed = METHODS[method]
    if _BUNDLE is not None:
        if _BUNDLE[0] != method:
            raise RuntimeError(
                f"this process already loaded the '{_BUNDLE[0]}' bundle; the bundles are "
                f"module-level singletons, so run '{method}' in a separate process")
        return armed
    bundle = os.path.join(ROOT, bundle_rel)
    # the bundle owns imports (harness, util, methods). Drop ROOT from sys.path so the
    # top-level datasets/ directory cannot shadow the HuggingFace `datasets` package.
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != ROOT]
    sys.path.insert(0, bundle)
    os.environ["ALETHEIA_METHOD"] = armed
    if backend == "local":
        torch.set_grad_enabled(False)   # remote saves come back detached; match locally
        import util
        # harness binds util.build_model at call time, so replacing it here is enough
        util.build_model = make_build_model(torch, judge_gpus, suspect_gpus)
    else:
        if "NDIF_API_KEY" not in os.environ:
            sys.exit("--backend ndif needs NDIF_API_KEY in the environment")
        from nnsight import CONFIG
        CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
    _BUNDLE = (method, armed)
    return armed


def score_dataset(method, dataset, *, suspect=None, backend="local", limit=None,
                  batch_size=8, blind=False, judge_gpus=None, suspect_gpus=None,
                  preflight=True):
    """Score one dataset with one bundled method.

    Returns {scores, labels, dist_id, metrics, health, nulls, suspect}. Writes
    submission.csv in the current directory (the harness's own output contract).
    """
    import torch
    if backend == "local" and preflight:
        preflight_local(torch, method)
    armed = select_bundle(method, backend, torch, judge_gpus, suspect_gpus)
    remote = backend != "local"

    suspect_key, suspect_hf, suspect_lora = resolve_suspect(suspect)
    man, dist_id, labels = {}, None, None
    battery = None if dataset in VARIANTS else resolve_battery(dataset)
    import util

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
        # loadable competition family; otherwise honour suspect= / the default.
        gms = {suspect_from_genmodel(r.get("gen_model")) for r in rows}
        if len(gms) == 1 and next(iter(gms)) is not None and suspect is None:
            suspect_key = next(iter(gms))
            suspect_hf, suspect_lora = SUSPECTS[suspect_key], None
        if blind:
            suspect_hf, suspect_lora = "", None
        from datasets import Dataset
        examples = Dataset.from_list([
            {"index": i, "messages": r["messages"], "model": suspect_hf, "lora": suspect_lora}
            for i, r in enumerate(rows)])
        run_name = f"battery/{battery_task(dist_id, genre)}-{dist_id}"
        util.load_examples = lambda _name, _ex=examples: _ex
    else:
        # ---- HF dataset: variant handle or full slug ----------------------------
        if dataset in VARIANTS:                         # variant -> pick edition by suspect
            slug = VARIANTS[dataset].format(suspect=suspect_key or DEFAULT_SUSPECT)
        else:
            slug = dataset
            slug_key = suspect_from_slug(slug)
            if slug_key is not None:                    # slug suffix is authoritative
                if suspect and suspect_key != slug_key:
                    print(f"[run] note: suspect {suspect} ignored; {slug} was generated "
                          f"on {slug_key}", flush=True)
                suspect_key, suspect_hf, suspect_lora = slug_key, SUSPECTS[slug_key], None

        from datasets import load_dataset
        ds = load_dataset(slug, split="test")
        if "index" not in ds.column_names:
            ds = ds.add_column("index", list(range(len(ds))))
        if blind:
            suspect_hf = ""
            ds = ds.map(lambda _: {"model": "", "lora": None})
        elif suspect_from_slug(slug) or dataset in VARIANTS or suspect:
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
        util.load_examples = lambda _name, _ex=ds: _ex

    import harness
    import methods
    print(f"[run] method={armed} dataset={dataset} "
          f"suspect={'blind' if blind else suspect_hf}"
          f"{(':' + suspect_lora) if suspect_lora else ''} backend={backend} "
          f"limit={limit}", flush=True)

    tee = _Tee(sys.stdout)
    real_stdout = sys.stdout
    sys.stdout = tee
    try:
        harness.run_submission(methods.load(armed), run_name, limit=limit,
                               batch_size=batch_size, max_len=256, remote=remote)
    finally:
        sys.stdout = real_stdout
    health = summarize_health(tee.lines)

    import pandas as pd
    sub = pd.read_csv("submission.csv")
    scores = [float(v) for v in sub["score"]]
    y = labels[:len(scores)] if labels is not None else None
    return {"scores": scores, "labels": y, "dist_id": dist_id,
            "metrics": metrics(scores, y) if y is not None else None,
            "health": health, "suspect": suspect_hf,
            "nulls": (man.get(dist_id, {}).get("nulls") if battery else None)}


# ---------------------------------------------------------------- CLI
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
                         "read the suspect (ampyx) — not recommended")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any leg of the method fell back instead of running")
    ap.add_argument("--judge-gpus", default=None, metavar="I,J",
                    help="pin the trusted judge to these CUDA devices")
    ap.add_argument("--suspect-gpus", default=None, metavar="I,J",
                    help="pin the suspect model to these CUDA devices")
    return ap.parse_args()


def _gpu_list(spec):
    return [int(x) for x in spec.split(",") if x.strip() != ""] if spec else None


def main():
    args = parse_args()
    res = score_dataset(args.method, args.dataset, suspect=args.suspect,
                        backend=args.backend, limit=args.limit,
                        batch_size=args.batch_size, blind=args.blind,
                        judge_gpus=_gpu_list(args.judge_gpus),
                        suspect_gpus=_gpu_list(args.suspect_gpus))

    n = len(res["scores"])
    mean = sum(res["scores"]) / n if n else float("nan")
    print(f"[run] submission.csv: {n} rows, mean score {mean:.3f}")
    print(f"[run] health: {format_health(res['health'])}")

    m = res["metrics"]
    if res["labels"] is None:
        print("[run] no labels for this dataset — scores only")
    elif m is None:
        print(f"[run] single-class slice — raise --limit for metrics")
    else:
        print(f"[run] n={m['n']} pos={m['n_pos']}  AUROC={m['auroc']:.4f}  "
              f"bal_acc@0.5={m['bal_acc']:.4f}")
        nulls = res["nulls"]
        if nulls:
            bn = nulls.get("blindnull", {}).get("auroc")
            ln = nulls.get("lennull", {}).get("auroc")
            print(f"[run] dataset nulls — blind-judge {bn:.3f}, length-only {ln:.3f} "
                  f"(beat the blind null to claim the method read the lie)")

    if args.strict and not res["health"]["ok"]:
        print("[run] --strict: a leg fell back, so these numbers are not this method's",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
