"""v9 checkpoint verifier: does this RL checkpoint load the way the eval expects?

v9_rwr_train.py writes v8's exact layout (config.json + model.safetensors + tokenizer
at the run root, plus latest/), so v8_eval_precompress.py needs no changes -- it just
takes MODEL_VOL_PATH. The one thing worth checking before pointing an eval sweep at a
checkpoint is that it actually loads with the eval's own two calls, on the volume,
with the right head shape and vocab. That is all this does.

Runs on CPU: loading a 140M encoder needs no GPU, and a GPU here would only queue.

Run:
    modal run v9_eval_export.py --exp-path exp-20260726-013000-v9-rwr
    modal run v9_eval_export.py --exp-path exp-20260726-013000-v9-rwr/latest

Then hand the SAME path to the existing eval:
    MODEL_VOL=otso-v8-training MODEL_VOL_PATH=<exp-path> MODEL_ALIAS=v9rwr \
        modal run precompress_v72_focus.py
"""

import json
from pathlib import Path

import modal

app = modal.App("otso-v9-eval-export")

output_vol = modal.Volume.from_name("otso-v8-training", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("scratch-compression-hf-cache",
                                      create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "safetensors", "sentencepiece", "protobuf")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TRANSFORMERS_VERBOSITY": "error"})
)

OUTPUT_VOL = "/output"
HF_CACHE = "/hf-cache"
EXPECTED_FILES = ("config.json", "tokenizer_config.json")


@app.function(image=image, timeout=900,
              volumes={OUTPUT_VOL: output_vol, HF_CACHE: hf_cache_vol})
def verify(exp_path: str):
    import os
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    path = Path(exp_path if exp_path.startswith("/") else f"{OUTPUT_VOL}/{exp_path}")
    if not path.exists():
        avail = sorted(d.name for d in Path(OUTPUT_VOL).iterdir() if d.is_dir())
        raise FileNotFoundError(f"{path} not on otso-v8-training. Available: {avail}")
    print(f"checkpoint: {path}")
    print("contents:", sorted(p.name for p in path.iterdir())[:20])
    missing = [f for f in EXPECTED_FILES if not (path / f).exists()]
    weights = [w for w in ("model.safetensors", "pytorch_model.bin")
               if (path / w).exists()]
    if missing or not weights:
        raise FileNotFoundError(f"incomplete checkpoint: missing {missing} "
                                f"weights_found={weights}")

    # The eval's own two calls, verbatim (precompress_v72_focus.py).
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForTokenClassification.from_pretrained(
        str(path), attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    head = tuple(model.classifier.weight.shape)
    print(f"params      : {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"vocab       : {model.config.vocab_size}")
    print(f"num_labels  : {model.config.num_labels}  head={head}")
    print(f"id2label    : {model.config.id2label}")
    print(f"max_pos     : {getattr(model.config, 'max_position_embeddings', '?')}")
    print(f"tokenizer   : cls/sep/pad="
          f"{tokenizer.cls_token_id}/{tokenizer.sep_token_id}/{tokenizer.pad_token_id}"
          f"  vocab={tokenizer.vocab_size}")

    if model.config.num_labels != 2:
        raise ValueError(f"eval reads keep-probs off a 2-label head, got "
                         f"{model.config.num_labels}")
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("tokenizer has no CLS/SEP -- the [CLS] q [SEP] chunk [SEP] "
                         "layout cannot be built")

    # One real forward in the eval's layout, so a shape bug surfaces here and not
    # 40 minutes into a sweep.
    import torch.nn.functional as F
    q = tokenizer("What was operating income?", add_special_tokens=False)["input_ids"]
    c = tokenizer("Operating income was $1,904 million for the quarter.",
                  add_special_tokens=False)["input_ids"]
    ids = ([tokenizer.cls_token_id] + q + [tokenizer.sep_token_id] + c
           + [tokenizer.sep_token_id])
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids]),
                       attention_mask=torch.ones(1, len(ids), dtype=torch.long)).logits
    p = F.softmax(logits.float(), dim=-1)[0, len(q) + 2:len(q) + 2 + len(c), 1]
    print(f"forward OK  : {len(ids)} tokens, chunk keep-prob "
          f"mean={p.mean():.3f} min={p.min():.3f} max={p.max():.3f}")
    if float(p.std()) < 1e-3:
        print("  WARN chunk keep-probs are near-constant on this probe -- possible "
              "collapse; check val answer_survival in best_metrics.json")

    for name in ("best_metrics.json", "metrics.json"):
        f = path / name
        if f.exists():
            d = json.loads(f.read_text())
            print(f"{name}: score={d.get('score')} step={d.get('global_step')} "
                  f"llm_calls={d.get('llm_calls')}")
            print("  " + json.dumps({k: v for k, v in (d.get("val") or d).items()
                                     if "@" in str(k)}, indent=2)[:600])

    print("\nOK -- loads with AutoModelForTokenClassification + AutoTokenizer. "
          f"Pass MODEL_VOL_PATH={exp_path} to the existing eval unchanged.")
    return {"exp_path": exp_path, "n_params": n_params, "head": list(head),
            "vocab_size": model.config.vocab_size}


@app.local_entrypoint()
def main(exp_path: str = ""):
    """--exp-path <dir under otso-v8-training> (the run root, or its latest/)"""
    if not exp_path:
        raise SystemExit("--exp-path is required, e.g. exp-20260726-013000-v9-rwr")
    print(json.dumps(verify.remote(exp_path), indent=2))
