"""
Modal training v8: query-adaptive finance compressor (FinanceBench Pareto push).

Fork of taaha's training/scratch_v4_0_focus/modal_train_scratch_v4_0_focus.py. His
loop, loss family, AUPRC instrumentation, checkpoint layout and W&B resume logic are
kept verbatim in spirit; the v8 deltas are:

  1. CHECKPOINT SELECTION BY val AUPRC (was -val_bce). On this imbalance val BCE is
     minimized by a well-calibrated near-constant predictor, so it happily selects a
     collapsed model; AUPRC is threshold-free and collapse-sensitive (a constant
     output scores == pos_prevalence). Early stop patience 4 on AUPRC.
  2. Per-epoch tau-swept DROP-class f1_best over {0.3..0.7}. taaha's v7.0 finding:
     selecting/stopping on F1 at a FIXED tau falsely early-stops a still-improving
     run whose operating point is drifting. Tau is chosen at serve time anyway.
  3. warm_from: a checkpoint under /models loaded STRICT (keeps its pretrained [2,384]
     keep/drop head), or cold start from a public encoder.
  4. Dropout injected on the AutoConfig BEFORE from_pretrained. Setting
     model.config.hidden_dropout_prob AFTER the model exists is a NO-OP -- the
     nn.Dropout modules are already built. That silent no-op is why bear-4.1's
     "proven" dropout 0.27 never actually ran.
  5. Collapse guards logged every epoch: prediction std across content tokens,
     keep-rate at 0.5, and AUPRC / pos_prevalence (warn under 1.5x).
  6. Own volumes, all prefixed otso-v8. compression-models is mounted READ-ONLY.

Data comes from v8_pretokenize.py: [CLS] question [SEP] filing_window [SEP], with
loss_mask 1 only on filing-content tokens.

Run:
    modal run v8_train.py --config r1_warm71_2048
    modal run v8_train.py --config r1_warm71_2048 --smoke 1     # 200 rows, 1 epoch
"""

import json
import time
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("otso-v8-finance-training")

# v8-owned volumes. NEVER write anything that is not otso-v8-*.
data_vol = modal.Volume.from_name("otso-v8-data", create_if_missing=True)
output_vol = modal.Volume.from_name("otso-v8-training", create_if_missing=True)
# Warm-start checkpoints live here. READ-ONLY mount -- this is a shared team volume.
models_vol = modal.Volume.from_name("compression-models").read_only()
# Shared HF cache (public encoder downloads persist across runs)
hf_cache_vol = modal.Volume.from_name("scratch-compression-hf-cache",
                                      create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "tqdm", "numpy", "wandb",
        "safetensors", "sentencepiece", "protobuf",
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_VERBOSITY": "error",
    })
)

DATA_VOL = "/data"
OUTPUT_VOL = "/output"
MODELS_DIR = "/models"
HF_CACHE = "/hf-cache"

KEEP_LABEL = 1
DROP_LABEL = 0


def _optional_secrets(names):
    """Secrets that may not exist in this workspace. A missing wandb-secret must not
    kill a 24h training run, so unresolvable names are dropped here and the run falls
    back to WANDB_MODE=offline inside train()."""
    out = []
    for n in names:
        try:
            s = modal.Secret.from_name(n)
            s.hydrate()
            out.append(s)
        except Exception as e:
            print(f"[secrets] '{n}' unavailable ({type(e).__name__}) -- continuing without it")
    return out


SECRETS = _optional_secrets(["wandb-secret", "huggingface-secret-refresh"])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ---- Data: /data/<data_subdir>/{train,val}.pt from v8_pretokenize.py ----
    data_subdir: str = "v8-ctx2048"
    max_len:     int = 2048

    # ---- Model init ----
    # warm_from: dir name under /models, loaded STRICT (keeps its pretrained keep/drop
    # head). Empty -> cold start from init_model (public HF encoder, fresh head).
    warm_from:      str = ""
    init_model:     str = "jhu-clsp/mmBERT-small"
    fallback_model: str = "jhu-clsp/mmBERT-small"
    # Dropout goes on the AutoConfig BEFORE construction (see module docstring).
    # attn_dropout < 0 mirrors dropout; 0.0 preserves the fused-attention fast path.
    dropout:      float = 0.15
    attn_dropout: float = -1.0

    # ---- Optimizer ----
    lr:            float = 2.0e-5
    weight_decay:  float = 0.01
    warmup_frac:   float = 0.07
    max_grad_norm: float = 1.0

    # ---- Loss ----
    #   ce            2-class cross-entropy + label_smoothing (the V43 recipe)
    #   focal         CE-based focal (1-p_t)^gamma -- the clean A/B against ce
    #   bce           taaha's soft-BCE on continuous targets
    #   tversky | bce_tversky | focal_tversky | listwise | listwise_tversky (taaha's)
    loss_type:       str   = "ce"
    label_smoothing: float = 0.05    # caps CE growth against saturated-wrong labels
    focal_gamma:     float = 1.3
    focal_alpha:     float = -1.0    # >=0 up-weights KEEP
    tversky_alpha:   float = 0.5
    tversky_beta:    float = 0.5
    tversky_weight:  float = 3.0
    listwise_weight: float = 1.0

    # ---- Training ----
    # V43: effective batch 16 (accum 1) + a cosine horizon matched to real convergence.
    # A 40-epoch cosine that never anneals leaves the LR hot past the val peak and the
    # model grinds CE against irreducible label noise (grad norms explode 100-1000x).
    epochs:               int = 15
    patience:             int = 4       # on val AUPRC
    batch_size:           int = 16
    effective_batch_size: int = 16
    seed:                 int = 42
    limit_rows:           int = 0       # 0 = full data; >0 caps train rows (smoke)
    eval_thresholds:      tuple = (0.3, 0.5)
    # tau sweep for the DROP-class f1_best (selection-adjacent diagnostic)
    tau_sweep: tuple = (0.3, 0.4, 0.5, 0.6, 0.7)
    # collapse guard: warn when val AUPRC is under this multiple of pos_prevalence
    auprc_ratio_warn: float = 1.5

    # ---- Output / logging ----
    run_name:          str = "v8-r1-warm71-2048"
    save_path:         str = ""
    log_every_n_steps: int = 5

    # ---- W&B ----
    wandb_project: str = "otso-v8-finance"
    wandb_entity:  str = ""
    wandb_tags:    str = ""

    def __post_init__(self):
        if not self.save_path:
            existing = sorted(Path(OUTPUT_VOL).glob(f"exp-*-{self.run_name}"))
            if existing:
                self.save_path = str(existing[-1])
            else:
                ts = time.strftime("%Y%m%d-%H%M%S")
                self.save_path = f"{OUTPUT_VOL}/exp-{ts}-{self.run_name}"


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 24,
    volumes={
        DATA_VOL:   data_vol,
        OUTPUT_VOL: output_vol,
        MODELS_DIR: models_vol,
        HF_CACHE:   hf_cache_vol,
    },
    secrets=SECRETS,
)
def train(config_overrides: dict = {}):
    import os
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
    # No wandb creds (secret absent) -> log offline instead of crashing the run.
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
        print("[wandb] no WANDB_API_KEY -> WANDB_MODE=offline")

    import numpy as np
    import torch
    import torch.nn.functional as F
    import wandb
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from transformers import (
        AutoConfig, AutoModelForTokenClassification, AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    cfg = TrainConfig(**config_overrides)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config:\n{json.dumps(asdict(cfg), indent=2)}")

    # ---- Data ----
    pretok_dir = Path(DATA_VOL) / cfg.data_subdir
    train_path, val_path = pretok_dir / "train.pt", pretok_dir / "val.pt"
    meta_path = pretok_dir / "meta.json"
    if not train_path.exists() or not val_path.exists():
        avail = sorted(d.name for d in Path(DATA_VOL).iterdir()) if Path(DATA_VOL).exists() else []
        raise FileNotFoundError(
            f"missing pretokenized data under {pretok_dir}. Available: {avail}")

    print(f"Loading pretokenized data from {pretok_dir}")
    train_data = torch.load(train_path, map_location="cpu", weights_only=False)
    val_data = torch.load(val_path, map_location="cpu", weights_only=False)
    pretok_meta = {}
    if meta_path.exists():
        pretok_meta = json.loads(meta_path.read_text())
        print(f"  pretok meta: max_len={pretok_meta.get('max_len')} "
              f"tokenizer={pretok_meta.get('tokenizer')} "
              f"n_train={pretok_meta.get('n_train')} n_val={pretok_meta.get('n_val')} "
              f"pos_prevalence={pretok_meta.get('pos_prevalence')}")
        if pretok_meta.get("max_len") not in (None, cfg.max_len):
            raise ValueError(f"cfg.max_len={cfg.max_len} != pretok max_len="
                             f"{pretok_meta['max_len']} ({pretok_dir})")

    def to_tensors(d):
        n = d["input_ids"].shape[0]
        src = d.get("source_id")
        return (
            d["input_ids"].long(),                       # [N, L]
            d["attention_mask"].long(),                  # [N, L]
            d["targets"].float(),                        # [N, L] keep target in [0,1]
            d["loss_mask"].bool(),                       # [N, L] 1 = filing-content
            (src.long() if src is not None else torch.zeros(n, dtype=torch.long)),
        )

    train_t = list(to_tensors(train_data))
    val_t = list(to_tensors(val_data))

    if cfg.limit_rows > 0:               # smoke-test cap (no-op at 0)
        n = cfg.limit_rows
        vn = max(1, n // 20)
        train_t = [t[:n] for t in train_t]
        val_t = [t[:vn] for t in val_t]
        print(f"  limit_rows={n}: train -> {len(train_t[0])}  val -> {len(val_t[0])}")

    seq_len = train_t[0].shape[1]
    print(f"Train: {len(train_t[0]):,}  Val: {len(val_t[0]):,}  seq_len={seq_len}")
    if seq_len != cfg.max_len:
        raise ValueError(f"data seq_len {seq_len} != cfg.max_len {cfg.max_len}")

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=cfg.batch_size,
                              shuffle=True, drop_last=False, num_workers=2,
                              pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)

    grad_accum = max(1, cfg.effective_batch_size // cfg.batch_size)
    print(f"Effective batch: {cfg.batch_size * grad_accum} "
          f"(bs={cfg.batch_size} * accum={grad_accum})  batches/epoch={len(train_loader)}")

    # ---- Output dir / resume detection ----
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    run_name = save_path.name
    print(f"Save path: {save_path}")

    existing_epochs = sorted([
        int(d.name.replace("epoch_", "")) for d in save_path.iterdir()
        if d.is_dir() and d.name.startswith("epoch_")
    ]) if save_path.exists() else []

    start_epoch = 0
    best_score = float("-inf")       # higher = better; score = val AUPRC
    epochs_no_improve = 0

    # ---- Model load ----
    # Dropout is set ON THE CONFIG so the nn.Dropout modules are BUILT with it.
    def build_model(path, allow_mismatch):
        mc = AutoConfig.from_pretrained(path, num_labels=2,
                                        id2label={0: "drop", 1: "keep"},
                                        label2id={"drop": 0, "keep": 1})
        attn_do = cfg.attn_dropout if cfg.attn_dropout >= 0 else cfg.dropout
        for k in ("mlp_dropout", "classifier_dropout", "hidden_dropout_prob"):
            if hasattr(mc, k):                    # ModernBERT keys + BERT fallbacks
                setattr(mc, k, cfg.dropout)
        for k in ("attention_dropout", "attention_probs_dropout_prob"):
            if hasattr(mc, k):
                setattr(mc, k, attn_do)
        return AutoModelForTokenClassification.from_pretrained(
            path, config=mc, attn_implementation="sdpa",
            ignore_mismatched_sizes=allow_mismatch)

    if existing_epochs:
        resume_path = save_path / f"epoch_{existing_epochs[-1]}"
        print(f"Resuming model from {resume_path}")
        model = build_model(resume_path, False)
        start_epoch = existing_epochs[-1]
        bm_path = save_path / "best_metrics.json"
        if bm_path.exists():
            best_score = json.loads(bm_path.read_text()).get("score", float("-inf"))
            print(f"  best_score (val AUPRC) so far: {best_score:.4f}")
        tok_src = str(resume_path)
    elif cfg.warm_from:
        warm_path = Path(MODELS_DIR) / cfg.warm_from
        if not warm_path.exists():
            avail = sorted(d.name for d in Path(MODELS_DIR).iterdir() if d.is_dir())
            raise FileNotFoundError(f"warm_from '{cfg.warm_from}' not on the models "
                                    f"volume. Available (first 40): {avail[:40]}")
        print(f"WARM start (strict) from {warm_path}")
        model = build_model(str(warm_path), False)   # strict: keep the pretrained head
        tok_src = str(warm_path)
    else:
        print(f"COLD start from {cfg.init_model} (fresh head)")
        try:
            model = build_model(cfg.init_model, True)
            tok_src = cfg.init_model
        except Exception as e:
            print(f"  cold start failed ({type(e).__name__}: {e}); "
                  f"falling back to {cfg.fallback_model}")
            model = build_model(cfg.fallback_model, True)
            tok_src = cfg.fallback_model

    # Vocab guard: the ettin checkpoints use a DIFFERENT tokenizer (vocab 50368) from
    # mmBERT (256000), so pairing a model with the wrong pretokenized build produces
    # silent garbage (or an index error deep in the embedding). Fail loudly, now.
    max_id = int(max(train_t[0].max().item(), val_t[0].max().item()))
    if max_id >= model.config.vocab_size:
        raise ValueError(
            f"TOKENIZER MISMATCH: data has token id {max_id} but the model vocab is "
            f"{model.config.vocab_size}. data_subdir='{cfg.data_subdir}' was built with "
            f"tokenizer={pretok_meta.get('tokenizer')!r}; re-run v8_pretokenize.py with "
            f"--tokenizer matching this model.")

    model.to(device)
    assert model.config.num_labels == 2, f"expected a 2-label head, got {model.config.num_labels}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params  vocab={model.config.vocab_size}  "
          f"head={tuple(model.classifier.weight.shape)}  "
          f"dropout={cfg.dropout}/attn={cfg.attn_dropout if cfg.attn_dropout>=0 else cfg.dropout}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg.fallback_model)

    # ---- W&B (resume-aware) ----
    wandb_id_path = save_path / "wandb_run_id.txt"
    prior_run_id = wandb_id_path.read_text().strip() if wandb_id_path.exists() else None
    tags = [t.strip() for t in cfg.wandb_tags.split(",") if t.strip()]
    tags += [f"ctx{cfg.max_len}", "v8", "finance", cfg.loss_type,
             "warm" if cfg.warm_from else "cold"]
    wandb_run = wandb.init(
        project=cfg.wandb_project, entity=cfg.wandb_entity or None,
        name=run_name, id=prior_run_id,
        resume="allow" if prior_run_id else None,
        config={**asdict(cfg), "n_params": n_params, "grad_accum": grad_accum,
                "n_train": len(train_t[0]), "n_val": len(val_t[0]), "seq_len": seq_len,
                "pretok_meta": pretok_meta},
        tags=tags, dir="/tmp")
    if not prior_run_id:
        wandb_id_path.write_text(wandb_run.id)
        output_vol.commit()
    print(f"W&B run: {wandb_run.url}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    remaining_epochs = max(1, cfg.epochs - start_epoch)
    total_optim_steps = max(1, (len(train_loader) // grad_accum) * remaining_epochs)
    warmup_steps = int(total_optim_steps * cfg.warmup_frac)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)
    print(f"Total optim steps: {total_optim_steps}  warmup: {warmup_steps}")

    # ---- Losses (taaha's family; "ce"/"focal" are the CE-based V43 pair) ----
    def _ce_none(logits, targets):
        """Per-position CE over the [2] (drop=0, keep=1) head, with label smoothing."""
        tcls = (targets >= 0.5).long()
        return F.cross_entropy(logits.permute(0, 2, 1), tcls, reduction="none",
                               label_smoothing=cfg.label_smoothing)

    def loss_fn(logits, targets, mask):
        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1)
        if cfg.loss_type in ("ce", "focal"):
            ce = _ce_none(logits, targets)
            if cfg.loss_type == "focal":
                p_t = torch.exp(-ce)                       # prob of the true class
                ce = (1 - p_t) ** cfg.focal_gamma * ce
                if cfg.focal_alpha >= 0:
                    tcls = (targets >= 0.5).float()
                    a_t = cfg.focal_alpha * tcls + (1 - cfg.focal_alpha) * (1 - tcls)
                    ce = a_t * ce
            return (ce * mask_f).sum() / denom

        # --- taaha's soft-target family (continuous targets) ---
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_drop = log_probs[..., DROP_LABEL]
        log_p_keep = log_probs[..., KEEP_LABEL]
        p_keep = log_p_keep.exp()
        keep_term = targets * log_p_keep
        drop_term = (1.0 - targets) * log_p_drop
        if cfg.loss_type == "focal_tversky":
            keep_term = keep_term * (1.0 - p_keep) ** cfg.focal_gamma
            drop_term = drop_term * p_keep ** cfg.focal_gamma
        bce = (-(keep_term + drop_term) * mask_f).sum() / denom
        if cfg.loss_type == "bce":
            return bce
        # soft Tversky over content tokens: all-drop -> TP=0 -> max loss (unlike BCE,
        # where all-drop is a minimum), so it acts as a keep-RATE anchor.
        t = targets * mask_f
        pm = p_keep * mask_f
        tp = (pm * t).sum()
        fp = (pm * (1.0 - t)).sum()
        fn = ((1.0 - p_keep) * t).sum()
        tversky = tp / (tp + cfg.tversky_alpha * fp + cfg.tversky_beta * fn + 1e-6)
        tv = ((1.0 - tversky) ** cfg.focal_gamma
              if cfg.loss_type == "focal_tversky" else (1.0 - tversky))
        if cfg.loss_type == "tversky":
            return tv
        if cfg.loss_type in ("listwise", "listwise_tversky"):
            # per-row ranking KL: optimizes the WITHIN-window ordering (threshold-free,
            # question-conditioned). Scale-free, so it CANNOT pin the keep-rate -- the
            # Tversky term is the rate anchor.
            z = (log_p_keep - log_p_drop).masked_fill(mask == 0, float("-inf"))
            log_q = F.log_softmax(z, dim=-1).nan_to_num(neginf=0.0)
            t_sum = t.sum(dim=-1, keepdim=True)
            qd = t / t_sum.clamp(min=1e-6)
            kl = (qd * (qd.clamp(min=1e-12).log() - log_q)).sum(dim=-1)
            has_pos = t_sum.squeeze(-1) > 0
            listwise = kl[has_pos].mean() if bool(has_pos.any()) else log_p_keep.sum() * 0.0
            if cfg.loss_type == "listwise":
                return cfg.listwise_weight * listwise
            return cfg.listwise_weight * listwise + cfg.tversky_weight * tv
        return bce + cfg.tversky_weight * tv                  # bce_tversky

    # ---- metrics ----
    def keepdrop_metrics(p, t):
        if p.numel() == 0:
            return {}
        pred, label = (p >= 0.5), (t >= 0.5)
        tp = (pred & label).sum().item(); fp = (pred & ~label).sum().item()
        fn = (~pred & label).sum().item(); tn = (~pred & ~label).sum().item()
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        p_drop = tn / max(1, tn + fn); r_drop = tn / max(1, tn + fp)
        f1_drop = 2 * p_drop * r_drop / max(1e-9, p_drop + r_drop)
        return {
            "f1": float(f1), "macro_f1": float(0.5 * (f1 + f1_drop)),
            "f1_drop": float(f1_drop), "precision": float(precision),
            "recall": float(recall), "acc": float((tp + tn) / max(1, tp + fp + fn + tn)),
            "l1": float((p - t).abs().mean().item()),
            "keep_ratio": float(pred.float().mean().item()),
            "label_keep_ratio": float(label.float().mean().item()),
            "p_keep_mean": float(p.mean().item()),
            "p_keep_std": float(p.std().item()) if p.numel() > 1 else 0.0,
            "target_keep_mean": float(t.mean().item()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "n_tokens": int(p.numel()),
        }

    def average_precision(p, t):
        """== sklearn average_precision. Positives = t>=0.5. A collapsed constant-output
        model scores ~pos_prevalence; a model that learned to RANK scores above it. This
        is the SELECTION metric."""
        label = (t >= 0.5).astype(np.float64)
        npos = label.sum()
        if npos == 0 or npos == label.size:
            return float("nan")
        ls = label[np.argsort(-p, kind="stable")]
        tp_c, fp_c = np.cumsum(ls), np.cumsum(1.0 - ls)
        precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
        return float((precision * ls).sum() / npos)

    def _prf(pred, label):
        """(f1, precision, recall) as plain floats -- the inputs are torch bool tensors,
        so the counts must be unwrapped or the metrics dict ends up holding Tensors and
        the per-epoch metrics.json write blows up."""
        tp = int((pred & label).sum()); fp = int((pred & ~label).sum())
        fn = int((~pred & label).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        return float(2 * prec * rec / max(1e-9, prec + rec)), float(prec), float(rec)

    @torch.no_grad()
    def batch_metrics(logits, targets, mask, step_loss):
        p_keep = F.softmax(logits, dim=-1)[..., KEEP_LABEL]
        p, t = p_keep[mask], targets[mask]
        md = keepdrop_metrics(p, t)
        if not md:
            return {}
        auprc = average_precision(p.detach().cpu().numpy(), t.detach().cpu().numpy())
        return {
            "train/loss": step_loss, "train/auprc": auprc,
            "train/pos_prevalence": float((t >= 0.5).float().mean().item()),
            "train/f1": md["f1"], "train/macro_f1": md["macro_f1"],
            "train/f1_drop": md["f1_drop"], "train/precision": md["precision"],
            "train/recall": md["recall"], "train/l1": md["l1"],
            "train/keep_ratio": md["keep_ratio"],
            "train/label_keep_ratio": md["label_keep_ratio"],
            "train/p_keep_mean": md["p_keep_mean"], "train/p_keep_std": md["p_keep_std"],
            "train/confident": float(((p < 0.1) | (p > 0.9)).float().mean().item()),
        }

    @torch.no_grad()
    def evaluate(loader, desc):
        model.eval()
        loss_sum, n_batches = 0.0, 0
        all_p, all_t, all_s = [], [], []
        for ids, mask, targets, lmask, src in tqdm(loader, desc=desc):
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            lmask = lmask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            logits = logits.float()
            loss_sum += loss_fn(logits, targets, lmask).item()
            n_batches += 1
            p_keep = F.softmax(logits, dim=-1)[..., KEEP_LABEL]
            all_p.append(p_keep[lmask].cpu())
            all_t.append(targets[lmask].cpu())
            # broadcast the per-window qtype onto its content tokens
            all_s.append(src.to(device).unsqueeze(1).expand_as(lmask)[lmask].cpu())
        model.train()
        all_p, all_t = torch.cat(all_p), torch.cat(all_t)
        all_s = torch.cat(all_s)
        pc = all_p.clamp(1e-6, 1 - 1e-6)

        sweep = {}
        for tau in cfg.eval_thresholds:
            pred, label = (all_p >= tau), (all_t >= tau)
            f1_t, p_t, r_t = _prf(pred, label)
            k = f"{tau:g}"
            sweep[f"f1@{k}"], sweep[f"precision@{k}"] = f1_t, p_t
            sweep[f"recall@{k}"] = r_t
            sweep[f"keep_ratio@{k}"] = float(pred.float().mean().item())

        # tau-swept DROP-class f1_best: robust to the operating point drifting between
        # epochs (a fixed-tau F1 is not, and falsely early-stops improving runs).
        label_drop = all_t < 0.5
        best_tau, best_f1 = None, -1.0
        for tau in cfg.tau_sweep:
            f, _, _ = _prf(all_p < tau, label_drop)
            sweep[f"f1_drop@{tau:g}"] = f
            if f > best_f1:
                best_tau, best_f1 = tau, f

        auprc = average_precision(all_p.numpy(), all_t.numpy())
        prevalence = float((all_t >= 0.5).float().mean().item())
        metrics = {
            "loss": float(loss_sum / max(1, n_batches)),
            "auprc": auprc, "pos_prevalence": prevalence,
            "auprc_ratio": float(auprc / prevalence) if prevalence > 0 else float("nan"),
            "f1_drop_best": float(best_f1), "f1_drop_best_tau": float(best_tau),
            "brier": float(((all_p - all_t) ** 2).mean().item()),
            "confident": float(((all_p < 0.1) | (all_p > 0.9)).float().mean().item()),
            "entropy": float(-(pc * pc.log() + (1 - pc) * (1 - pc).log()).mean().item()),
            **keepdrop_metrics(all_p, all_t),
            **sweep,
            "_probs": all_p.numpy(), "_targets": all_t.numpy(),
        }
        for si, qt in enumerate(pretok_meta.get("qtype_source_ids", []) or []):
            sel = all_s == si
            if bool(sel.any()):
                metrics[f"keep_ratio_{qt}"] = float((all_p[sel] >= 0.5).float().mean().item())
                metrics[f"target_keep_ratio_{qt}"] = float((all_t[sel] >= 0.5).float().mean().item())
        return metrics

    # ---- Training loop ----
    global_step = optim_step = 0
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}\nEpoch {epoch+1}/{cfg.epochs}\n{'='*60}")
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        step = -1

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            ids, mask, targets, lmask, _src = [t.to(device, non_blocking=True) for t in batch]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            logits = logits.float()
            raw = loss_fn(logits, targets, lmask)
            (raw / grad_accum).backward()
            step_loss = raw.item()
            total_loss += step_loss

            if (step + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm).item()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                optim_step += 1
                if optim_step % cfg.log_every_n_steps == 0:
                    bm = batch_metrics(logits, targets, lmask, step_loss)
                    wandb.log({
                        "train/step_loss": step_loss, "train/grad_norm": grad_norm,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch_frac": epoch + (step + 1) / max(1, len(train_loader)),
                        "optim_step": optim_step, **bm,
                    }, step=global_step)
            global_step += 1

        if step >= 0 and (step + 1) % grad_accum != 0:      # tail accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            optim_step += 1

        avg_train_loss = total_loss / max(1, len(train_loader))

        # ---- Validation ----
        val = evaluate(val_loader, "Validating")
        epoch_time = time.time() - epoch_start
        score = float(val["auprc"])        # SELECTION: maximize val AUPRC

        print(f"  train_loss={avg_train_loss:.4f}  val_loss={val['loss']:.4f}  "
              f"val_auprc={val['auprc']:.4f} (base {val['pos_prevalence']:.4f}, "
              f"{val['auprc_ratio']:.2f}x)  val_f1={val['f1']:.4f}  ({epoch_time:.0f}s)")
        print(f"  P={val['precision']:.4f}  R={val['recall']:.4f}  "
              f"drop-f1_best={val['f1_drop_best']:.4f}@tau{val['f1_drop_best_tau']:g}  "
              f"macro_f1={val['macro_f1']:.4f}")
        # ---- collapse guards ----
        print(f"  GUARDS keep@0.5={val['keep_ratio']:.4f} (target "
              f"{val['label_keep_ratio']:.4f})  p_keep_std={val['p_keep_std']:.4f}  "
              f"auprc/prev={val['auprc_ratio']:.2f}x")
        warns = []
        if not (val["auprc_ratio"] >= cfg.auprc_ratio_warn):
            warns.append(f"AUPRC only {val['auprc_ratio']:.2f}x prevalence "
                         f"(< {cfg.auprc_ratio_warn}) -- not ranking, likely COLLAPSED")
        if val["p_keep_std"] < 0.02:
            warns.append(f"p_keep std {val['p_keep_std']:.4f} -- near-CONSTANT output")
        if val["keep_ratio"] < 0.005:
            warns.append(f"keep-rate {val['keep_ratio']:.4f} -- ALL-DROP collapse")
        if val["keep_ratio"] > 0.95:
            warns.append(f"keep-rate {val['keep_ratio']:.4f} -- ALL-KEEP collapse")
        for w in warns:
            print(f"  WARN {w}")

        log_dict = {
            "epoch": epoch + 1, "train/epoch_loss": avg_train_loss,
            "val/loss": val["loss"], "val/auprc": val["auprc"],
            "val/pos_prevalence": val["pos_prevalence"],
            "val/auprc_ratio": val["auprc_ratio"],
            "val/f1_drop_best": val["f1_drop_best"],
            "val/f1_drop_best_tau": val["f1_drop_best_tau"],
            "val/l1": val["l1"], "val/brier": val["brier"], "val/acc": val["acc"],
            "val/f1": val["f1"], "val/macro_f1": val["macro_f1"],
            "val/f1_drop": val["f1_drop"], "val/precision": val["precision"],
            "val/recall": val["recall"], "val/keep_ratio": val["keep_ratio"],
            "val/label_keep_ratio": val["label_keep_ratio"],
            "val/p_keep_mean": val["p_keep_mean"], "val/p_keep_std": val["p_keep_std"],
            "val/target_keep_mean": val["target_keep_mean"],
            "val/confident": val["confident"], "val/entropy": val["entropy"],
            "val/tp": val["tp"], "val/fp": val["fp"], "val/fn": val["fn"], "val/tn": val["tn"],
            "val/n_collapse_warnings": len(warns),
            **{f"val/{k}": v for k, v in val.items() if "@" in k},
            **{f"val/{k}": v for k, v in val.items() if k.startswith("keep_ratio_")
               or k.startswith("target_keep_ratio_")},
            "best/score": best_score, "time/epoch_seconds": epoch_time,
            "lr_current": scheduler.get_last_lr()[0], "epochs_no_improve": epochs_no_improve,
        }
        for hk, arr in (("val/prob_hist", val["_probs"]), ("val/target_hist", val["_targets"])):
            try:
                log_dict[hk] = wandb.Histogram(arr)
            except Exception:
                pass                    # degenerate distribution -> never crash the run
        wandb.log(log_dict, step=global_step)

        val_json = {k: v for k, v in val.items() if not k.startswith("_")}

        # ---- Per-epoch checkpoint ----
        epoch_path = save_path / f"epoch_{epoch+1}"
        epoch_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)
        with open(epoch_path / "metrics.json", "w") as f:
            json.dump({"epoch": epoch + 1, "train_loss": avg_train_loss,
                       "val": val_json, "score": score}, f, indent=2)
        output_vol.commit()

        # ---- Best handling (canonical "best" at save_path root), by val AUPRC ----
        if score > best_score:
            best_score = score
            epochs_no_improve = 0
            print(f"  *** New best val AUPRC: {score:.4f} ***")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            with open(save_path / "best_metrics.json", "w") as f:
                json.dump({"epoch": epoch + 1, "score": score, "val": val_json,
                           "train_loss": avg_train_loss, "hyperparams": asdict(cfg)},
                          f, indent=2)
            output_vol.commit()
            wandb.summary["best/epoch"] = epoch + 1
            wandb.summary["best/score"] = best_score
            wandb.summary["best/val_auprc"] = val["auprc"]
            wandb.summary["best/val_auprc_ratio"] = val["auprc_ratio"]
            wandb.summary["best/val_f1_drop_best"] = val["f1_drop_best"]
            wandb.summary["best/val_macro_f1"] = val["macro_f1"]
        else:
            epochs_no_improve += 1
            print(f"  No AUPRC improvement for {epochs_no_improve}/{cfg.patience} epochs")
            if epochs_no_improve >= cfg.patience:
                print("Early stopping!")
                wandb.summary["stopped_early"] = True
                wandb.summary["stopped_epoch"] = epoch + 1
                break

    print(f"\nDone. best val AUPRC={best_score:.4f}  -> {save_path}")
    wandb.summary["final/best_score"] = best_score
    wandb.finish()
    return {"run_name": run_name, "best_auprc": best_score, "save_path": str(save_path)}


# ---------------------------------------------------------------------------
# Run presets
# ---------------------------------------------------------------------------
# What to watch, in priority order:
#   1. val/auprc_ratio (auprc / pos_prevalence) climbing well past 1.5x -- the only
#      metric that separates real ranking from collapse on this imbalance.
#   2. val/keep_ratio tracking val/label_keep_ratio (calibration = adaptive mass).
#   3. val/f1_drop_best trending up; its tau tells you the serve-time operating point.
#   4. val/p_keep_std > 0.02 (a near-constant predictor is a dead run).

# V43 recipe (taaha, 02.07.2026): eff-batch 16, epochs 15 so the cosine actually
# anneals, label_smoothing 0.05, dropout 0.15, lr 2e-5 + clip 1.0. His v4.1-copied
# lr 4.1e-5 / clip 10 collapsed EVERY run; r7 re-tests the LR alone, with clip held
# at 1.0, because our label set is drop-majority (not keep-majority like his meetings).
_V43 = {
    "loss_type": "ce", "lr": 2.0e-5, "max_grad_norm": 1.0, "label_smoothing": 0.05,
    "dropout": 0.15, "attn_dropout": 0.15, "epochs": 15, "patience": 4,
    "batch_size": 16, "effective_batch_size": 16, "warmup_frac": 0.07,
    "log_every_n_steps": 5, "data_subdir": "v8-ctx2048", "max_len": 2048,
}

PRESETS = {
    "w2_140": {**_V43, "run_name": "w2-140", "warm_from": "bear-v7.1-rl-max-v2",
               "data_subdir": "v8-w2-ctx2048", "epochs": 12},
    "w2_140_s2": {**_V43, "run_name": "w2-140-s2", "warm_from": "bear-v7.1-rl-max-v2",
                  "data_subdir": "v8-w2-ctx2048", "epochs": 12, "seed": 2},
    "w2_ettin": {**_V43, "run_name": "w2-ettin", "warm_from": "",
                 "init_model": "jhu-clsp/ettin-encoder-400m",
                 "fallback_model": "jhu-clsp/ettin-encoder-400m",
                 "data_subdir": "v8-w2-ettin", "epochs": 12,
                 "batch_size": 8, "effective_batch_size": 16},
    # R1 -- the reference run: warm-start the v7.1 RL compressor, V43 recipe.
    "r1_warm71_2048": {**_V43, "run_name": "v8-r1-warm71-2048",
                       "warm_from": "bear-v7.1-rl-max-v2"},

    # R2 -- cold mmBERT-small: how much the warm keep/drop head is actually worth.
    "r2_scratch_2048": {**_V43, "run_name": "v8-r2-scratch-2048",
                        "warm_from": "", "init_model": "jhu-clsp/mmBERT-small"},

    # R3 -- warm the question-conditioned focus model instead of the RL one. Closest
    # existing init to this task (it already reads a query + content pair).
    "r3_warmfocus_2048": {**_V43, "run_name": "v8-r3-warmfocus-2048",
                          "warm_from": "bear-4.0-focus-1-ctx2048"},

    # R4 -- ctx512 (serving-compatible). Needs the 512 pretok build. Shorter windows
    # tile the same filing more finely -> more windows and denser positives.
    "r4_warm71_512": {**_V43, "run_name": "v8-r4-warm71-512",
                      "warm_from": "bear-v7.1-rl-max-v2",
                      "data_subdir": "v8-ctx512", "max_len": 512,
                      "batch_size": 32, "effective_batch_size": 32},

    # R5 -- focal (gamma 1.3, the bear-4.1 value) instead of CE. Tests whether
    # down-weighting the easy drops helps at our ~5-30% positive rate.
    "r5_focal": {**_V43, "run_name": "v8-r5-focal", "warm_from": "bear-v7.1-rl-max-v2",
                 "loss_type": "focal", "focal_gamma": 1.3},

    # R6 -- taaha's bce_tversky combo. On his focus data BCE alone went all-drop and
    # Tversky alone all-keep; only the combo landed a sane keep-rate.
    "r6_bce_tversky": {**_V43, "run_name": "v8-r6-bce-tversky",
                       "warm_from": "bear-v7.1-rl-max-v2",
                       "loss_type": "bce_tversky", "label_smoothing": 0.0,
                       "tversky_alpha": 0.5, "tversky_beta": 0.5, "tversky_weight": 3.0},

    # R7 -- lr 4.1e-5 with clip HELD at 1.0. Isolates the LR from the loose clip that
    # taaha's collapsed sweep confounded it with.
    "r7_highlr_guard": {**_V43, "run_name": "v8-r7-highlr-guard",
                        "warm_from": "bear-v7.1-rl-max-v2", "lr": 4.1e-5,
                        "max_grad_norm": 1.0},

    # R8 -- ettin-encoder-400m capacity probe (1024 hidden, 28 layers vs small's
    # 384/22). DIFFERENT TOKENIZER (vocab 50368, not mmBERT's 256000), so it needs its
    # OWN pretokenized build -- see v8-ettin-ctx2048. Heavier per sample: micro-batch 8
    # with accum 2 to hold the effective batch at 16.
    "r8_ettin400": {**_V43, "run_name": "v8-r8-ettin400",
                    "warm_from": "", "init_model": "jhu-clsp/ettin-encoder-400m",
                    "fallback_model": "jhu-clsp/ettin-encoder-400m",
                    "data_subdir": "v8-ettin-ctx2048",
                    "batch_size": 8, "effective_batch_size": 16},

    # R9 -- R1 at seed 2: how much of any R1-vs-rest gap is just seed noise.
    "r9_warm71_seed2": {**_V43, "run_name": "v8-r9-warm71-seed2",
                        "warm_from": "bear-v7.1-rl-max-v2", "seed": 2},

    # R10 -- longer horizon at a lower LR (cosine annealed over 30 epochs).
    "r10_longtrain": {**_V43, "run_name": "v8-r10-longtrain",
                      "warm_from": "bear-v7.1-rl-max-v2",
                      "epochs": 30, "lr": 1.5e-5},
}


@app.local_entrypoint()
def main(config: str = "r1_warm71_2048", smoke: int = 0, data_subdir: str = ""):
    """--config <preset>  [--smoke 1]  [--data-subdir <name>]

    smoke = 200 train rows / 1 epoch. data_subdir overrides the preset's dataset,
    which is how the smoke test points at the tiny fixture build instead of the real
    one (so a smoke run never needs the real subdir name to exist)."""
    if config not in PRESETS:
        raise SystemExit(f"unknown --config '{config}'. Choices: {sorted(PRESETS)}")
    cfg = dict(PRESETS[config])
    if data_subdir:
        cfg["data_subdir"] = data_subdir
    if smoke:
        cfg.update(run_name=f"{cfg['run_name']}-smoke", epochs=1, patience=1,
                   limit_rows=200, batch_size=min(cfg["batch_size"], 4),
                   effective_batch_size=min(cfg["batch_size"], 4),
                   log_every_n_steps=1)
    print(f"config={config}  run_name={cfg['run_name']}  data={cfg['data_subdir']}  "
          f"warm_from={cfg.get('warm_from') or '(cold ' + cfg.get('init_model', '') + ')'}"
          f"{'  [SMOKE]' if smoke else ''}")
    train.remote(config_overrides=cfg)
