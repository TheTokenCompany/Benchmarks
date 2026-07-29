"""
v9 RL: reward-weighted regression (RWR) for the query-adaptive finance compressor.

Fork of v8_train.py's Modal scaffolding (same volumes, same checkpoint layout, same
"thread every parameter as a function argument" discipline) with taaha's RWR
mechanics from training/bear_rl/modal_train_bear_rl_rwr.py grafted in. His plain
REINFORCE collapsed -- one scalar reward supervising hundreds of binary token
decisions with no trust region -- and RWR did not, so the parts that make RWR stable
are kept verbatim in spirit:

  * K candidate masks sampled from the CURRENT policy per item,
  * rewards softmaxed into weights (reward_temp), never used as a return gradient,
  * weighted-imitation CE toward the sampled masks,
  * KL to a FROZEN reference as the trust region,
  * an anti-degeneracy keep floor that kills candidates below it.

The v9 deltas over taaha's RWR:

  1. REWARD IS END-TO-END, not gold-token recall. A candidate is rendered back into
     text and a reader model (gemini-3.1-flash-lite) is asked the question against
     it; answer_survival is a DETERMINISTIC normalized comparison of its reply to the
     gold answer (no LLM judge, no rubric, nothing to game). Recall of gold tokens
     says the right tokens ranked high; answer_survival says the compressed text is
     still ANSWERABLE, which is the metric FinanceBench actually scores.
  2. Decisions are per WORD, not per token. Serving pools subwords to words and
     thresholds the word, so sampling per token would train a policy that can emit
     half a number. Pooling is done on the word_id map from v9_rl_prep.py, with
     numbers atomic per otsofier PR #714.
  3. keep_frac is TOKEN-weighted (tokens under kept words / content tokens), because
     that -- not the word count -- is the retention the compressor is billed on.
  4. The reference log-probs are PRECOMPUTED ONCE and cached on CPU instead of
     running a frozen ref forward every step. The ref never changes and the dataset
     is fixed, so this is exact, and it buys back a third of the step cost.
  5. Reader verdicts are cached on the data volume keyed (qa_id, md5(rendered)), so
     re-runs and the recurring deterministic candidate cost nothing.
  6. Hard budget guards: a cap on reader calls and a wall-clock cap, both of which
     stop the run CLEANLY (checkpoint written, reason logged) rather than by timeout.

Data contract: /data/v9-rl/{train,val}.pt + {train,val}_meta.jsonl from v9_rl_prep.py.
Checkpoint layout is v8's exactly -- best at the save_path root (config.json +
model.safetensors + tokenizer + best_metrics.json), plus latest/ -- so
v8_eval_precompress.py consumes it with no changes.

Run:
    modal run v9_rwr_train.py --init-from exp-20260725-235952-w2-140-s2/best
    modal run v9_rwr_train.py --init-from <ckpt> --smoke 1     # 8 items, stub reader
"""

import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup (v8_train.py's, unchanged: same volumes, same optional secrets)
# ---------------------------------------------------------------------------

app = modal.App("otso-v9-rl")

data_vol = modal.Volume.from_name("otso-v8-data", create_if_missing=True)
output_vol = modal.Volume.from_name("otso-v8-training", create_if_missing=True)
# Shared team volume. READ-ONLY -- never write anything that is not otso-v8-*.
models_vol = modal.Volume.from_name("compression-models").read_only()
hf_cache_vol = modal.Volume.from_name("scratch-compression-hf-cache",
                                      create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "tqdm", "numpy", "wandb",
        "safetensors", "sentencepiece", "protobuf", "openai",
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_VERBOSITY": "error",
    })
    # render_mask / fact_survival have ONE definition, in the prep script, and the
    # reward must render candidates exactly the way the dataset was built.
    .add_local_python_source("v9_rl_prep")
)

DATA_VOL = "/data"
OUTPUT_VOL = "/output"
MODELS_DIR = "/models"
HF_CACHE = "/hf-cache"

KEEP_LABEL = 1
DROP_LABEL = 0


def _optional_secrets(names):
    """Secrets that may not exist in this workspace. A missing wandb-secret must not
    kill a 6h RL run, so unresolvable names are dropped and the run falls back to
    WANDB_MODE=offline inside train()."""
    out = []
    for n in names:
        try:
            s = modal.Secret.from_name(n)
            s.hydrate()
            out.append(s)
        except Exception as e:
            print(f"[secrets] '{n}' unavailable ({type(e).__name__}) -- continuing without it")
    return out


# gemini-api-secret carries GEMINI_API_KEY into the container. It is NOT the only
# copy of that key -- the earlier note here saying so was wrong and cost an agent a
# blocked fleet run. For anything running LOCALLY, pull it from AWS Secrets Manager
# instead (verified 27.07.2026): secret `otsofy-llm-api-keys-usw2` is a JSON blob and
# the key is the `google_ai_api_key` field, which is why searching SM for a secret
# NAMED gemini finds nothing.
#   export GEMINI_API_KEY="$(aws secretsmanager get-secret-value \
#     --secret-id otsofy-llm-api-keys-usw2 --profile default --region us-west-2 \
#     --query SecretString --output text \
#     | python3 -c 'import json,sys;print(json.load(sys.stdin)["google_ai_api_key"])')"
SECRETS = _optional_secrets(["gemini-api-secret", "wandb-secret",
                             "huggingface-secret-refresh"])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    # ---- Data ----
    data_subdir: str = "v9-rl"
    max_len:     int = 2048
    limit_train: int = 0          # 0 = all; >0 caps items (smoke)
    limit_val:   int = 0

    # ---- Policy init ----
    # Resolved against /output then /models; absolute paths are honoured as-is.
    # NEVER hardcode a checkpoint here -- it is chosen at launch.
    init_from:      str = ""
    fallback_model: str = "jhu-clsp/mmBERT-small"   # tokenizer fallback only

    # ---- Candidate sampling ----
    k:            int   = 6       # sampled masks per item (+1 deterministic)
    sample_temp:  float = 1.0     # logit temperature on the word keep-prob
    target_keep:  float = 0.22    # the deterministic candidate's token budget
    # pool: subword -> word probability aggregation, min | max.
    # MIN by default to match v8_eval_precompress.py / precompress_v72_focus.py, which
    # min-pool. Tonight's benchmark numbers come off that path, and training against a
    # statistic the eval does not threshold means optimizing something nobody measures.
    # `max` stays available for a later A/B.
    # NOTE before any prod ship: otsofier's merge_words (PR #714) MAX-pools glued words,
    # so an RL checkpoint headed for prod has to revisit this. PR #714's atomic-number
    # MERGE is independent of the choice and lives in v9_rl_prep.segment_words.
    pool:         str   = "min"
    # sampler: how the K candidates are drawn.
    #   gumbel    -- Gumbel-perturbed keep-logit, top-k to a TOKEN BUDGET drawn from
    #                cand_keeps (taaha's fixed-budget mechanic, the one that did not
    #                collapse). Candidates differ in WHICH words, at the retentions
    #                the eval actually measures.
    #   bernoulli -- independent Bernoulli(word_prob) per word.
    # Measured on the real SFT init with real reader calls (10 val items), comparing
    # each sampler's best exploration candidate against the deterministic anchor:
    #   bernoulli    wins 40% of items, best R 1.276, winner sits at keep_frac 0.411
    #   gumbel 0.3   wins 70% of items, best R 1.681, winner sits at keep_frac 0.191
    # Bernoulli's problem is not diversity, it is WHERE the diversity lands: summing
    # ~1100 independent coin flips concentrates the keep rate, so its winners sit at
    # roughly twice the 0.22 target and imitating them drags retention UP, away from
    # the operating point the eval scores. gumbel is the default for that reason.
    sampler:      str   = "gumbel"
    # gumbel_temp 0.3 measured on the real SFT init (10 val items, real reader calls):
    # exploration beats the deterministic anchor on 70% of items at 0.3, 50% at 1.0,
    # 40% at 2.0, 10% at 4.0 (too noisy to be useful). 0.3 also gives the best
    # exploration reward (1.681 vs the anchor's 1.439) and the lowest anchor weight
    # (0.194) -- i.e. the most actual learning signal rather than self-imitation.
    gumbel_temp:  float = 0.3
    cand_keeps:   tuple = (0.12, 0.22, 0.33)

    # ---- Reward ----
    #   R = w_answer*answer_survival + w_fact*fact_survival
    #       - lam*keep_frac - degen_pen*[keep_frac < keep_floor]
    w_answer:  float = 2.0
    w_fact:    float = 0.5
    lam:       float = 1.0
    keep_floor: float = 0.04
    degen_pen: float = 2.0

    # ---- Loss ----
    reward_temp: float = 0.10
    kl_coef:     float = 0.05

    # ---- Optimizer ----
    lr:            float = 5.0e-6
    weight_decay:  float = 0.01
    max_grad_norm: float = 1.0

    # ---- Training ----
    epochs:     int = 4
    batch_size: int = 4
    seed:       int = 0

    # ---- Reader model (answer_survival) ----
    reader_model:   str = "gemini-3.1-flash-lite"
    reader_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    reader_concurrency: int = 16
    reader_max_tokens:  int = 96
    reader_timeout:     float = 60.0
    reader_retries:     int = 5
    # stub_reader: answer_survival from deterministic gold-answer containment in the
    # rendered text, no API calls. For GPU smoke tests and for reasoning about the
    # loop without spending budget. NEVER a real training setting.
    stub_reader: bool = False

    # prefilter_full: drop train items the reader cannot answer even from the
    # UNCOMPRESSED chunk. Measured on val: the ceiling is ~0.83, so ~1 item in 6
    # scores answer_survival 0 for EVERY candidate. Those items still contribute a
    # reward gradient -- just not one about preserving answers: with the answer term
    # pinned at 0, the only thing left to optimize is -lam*keep_frac, i.e. "drop
    # more". Costs one call per train item (~1.3% of the default budget), and the
    # verdicts are cached so a resume pays nothing.
    prefilter_full:       bool = True
    measure_val_ceiling:  bool = True

    # ---- Validation ----
    val_keeps:      tuple = (0.12, 0.22, 0.33)
    val_every_frac: float = 0.5    # validate every half epoch
    select_keep:    float = 0.22   # checkpoint on val answer_survival at this keep

    # ---- Budget guards (clean stop, not a timeout) ----
    max_llm_calls: int = 150_000
    max_hours:     float = 6.0

    # ---- Output / logging ----
    run_name:          str = "v9-rwr"
    save_path:         str = ""
    cache_subdir:      str = "v9-rl-cache"
    log_every_n_steps: int = 5

    # ---- W&B ----
    wandb_project: str = "otso-v9-rl"
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
# Reward: answer_survival matching (pure, deterministic, locally unit-testable)
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
SCALES = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
ANSWER_STOP = {
    "the", "and", "for", "with", "was", "were", "are", "our", "its", "from",
    "that", "this", "these", "those", "than", "into", "per", "not", "but", "all",
    "any", "has", "had", "have", "been", "which", "such", "also", "other",
    "million", "millions", "billion", "billions", "thousand", "thousands",
    "approximately", "about", "total", "increase", "decrease", "increased",
    "decreased", "primarily", "mainly", "including", "respectively",
}
NOT_FOUND = "NOT FOUND"

READER_SYSTEM = (
    "You answer questions about excerpts of SEC filings. The excerpt may be "
    "compressed: words are missing and the text may read as fragments. Use ONLY the "
    "excerpt. Reply with the answer alone -- a number, a phrase, a short list -- and "
    f"no explanation. If the excerpt does not contain the answer, reply exactly: {NOT_FOUND}"
)


def parse_numbers(text):
    """Numeric literals as floats, with $ % and thousands separators stripped."""
    out = []
    for m in NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def num_match(gold, cand, tol=0.02):
    """Is `cand` the same quantity as `gold` within `tol`, allowing a scale swap?

    "$1,904 million" vs a reply of "1.904 billion" is the SAME fact reported at a
    different scale; without the scale sweep that reads as a miss and the reward
    punishes a candidate that in fact preserved the answer."""
    if gold == 0:
        return abs(cand) <= tol
    for s in SCALES:
        g = gold * s
        if g != 0 and abs(cand - g) <= tol * abs(g):
            return True
    return False


def content_words(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-']*", (text or "").lower())
            if len(w) > 3 and w not in ANSWER_STOP]


def answer_hit(gold_answer, reply, tol=0.02, phrase_frac=0.6):
    """1 if the reply carries the gold answer's key number/phrase, else 0.

    Numeric golds are compared numerically (scale-tolerant, `tol` relative); every
    number in the gold must be matched, which is what makes a multistep answer like
    "$1,904 million, up 12%" require both halves. Non-numeric golds fall back to
    content-word overlap, and a gold with no content words at all (a bare "Yes") to
    plain containment. Deliberately NOT an LLM judge: the reward has to be stable
    across epochs or RWR is optimizing a moving target."""
    gold = (gold_answer or "").strip()
    reply = (reply or "").strip()
    if not gold or not reply or reply.upper().startswith(NOT_FOUND):
        return 0.0
    gnums = parse_numbers(gold)
    if gnums:
        rnums = parse_numbers(reply)
        if not rnums:
            return 0.0
        return float(all(any(num_match(g, r, tol) for r in rnums) for g in gnums))
    gwords = content_words(gold)
    low = reply.lower()
    if not gwords:
        return float(gold.lower() in low)
    hit = sum(1 for w in gwords if w in low)
    return float(hit / len(gwords) >= phrase_frac)


def cache_key(qa_id, rendered):
    return f"{qa_id}:{hashlib.md5(rendered.encode('utf-8')).hexdigest()}"


def ask_reader(client, model, question, rendered, max_tokens, retries, rng):
    """One reader call, retried on 429/5xx/timeout with exponential backoff + jitter.

    Module level so the CPU-only reader_probe and the training loop cannot drift apart
    on prompt or retry policy -- if the probe's numbers are to mean anything for a
    launch decision, it has to be asking exactly what training asks."""
    msgs = [{"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": f"Excerpt:\n{rendered}\n\n"
                                        f"Question: {question}\nAnswer:"}]
    delay = 1.0
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs, temperature=0.0, max_tokens=max_tokens)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            s = str(e)
            retryable = any(c in s for c in ("429", "500", "502", "503", "504")) \
                or "timeout" in s.lower() or "overloaded" in s.lower()
            if attempt == retries - 1 or not retryable:
                raise
            time.sleep(delay * (1.0 + rng.random()))
            delay = min(delay * 2, 30.0)
    return ""


def make_reader_client(base_url, timeout):
    import os
    from openai import OpenAI
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set -- the Modal secret "
                           "'gemini-api-secret' did not attach.")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=["B200", "H100"],          # B200 first, H100 when B200 capacity is gone
    timeout=60 * 60 * 12,
    volumes={
        DATA_VOL:   data_vol,
        OUTPUT_VOL: output_vol,
        MODELS_DIR: models_vol,
        HF_CACHE:   hf_cache_vol,
    },
    secrets=SECRETS,
)
def train(config_overrides: dict = {}):
    """Every knob arrives in config_overrides. NOTHING is read from module-level env:
    Modal evaluates module scope in a fresh container, so an env var read at import
    time on the laptop never reaches here (this bit us twice on v8)."""
    import os
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
        print("[wandb] no WANDB_API_KEY -> WANDB_MODE=offline")

    import numpy as np
    import torch
    import torch.nn.functional as F
    import wandb
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from transformers import (AutoConfig, AutoModelForTokenClassification,
                              AutoTokenizer)

    from v9_rl_prep import fact_survival, render_mask

    cfg = RLConfig(**config_overrides)
    rng = random.Random(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config:\n{json.dumps(asdict(cfg), indent=2)}")
    t_start = time.time()

    # ---- Data ----
    ddir = Path(DATA_VOL) / cfg.data_subdir
    if not (ddir / "train.pt").exists():
        avail = sorted(d.name for d in Path(DATA_VOL).iterdir()) if Path(DATA_VOL).exists() else []
        raise FileNotFoundError(f"missing RL data under {ddir} (run v9_rl_prep.py "
                                f"--upload). Available: {avail}")
    prep_meta = {}
    if (ddir / "meta.json").exists():
        prep_meta = json.loads((ddir / "meta.json").read_text())
        if prep_meta.get("max_len") not in (None, cfg.max_len):
            raise ValueError(f"cfg.max_len={cfg.max_len} != prep max_len="
                             f"{prep_meta['max_len']} ({ddir})")

    def load_split(name, limit):
        d = torch.load(ddir / f"{name}.pt", map_location="cpu", weights_only=False)
        items = [json.loads(l) for l in (ddir / f"{name}_meta.jsonl").read_text().splitlines() if l.strip()]
        n = d["input_ids"].shape[0]
        if len(items) != n:
            raise ValueError(f"{name}: {n} rows but {len(items)} meta records")
        if d["qa_id"] != [it["qa_id"] for it in items]:
            raise ValueError(f"{name}: .pt / meta.jsonl qa_id order mismatch")
        if limit > 0:
            d = {k: (v[:limit] if hasattr(v, "__getitem__") else v) for k, v in d.items()}
            items = items[:limit]
        return d, items

    train_d, train_items = load_split("train", cfg.limit_train)
    val_d, val_items = load_split("val", cfg.limit_val)
    n_train, n_val = len(train_items), len(val_items)
    seq_len = train_d["input_ids"].shape[1]
    if seq_len != cfg.max_len:
        raise ValueError(f"data seq_len {seq_len} != cfg.max_len {cfg.max_len}")
    print(f"RL data: {n_train} train / {n_val} val items  seq_len={seq_len}  "
          f"mean_words={prep_meta.get('train_mean_words')}")

    # ---- Output dir / resume ----
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    run_name = save_path.name
    latest_path = save_path / "latest"
    state_path = save_path / "state.json"
    metrics_path = save_path / "metrics.jsonl"
    print(f"Save path: {save_path}")

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    start_epoch = int(state.get("epoch", 0))
    llm_calls = int(state.get("llm_calls", 0))
    best_score = float("-inf")
    if (save_path / "best_metrics.json").exists():
        best_score = json.loads((save_path / "best_metrics.json").read_text()).get(
            "score", float("-inf"))
        print(f"  best val answer_survival@{cfg.select_keep} so far: {best_score:.4f}")

    # ---- Policy init ----
    def resolve_init(name):
        if not name:
            raise ValueError("init_from is required -- pass the SFT checkpoint to "
                             "start from (--init-from). It is never hardcoded.")
        cands = ([Path(name)] if name.startswith("/")
                 else [Path(OUTPUT_VOL) / name, Path(MODELS_DIR) / name])
        for c in cands:
            if (c / "config.json").exists():
                return c
        avail_out = sorted(d.name for d in Path(OUTPUT_VOL).iterdir() if d.is_dir())
        raise FileNotFoundError(
            f"init_from '{name}' not found (tried {[str(c) for c in cands]}). "
            f"On {OUTPUT_VOL}: {avail_out[:40]}")

    resuming = latest_path.exists() and (latest_path / "config.json").exists()
    # The KL anchor is the ORIGINAL init, never `latest` -- re-anchoring to the current
    # policy on every resume turns the trust region into a no-op and lets the run drift
    # arbitrarily far from the SFT model across restarts. state.json therefore records
    # init_from so a resume can still find it without the launch flag.
    anchor_from = state.get("init_from") or cfg.init_from
    if resuming:
        init_path = latest_path
        print(f"RESUMING policy from {init_path} (epoch {start_epoch}, "
              f"{llm_calls} reader calls spent)")
        if not anchor_from:
            anchor_from = str(latest_path)
            print("  WARN no init_from recorded for this run and none passed: "
                  "anchoring KL to `latest` instead of the original SFT init")
    else:
        init_path = resolve_init(cfg.init_from)
        anchor_from = cfg.init_from
        print(f"INIT policy + frozen ref from {init_path}")

    mc = AutoConfig.from_pretrained(init_path, num_labels=2,
                                    id2label={0: "drop", 1: "keep"},
                                    label2id={"drop": 0, "keep": 1})
    policy = AutoModelForTokenClassification.from_pretrained(
        init_path, config=mc, attn_implementation="sdpa").to(device)
    assert policy.config.num_labels == 2, \
        f"expected a 2-label keep/drop head, got {policy.config.num_labels}"
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params/1e6:.1f}M params  vocab={policy.config.vocab_size}")

    max_id = int(max(train_d["input_ids"].max().item(), val_d["input_ids"].max().item()))
    if max_id >= policy.config.vocab_size:
        raise ValueError(f"TOKENIZER MISMATCH: data token id {max_id} >= model vocab "
                         f"{policy.config.vocab_size}. The RL data was built with "
                         f"tokenizer={prep_meta.get('tokenizer')!r}.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(init_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg.fallback_model)

    # ---- Frozen reference log-probs, computed ONCE ----
    # The ref is frozen and the dataset is fixed, so a per-step ref forward recomputes
    # the same numbers ~500 times per epoch. Cache them on CPU in fp16 (2000 x 2048 x 2
    # = ~16 MB) and drop the ref model.
    ref_src = resolve_init(anchor_from) if resuming else init_path
    print(f"Precomputing frozen-ref log-probs from {ref_src} ...")
    ref = AutoModelForTokenClassification.from_pretrained(
        ref_src, config=mc, attn_implementation="sdpa").to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def ref_logprobs(d):
        out = torch.zeros((d["input_ids"].shape[0], seq_len, 2), dtype=torch.float16)
        bs = max(1, cfg.batch_size * 4)
        for i in tqdm(range(0, out.shape[0], bs), desc="ref logprobs"):
            ids = d["input_ids"][i:i + bs].long().to(device)
            am = d["attention_mask"][i:i + bs].long().to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = ref(input_ids=ids, attention_mask=am).logits
            out[i:i + bs] = F.log_softmax(lg.float(), dim=-1).half().cpu()
        return out

    train_ref = ref_logprobs(train_d)
    del ref
    torch.cuda.empty_cache()

    # ---- Reader (answer_survival) ----
    cache_dir = Path(DATA_VOL) / cfg.cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cfg.reader_model}.jsonl"
    verdict_cache = {}
    if cache_file.exists():
        for line in cache_file.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                verdict_cache[rec["k"]] = rec["hit"]
            except Exception:
                continue
    print(f"reader cache: {len(verdict_cache)} verdicts at {cache_file}")
    cache_fh = open(cache_file, "a")
    cache_hits = cache_misses = reader_errors = 0

    client = None if cfg.stub_reader else make_reader_client(cfg.reader_base_url,
                                                             cfg.reader_timeout)
    pool_exec = ThreadPoolExecutor(max_workers=cfg.reader_concurrency)

    def answer_survival(jobs):
        """[(qa_id, question, gold_answer, rendered)] -> [0/1], cached + concurrent.

        A reader error scores 0.0 rather than crashing: an overnight run must survive
        a transient upstream failure, and a zero on one candidate only removes it from
        that item's reward softmax."""
        nonlocal llm_calls, cache_hits, cache_misses, reader_errors
        out = [None] * len(jobs)
        todo = []
        for i, (qa_id, _q, _g, rendered) in enumerate(jobs):
            if not rendered.strip():
                out[i] = 0.0                    # nothing kept -> nothing answerable
                continue
            k = cache_key(qa_id, rendered)
            if k in verdict_cache:
                out[i] = float(verdict_cache[k])
                cache_hits += 1
            else:
                todo.append((i, k))
        if cfg.stub_reader:
            # Deterministic stand-in: does the gold answer survive in the text at all?
            for i, k in todo:
                _qa, _q, gold, rendered = jobs[i]
                hit = answer_hit(gold, rendered)
                out[i] = hit
                verdict_cache[k] = hit
            return out

        def work(pair):
            i, k = pair
            qa_id, question, gold, rendered = jobs[i]
            try:
                reply = ask_reader(client, cfg.reader_model, question, rendered,
                                   cfg.reader_max_tokens, cfg.reader_retries, rng)
                return i, k, answer_hit(gold, reply), reply[:200], None
            except Exception as e:
                return i, k, 0.0, "", f"{type(e).__name__}: {e}"

        for i, k, hit, reply, err in pool_exec.map(work, todo):
            out[i] = hit
            llm_calls += 1
            cache_misses += 1
            if err:
                reader_errors += 1
                if reader_errors <= 5 or reader_errors % 100 == 0:
                    print(f"  [reader] error #{reader_errors}: {err}")
                continue                        # a failed call is not a cached verdict
            verdict_cache[k] = hit
            cache_fh.write(json.dumps({"k": k, "hit": hit, "reply": reply}) + "\n")
        cache_fh.flush()
        return [0.0 if o is None else o for o in out]

    # ---- Prefilter: keep only items whose answer is reachable at all ----
    def full_render_jobs(items):
        return [(it["qa_id"], it["question"], it["answer"],
                 render_mask(it["words"], it["nl_after"], [True] * it["n_words"]))
                for it in items]

    ceilings = {}
    if cfg.prefilter_full:
        print(f"Prefilter: reading all {n_train} train chunks UNCOMPRESSED "
              f"(cached across runs) ...")
        hits = answer_survival(full_render_jobs(train_items))
        keep_idx = [i for i, h in enumerate(hits) if h > 0]
        ceilings["train_ceiling"] = len(keep_idx) / max(1, n_train)
        print(f"  train answer ceiling = {ceilings['train_ceiling']:.3f} "
              f"({len(keep_idx)}/{n_train} answerable) -> dropping "
              f"{n_train - len(keep_idx)} unanswerable items")
        if len(keep_idx) < 0.4 * n_train:
            raise RuntimeError(
                f"only {len(keep_idx)}/{n_train} train items are answerable from the "
                f"UNCOMPRESSED chunk. That is a data or reader problem, not something "
                f"RL can fix -- check reader_probe before spending GPU hours.")
        if keep_idx and len(keep_idx) < n_train:
            sel = torch.tensor(keep_idx)
            train_d = {k: (v[sel] if torch.is_tensor(v) else [v[i] for i in keep_idx])
                       for k, v in train_d.items()}
            train_items = [train_items[i] for i in keep_idx]
            train_ref = train_ref[sel]
            n_train = len(train_items)
            print(f"  train set now {n_train} items")

    if cfg.measure_val_ceiling:
        # Not filtered -- val must stay comparable to the eval harness, which scores
        # every question. The ceiling is logged so the val curve can be read against
        # what is actually achievable rather than against 1.0.
        vh = answer_survival(full_render_jobs(val_items))
        ceilings["val_ceiling"] = float(np.mean(vh)) if vh else 0.0
        print(f"  val answer ceiling = {ceilings['val_ceiling']:.3f} "
              f"(answer_survival@* cannot exceed this)")

    # ---- word pooling / mask helpers ----
    def pool_words(p_keep, word_id, n_words_max):
        """token keep-probs -> per-word keep-probs + token counts + validity.

        Non-content tokens (word_id -1: question, specials, padding) are scattered
        into a sink column that is then sliced off, so they can never win the pool
        for word 0 -- which is exactly the bug that would make every chunk's first
        word look confidently keepable."""
        B = p_keep.shape[0]
        valid = word_id >= 0
        idx = torch.where(valid, word_id.long(), torch.full_like(word_id.long(),
                                                                n_words_max))
        if cfg.pool == "min":
            init, reduce = 2.0, "amin"
        else:
            init, reduce = -1.0, "amax"
        wp = torch.full((B, n_words_max + 1), init, device=p_keep.device,
                        dtype=p_keep.dtype)
        wp.scatter_reduce_(1, idx, p_keep, reduce=reduce, include_self=True)
        wp = wp[:, :n_words_max]
        tpw = torch.zeros((B, n_words_max + 1), device=p_keep.device, dtype=p_keep.dtype)
        tpw.scatter_add_(1, idx, valid.to(p_keep.dtype))
        tpw = tpw[:, :n_words_max]
        wvalid = tpw > 0
        wp = torch.where(wvalid, wp, torch.zeros_like(wp)).clamp(0.0, 1.0)
        return wp, tpw, wvalid

    def budget_mask_from(scores, tpw, wvalid, keep_frac):
        """Top-scoring words until the TOKEN budget is spent.

        Token-weighted because retention is measured in tokens: a policy rewarded on a
        word count learns to keep cheap one-token words and drop the 4-subword numbers
        that carry the answer. Invalid words are sent to -inf rather than a magic low
        constant, because `scores` is a keep-LOGIT for the Gumbel sampler and logits
        are freely negative."""
        scores = torch.where(wvalid, scores,
                             torch.full_like(scores, float("-inf")))
        order = scores.argsort(dim=1, descending=True)
        tok_sorted = torch.gather(tpw, 1, order)
        valid_sorted = torch.gather(wvalid, 1, order)
        cum = (tok_sorted * valid_sorted).cumsum(dim=1)
        budget = (keep_frac * (tpw * wvalid).sum(dim=1, keepdim=True)).clamp(min=1.0)
        take_sorted = (cum <= budget) & valid_sorted
        take_sorted[:, 0] |= valid_sorted[:, 0]        # never emit an empty candidate
        mask = torch.zeros_like(take_sorted)
        mask.scatter_(1, order, take_sorted)
        return mask

    def budget_mask(word_probs, tpw, wvalid, keep_frac):
        """The deterministic mask serving emits: top words by keep-prob to budget."""
        return budget_mask_from(word_probs, tpw, wvalid, keep_frac)

    def sample_candidates(wp, tpw, wvalid):
        """K exploration candidates + the deterministic anchor at target_keep.

        The anchor is always included because it is what serving actually emits: if
        the imitation target set never contains it, training optimizes a distribution
        the eval never draws from."""
        pc = wp.clamp(1e-4, 1 - 1e-4)
        wlogit = torch.log(pc / (1 - pc))
        out = []
        if cfg.sampler == "gumbel":
            keeps = list(cfg.cand_keeps) or [cfg.target_keep]
            for i in range(cfg.k):
                # Gumbel(0,1). Clamp u on BOTH sides and take no clamp inside:
                # `-torch.log(u).clamp(min=1e-9)` -- taaha's expression, which this was
                # forked from -- parses as `-(torch.log(u).clamp(min=1e-9))`. log(u) is
                # negative, clamp(min=) raises it to +1e-9, the unary minus makes it
                # -1e-9, and the outer log of a negative number is NaN. Every Gumbel
                # draw comes out NaN, argsort then orders by nothing, and the sampler
                # silently returns arbitrary masks that do not respond to gumbel_temp
                # at all (measured: top-22% Jaccard 0.005 vs the anchor at EVERY temp
                # from 0.01 to 10). Verified fixed: mean 0.553 / std 1.265 against
                # Gumbel's 0.577 / 1.283.
                u = torch.rand_like(wlogit).clamp(1e-9, 1 - 1e-9)
                g = -torch.log(-torch.log(u))
                kf = keeps[i % len(keeps)]
                out.append(budget_mask_from(wlogit + cfg.gumbel_temp * g,
                                            tpw, wvalid, kf))
        else:
            ps = torch.sigmoid(wlogit / max(cfg.sample_temp, 1e-3))
            out = [(torch.rand_like(ps) < ps) & wvalid for _ in range(cfg.k)]
        out.append(budget_mask(wp, tpw, wvalid, cfg.target_keep))
        return out

    def render_candidate(item, mask_row):
        keep = mask_row[:item["n_words"]].tolist()
        return render_mask(item["words"], item["nl_after"], keep)

    def rewards_for(items, masks, tpw, wvalid, tag):
        """[K][B] word masks -> reward tensor [B,K] + the per-term breakdown.

        The reward is the point of v9, so all four terms are logged separately: a run
        that improves only by shrinking keep_frac is not the same result as one that
        improves answer_survival, and the aggregate hides which happened."""
        n_cand = len(masks)
        B = masks[0].shape[0]
        content = (tpw * wvalid).sum(dim=1).clamp(min=1.0)
        keep_frac = torch.stack([(tpw * m.to(tpw.dtype)).sum(dim=1) / content
                                 for m in masks], dim=1)          # [B,K]
        jobs, facts = [], []
        for k in range(n_cand):
            mk = masks[k].cpu()
            for b in range(B):
                it = items[b]
                rendered = render_candidate(it, mk[b])
                jobs.append((it["qa_id"], it["question"], it["answer"], rendered))
                facts.append(fact_survival(it["gold_fact_words"], rendered))
        ans = answer_survival(jobs)
        ans_t = torch.tensor(ans, dtype=torch.float32).view(n_cand, B).t().contiguous()
        fac_t = torch.tensor(facts, dtype=torch.float32).view(n_cand, B).t().contiguous()
        kf = keep_frac.detach().float().cpu()
        degen = (kf < cfg.keep_floor).float()
        R = (cfg.w_answer * ans_t + cfg.w_fact * fac_t
             - cfg.lam * kf - cfg.degen_pen * degen)
        return R, {f"{tag}/answer_survival": ans_t.mean().item(),
                   f"{tag}/fact_survival": fac_t.mean().item(),
                   f"{tag}/keep_frac": kf.mean().item(),
                   f"{tag}/degenerate_frac": degen.mean().item(),
                   f"{tag}/reward_mean": R.mean().item(),
                   f"{tag}/reward_best": R.max(dim=1).values.mean().item(),
                   f"{tag}/reward_spread": (R.max(dim=1).values
                                            - R.min(dim=1).values).mean().item()}

    # ---- W&B ----
    wandb_id_path = save_path / "wandb_run_id.txt"
    prior_run_id = wandb_id_path.read_text().strip() if wandb_id_path.exists() else None
    tags = [t.strip() for t in cfg.wandb_tags.split(",") if t.strip()]
    tags += [f"ctx{cfg.max_len}", "v9", "rl", "rwr", "finance"]
    wandb_run = wandb.init(
        project=cfg.wandb_project, entity=cfg.wandb_entity or None,
        name=run_name, id=prior_run_id, resume="allow" if prior_run_id else None,
        config={**asdict(cfg), "n_params": n_params, "n_train": n_train,
                "n_val": n_val, "seq_len": seq_len, "prep_meta": prep_meta,
                "gpu": torch.cuda.get_device_name(0), **ceilings},
        tags=tags, dir="/tmp")
    if not prior_run_id:
        wandb_id_path.write_text(wandb_run.id)
        output_vol.commit()
    print(f"W&B run: {wandb_run.url}")

    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- Validation ----
    @torch.no_grad()
    def validate():
        """Deterministic masks at each target keep -> fact + answer survival.

        Deterministic (not sampled) because this has to be comparable across
        half-epochs, and at the SAME budgets the eval harness uses."""
        policy.eval()
        acc = {kf: {"ans": [], "fact": [], "keep": []} for kf in cfg.val_keeps}
        bs = cfg.batch_size
        for i in range(0, n_val, bs):
            ids = val_d["input_ids"][i:i + bs].long().to(device)
            am = val_d["attention_mask"][i:i + bs].long().to(device)
            wid = val_d["word_id"][i:i + bs].to(device)
            items = val_items[i:i + bs]
            nwmax = int(val_d["n_words"][i:i + bs].max().item())
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = policy(input_ids=ids, attention_mask=am).logits
            p_keep = F.softmax(logits.float(), dim=-1)[..., KEEP_LABEL]
            wp, tpw, wvalid = pool_words(p_keep, wid, nwmax)
            content = (tpw * wvalid).sum(dim=1).clamp(min=1.0)
            for kf in cfg.val_keeps:
                m = budget_mask(wp, tpw, wvalid, kf).cpu()
                jobs, fs = [], []
                for b, it in enumerate(items):
                    rendered = render_candidate(it, m[b])
                    jobs.append((it["qa_id"], it["question"], it["answer"], rendered))
                    fs.append(fact_survival(it["gold_fact_words"], rendered))
                acc[kf]["ans"].extend(answer_survival(jobs))
                acc[kf]["fact"].extend(fs)
                real = (tpw.cpu() * m.to(tpw.dtype).cpu()).sum(dim=1) / content.cpu()
                acc[kf]["keep"].extend(real.tolist())
        policy.train()
        out = {}
        for kf, d in acc.items():
            k = f"{kf:g}"
            out[f"answer_survival@{k}"] = float(np.mean(d["ans"])) if d["ans"] else 0.0
            out[f"fact_survival@{k}"] = float(np.mean(d["fact"])) if d["fact"] else 0.0
            out[f"real_keep@{k}"] = float(np.mean(d["keep"])) if d["keep"] else 0.0
        return out

    def save_ckpt(dest, extra=None):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(dest)
        tokenizer.save_pretrained(dest)
        if extra:
            with open(dest / "metrics.json", "w") as f:
                json.dump(extra, f, indent=2)

    def log_metrics(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def budget_left():
        """(ok, reason) -- a clean stop, checkpoint written, never a Modal timeout."""
        hours = (time.time() - t_start) / 3600.0
        if llm_calls >= cfg.max_llm_calls:
            return False, f"reader-call budget spent ({llm_calls} >= {cfg.max_llm_calls})"
        if hours >= cfg.max_hours:
            return False, f"wall-clock budget spent ({hours:.2f}h >= {cfg.max_hours}h)"
        return True, ""

    # ---- Training loop ----
    steps_per_epoch = max(1, (n_train + cfg.batch_size - 1) // cfg.batch_size)
    val_every = max(1, int(steps_per_epoch * cfg.val_every_frac))
    print(f"{steps_per_epoch} steps/epoch (batch {cfg.batch_size}), "
          f"validating every {val_every} steps, K={cfg.k}+1 candidates "
          f"-> ~{(cfg.k+1)*cfg.batch_size} reader calls/step")

    global_step = int(state.get("global_step", 0))
    stop_reason = "completed"
    order = list(range(n_train))

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{cfg.epochs}\n{'='*60}")
        rng.shuffle(order)
        policy.train()
        ep_loss = ep_n = 0.0

        for si in tqdm(range(steps_per_epoch), desc=f"RWR epoch {epoch+1}"):
            ok, reason = budget_left()
            if not ok:
                stop_reason = reason
                print(f"\nSTOP: {reason}")
                break

            sel = order[si * cfg.batch_size:(si + 1) * cfg.batch_size]
            if not sel:
                continue
            sel_t = torch.tensor(sel)
            ids = train_d["input_ids"][sel_t].long().to(device)
            am = train_d["attention_mask"][sel_t].long().to(device)
            wid = train_d["word_id"][sel_t].to(device)
            items = [train_items[i] for i in sel]
            nwmax = int(train_d["n_words"][sel_t].max().item())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = policy(input_ids=ids, attention_mask=am).logits
            logits = logits.float()
            logp = F.log_softmax(logits, dim=-1)
            lk, ld = logp[..., KEEP_LABEL], logp[..., DROP_LABEL]
            p_keep = lk.exp()

            wp, tpw, wvalid = pool_words(p_keep.detach(), wid, nwmax)

            masks = sample_candidates(wp, tpw, wvalid)

            R, rmeta = rewards_for(items, masks, tpw, wvalid, "train")
            R = R.to(device)
            # anti-degeneracy: a candidate under the keep floor is removed from the
            # softmax outright, not merely penalized, so it can never be the target
            # the policy imitates even when every candidate is bad.
            content = (tpw * wvalid).sum(dim=1).clamp(min=1.0)
            kfrac = torch.stack([(tpw * m.to(tpw.dtype)).sum(dim=1) / content
                                 for m in masks], dim=1)
            alive = kfrac >= cfg.keep_floor
            alive[:, -1] = True                     # keep the deterministic anchor
            Rm = R.masked_fill(~alive, float("-inf"))
            w = F.softmax((Rm - Rm.max(dim=1, keepdim=True).values) / cfg.reward_temp,
                          dim=1)
            w = torch.nan_to_num(w, nan=0.0).detach()

            # Weighted imitation, at TOKEN level: the word decision is broadcast back
            # through word_id so the gradient reaches the token logits that serving
            # will actually pool.
            valid = wid >= 0
            idx = torch.where(valid, wid.long(), torch.zeros_like(wid.long()))
            content_f = valid.float()
            ntok = content_f.sum(dim=1).clamp(min=1.0)
            imit = 0.0
            for k in range(len(masks)):
                a_tok = torch.gather(masks[k].float(), 1, idx) * content_f
                bce = -(a_tok * lk + (1 - a_tok) * ld) * content_f
                imit = imit + (w[:, k].unsqueeze(1) * bce).sum(dim=1) / ntok
            imit = imit.mean()

            ref_lp = train_ref[sel_t].to(device).float()
            kl = (p_keep * (lk - ref_lp[..., KEEP_LABEL])
                  + (1 - p_keep) * (ld - ref_lp[..., DROP_LABEL]))
            kl = (kl * content_f).sum(dim=1) / ntok
            loss = imit + cfg.kl_coef * kl.mean()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                                       cfg.max_grad_norm).item()
            optimizer.step()

            ep_loss += loss.item()
            ep_n += 1
            global_step += 1

            if global_step % cfg.log_every_n_steps == 0:
                wandb.log({
                    "train/loss": loss.item(), "train/imit": imit.item(),
                    "train/kl": kl.mean().item(), "train/grad_norm": grad_norm,
                    "train/p_keep_mean": p_keep[valid].mean().item(),
                    "train/p_keep_std": p_keep[valid].std().item(),
                    "train/word_prob_mean": wp[wvalid].mean().item(),
                    "train/alive_frac": alive.float().mean().item(),
                    "train/weight_entropy": float(
                        -(w.clamp(min=1e-9) * w.clamp(min=1e-9).log()).sum(1).mean()),
                    "reader/llm_calls": llm_calls,
                    "reader/cache_hit_rate": cache_hits / max(1, cache_hits + cache_misses),
                    "reader/errors": reader_errors,
                    "epoch_frac": epoch + (si + 1) / steps_per_epoch,
                    **rmeta,
                }, step=global_step)

            if (si + 1) % val_every == 0 or si == steps_per_epoch - 1:
                v = validate()
                score = v.get(f"answer_survival@{cfg.select_keep:g}", 0.0)
                print(f"\n  [val] step {global_step} " + "  ".join(
                    f"{k}={val:.4f}" for k, val in sorted(v.items())))
                rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "epoch": epoch + 1,
                       "epoch_frac": epoch + (si + 1) / steps_per_epoch,
                       "global_step": global_step, "score": score,
                       "llm_calls": llm_calls, "train_loss": ep_loss / max(1, ep_n),
                       **ceilings, **v, **rmeta}
                log_metrics(rec)
                wandb.log({**{f"val/{k}": val for k, val in v.items()},
                           "val/score": score}, step=global_step)

                save_ckpt(latest_path, rec)
                state_path.write_text(json.dumps(
                    {"epoch": epoch, "global_step": global_step,
                     "llm_calls": llm_calls, "init_from": anchor_from,
                     "last": rec}, indent=2))
                if score > best_score:
                    best_score = score
                    print(f"  *** New best val answer_survival@{cfg.select_keep:g}: "
                          f"{score:.4f} ***")
                    save_ckpt(save_path)
                    with open(save_path / "best_metrics.json", "w") as f:
                        json.dump({"epoch": epoch + 1, "score": score, "val": v,
                                   "global_step": global_step,
                                   "llm_calls": llm_calls,
                                   "hyperparams": asdict(cfg)}, f, indent=2)
                    wandb.summary["best/score"] = best_score
                    wandb.summary["best/step"] = global_step
                    for k, val in v.items():
                        wandb.summary[f"best/{k}"] = val
                output_vol.commit()
                data_vol.commit()          # persist the reader cache

        print(f"  epoch {epoch+1}: mean loss {ep_loss / max(1, ep_n):.4f}  "
              f"reader calls {llm_calls}")
        state_path.write_text(json.dumps({"epoch": epoch + 1,
                                          "global_step": global_step,
                                          "llm_calls": llm_calls,
                                          "init_from": anchor_from}, indent=2))
        output_vol.commit()
        if stop_reason != "completed":
            break

    cache_fh.close()
    pool_exec.shutdown(wait=False)
    output_vol.commit()
    data_vol.commit()
    print(f"\nDone ({stop_reason}). best val answer_survival@{cfg.select_keep:g}="
          f"{best_score:.4f}  reader calls={llm_calls}  "
          f"cache_hit_rate={cache_hits / max(1, cache_hits + cache_misses):.3f}  "
          f"-> {save_path}")
    wandb.summary["final/best_score"] = best_score
    wandb.summary["final/llm_calls"] = llm_calls
    wandb.summary["final/stop_reason"] = stop_reason
    wandb.finish()
    return {"run_name": run_name, "best_score": best_score,
            "save_path": str(save_path), "llm_calls": llm_calls,
            "stop_reason": stop_reason}


# ---------------------------------------------------------------------------
# Reader probe (CPU, no GPU, no policy) -- run this BEFORE a real launch
# ---------------------------------------------------------------------------

@app.function(image=image, timeout=1800, volumes={DATA_VOL: data_vol}, secrets=SECRETS)
def reader_probe(n: int = 12, keeps: str = "1.0,0.33,0.22,0.12",
                 reader_model: str = "gemini-3.1-flash-lite",
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
                 concurrency: int = 16, data_subdir: str = "v9-rl", seed: int = 0,
                 show: int = 3):
    """Measure answer_survival on val items with RANDOM word masks at several keeps.

    This costs no GPU and answers the two questions that decide whether the reward is
    worth training against:

      keep=1.0  -- the CEILING. If the reader cannot answer from the UNCOMPRESSED
                   chunk, answer_survival can never reach 1.0 and the reward's dynamic
                   range is whatever this number is. A low ceiling means the questions
                   or the chunk windows are wrong, not the policy.
      keep<1.0  -- the FLOOR from random masks. If random keeps score as well as full
                   text, the reward cannot distinguish a good policy from a coin flip
                   and RWR has nothing to optimize.

    It also verifies the secret attaches, the OpenAI-compat endpoint answers, and
    answer_hit's normalization agrees with what the model actually replies."""
    import random as _rnd
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from v9_rl_prep import fact_survival, render_mask

    rng = _rnd.Random(seed)
    ddir = Path(DATA_VOL) / data_subdir
    items = [json.loads(l) for l in
             (ddir / "val_meta.jsonl").read_text().splitlines() if l.strip()][:n]
    keep_list = [float(x) for x in keeps.split(",") if x.strip()]
    client = make_reader_client(base_url, 60.0)
    pool = ThreadPoolExecutor(max_workers=concurrency)
    print(f"reader probe: {len(items)} val items x keeps {keep_list} "
          f"= {len(items)*len(keep_list)} calls to {reader_model}")

    for kf in keep_list:
        jobs = []
        for it in items:
            nw = it["n_words"]
            keep = ([True] * nw if kf >= 1.0
                    else [rng.random() < kf for _ in range(nw)])
            jobs.append((it, render_mask(it["words"], it["nl_after"], keep)))

        def work(job):
            it, rendered = job
            try:
                reply = ask_reader(client, reader_model, it["question"], rendered,
                                   96, 5, rng)
                return answer_hit(it["answer"], reply), reply, rendered, None
            except Exception as e:
                return 0.0, "", rendered, f"{type(e).__name__}: {e}"

        res = list(pool.map(work, jobs))
        errs = [r[3] for r in res if r[3]]
        ans = float(np.mean([r[0] for r in res]))
        fac = float(np.mean([fact_survival(it["gold_fact_words"], rd)
                             for (it, rd) in jobs]))
        chars = float(np.mean([len(rd) for _it, rd in jobs]))
        print(f"\nkeep={kf:g}  answer_survival={ans:.3f}  fact_survival={fac:.3f}  "
              f"mean_chars={chars:.0f}  errors={len(errs)}")
        for e in errs[:2]:
            print(f"    ERROR {e}")
        for i in range(min(show, len(res))):
            hit, reply, _rd, _e = res[i]
            print(f"    [{'HIT ' if hit else 'MISS'}] gold={items[i]['answer'][:70]!r} "
                  f"reply={reply[:70]!r}")
    pool.shutdown(wait=False)
    print("\nProbe done. Expect answer_survival to FALL with keep; if keep=1.0 is low, "
          "fix the data before spending GPU hours.")


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# What to watch, in priority order:
#   1. val/answer_survival@0.22 -- the selection metric and the thing FinanceBench
#      scores. Everything else is diagnosis.
#   2. train/keep_frac: if it slides toward keep_floor while answer_survival is flat,
#      lam is too high (or the reader is failing and scoring everything 0).
#   3. train/reward_spread: near zero means the K candidates are indistinguishable and
#      the softmax has nothing to weight -- raise gumbel_temp (or sample_temp under
#      --sampler bernoulli). Measured ~1.7 at init with gumbel, ~0.9 with bernoulli.
#   4. train/kl: should stay small and bounded. Growing without bound means kl_coef is
#      too low for this lr and the trust region is gone.
#   5. reader/errors and reader/cache_hit_rate -- a silent reader outage looks exactly
#      like a policy that stopped preserving answers.

@app.local_entrypoint()
def main(init_from: str = "", lam: float = 1.0, kl_coef: float = 0.05,
         lr: float = 5.0e-6, k: int = 6, reward_temp: float = 0.10,
         sample_temp: float = 1.0, target_keep: float = 0.22, keep_floor: float = 0.04,
         degen_pen: float = 2.0, w_answer: float = 2.0, w_fact: float = 0.5,
         sampler: str = "gumbel", gumbel_temp: float = 0.3,
         epochs: int = 4, batch: int = 4, pool: str = "min",
         reader_model: str = "gemini-3.1-flash-lite", concurrency: int = 16,
         max_llm_calls: int = 150_000, max_hours: float = 6.0,
         data_subdir: str = "v9-rl", tag: str = "", seed: int = 0,
         stub_reader: int = 0, smoke: int = 0):
    """--init-from <ckpt-under-/output-or-/models>  [knobs]

    smoke = 8 train / 4 val items, 1 epoch, stub reader, tiny budgets: exercises
    pooling, sampling, rendering, reward, loss, val and checkpointing end to end on
    the GPU without spending a single reader call."""
    cfg = dict(
        init_from=init_from, lam=lam, kl_coef=kl_coef, lr=lr, k=k,
        reward_temp=reward_temp, sample_temp=sample_temp, target_keep=target_keep,
        keep_floor=keep_floor, degen_pen=degen_pen, w_answer=w_answer, w_fact=w_fact,
        sampler=sampler, gumbel_temp=gumbel_temp,
        epochs=epochs, batch_size=batch, pool=pool, reader_model=reader_model,
        reader_concurrency=concurrency, max_llm_calls=max_llm_calls,
        max_hours=max_hours, data_subdir=data_subdir, seed=seed,
        stub_reader=bool(stub_reader),
        run_name=f"v9-rwr-{tag}" if tag else "v9-rwr",
    )
    if not init_from:
        raise SystemExit("--init-from is required (the wave-2 SFT checkpoint to start "
                         "from, e.g. exp-20260725-235952-w2-140-s2/best)")
    if smoke:
        cfg.update(run_name=f"{cfg['run_name']}-smoke", epochs=1, limit_train=8,
                   limit_val=4, batch_size=2, k=3, stub_reader=True,
                   max_llm_calls=200, max_hours=0.5, val_every_frac=1.0,
                   log_every_n_steps=1)
    print(f"init_from={init_from}  run_name={cfg['run_name']}  data={data_subdir}  "
          f"K={cfg['k']}  lam={lam}  kl={kl_coef}  reader={cfg['reader_model']}"
          f"{'  [SMOKE, stub reader]' if smoke else ''}")
    train.remote(config_overrides=cfg)
