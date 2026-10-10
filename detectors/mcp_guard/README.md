# MCP-Guard

MCP-Guard (arXiv 2508.10991) wrapped for our experiments: Stage I regex rules → Stage II neural classifier → Stage III LLM arbiter, run over tool descriptions. We train nothing; tuning means thresholds, prompts and Stage I rule selection, on `tune`/`val` only.

## What we use from upstream

| Piece | Source | Notes |
|---|---|---|
| Stage I rules | `upstream/` = github.com/GenTelLab/MCP-Guard @ `4bc9779`, vendored unmodified | All 7 detectors, upstream's rule JSONs. Left out: `configs/models/model_registry.json` (see issues) and `test_data/dev.csv` (contains a Stripe test key that GitHub push protection blocks; `reproduce_upstream_dev.py` downloads it at the pinned commit). |
| Stage II weights | HF `GenTelLab/MCP-Guard` @ `861a928` (`model.onnx` + `finetuned_from_onnx_v3.pth`) | Downloaded to the HF cache on first use, not committed. |
| Stage II tokenizer | `google-bert/bert-base-uncased` @ `86b5e09` | What upstream's code loads; reproduces the paper (below). |
| Stage III prompt | upstream `llm_detector.py` system prompt (default) or the paper's Box 1 | Any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama). |

Our code: `pipeline.py` (the cascade), `stage1.py` (calls the upstream detectors), `stage2.py` (loads the checkpoint), `stage3.py` (LLM client), `evaluate.py` (MCPTox runs), `reproduce_upstream_dev.py` (sanity check), `carc/` (Slurm jobs).

## Setup

```bash
uv sync
uv pip install -r detectors/mcp_guard/requirements.txt   # torch, transformers, onnx2pytorch, fastapi, ...
# data: see the repo README (MCPTox clone + scripts/build_splits.py)
# Stage III with OpenAI: OPENAI_API_KEY in .env
```

These extras are kept out of the shared `pyproject.toml` for now; note that a plain `uv sync` removes them again.

## Run

```bash
# MCPTox dev splits (test stays locked unless FINAL_EVAL=1)
uv run python -m detectors.mcp_guard.evaluate --split tune --stages s1,s2 --sweep
uv run python -m detectors.mcp_guard.evaluate --split val                       # full cascade, gpt-4o-mini
uv run python -m detectors.mcp_guard.evaluate --split val --mode released
uv run python -m detectors.mcp_guard.evaluate --split val --s3-model Qwen/Qwen2.5-7B-Instruct --s3-base-url http://localhost:8000/v1

# Reproduction check on upstream's own sample file
uv run python -m detectors.mcp_guard.reproduce_upstream_dev

uv run pytest tests/test_mcp_guard.py                      # MCP_GUARD_SLOW=1 also checks the checkpoint
```

Outputs go to `results/mcp_guard/<split>_<mode>_<stages>[_tag]/` (`predictions.jsonl` has ids, labels, scores and verdicts, no description text). Stage II scores and Stage III verdicts are cached in `results/mcp_guard/cache.sqlite`.

On CARC (Discovery, account `xiangren_1987`): `sbatch detectors/mcp_guard/carc/setup.slurm` once, then `sbatch --export=ALL,MODEL=...,SPLIT=val detectors/mcp_guard/carc/eval_s3.slurm` to run Stage III with an open model served by vLLM on the same GPU.

## API (for the proxy)

```python
from detectors.mcp_guard import MCPGuard
guard = MCPGuard.from_config()                      # configs/mcp_guard.yaml
v = guard.scan(tool["name"], tool.get("description"))
v.blocked, v.stage, v.s1_hits, v.s2_score, v.s3_verdict, v.latency_ms
verdicts = guard.scan_many([{"tool_name": ..., "description": ...}, ...])   # batched Stage II
```

Empty or missing descriptions are allowed without scanning (`stage="empty"`), as upstream does.

## Decision modes (`mode` in the config)

- **`paper`** (default): follows Sec. 3.2–3.3 and Eq. 2. Stage I hit → block. Stage II score P; if 0.45 < P < 0.55 the LLM decides (unsafe → block, safe → allow, uncertain → block iff P > T_u); otherwise block iff P > T_u (T_u = 0.45).
- **`released`**: what the GitHub code does. Stage I hit → block; P ≥ 0.5 → block; *every* other case goes to the LLM; unsure and LLM errors are allowed.

`stages` (e.g. `s1,s2`) runs ablations; without Stage III, Stage II decides.

## Reproduction check (upstream `test_data/dev.csv` @ `4bc9779`, 1,053 samples, 432 malicious)

| Config | Acc | Prec | Rec | F1 |
|---|---:|---:|---:|---:|
| Stage I only | 74.5 | 97.7 | 38.9 | 55.6 |
| Stage II only, BERT tokenizer | 96.0 | 96.7 | 93.5 | 95.1 |
| Stage II only, XLM-R tokenizer shipped with the checkpoint | 66.9 | 91.1 | 21.3 | 34.5 |
| Stage I+II, BERT tokenizer | 95.9 | 96.0 | 94.0 | 95.0 |
| *Paper Table 4a (Stage I / Stage II)* | *74.6 / 96.0* | *97.7 / 96.7* | *38.9 / 93.5* | *55.6 / 95.1* |

Stage I and Stage II match the paper's internal-stage rows to the decimal, so the checkpoint and rules are reproduced. The exact match also suggests those two rows were computed on this file, though the paper describes a 5,258-sample test set (unverified).

## First MCPTox numbers (dev splits only, Stage I+II, paper mode, T_u = 0.45, 2026-10-08)

| Split | n (poisoned / clean) | Prec | Rec | F1 | FPR on clean |
|---|---|---:|---:|---:|---:|
| tune | 759 / 240 | 95.1 | 99.2 | 97.1 | 16.2 |
| val | 171 / 49 | 91.4 | 100.0 | 95.5 | 32.7 |

On `tune`, Stage I alone blocks 546/759 poisoned cases (paper: 38.9% recall in-domain), mostly via `shell_injection` (342) and `important_tag` (251). That points to template artifacts in MCPTox that the regexes match, which matters for the adaptive-attacker experiment. Of the 39 false positives on `tune`, 30 come from Stage II and 9 from Stage I. The T_u sweep barely moves F1 (96.4–97.2 over 0.05–0.95) because Stage II scores are nearly binary. Stage III is not in these numbers yet.

## Fine-tuned Stage II (extra condition)

Frozen MCP-Guard stays the main "existing detector" result. As an extra condition we fine-tune its Stage II on MCPTox `tune` and pick the epoch on `val`; `test` is never loaded during training. Only the description text is used as input (first 128 tokens); Stage I rules are not changed.

| Option | Choices |
|---|---|
| `--init` | `mcpguard`: GenTelLab's released weights + bert-base-uncased tokenizer. `e5base`: `intfloat/multilingual-e5-base@d128750` + its own tokenizer, fresh head (the paper's recipe, Sec. 3.2). |
| `--scope` | `full`: all 278M weights (lr 2e-5, 5 epochs). `head`: encoder frozen, 0.6M head weights (lr 1e-3, 10 epochs). |
| Loss | Cross-entropy, inverse-frequency class weights (`tune` is 759 poisoned : 223 clean after dropping 17 empty descriptions). |
| Selection | Best epoch by val F1 of Stage II alone at 0.5 (`--select-metric balanced_accuracy` also available). |

```bash
uv run python -m detectors.mcp_guard.train --init mcpguard --scope full --seed 0
uv run python -m detectors.mcp_guard.evaluate --split val --stages s1,s2 --s2-path results/mcp_guard/finetune/mcpguard_full_seed0
sbatch detectors/mcp_guard/carc/train_sweep.slurm        # CARC: 2 inits x 2 scopes x 5 seeds + frozen reference
uv run python -m detectors.mcp_guard.summarize_finetune  # mean ± std table
```

Each run saves the best checkpoint (~1.1 GB, HF format) plus `train_meta.json` (settings, per-epoch val metrics, data counts, git commit). The `finetuned` Stage II loader reads that directory: `stage2: {loader: finetuned, path: ...}` in the config.

### Results (CARC job 12848251, one A40, 2026-10-08; 20 runs in 21 min)

Data: `tune` 982 descriptions used for training (759 poisoned / 223 clean; 30 servers), `val` 220 (171 / 49; 6 other servers) for epoch selection only.

Full cascade, Stage I+II, paper mode, T_u = 0.45, on `val` (mean ± std over 5 seeds):

| Condition | Precision | Recall | F1 | FPR on clean |
|---|---:|---:|---:|---:|
| Frozen MCP-Guard | 91.4 | 100.0 | 95.5 | 32.7 |
| E5-base, full | 93.3 ± 0.2 | 100.0 | 96.6 ± 0.1 | 24.9 ± 0.9 |
| E5-base, head only | 92.0 ± 0.4 | 100.0 | 95.9 ± 0.2 | 30.2 ± 1.7 |
| MCP-Guard weights, full | 91.9 ± 0.5 | 100.0 | 95.8 ± 0.3 | 30.6 ± 2.0 |
| MCP-Guard weights, head only | 91.3 ± 0.2 | 100.0 | 95.5 ± 0.1 | 33.1 ± 0.9 |

Stage II alone on `val` (threshold 0.5, mean over seeds):

| Condition | F1 before → after | FPR before → after |
|---|---|---|
| E5-base, full | 55.4 → 99.8 | 54.7 → 1.2 |
| E5-base, head only | 55.4 → 97.8 | 54.7 → 12.7 |
| MCP-Guard weights, full | 95.5 → 96.6 | 30.6 → 24.5 |
| MCP-Guard weights, head only | 95.5 → 95.5 | 30.6 → 30.6 |

What this shows and what it doesn't:
- **Stage I sets the false-positive floor.** 12 of the 49 clean `val` tools are blocked by Stage I (`sensitive_file` 11, `shadow_hijack` 7, `sql_injection` 2; some hit several rules), so no Stage II change can bring cascade FPR below ~24.5%. The best fine-tune removes all Stage II false positives.
- **E5-base fine-tuned fully fits MCPTox almost perfectly; MCP-Guard's own weights barely move.**
- **Caveats.** `val` chose the epoch, so its numbers are optimistic; it has only 49 clean tools (1 tool ≈ 2 FPR points) and contains no held-out categories. Clean descriptions are much shorter than poisoned ones (median 14 vs 60 BERT tokens), and many poisoned ones share template wording (`<IMPORTANT>`, "you MUST FIRST call"), so a fine-tuned model may be learning length or template rather than intent; held-out categories in `test` and the adaptive attacker are the real check. 39/759 poisoned and 20/223 clean `tune` descriptions are longer than 128 tokens and get truncated. On `tune` itself fine-tuned FPR drops to ~4–5% vs ~25% on `val`, i.e. some overfitting.

Checkpoints: `/project2/xiangren_1987/group_12/checkpoints/mcp_guard/<init>_<scope>_seed<k>/` on CARC (21 GB).

## Differences between the paper and the released code

From running the code (verified):
1. **Tokenizer.** Stage II only works with `bert-base-uncased` ids, which upstream uses; the XLM-R tokenizer files in the HF repo give F1 34.5 on upstream's own data.
2. **Backbone size.** The checkpoint is E5-*base* (12 layers, hidden 768; `config.json` `_name_or_path` mentions `multilingual-e5-base`); the paper says Multilingual-E5-large (Sec. 5.2).
3. **Checkpoint loading.** Upstream's `strict=True` load fails (the converted model lists fused layers under duplicate names) and falls back to `strict=False`; we checked that all 148 fine-tuned tensors still load. Our `hf` loader maps the same weights onto `XLMRobertaForSequenceClassification` and matches upstream's loader to ~3e-6 in the logits.

From reading the code (not executed):
4. **Stage III routing.** The router sends every case Stage II passes to the LLM, not only 0.45 < P < 0.55 (`api/routers/guardrail.py`).
5. **Dead fallback.** The Stage II score is never passed to the LLM detector, so "unsure → use P > 0.45" can't trigger. An unsure verdict (or a failed LLM call) returns an empty list, then `model_issues[0]` raises IndexError; `main.py` catches it and drops that sample from the metrics. Our `released` mode allows such cases instead of dropping them.
6. **API key overwrite.** `configs/models/model_registry.json` sets `OPENAI_API_KEY="REDACTED"` via `os.environ`, overwriting a real key; we don't vendor that file or use upstream's ModelManager.
7. **Extra LLM call.** Upstream makes a second "explain the verdict" LLM call per case that never affects the decision (and is excluded from its timing); we skip it.
8. **Temperature.** Upstream's GPT detector uses temperature 0.1; the paper reports 0.7 (Sec. 5.2). Ours defaults to 0.1 (configurable).

License: upstream's README badge and the HF model card say Apache-2.0; the GitHub repo itself has no LICENSE file.
