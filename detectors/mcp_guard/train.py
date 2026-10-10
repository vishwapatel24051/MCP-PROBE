"""Fine-tune MCP-Guard's Stage II classifier on the MCPTox `tune` split.

An extra condition next to the frozen MCP-Guard (team decision pending in
DECISIONS.md). Trains on `tune` only, picks the epoch on `val`, never loads `test`.

--init mcpguard : GenTelLab's released Stage II weights (HF GenTelLab/MCP-Guard@861a928)
                  with the bert-base-uncased tokenizer they were trained with.
--init e5base   : intfloat/multilingual-e5-base@d128750 with its own tokenizer and a
                  fresh classification head (the paper's recipe, Sec. 3.2, on our data).
--scope full    : all weights (as the paper does).
--scope head    : encoder frozen, only the classifier head trains.

Loss: cross-entropy over {benign, poisoned} (the 2-logit form of the paper's Eq. 1),
class-weighted by inverse frequency because `tune` is ~3:1 poisoned:clean.

Usage:
  uv run python -m detectors.mcp_guard.train --init mcpguard --scope full --seed 0
  uv run python -m detectors.mcp_guard.train --init e5base --scope head --seed 1 --out-root /path/to/checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data.loader import load_cases

from .metrics import binary_metrics
from .pipeline import ROOT, _norm
from .stage2 import BERT_TOKENIZER, MAX_LENGTH, NeuralStage, pick_device

E5_BASE = ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a")
DEFAULTS = {  # (lr, epochs) per scope; head-only needs a larger lr
    "full": (2e-5, 5),
    "head": (1e-3, 10),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(split: str) -> tuple[list[str], list[int]]:
    cases = [c for c in load_cases(split) if c["description"] and c["description"].strip()]
    return [_norm(c["description"]) for c in cases], [int(c["is_poisoned"]) for c in cases]


def build_model(init: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if init == "mcpguard":
        stage = NeuralStage(loader="hf", tokenizer="bert", device="cpu")
        return stage.model, stage.tokenizer, {"weights": "GenTelLab/MCP-Guard@861a928", "tokenizer": "@".join(BERT_TOKENIZER)}
    model = AutoModelForSequenceClassification.from_pretrained(E5_BASE[0], revision=E5_BASE[1], num_labels=2)
    tok = AutoTokenizer.from_pretrained(E5_BASE[0], revision=E5_BASE[1])
    return model, tok, {"weights": "@".join(E5_BASE), "tokenizer": "@".join(E5_BASE)}


def freeze_encoder(model) -> None:
    base = getattr(model, model.base_model_prefix)
    for p in base.parameters():
        p.requires_grad = False


@torch.no_grad()
def predict(model, tok, texts: list[str], device, batch_size: int = 64) -> list[float]:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        out.extend(torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist())
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune MCP-Guard Stage II on MCPTox tune.")
    ap.add_argument("--init", required=True, choices=["mcpguard", "e5base"])
    ap.add_argument("--scope", required=True, choices=["full", "head"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of steps for linear warmup")
    ap.add_argument("--select-metric", default="f1", choices=["f1", "balanced_accuracy"],
                    help="val metric used to keep the best epoch (Stage II alone, threshold 0.5)")
    ap.add_argument("--limit", type=int, help="first N tune/val cases only (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "mcp_guard" / "finetune"))
    args = ap.parse_args()

    lr = args.lr or DEFAULTS[args.scope][0]
    epochs = args.epochs or DEFAULTS[args.scope][1]
    set_seed(args.seed)
    device = pick_device(args.device)

    x_tr, y_tr = load_split("tune")
    x_va, y_va = load_split("val")
    if args.limit:
        x_tr, y_tr, x_va, y_va = x_tr[: args.limit], y_tr[: args.limit], x_va[: args.limit], y_va[: args.limit]
    n_pos, n_neg = sum(y_tr), len(y_tr) - sum(y_tr)
    print(f"tune {len(y_tr)} ({n_pos} poisoned / {n_neg} clean)  val {len(y_va)} ({sum(y_va)} / {len(y_va) - sum(y_va)})")

    model, tok, source = build_model(args.init)
    if args.scope == "head":
        freeze_encoder(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"init={args.init} scope={args.scope} lr={lr} epochs={epochs} seed={args.seed} "
          f"trainable={sum(p.numel() for p in trainable) / 1e6:.1f}M device={device}")

    # Inverse-frequency class weights: each class contributes equally to the loss.
    w = torch.tensor([len(y_tr) / (2 * max(n_neg, 1)), len(y_tr) / (2 * max(n_pos, 1))], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)
    steps = epochs * math.ceil(len(x_tr) / args.batch_size)
    warm = int(args.warmup * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else max(0.0, (steps - s) / max(steps - warm, 1))
    )
    use_amp = device.type == "cuda"

    run = f"{args.init}_{args.scope}_seed{args.seed}"
    out_dir = Path(args.out_root) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    def val_metrics() -> dict:
        scores = predict(model, tok, x_va, device)
        m = binary_metrics([bool(v) for v in y_va], [s >= 0.5 for s in scores])
        rec, spec = m["recall"] or 0.0, 1 - (m["fpr"] or 0.0)
        m["balanced_accuracy"] = (rec + spec) / 2
        return m

    history, best, best_epoch = [], -1.0, None
    m0 = val_metrics()
    history.append({"epoch": 0, "train_loss": None, "val": m0})
    print(f"epoch 0 (before training)  val f1 {m0['f1']}  fpr {m0['fpr']}")
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(x_tr)))
        random.shuffle(order)
        total, n = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            enc = tok([x_tr[j] for j in idx], return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)
            enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            labels = torch.tensor([y_tr[j] for j in idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
            n += len(idx)
        m = val_metrics()
        history.append({"epoch": epoch, "train_loss": total / n, "val": m})
        print(f"epoch {epoch}  loss {total / n:.4f}  val f1 {m['f1']:.4f}  fpr {m['fpr']:.4f}  recall {m['recall']:.4f}")
        score = m[args.select_metric] or 0.0
        if score > best:
            best, best_epoch = score, epoch
            model.save_pretrained(out_dir, safe_serialization=True)
            tok.save_pretrained(out_dir)

    meta = {
        "run": run, "init": args.init, "scope": args.scope, "seed": args.seed, "lr": lr, "epochs": epochs,
        "batch_size": args.batch_size, "weight_decay": args.weight_decay, "warmup": args.warmup,
        "class_weights": w.tolist(), "max_length": MAX_LENGTH, "select_metric": args.select_metric,
        "best_epoch": best_epoch, "best_val": history[best_epoch]["val"] if best_epoch is not None else None,
        "history": history, "source": source, "train_split": "tune", "selection_split": "val",
        "n_train": len(y_tr), "n_val": len(y_va), "limit": args.limit, "git_commit": git_commit(),
        "train_seconds": time.perf_counter() - t_start, "device": str(device),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"best epoch {best_epoch} (val {args.select_metric} {best:.4f}) -> {out_dir}")


if __name__ == "__main__":
    main()
