"""Run MCP-Guard over an MCPTox split and report detection metrics.

Uses data.loader.load_cases, so the test split stays locked unless FINAL_EVAL=1.

Examples
  uv run python -m detectors.mcp_guard.evaluate --split tune --stages s1,s2
  uv run python -m detectors.mcp_guard.evaluate --split val --mode released
  uv run python -m detectors.mcp_guard.evaluate --split tune --stages s1,s2 --sweep

Writes results/mcp_guard/<run>/{predictions.jsonl,metrics.json,run.json}.
Predictions hold ids, labels and verdicts only, never description text.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data.loader import load_cases

from .metrics import binary_metrics, summarize
from .pipeline import DEFAULT_CONFIG, ROOT, MCPGuard


def _pct(x) -> str:
    return f"{100 * x:5.1f}" if x is not None else "    -"


def print_summary(m: dict) -> None:
    o = m["overall"]
    print(f"\nn={o['n']}  TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']}")
    print(f"acc {_pct(o['accuracy'])}  prec {_pct(o['precision'])}  rec {_pct(o['recall'])}  "
          f"f1 {_pct(o['f1'])}  FPR(clean) {_pct(o['fpr'])}")
    for field in ("by_risk_category", "by_paradigm", "by_test_cell"):
        if field not in m:
            continue
        print(f"\n{field[3:]:28s}     n  recall")
        for k, g in m[field].items():
            rate = g["recall"] if g["recall"] is not None else (1 - g["fpr"] if g["fpr"] is not None else None)
            print(f"{k:28s} {g['n']:5d}  {_pct(rate)}")
    print(f"\ndecided at stage: {m['decided_at_stage']}")
    if "latency_ms" in m:
        lat = m["latency_ms"]
        print(f"latency ms: mean {lat['mean']:.1f}  p50 {lat['p50']:.1f}  p95 {lat['p95']:.1f}")


def sweep(rows: list[dict], grid: list[float]) -> list[dict]:
    """Stage I + Stage II at different T_u, from stored scores (no re-inference)."""
    out = []
    y = [bool(r["is_poisoned"]) for r in rows]
    for tu in grid:
        pred = [bool(r["s1_hits"]) or (r["s2_score"] is not None and r["s2_score"] > tu) for r in rows]
        out.append({"tu": tu, **binary_metrics(y, pred)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MCP-Guard on an MCPTox split.")
    ap.add_argument("--split", required=True, choices=["tune", "val", "test"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--mode", choices=["paper", "released"])
    ap.add_argument("--stages", help="comma list, e.g. s1,s2 (default: from config)")
    ap.add_argument("--tu", type=float)
    ap.add_argument("--limit", type=int, help="first N cases only (smoke tests)")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    ap.add_argument("--sweep", action="store_true", help="also report Stage I+II metrics over a T_u grid")
    ap.add_argument("--s3-model", help="override stage3.model (e.g. a vLLM-served model name)")
    ap.add_argument("--s3-base-url", help="override stage3.base_url (e.g. http://localhost:8000/v1)")
    ap.add_argument("--s3-prompt", choices=["upstream", "paper"])
    ap.add_argument("--s2-path", help="use a fine-tuned Stage II from train.py (directory)")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    cases = load_cases(args.split)
    if args.limit:
        cases = cases[: args.limit]

    stages = tuple(args.stages.split(",")) if args.stages else None
    guard = MCPGuard.from_config(
        args.config, mode=args.mode, stages=stages, tu=args.tu,
        stage3={"model": args.s3_model, "base_url": args.s3_base_url, "prompt": args.s3_prompt},
        stage2={"loader": "finetuned", "path": args.s2_path} if args.s2_path else None,
    )
    run = f"{args.split}_{guard.mode}_{'-'.join(guard.stages)}" + (f"_{args.tag}" if args.tag else "")
    out_dir = ROOT / "results" / "mcp_guard" / run
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{run}: {len(cases)} cases ({sum(c['is_poisoned'] for c in cases)} poisoned)")
    t0 = time.perf_counter()
    verdicts = guard.scan_many(cases)
    wall = time.perf_counter() - t0

    rows = []
    for c, v in zip(cases, verdicts):
        rows.append({
            "case_id": c["case_id"], "server": c["server"], "is_poisoned": c["is_poisoned"],
            "risk_category": c["risk_category"], "paradigm": c["paradigm"], "test_cell": c["test_cell"],
            "blocked": v.blocked, "stage": v.stage, "s2_score": v.s2_score, "s3_verdict": v.s3_verdict,
            "s1_hits": sorted({h["detector"] for h in v.s1_hits}), "latency_ms": v.latency_ms,
        })
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    metrics = summarize(rows)
    if args.sweep:
        metrics["tu_sweep"] = sweep(rows, [round(0.05 * i, 2) for i in range(1, 20)])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "run.json").write_text(json.dumps({
        "split": args.split, "mode": guard.mode, "stages": guard.stages, "tu": guard.tu, "band": guard.band,
        "config": str(Path(args.config).resolve().relative_to(ROOT)), "n_cases": len(cases),
        "wall_seconds": wall,
        "stage2": getattr(guard.s2, "cache_tag", None), "stage3": getattr(guard.s3, "cache_tag", None),
    }, indent=2), encoding="utf-8")

    print_summary(metrics)
    if args.sweep:
        print(f"\n{'T_u':>5s}  prec   rec    f1   FPR")
        for s in metrics["tu_sweep"]:
            print(f"{s['tu']:5.2f} {_pct(s['precision'])} {_pct(s['recall'])} {_pct(s['f1'])} {_pct(s['fpr'])}")
    print(f"\nwrote {out_dir.relative_to(ROOT)}/  ({wall:.1f}s)")


if __name__ == "__main__":
    main()
