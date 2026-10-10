"""Reproduction check on upstream's own data (GenTelLab/MCP-Guard test_data/dev.csv).

This is GenTelLab's sample file, not MCPTox, so it is safe to inspect. It is
downloaded at the pinned commit rather than vendored: one of its attack samples
contains a Stripe test key that GitHub push protection blocks. It tells us
whether our setup reproduces the paper's in-domain numbers (arXiv 2508.10991,
Table 4a: Stage I F1 55.6, Stage II acc 96.0 / F1 95.1) before we test on MCPTox.

Usage: uv run python -m detectors.mcp_guard.reproduce_upstream_dev [--limit N] [--out results/mcp_guard/upstream_dev.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import urllib.request

from ._upstream import UPSTREAM_COMMIT
from .metrics import binary_metrics
from .pipeline import MCPGuard, ROOT
from .stage1 import PatternStage
from .stage2 import NeuralStage

DEV_URL = f"https://raw.githubusercontent.com/GenTelLab/MCP-Guard/{UPSTREAM_COMMIT}/test_data/dev.csv"
DEV_CSV = ROOT / "results" / "mcp_guard" / "upstream_dev.csv"  # results/ is gitignored


def fetch_dev_csv() -> Path:
    if not DEV_CSV.exists():
        DEV_CSV.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DEV_URL, DEV_CSV)
    return DEV_CSV


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/mcp_guard/upstream_dev.json")
    args = ap.parse_args()

    with fetch_dev_csv().open(encoding="utf-8") as f:
        rows = [(r["description"].strip(), int(r["Label"]) == 1) for r in csv.DictReader(f)]
    rows = rows[: args.limit] if args.limit else rows
    tools = [{"tool_name": "test_tool", "description": d} for d, _ in rows]  # main.py uses "test_tool"
    y = [lab for _, lab in rows]
    print(f"{len(rows)} samples ({sum(y)} malicious)")

    s1 = PatternStage()
    results = {}
    t0 = time.perf_counter()
    v = MCPGuard(mode="released", stages=("s1",), stage1=s1).scan_many(tools)
    results["stage1_only"] = binary_metrics(y, [x.blocked for x in v]) | {"secs": time.perf_counter() - t0}

    for tok in ("bert", "checkpoint_xlmr"):
        s2 = NeuralStage(tokenizer=tok, device=args.device)
        t0 = time.perf_counter()
        v = MCPGuard(mode="released", stages=("s2",), stage2=s2).scan_many(tools)
        results[f"stage2_only[{tok}]"] = binary_metrics(y, [x.blocked for x in v]) | {"secs": time.perf_counter() - t0}
        t0 = time.perf_counter()
        v = MCPGuard(mode="released", stages=("s1", "s2"), stage1=s1, stage2=s2).scan_many(tools)
        results[f"stage1+2[{tok}]"] = binary_metrics(y, [x.blocked for x in v]) | {"secs": time.perf_counter() - t0}
        del s2

    print(f"\n{'config':28s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'secs':>6s}")
    for k, m in results.items():
        f = lambda x: f"{100 * x:6.1f}" if x is not None else "     -"
        print(f"{k:28s} {f(m['accuracy'])} {f(m['precision'])} {f(m['recall'])} {f(m['f1'])} {m['secs']:6.1f}")
    print("\npaper (Table 4a, MCP-AttackBench): Stage I acc 74.6 / F1 55.6; Stage II acc 96.0 / F1 95.1")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(rows), "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
