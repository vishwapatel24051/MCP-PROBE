"""Table of frozen vs fine-tuned MCP-Guard (Stage I+II) on tune and val, mean ± std over seeds.

Reads results/mcp_guard/<split>_paper_s1-s2_{frozen,ft_<init>_<scope>_seed<k>}/metrics.json
and writes results/mcp_guard/finetune_summary.md.

Usage: uv run python -m detectors.mcp_guard.summarize_finetune
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict

from .pipeline import ROOT

RES = ROOT / "results" / "mcp_guard"
METRICS = ("precision", "recall", "f1", "fpr")


def fmt(vals: list[float]) -> str:
    vals = [100 * v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f}"


def main() -> None:
    groups = defaultdict(lambda: defaultdict(list))  # (split, condition) -> metric -> values
    for d in sorted(RES.glob("*_paper_s1-s2_*")):
        m = re.fullmatch(r"(tune|val)_paper_s1-s2_(frozen|ft_(\w+?)_(full|head)_seed\d+)", d.name)
        if not m or not (d / "metrics.json").exists():
            continue
        split = m[1]
        cond = "frozen" if m[2] == "frozen" else f"{m[3]} / {m[4]}"
        o = json.loads((d / "metrics.json").read_text())["overall"]
        for k in METRICS:
            groups[(split, cond)][k].append(o[k])

    lines = ["# MCP-Guard Stage I+II: frozen vs fine-tuned on MCPTox tune", "",
             "Paper mode, T_u = 0.45. Fine-tuned rows: mean ± std over seeds. FPR is on clean tools.", ""]
    for split in ("tune", "val"):
        lines += [f"## {split}", "", "| Condition | Seeds | Precision | Recall | F1 | FPR |", "|---|---:|---:|---:|---:|---:|"]
        conds = sorted({c for s, c in groups if s == split}, key=lambda c: (c != "frozen", c))
        for c in conds:
            g = groups[(split, c)]
            lines.append(f"| {c} | {len(g['f1'])} | " + " | ".join(fmt(g[k]) for k in METRICS) + " |")
        lines.append("")
    lines.append("tune is the training split, so its rows show fit, not generalization; compare on val.")
    out = RES / "finetune_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
