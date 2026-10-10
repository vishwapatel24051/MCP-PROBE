"""Binary detection metrics with breakdowns by MCPTox metadata.

Positive class = poisoned. Per-group rows for poisoned-only groups (risk
category, paradigm, poisoned test cells) report recall, since precision and
FPR are undefined without clean tools in the group.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / n if n else None,
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


def summarize(rows: list[dict]) -> dict:
    """rows: case dicts (from data.loader) merged with a boolean `blocked` and a `stage`."""
    y_true = [bool(r["is_poisoned"]) for r in rows]
    y_pred = [bool(r["blocked"]) for r in rows]
    out = {"overall": binary_metrics(y_true, y_pred)}

    for field in ("risk_category", "paradigm", "test_cell"):
        groups = defaultdict(lambda: ([], []))
        for r, t, p in zip(rows, y_true, y_pred):
            key = r.get(field)
            if key is None:
                continue
            groups[key][0].append(t)
            groups[key][1].append(p)
        if groups:
            out[f"by_{field}"] = {k: binary_metrics(*v) for k, v in sorted(groups.items())}

    out["decided_at_stage"] = dict(Counter(r["stage"] for r in rows))
    out["blocked_at_stage"] = dict(Counter(r["stage"] for r in rows if r["blocked"]))
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    if lat:
        lat_sorted = sorted(lat)
        out["latency_ms"] = {
            "mean": sum(lat) / len(lat),
            "p50": lat_sorted[len(lat) // 2],
            "p95": lat_sorted[min(len(lat) - 1, int(0.95 * len(lat)))],
        }
    return out
