"""MCP-Guard three-stage cascade over a tool description.

Two decision modes, because the released code does not do what the paper says:

mode="paper"  (arXiv 2508.10991, Sec. 3.2-3.3, Eq. 2, Fig. 5)
    Stage I hit -> block.
    Stage II score P. If band_low < P < band_high and Stage III is on, ask the LLM:
        unsafe -> block, safe -> allow, uncertain/error -> block iff P > tu.
    Otherwise block iff P > tu.

mode="released"  (GenTelLab/MCP-Guard @ 4bc9779, api/routers/guardrail.py)
    Stage I hit -> block.
    Stage II: P >= 0.5 -> block.
    Every case Stage II passes goes to the LLM: unsafe -> block, safe -> allow.
    Upstream never hands the Stage II score to the LLM stage, so its
    "unsure and score > 0.45" fallback can't fire: unsure is allowed (in the
    upstream code this path actually raises IndexError, and its main.py then
    drops the sample from the metrics). LLM errors are allowed too (fail-open).

`stages` limits which stages run (for ablations). Without Stage III, Stage II decides.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .cache import ResultCache
from .stage1 import PatternStage
from .stage3 import ERROR, SAFE, UNSAFE, LLMArbiter

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "mcp_guard.yaml"


@dataclass
class Verdict:
    blocked: bool
    stage: str  # s1 | s2 | s3 | s3_fallback | empty
    s1_hits: list[dict] = field(default_factory=list)
    s2_score: float | None = None
    s3_verdict: str | None = None
    s3_raw: str | None = None
    latency_ms: float = 0.0
    stage_latency_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text: str) -> str:
    # Upstream's preprocess_data_for_next_layer("second_layer").
    return re.sub(r"\s+", " ", text).strip()


class MCPGuard:
    def __init__(
        self,
        mode: str = "paper",
        stages: tuple[str, ...] = ("s1", "s2", "s3"),
        tu: float = 0.45,
        band: tuple[float, float] = (0.45, 0.55),
        released_s2_threshold: float = 0.5,
        stage1: PatternStage | None = None,
        stage2=None,
        stage3: LLMArbiter | None = None,
        cache: ResultCache | None = None,
    ):
        if mode not in ("paper", "released"):
            raise ValueError("mode must be 'paper' or 'released'")
        self.mode = mode
        self.stages = tuple(stages)
        self.tu = tu
        self.band = band
        self.released_s2_threshold = released_s2_threshold
        self.s1 = stage1 if "s1" in self.stages else None
        self.s2 = stage2 if "s2" in self.stages else None
        self.s3 = stage3 if "s3" in self.stages else None
        if "s2" in self.stages and self.s2 is None:
            raise ValueError("stage s2 requested but no Stage II model given")
        self.cache = cache

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG, **overrides) -> "MCPGuard":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        nested = {k: overrides.pop(k, None) or {} for k in ("stage2", "stage3")}
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        for k, over in nested.items():
            cfg[k] = {**(cfg.get(k) or {}), **{kk: vv for kk, vv in over.items() if vv is not None}}
        stages = tuple(cfg.get("stages", ["s1", "s2", "s3"]))

        s1 = PatternStage(cfg.get("stage1", {}).get("detectors")) if "s1" in stages else None
        s2 = None
        if "s2" in stages:
            from .stage2 import NeuralStage  # torch import is slow; only when needed
            s2 = NeuralStage(**cfg.get("stage2", {}))
        s3 = None
        s3_cfg = cfg.get("stage3") or {}
        if "s3" in stages and s3_cfg.get("model"):
            s3 = LLMArbiter(**s3_cfg)
        elif "s3" in stages:
            log.warning("Stage III requested but stage3.model is empty; running without it")

        cache_path = cfg.get("cache_path")
        return cls(
            mode=cfg.get("mode", "paper"),
            stages=stages,
            tu=cfg.get("tu", 0.45),
            band=tuple(cfg.get("band", (0.45, 0.55))),
            released_s2_threshold=cfg.get("released_s2_threshold", 0.5),
            stage1=s1, stage2=s2, stage3=s3,
            cache=ResultCache(ROOT / cache_path) if cache_path else None,
        )

    # ---- stage helpers ------------------------------------------------------

    def _s2_scores(self, texts: list[str]) -> list[float]:
        out: list[float | None] = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            hit = self.cache.get(self.s2.cache_tag, t) if self.cache else None
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        if todo:
            scores = self.s2.score_many([texts[i] for i in todo])
            for i, s in zip(todo, scores):
                out[i] = s
                if self.cache:
                    self.cache.put(self.s2.cache_tag, texts[i], s)
        return out  # type: ignore[return-value]

    def _s3(self, tool_name: str | None, text: str) -> tuple[str, str, float]:
        key = f"{tool_name}\x00{text}"
        if self.cache:
            hit = self.cache.get(self.s3.cache_tag, key)
            if hit is not None:
                return hit["verdict"], hit["raw"], 0.0
        res = self.s3.judge(tool_name, text)
        if self.cache and res.verdict != ERROR:
            self.cache.put(self.s3.cache_tag, key, {"verdict": res.verdict, "raw": res.raw})
        return res.verdict, res.raw, res.latency_ms

    def _decide_after_s2(self, v: Verdict, tool_name: str | None, text: str) -> None:
        p = v.s2_score
        if self.mode == "released":
            if p is not None and p >= self.released_s2_threshold:
                v.blocked, v.stage = True, "s2"
                return
            if self.s3 is None:
                v.blocked, v.stage = False, "s2"
                return
            t0 = time.perf_counter()
            v.s3_verdict, v.s3_raw, _ = self._s3(tool_name, text[:2000])  # upstream third_layer cut
            v.stage_latency_ms["s3"] = (time.perf_counter() - t0) * 1000
            v.blocked, v.stage = v.s3_verdict == UNSAFE, "s3"
            return

        # paper mode
        if p is None:  # Stage II not run: only Stage I (and maybe III) decide
            p_block = False
        else:
            p_block = p > self.tu
        in_band = p is not None and self.band[0] < p < self.band[1]
        if self.s3 is not None and (in_band or self.s2 is None):
            t0 = time.perf_counter()
            v.s3_verdict, v.s3_raw, _ = self._s3(tool_name, text)
            v.stage_latency_ms["s3"] = (time.perf_counter() - t0) * 1000
            if v.s3_verdict == UNSAFE:
                v.blocked, v.stage = True, "s3"
            elif v.s3_verdict == SAFE:
                v.blocked, v.stage = False, "s3"
            else:  # uncertain or error -> neural fallback (Eq. 2)
                v.blocked, v.stage = p_block, "s3_fallback"
            return
        v.blocked, v.stage = p_block, ("s2" if p is not None else "s1")

    # ---- public API ---------------------------------------------------------

    def scan(self, tool_name: str | None, description: str | None) -> Verdict:
        """Scan one tool description (what the proxy calls per tool from tools/list)."""
        return self.scan_many([{"tool_name": tool_name, "description": description}])[0]

    def scan_many(self, tools: list[dict]) -> list[Verdict]:
        """Scan many tools; Stage II runs batched. Each dict needs tool_name and description.

        Per-item latency splits batched Stage II time evenly across the batch.
        """
        verdicts: list[Verdict | None] = [None] * len(tools)
        pending: list[int] = []
        texts = [(t.get("description") or "") for t in tools]

        for i, text in enumerate(texts):
            if not text.strip():
                # Upstream returns "safe" for empty input; nothing to scan.
                verdicts[i] = Verdict(blocked=False, stage="empty")
                continue
            v = Verdict(blocked=False, stage="s1")
            if self.s1 is not None:
                t0 = time.perf_counter()
                v.s1_hits = [asdict(h) for h in self.s1.scan(text)]
                v.stage_latency_ms["s1"] = (time.perf_counter() - t0) * 1000
                if v.s1_hits:
                    v.blocked = True
                    verdicts[i] = v
                    continue
            verdicts[i] = v
            pending.append(i)

        if pending and self.s2 is not None:
            normed = [_norm(texts[i]) for i in pending]
            t0 = time.perf_counter()
            scores = self._s2_scores(normed)
            per = (time.perf_counter() - t0) * 1000 / len(pending)
            for i, s in zip(pending, scores):
                verdicts[i].s2_score = float(s)
                verdicts[i].stage_latency_ms["s2"] = per

        for i in pending:
            text = _norm(texts[i]) if self.s2 is not None else texts[i]
            self._decide_after_s2(verdicts[i], tools[i].get("tool_name"), text)

        for v in verdicts:
            v.latency_ms = sum(v.stage_latency_ms.values())
        return verdicts  # type: ignore[return-value]
