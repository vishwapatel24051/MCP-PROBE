"""Stage II: the released MCP-Guard neural classifier (HF GenTelLab/MCP-Guard).

The checkpoint is an XLM-RoBERTa sequence classifier (config.json: 12 layers,
hidden 768, `_name_or_path` .../multilingual-e5-base...), i.e. E5-*base*; the
paper (arXiv 2508.10991, Sec. 5.2) says Multilingual-E5-large. Weights ship as
`model.onnx` plus `finetuned_from_onnx_v3.pth`, a state_dict for the
onnx2pytorch conversion of that ONNX graph.

Two loaders:
- loader="onnx2pytorch": upstream's exact path (ConvertModel + load_state_dict).
- loader="hf": the same weights mapped onto transformers'
  XLMRobertaForSequenceClassification, which batches and runs on GPU cleanly.
  `verify_loaders()` checks both give the same logits before we rely on it.
- loader="finetuned": a directory written by detectors/mcp_guard/train.py
  (our MCPTox fine-tunes); model and tokenizer both come from `path`.

Two tokenizers:
- tokenizer="bert" (default): bert-base-uncased, which upstream's
  learnableshield_local_detector.py loads. Although the weights are XLM-R shaped,
  this is the tokenizer the classifier was evidently trained with: on upstream's
  test_data/dev.csv it reproduces the paper's Stage II numbers exactly
  (acc 96.0 / P 96.7 / R 93.5 / F1 95.1, Table 4a). See reproduce_upstream_dev.py.
- tokenizer="checkpoint_xlmr": the XLM-R sentencepiece tokenizer shipped next to
  the checkpoint. Gives F1 34.5 on the same file, so it is not what was used.

Score = softmax(logits)[1] = P(malicious), max_length 128 as upstream.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import torch

log = logging.getLogger(__name__)

HF_REPO = "GenTelLab/MCP-Guard"
HF_REVISION = "861a928763c66f7cbb7e77caa88000fecb14d9d8"
MAX_LENGTH = 128
BERT_TOKENIZER = ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594")


def checkpoint_dir(revision: str = HF_REVISION) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=HF_REPO, revision=revision)


def pick_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hf_state_dict_from_pth(pth: dict) -> dict:
    """Map the onnx2pytorch state_dict onto XLMRobertaForSequenceClassification names.

    Linear layers come from the fused MatMul_/Gemm_ modules (the ones the converted
    graph runs); embeddings and LayerNorms from the `_initializer_*` tensors.
    """
    out = {}
    for k, v in pth.items():
        m = re.fullmatch(r"MatMul_/roberta/encoder/layer\.(\d+)/(.+)/Add_output_0\.(weight|bias)", k)
        if m:
            out[f"roberta.encoder.layer.{m[1]}.{m[2].replace('/', '.')}.{m[3]}"] = v
            continue
        m = re.fullmatch(r"Gemm_/classifier/dense/Gemm_output_0\.(weight|bias)", k)
        if m:
            out[f"classifier.dense.{m[1]}"] = v
            continue
        m = re.fullmatch(r"Gemm_logits\.(weight|bias)", k)
        if m:
            out[f"classifier.out_proj.{m[1]}"] = v
            continue
        m = re.fullmatch(r"_initializer_roberta_embeddings_(.+)_(weight|bias)", k)
        if m:
            out[f"roberta.embeddings.{m[1]}.{m[2]}"] = v
            continue
        m = re.fullmatch(r"_initializer_roberta_encoder_layer[._](\d+)_(attention_output|output)_LayerNorm_(weight|bias)", k)
        if m:
            out[f"roberta.encoder.layer.{m[1]}.{m[2].replace('_', '.')}.LayerNorm.{m[3]}"] = v
    return out


class NeuralStage:
    def __init__(
        self,
        loader: str = "hf",
        tokenizer: str = "bert",
        device: str = "auto",
        batch_size: int = 32,
        revision: str = HF_REVISION,
        path: str | None = None,
    ):
        if loader not in ("hf", "onnx2pytorch", "finetuned"):
            raise ValueError("loader must be 'hf', 'onnx2pytorch' or 'finetuned'")
        if tokenizer not in ("bert", "checkpoint_xlmr"):
            raise ValueError("tokenizer must be 'bert' or 'checkpoint_xlmr'")
        self.loader = loader
        self.tokenizer_name = tokenizer
        self.revision = revision
        self.batch_size = batch_size if loader != "onnx2pytorch" else 1  # converted graph: one at a time
        self.device = pick_device(device)

        from transformers import AutoTokenizer
        if loader == "finetuned":
            if not path:
                raise ValueError("loader='finetuned' needs stage2.path")
            from transformers import AutoModelForSequenceClassification
            self.path = str(Path(path).resolve())
            meta = Path(self.path) / "train_meta.json"
            self._run_id = hashlib.sha256(meta.read_bytes()).hexdigest()[:12] if meta.exists() else "nometa"
            self.tokenizer_name = "saved"
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path)
            self.model.to(self.device).eval()
            return

        self.ckpt = checkpoint_dir(revision)
        if tokenizer == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER[0], revision=BERT_TOKENIZER[1])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.model = self._load_hf() if loader == "hf" else self._load_onnx2pytorch()
        self.model.to(self.device).eval()

    @property
    def cache_tag(self) -> str:
        if self.loader == "finetuned":
            return f"s2|finetuned|{self.path}|{self._run_id}|len={MAX_LENGTH}"
        # Both loaders compute the same function once verified; the tokenizer changes the inputs.
        return f"s2|{HF_REPO}@{self.revision}|tok={self.tokenizer_name}|len={MAX_LENGTH}"

    def _load_pth(self) -> dict:
        return torch.load(f"{self.ckpt}/finetuned_from_onnx_v3.pth", map_location="cpu", weights_only=False)

    def _load_hf(self):
        from transformers import AutoConfig, XLMRobertaForSequenceClassification
        model = XLMRobertaForSequenceClassification(AutoConfig.from_pretrained(self.ckpt))
        mapped = hf_state_dict_from_pth(self._load_pth())
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        # position_ids is a non-persistent buffer in recent transformers; anything else is a real gap.
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mapping incomplete: missing={missing} unexpected={unexpected}")
        return model

    def _load_onnx2pytorch(self):
        import onnx
        from onnx2pytorch import ConvertModel
        model = ConvertModel(onnx.load(f"{self.ckpt}/model.onnx"))
        pth = self._load_pth()
        # Like upstream: strict=True fails (the converted model exposes each fused layer
        # under duplicate names, reported as "unexpected"), so it falls back to strict=False.
        # Check every checkpoint tensor really landed rather than trusting the fallback.
        try:
            model.load_state_dict(pth, strict=True)
        except RuntimeError:
            model.load_state_dict(pth, strict=False)
        now = model.state_dict()
        not_applied = [k for k, v in pth.items() if not torch.equal(now[k], v)]
        if not_applied:
            raise RuntimeError(f"{len(not_applied)} checkpoint tensors not applied, e.g. {not_applied[:3]}")
        return model

    def _encode(self, texts: list[str]) -> dict:
        enc = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)
        keep = ("input_ids", "attention_mask")  # the ONNX graph takes only these two
        return {k: v.to(self.device) for k, v in enc.items() if k in keep}

    @torch.no_grad()
    def logits(self, texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self._encode(texts[i : i + self.batch_size])
            res = self.model(**enc)
            out.append((res if isinstance(res, torch.Tensor) else res.logits).float().cpu())
        return torch.cat(out)

    def score_many(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return torch.softmax(self.logits(texts), dim=-1)[:, 1].tolist()


def verify_loaders(texts: list[str], tokenizer: str = "bert", atol: float = 1e-3) -> float:
    """Max |logit difference| between the hf and onnx2pytorch loaders on `texts` (CPU)."""
    a = NeuralStage(loader="hf", tokenizer=tokenizer, device="cpu").logits(texts)
    b = NeuralStage(loader="onnx2pytorch", tokenizer=tokenizer, device="cpu").logits(texts)
    diff = float((a - b).abs().max())
    if diff > atol:
        raise AssertionError(f"loaders disagree: max |dlogit| = {diff:.4g}")
    return diff
