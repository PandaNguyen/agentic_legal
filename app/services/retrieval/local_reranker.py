from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.schemas.search import SearchHit


class LocalReranker:
    def __init__(
        self,
        settings: Settings,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.settings = settings
        self._tokenizer = tokenizer
        self._model = model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits or top_k <= 0:
            return []
        if len(hits) <= 1:
            return hits[:top_k]

        try:
            scores = self._score_pairs(query, hits)
        except Exception:
            return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

        ranked: list[SearchHit] = []
        for index, score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]:
            hit = hits[index].model_copy(deep=True)
            hit.score = float(score)
            hit.metadata["rerank_score"] = float(score)
            hit.metadata["retrieval_stage"] = "local_rerank"
            ranked.append(hit)
        return ranked

    def _score_pairs(self, query: str, hits: list[SearchHit]) -> list[float]:
        import torch

        tokenizer = self._get_tokenizer()
        model = self._get_model()
        device = self.settings.rerank_device
        if device:
            model = model.to(device)

        scores: list[float] = []
        batch_size = max(1, int(self.settings.rerank_batch_size))
        for start in range(0, len(hits), batch_size):
            batch_hits = hits[start : start + batch_size]
            pairs = [[query, hit.context_text or hit.text] for hit in batch_hits]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=int(self.settings.rerank_max_length),
            )
            if device:
                inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(logits.detach().cpu().tolist())
        return scores

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.settings.rerank_model_name,
            cache_folder=self.settings.rerank_cache_folder,
        )
        return self._tokenizer

    def _get_model(self):
        if self._model is not None:
            return self._model

        from transformers import AutoModelForSequenceClassification

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.settings.rerank_model_name,
            cache_folder=self.settings.rerank_cache_folder,
        )
        self._model.eval()
        return self._model
