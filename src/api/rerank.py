"""Reranking seam. Both implementations satisfy the same Reranker protocol -
return survivors re-ordered by relevance, each paired with the score that
justifies its position - so pipeline.py's call site and response mapping
never change regardless of which implementation is active.

NoopReranker is a passthrough: keeps Weaviate's original hybrid score and
order unchanged.

CrossEncoderReranker reads the query and each chunk's text together (unlike
the hybrid score, which compares independently pre-computed vectors), which
catches cases where hybrid search over- or under-ranks a chunk based on
incidental keyword/embedding overlap rather than actually answering the
query. Uses BAAI/bge-reranker-base by default - same family as the embedding
model, so it's tuned to complement it.
"""

from typing import Protocol

import torch
from sentence_transformers import CrossEncoder
from weaviate.collections.classes.internal import Object


class Reranker(Protocol):
    def rerank(self, query: str, objects: list[Object]) -> list[tuple[Object, float]]: ...


class NoopReranker:
    def rerank(self, query: str, objects: list[Object]) -> list[tuple[Object, float]]:
        return [(obj, obj.metadata.score or 0.0) for obj in objects]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, objects: list[Object]) -> list[tuple[Object, float]]:
        if not objects:
            return []
        pairs = [(query, obj.properties.get("chunk_text") or "") for obj in objects]
        scores = self._model.predict(pairs, activation_fn=torch.nn.Sigmoid())
        ranked = sorted(zip(objects, scores), key=lambda pair: pair[1], reverse=True)
        return [(obj, float(score)) for obj, score in ranked]
