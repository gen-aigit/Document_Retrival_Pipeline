"""Reranking seam. v1 ships a no-op so pipeline.py's call site never has to
change when a real reranker (e.g. a sentence-transformers CrossEncoder) is
added later - only the implementation swapped in here, once measurement
shows a specific collection needs it.
"""

from typing import Protocol

from weaviate.collections.classes.internal import Object


class Reranker(Protocol):
    def rerank(self, query: str, objects: list[Object]) -> list[Object]: ...


class NoopReranker:
    def rerank(self, query: str, objects: list[Object]) -> list[Object]:
        return objects
