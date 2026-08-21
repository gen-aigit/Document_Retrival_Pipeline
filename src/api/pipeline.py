"""Orchestrates one retrieval request: embed -> hybrid search -> threshold
filter -> rerank -> top_k truncation -> response mapping. Logs per-stage
latency and returned scores, since that's also what future Recall@k/MRR
drift tracking would depend on.

The threshold filter runs on Weaviate's hybrid score (search_strategy.py's
SCORE_THRESHOLD_* config), before reranking - that's the calibrated cutoff
for "is this worth reranking at all". The score returned to the caller,
however, is whatever the active reranker assigns (for NoopReranker, that's
just the same hybrid score passed through) - so the returned score always
matches the returned order.
"""

import asyncio
import time

from src.api.rerank import Reranker
from src.api.schemas import Category, ChunkResult, RetrieveResponse
from src.api.search_strategy import run_hybrid_search, score_threshold_for
from src.config import settings
from src.embeddings.embedder import Embedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_chunk_result(obj, score: float) -> ChunkResult:
    props = obj.properties
    ingested_at = props.get("ingested_at")
    return ChunkResult(
        chunk_id=props.get("chunk_id"),
        chunk_text=props.get("chunk_text"),
        file_name=props.get("file_name"),
        page_no=props.get("page_no"),
        is_table=props.get("is_table"),
        section_title=props.get("section_title"),
        product_name=props.get("product_name"),
        ingested_at=str(ingested_at) if ingested_at is not None else None,
        score=score,
    )


async def retrieve(
    *,
    embedder: Embedder,
    reranker: Reranker,
    collection,
    query: str,
    category: Category,
    top_k: int,
) -> RetrieveResponse:
    loop = asyncio.get_running_loop()

    embed_start = time.perf_counter()
    vector = await loop.run_in_executor(None, embedder.embed_query, query)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    candidate_k = top_k * settings.candidate_k_multiplier
    search_start = time.perf_counter()
    results = await run_hybrid_search(collection, query, vector, category, candidate_k)
    search_ms = (time.perf_counter() - search_start) * 1000

    rerank_start = time.perf_counter()
    reranked = await loop.run_in_executor(None, reranker.rerank, query, results.objects)
    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    threshold = score_threshold_for(category)
    survivors = [(obj, score) for obj, score in reranked if score >= threshold]

    truncated = survivors[:top_k]

    logger.info(
        "retrieve category=%s query_chars=%d candidates=%d survivors=%d returned=%d "
        "embed_ms=%.1f search_ms=%.1f rerank_ms=%.1f scores=%s",
        category,
        len(query),
        len(results.objects),
        len(survivors),
        len(truncated),
        embed_ms,
        search_ms,
        rerank_ms,
        [round(score, 4) for _, score in truncated],
    )

    chunk_results = [_to_chunk_result(obj, score) for obj, score in truncated]
    return RetrieveResponse(
        results=chunk_results,
        result_count=len(chunk_results),
        category=category,
        query=query,
    )
