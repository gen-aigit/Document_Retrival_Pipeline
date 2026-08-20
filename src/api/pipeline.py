"""Orchestrates one retrieval request: embed -> hybrid search -> threshold
filter -> rerank seam (no-op in v1) -> top_k truncation -> response
mapping. Logs per-stage latency and returned scores, since that's also
what future Recall@k/MRR drift tracking would depend on.
"""

import asyncio
import time

from src.api.rerank import NoopReranker, Reranker
from src.api.schemas import Category, ChunkResult, RetrieveResponse
from src.api.search_strategy import run_hybrid_search, score_threshold_for
from src.config import settings
from src.embeddings.embedder import Embedder
from src.utils.logger import get_logger

logger = get_logger(__name__)

_reranker: Reranker = NoopReranker()


def _to_chunk_result(obj) -> ChunkResult:
    props = obj.properties
    ingested_at = props.get("ingested_at")
    score = obj.metadata.score
    return ChunkResult(
        chunk_id=props.get("chunk_id"),
        chunk_text=props.get("chunk_text"),
        file_name=props.get("file_name"),
        page_no=props.get("page_no"),
        is_table=props.get("is_table"),
        section_title=props.get("section_title"),
        product_name=props.get("product_name"),
        ingested_at=str(ingested_at) if ingested_at is not None else None,
        score=score if score is not None else 0.0,
    )


async def retrieve(
    *,
    embedder: Embedder,
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

    threshold = score_threshold_for(category)
    survivors = [obj for obj in results.objects if (obj.metadata.score or 0.0) >= threshold]

    reranked = _reranker.rerank(query, survivors)
    truncated = reranked[:top_k]

    logger.info(
        "retrieve category=%s query_chars=%d candidates=%d survivors=%d returned=%d "
        "embed_ms=%.1f search_ms=%.1f scores=%s",
        category,
        len(query),
        len(results.objects),
        len(survivors),
        len(truncated),
        embed_ms,
        search_ms,
        [round(obj.metadata.score or 0.0, 4) for obj in truncated],
    )

    chunk_results = [_to_chunk_result(obj) for obj in truncated]
    return RetrieveResponse(
        results=chunk_results,
        result_count=len(chunk_results),
        category=category,
        query=query,
    )
