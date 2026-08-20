"""POST /retrieve - the only endpoint this service exposes. No answer
generation happens here; this is retrieval only, for an agentic tool to
consume as context.
"""

import asyncio

from fastapi import APIRouter, Depends

from src.api.dependencies import get_collections, get_embedder, require_api_key
from src.api.pipeline import retrieve
from src.api.schemas import RetrieveRequest, RetrieveResponse
from src.config import settings
from src.embeddings.embedder import Embedder

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
async def retrieve_endpoint(
    body: RetrieveRequest,
    embedder: Embedder = Depends(get_embedder),
    collections: dict = Depends(get_collections),
) -> RetrieveResponse:
    top_k = body.top_k or settings.default_top_k
    collection = collections[body.category]

    return await asyncio.wait_for(
        retrieve(
            embedder=embedder,
            collection=collection,
            query=body.query,
            category=body.category,
            top_k=top_k,
        ),
        timeout=settings.request_timeout_seconds,
    )
