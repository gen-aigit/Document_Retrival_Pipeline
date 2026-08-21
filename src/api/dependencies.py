"""FastAPI Depends() providers: read the singletons created once in the
lifespan handler off app.state, plus the static API-key auth dependency."""

from typing import Optional

from fastapi import Header, HTTPException, Request, status

from src.api.rerank import Reranker
from src.config import settings
from src.embeddings.embedder import Embedder


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_reranker(request: Request) -> Reranker:
    return request.app.state.reranker


def get_collections(request: Request) -> dict:
    return request.app.state.collections


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
