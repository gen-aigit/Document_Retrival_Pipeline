"""FastAPI app: creates the embedder, the reranker, and the long-lived async
Weaviate client once at startup (all expensive to construct - model loads
and a gRPC connection - and must never be re-created per request), and
verifies all three configured collections exist before accepting traffic
(fail fast rather than deferring to the first request).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.api.errors import install_exception_handlers
from src.api.rerank import CrossEncoderReranker
from src.api.routes import router
from src.config import settings
from src.embeddings.embedder import Embedder
from src.utils.logger import get_logger
from src.weaviate_store.client import weaviate_async_client

logger = get_logger(__name__)

_COLLECTIONS_BY_CATEGORY = {
    "product": settings.product_specs_collection,
    "technical": settings.technical_specs_collection,
    "refund": settings.refund_specs_collection,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model '%s'...", settings.embedding_model_name)
    app.state.embedder = Embedder()

    logger.info("Loading reranker model '%s'...", settings.reranker_model_name)
    app.state.reranker = CrossEncoderReranker(settings.reranker_model_name)

    async with weaviate_async_client() as client:
        for category, name in _COLLECTIONS_BY_CATEGORY.items():
            if not await client.collections.exists(name):
                raise RuntimeError(
                    f"Collection '{name}' (category={category!r}) does not exist in "
                    "Weaviate. Run the ingestion pipeline before starting this service."
                )
        app.state.collections = {
            category: client.collections.get(name) for category, name in _COLLECTIONS_BY_CATEGORY.items()
        }
        logger.info("All configured collections verified. Retrieval API ready.")
        yield

    logger.info("Shutting down retrieval API.")


app = FastAPI(title="Document Retrieval Service", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

install_exception_handlers(app)
app.include_router(router)
