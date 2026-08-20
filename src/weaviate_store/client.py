"""Async Weaviate connection handling (local Docker instance). This client
is long-lived - opened once in the API's lifespan and closed once at
shutdown, not per-request - so it uses the async gRPC client rather than
the sync one, letting concurrent requests share a single connection
instead of each blocking a thread-pool slot.
"""

from contextlib import asynccontextmanager

import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig, Timeout

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def weaviate_async_client():
    """Connects to the local Weaviate instance and fails fast (with a clear
    message) if it isn't up, instead of letting later calls fail with an
    opaque connection error."""

    auth = AuthApiKey(settings.weaviate_api_key) if settings.weaviate_api_key else None

    client = weaviate.use_async_with_local(
        host=settings.weaviate_http_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
        auth_credentials=auth,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=settings.request_timeout_seconds, insert=120)
        ),
    )

    try:
        await client.connect()
    except Exception as exc:
        raise ConnectionError(
            f"Could not connect to Weaviate at "
            f"{settings.weaviate_http_host}:{settings.weaviate_http_port} "
            "(gRPC port "
            f"{settings.weaviate_grpc_port}). Is the local Docker instance running?"
        ) from exc

    try:
        if not await client.is_ready():
            raise ConnectionError(
                f"Weaviate at {settings.weaviate_http_host}:{settings.weaviate_http_port} "
                "responded but is not ready. Is the local Docker instance still starting up?"
            )
        logger.info(
            "Connected to Weaviate at %s:%s (gRPC %s)",
            settings.weaviate_http_host,
            settings.weaviate_http_port,
            settings.weaviate_grpc_port,
        )
        yield client
    finally:
        await client.close()
