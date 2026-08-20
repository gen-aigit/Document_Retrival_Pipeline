"""Maps internal exceptions to HTTP responses. Short/no retries here -
unlike ingestion's multi-attempt backoff (appropriate for background batch
writes), this is a synchronous read path with a caller-side timeout
expectation: fail fast and let the caller retry.
"""

import asyncio

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.utils.logger import get_logger

logger = get_logger(__name__)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(asyncio.TimeoutError)
    async def _timeout_handler(request: Request, exc: asyncio.TimeoutError) -> JSONResponse:
        logger.warning("Request timed out: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"detail": "Retrieval request timed out"},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
