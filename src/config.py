"""Central settings, loaded once from .env."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    weaviate_http_host: str
    weaviate_http_port: int
    weaviate_grpc_host: str
    weaviate_grpc_port: int
    weaviate_api_key: Optional[str]

    embedding_model_name: str

    product_specs_collection: str
    technical_specs_collection: str
    refund_specs_collection: str

    api_host: str
    api_port: int
    api_key: Optional[str]
    request_timeout_seconds: float
    max_query_chars: int
    rate_limit_per_minute: int

    default_top_k: int
    max_top_k: int
    candidate_k_multiplier: int

    hybrid_alpha_product: float
    hybrid_alpha_technical: float
    hybrid_alpha_refund: float

    score_threshold_product: float
    score_threshold_technical: float
    score_threshold_refund: float

    log_file_path: Path
    log_level: str


def _load_settings() -> Settings:
    return Settings(
        weaviate_http_host=os.getenv("WEAVIATE_HTTP_HOST", "localhost"),
        weaviate_http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
        weaviate_grpc_host=os.getenv("WEAVIATE_GRPC_HOST", "localhost"),
        weaviate_grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY") or None,
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5"),
        product_specs_collection=os.getenv("PRODUCT_SPECS_COLLECTION", "product_specs"),
        technical_specs_collection=os.getenv("TECHNICAL_SPECS_COLLECTION", "technical_specs"),
        refund_specs_collection=os.getenv("REFUND_SPECS_COLLECTION", "refund_specs"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        api_key=os.getenv("API_KEY") or None,
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10")),
        max_query_chars=int(os.getenv("MAX_QUERY_CHARS", "1000")),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        max_top_k=int(os.getenv("MAX_TOP_K", "20")),
        candidate_k_multiplier=int(os.getenv("CANDIDATE_K_MULTIPLIER", "4")),
        hybrid_alpha_product=float(os.getenv("HYBRID_ALPHA_PRODUCT", "0.4")),
        hybrid_alpha_technical=float(os.getenv("HYBRID_ALPHA_TECHNICAL", "0.7")),
        hybrid_alpha_refund=float(os.getenv("HYBRID_ALPHA_REFUND", "0.4")),
        score_threshold_product=float(os.getenv("SCORE_THRESHOLD_PRODUCT", "0.60")),
        score_threshold_technical=float(os.getenv("SCORE_THRESHOLD_TECHNICAL", "0.60")),
        score_threshold_refund=float(os.getenv("SCORE_THRESHOLD_REFUND", "0.60")),
        log_file_path=PROJECT_ROOT / os.getenv("LOG_FILE_PATH", "logs/retrieval_api.log"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


settings = _load_settings()
