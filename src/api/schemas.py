"""Request/response contracts for POST /retrieve."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.config import settings

Category = Literal["product", "technical", "refund"]


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: Category
    top_k: Optional[int] = Field(default=None, gt=0, le=settings.max_top_k)

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace-only")
        if len(v) > settings.max_query_chars:
            raise ValueError(f"query exceeds max length of {settings.max_query_chars} characters")
        return v


class ChunkResult(BaseModel):
    chunk_id: str
    chunk_text: str
    file_name: str
    page_no: int
    is_table: bool
    section_title: str
    product_name: Optional[str] = None
    ingested_at: Optional[str] = None
    score: float


class RetrieveResponse(BaseModel):
    results: list[ChunkResult]
    result_count: int
    category: Category
    query: str
