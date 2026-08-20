"""Builds the hybrid search call for a given category. Alpha (0 = pure
BM25/keyword, 1 = pure vector) is tuned per collection: product_specs and
refund_specs contain exact identifiers (product IDs, category names) that
keyword matching handles more precisely than vector similarity, while
technical_specs' own source material notes that customers describe
symptoms in their own words rather than document phrasing, so it leans
more heavily on the vector side.

query_properties is restricted to chunk_text and section_title - leaving
it at the client default searches every TEXT property, including
product_name/file_name, which dilutes the BM25 match with irrelevant
token hits.
"""

from weaviate.classes.query import MetadataQuery

from src.api.schemas import Category
from src.config import settings

QUERY_PROPERTIES = ["chunk_text", "section_title"]

_ALPHA_BY_CATEGORY: dict[Category, float] = {
    "product": settings.hybrid_alpha_product,
    "technical": settings.hybrid_alpha_technical,
    "refund": settings.hybrid_alpha_refund,
}

_SCORE_THRESHOLD_BY_CATEGORY: dict[Category, float] = {
    "product": settings.score_threshold_product,
    "technical": settings.score_threshold_technical,
    "refund": settings.score_threshold_refund,
}


def alpha_for(category: Category) -> float:
    return _ALPHA_BY_CATEGORY[category]


def score_threshold_for(category: Category) -> float:
    return _SCORE_THRESHOLD_BY_CATEGORY[category]


async def run_hybrid_search(collection, query: str, vector: list[float], category: Category, candidate_k: int):
    return await collection.query.hybrid(
        query=query,
        vector=vector,
        alpha=alpha_for(category),
        query_properties=QUERY_PROPERTIES,
        limit=candidate_k,
        return_metadata=MetadataQuery(score=True, certainty=True, distance=True),
    )
