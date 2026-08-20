"""Query embedding via the same local Hugging Face model used at ingestion
time (BAAI/bge-base-en-v1.5, 768-dim by default). Query and document
vectors must come from the same model, so EMBEDDING_MODEL_NAME here must
match the ingestion pipeline's setting.
"""

from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# BGE models are trained for asymmetric retrieval: queries should carry this
# instruction prefix, stored documents should not.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model_name
        logger.info("Loading embedding model '%s' (downloads on first run, then cached)", self.model_name)
        try:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model '{self.model_name}'. "
                "Check that sentence-transformers/torch installed correctly "
                "for this Python version and that there's network access to "
                "download the model on first run."
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(QUERY_INSTRUCTION + text)
