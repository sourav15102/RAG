from __future__ import annotations

from sentence_transformers import SentenceTransformer

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DIMENSION = 768


class NomicEmbedder:
    def __init__(self, batch_size: int = 32):
        print(f"Loading embedding model: {MODEL_NAME}")
        self._model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [DOCUMENT_PREFIX + t for t in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            QUERY_PREFIX + text,
            normalize_embeddings=True,
        )
        return vector.tolist()
