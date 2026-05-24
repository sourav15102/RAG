from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.indexing.embedder import DIMENSION
from src.models import ChildChunk

COLLECTION = "child_chunks"
UPSERT_BATCH = 64


def _point_id(child_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, child_id))


class QdrantVectorStore:
    def __init__(self, url: str = "http://localhost:6333"):
        self._client = QdrantClient(url=url, check_compatibility=False)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION not in existing:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
            )

    def upsert(self, children: list[ChildChunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=_point_id(child.id),
                vector=vectors[i],
                payload={
                    "child_id": child.id,
                    "parent_id": child.parent_id,
                    "doc_id": child.doc_id,
                    "content": child.content,
                    **child.metadata,
                },
            )
            for i, child in enumerate(children)
        ]

        for start in range(0, len(points), UPSERT_BATCH):
            batch = points[start : start + UPSERT_BATCH]
            self._client.upsert(collection_name=COLLECTION, points=batch)
        print(f"Upserted {len(points)} points to Qdrant collection '{COLLECTION}'.")

    def search(self, query_vector: list[float], top_k: int = 20) -> list[dict]:
        results = self._client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {**hit.payload, "score": hit.score}
            for hit in results
        ]
