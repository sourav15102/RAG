from embedder.code_embedder import CodeEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .vector_store import VectorStore


def build_chunk_id(chunk, repo: str = "") -> str:
    prefix = f"{repo}/" if repo else ""
    return f"{prefix}{chunk.file_path}::{chunk.name}"


class QdrantStore(VectorStore):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "code_chunks",
        vector_size: int = 1024,
        repo: str = "",
    ):
        self._client = QdrantClient(host=host, port=port)
        self.collection = collection
        self.vector_size = vector_size
        self.repo = repo
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        if not any(c.name == self.collection for c in collections):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "code": models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                },
            )

    def upsert(self, records: list[CodeEmbedding]) -> None:
        points = []
        for rec in records:
            chunk = rec.chunk
            chunk_id = build_chunk_id(chunk, repo=self.repo)
            point_id = hash(chunk_id)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"code": rec.embedding},
                    payload={
                        "chunk_id": chunk_id,
                        "name": chunk.name,
                        "chunk_type": chunk.chunk_type,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "parent_class": chunk.parent_class,
                        "docstring": chunk.docstring,
                        "decorators": chunk.decorators,
                        "calls": chunk.calls,
                        "line_count": chunk.line_count,
                        "is_fallback_split": chunk.is_fallback_split,
                        "content": chunk.content,
                    },
                )
            )
        self._client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        vector_name: str = "code",
    ) -> list[tuple[str, float]]:
        results = self._client.search(
            collection_name=self.collection,
            query_vector=models.NamedVector(
                name=vector_name,
                vector=query_vector,
            ),
            limit=top_k,
        )
        return [(r.payload["chunk_id"], r.score) for r in results]
