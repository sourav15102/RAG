from __future__ import annotations

from src.chunking.parent_child import ChunkingConfig, ParentChildChunker
from src.indexing.base import BaseIndexer
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import NomicEmbedder
from src.indexing.parent_store import ParentStore
from src.indexing.qdrant_store import QdrantVectorStore
from src.models import ChildChunk, ParentChunk, RawDocument


class ChunkingIndexer(BaseIndexer):
    """
    Indexes documents via parent-child chunking → Qdrant (semantic) + BM25 (keyword).
    Retrieval is handled separately by HybridRetriever.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        chunking_config: ChunkingConfig | None = None,
    ):
        self._chunker = ParentChildChunker(chunking_config or ChunkingConfig(
            parent_tokens=1500,
            child_tokens=300,
            child_overlap_tokens=50,
        ))
        self._embedder = NomicEmbedder()
        self._vector_store = QdrantVectorStore(url=qdrant_url)
        self._bm25 = BM25Store()
        self._parents = ParentStore()
        self._queued_children: list[ChildChunk] = []

    def add_document(self, doc: RawDocument) -> None:
        parents, children = self._chunker.chunk(doc)
        print(f"  → {len(parents)} parents, {len(children)} children")
        self._parents.add(parents)
        self._queued_children.extend(children)

    def finalize(self) -> None:
        if not self._queued_children:
            print("Nothing to index.")
            return

        print(f"Embedding {len(self._queued_children)} child chunks...")
        vectors = self._embedder.embed_documents([c.content for c in self._queued_children])
        self._vector_store.upsert(self._queued_children, vectors)

        self._bm25.build(self._queued_children)
        self._parents.save()
        self._bm25.save()

        print(
            f"Done. Indexed {len(self._queued_children)} child chunks "
            f"from {len(self._parents._store)} parent chunks."
        )
