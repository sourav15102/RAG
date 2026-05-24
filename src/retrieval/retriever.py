from __future__ import annotations

from dataclasses import dataclass

from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import NomicEmbedder
from src.indexing.parent_store import ParentStore
from src.indexing.qdrant_store import QdrantVectorStore
from src.models import ParentChunk
from src.retrieval.hyde import HyDEGenerator

RRF_K = 60  # standard constant — dampens the impact of high ranks


@dataclass
class RetrievalResult:
    child_id: str
    child_content: str
    parent: ParentChunk
    rrf_score: float
    hypothetical_doc: str | None = None  # set when use_hyde=True
    rerank_score: float | None = None    # set after cross-encoder re-ranking


def _rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked lists of IDs."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    """
    Combines semantic (Qdrant) and keyword (BM25) search via Reciprocal Rank Fusion.
    Returns RetrievalResult objects containing child content (for re-ranking)
    and the parent chunk (for LLM context).
    """

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self._embedder = NomicEmbedder()
        self._hyde = HyDEGenerator()
        self._vector_store = QdrantVectorStore(url=qdrant_url)
        self._bm25 = BM25Store()
        self._bm25.load()
        self._parents = ParentStore()
        self._parents.load()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        final_k: int = 10,
        use_hyde: bool = False,
    ) -> list[RetrievalResult]:
        # 1. Semantic search — HyDE replaces query vector with hypothetical doc embedding
        if use_hyde:
            hypothetical_doc = self._hyde.generate(query)
            # embed as a document so it lands in the same space as indexed chunks
            query_vector = self._embedder.embed_documents([hypothetical_doc])[0]
        else:
            hypothetical_doc = None
            query_vector = self._embedder.embed_query(query)
        semantic_hits = self._vector_store.search(query_vector, top_k=top_k)

        # 2. BM25 search
        bm25_hits = self._bm25.search(query, top_k=top_k)

        # 3. Build ranked lists for RRF
        semantic_ranking = [h["child_id"] for h in semantic_hits]
        bm25_ranking = [h["child_id"] for h in bm25_hits]
        rrf_scores = _rrf([semantic_ranking, bm25_ranking])

        # 4. Index child content from both result sets (semantic payload is authoritative)
        child_map: dict[str, dict] = {}
        for hit in bm25_hits + semantic_hits:  # semantic overwrites BM25 if duplicate
            child_map[hit["child_id"]] = hit

        # 5. Sort by RRF score and take top final_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:final_k]

        # 6. Resolve parent chunks, preserving RRF order
        results: list[RetrievalResult] = []
        for child_id, score in ranked:
            child = child_map.get(child_id)
            if child is None:
                continue
            parent = self._parents.get(child["parent_id"])
            if parent is None:
                continue
            results.append(RetrievalResult(
                child_id=child_id,
                child_content=child["content"],
                parent=parent,
                rrf_score=score,
                hypothetical_doc=hypothetical_doc,
            ))

        return results
