from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ingester.step import PipelineContext, Step
from ingester.storage.bm25_store import BM25Store
from ingester.storage.qdrant_store import QdrantStore
from embedder.code_embedder import CodeEmbedder
from search.rrf import rrf_fuse
from search.fetcher import fetch_chunks
from search.generator import generate


class SearchStep(Step):
    name = "search"

    def __init__(
        self,
        bm25_store: BM25Store,
        qdrant_store: QdrantStore,
        embedder: CodeEmbedder,
        rrf_k: int = 60,
        top_k: int = 10,
        llm_api_key: str | None = None,
    ):
        self._bm25 = bm25_store
        self._qdrant = qdrant_store
        self._embedder = embedder
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.llm_api_key = llm_api_key

    def execute(self, ctx: PipelineContext, data: str) -> Any:
        query = data

        with ThreadPoolExecutor() as ex:
            futures = {
                ex.submit(self._bm25.search, query, self.top_k): "bm25",
                ex.submit(self._vector_search, query): "vector",
            }
            results: dict[str, list[str]] = {}
            for future in as_completed(futures):
                label = futures[future]
                results[label] = future.result()

        fused = rrf_fuse(list(results.values()), k=self.rrf_k)[: self.top_k]

        chunks = fetch_chunks(self._qdrant, fused)

        answer = generate(
            query=query,
            chunks=chunks,
            api_key=self.llm_api_key,
        )

        return {
            "answer": answer,
            "chunk_ids": fused,
            "chunks": chunks,
        }

    def _vector_search(self, query: str) -> list[str]:
        query_vector = self._embedder.embed_query(query)
        results = self._qdrant.search(query_vector, top_k=self.top_k)
        return [cid for cid, _ in results]
