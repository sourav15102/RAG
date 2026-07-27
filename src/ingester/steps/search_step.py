from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ingester.step import PipelineContext, Step
from ingester.storage.bm25_store import BM25Store
from ingester.storage.qdrant_store import QdrantStore
from embedder.code_embedder import CodeEmbedder
from search.rrf import rrf_fuse
from search.fetcher import fetch_chunks
from search.generator import generate
from search.reranker import CrossEncoderReranker
from search.query_rewriter import rewrite_query
from search.hyde import generate_hypothetical_docstring


class SearchStep(Step):
    name = "search"

    def __init__(
        self,
        bm25_store: BM25Store,
        qdrant_store: QdrantStore,
        embedder: CodeEmbedder,
        rrf_k: int = 60,
        top_k: int = 20,
        rerank_top_n: int = 5,
        llm_api_key: str | None = None,
        reranker: CrossEncoderReranker | None = None,
        rewrite: bool = True,
        use_hyde: bool = True,
    ):
        self._bm25 = bm25_store
        self._qdrant = qdrant_store
        self._embedder = embedder
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.llm_api_key = llm_api_key
        self._reranker = reranker or CrossEncoderReranker()
        self.rewrite = rewrite
        self.use_hyde = use_hyde

    def execute(self, ctx: PipelineContext, data: str) -> Any:
        query = data

        if self.rewrite and self.llm_api_key:
            query = rewrite_query(query, api_key=self.llm_api_key)

        if self.use_hyde and self.llm_api_key:
            hyde_query = generate_hypothetical_docstring(query, api_key=self.llm_api_key)
            if hyde_query:
                query = hyde_query

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

        chunks = self._reranker.rerank(query, chunks, top_n=self.rerank_top_n)

        result = generate(
            query=data,
            chunks=chunks,
            api_key=self.llm_api_key,
        )

        return {
            "answer": result.answer,
            "claims": result.claims,
            "unanswered_parts": result.unanswered_parts,
            "chunk_ids": [f"{c.file_path}::{c.name}" for c in chunks],
            "chunks": chunks,
        }

    def _vector_search(self, query: str) -> list[str]:
        query_vector = self._embedder.embed_query(query)
        results = self._qdrant.search(query_vector, top_k=self.top_k)
        return [cid for cid, _ in results]
