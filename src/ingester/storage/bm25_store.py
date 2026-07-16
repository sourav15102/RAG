from elasticsearch import Elasticsearch

from code_chunker.ast_chunker import CodeChunk
from storage.es_store import create_index, index_chunks, search_bm25 as _search_bm25


class BM25Store:
    def __init__(
        self,
        es: Elasticsearch,
        index: str = "code_chunks",
        tenant_id: str = "default",
        repo: str = "",
    ):
        self._es = es
        self.index = index
        self.tenant_id = tenant_id
        self.repo = repo

    def ensure_index(self) -> None:
        if not self._es.indices.exists(index=self.index):
            create_index(self._es, index=self.index)

    def index(self, chunks: list[CodeChunk]) -> None:
        index_chunks(self._es, chunks, index=self.index, tenant_id=self.tenant_id, repo=self.repo)

    def search(self, query: str, top_k: int = 10) -> list[str]:
        return _search_bm25(
            self._es,
            query,
            top_k=top_k,
            index=self.index,
            tenant_id=self.tenant_id,
        )
