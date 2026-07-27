from datetime import datetime, timezone

from elasticsearch import Elasticsearch

from code_chunker.ast_chunker import CodeChunk

INDEX_NAME = "code_chunks"
ANALYZER_NAME = "code_analyzer"

def create_index(es: Elasticsearch, index: str = INDEX_NAME) -> None:
    es.indices.create(
        index=index,
        body={
            "settings": {
                "analysis": {
                    "analyzer": {
                        ANALYZER_NAME: {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"],
                        }
                    },
                },
                "similarity": {
                    "bm25_custom": {
                        "type": "BM25",
                        "k1": 1.5,
                        "b": 0.75,
                    }
                },
            },
            "mappings": {
                "properties": {
                    "bm25_text": {
                        "type": "text",
                        "analyzer": ANALYZER_NAME,
                        "similarity": "bm25_custom",
                    },
                    "chunk_id": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "name": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "source_type": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "last_updated": {"type": "date"},
                }
            },
        },
    )


def index_chunks(
    es: Elasticsearch,
    chunks: list[CodeChunk],
    index: str = INDEX_NAME,
    tenant_id: str = "default",
    repo: str = "",
) -> None:
    prefix = f"{repo}/" if repo else ""
    for chunk in chunks:
        chunk_id = f"{prefix}{chunk.file_path}::{chunk.name}"
        doc = {
            "bm25_text": chunk.content,
            "chunk_id": chunk_id,
            "file_path": chunk.file_path,
            "name": chunk.name,
            "chunk_type": chunk.chunk_type,
            "source_type": "code",
            "tenant_id": tenant_id,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        es.index(index=index, document=doc)


def search_bm25(
    es: Elasticsearch,
    query: str,
    top_k: int = 10,
    index: str = INDEX_NAME,
    tenant_id: str | None = None,
) -> list[str]:
    must = [{"match": {"bm25_text": query}}]
    if tenant_id:
        must.append({"term": {"tenant_id": tenant_id}})

    resp = es.search(
        index=index,
        query={"bool": {"must": must}},
        size=top_k,
        _source=["chunk_id"],
    )
    return [hit["_source"]["chunk_id"] for hit in resp["hits"]["hits"]]
