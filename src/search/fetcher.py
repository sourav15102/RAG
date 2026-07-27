from code_chunker.ast_chunker import CodeChunk
from ingester.storage.qdrant_store import QdrantStore, _point_id


def fetch_chunks(
    store: QdrantStore,
    chunk_ids: list[str],
) -> list[CodeChunk]:
    results = store._client.retrieve(
        collection_name=store.collection,
        ids=[_point_id(cid) for cid in chunk_ids],
    )
    id_map = {r.payload["chunk_id"]: r for r in results}

    chunks: list[CodeChunk] = []
    for cid in chunk_ids:
        point = id_map.get(cid)
        if point is None:
            continue
        p = point.payload
        chunks.append(CodeChunk(
            content=p.get("content", ""),
            chunk_type=p.get("chunk_type", ""),
            name=p.get("name", ""),
            file_path=p.get("file_path", ""),
            start_line=p.get("start_line", 0),
            end_line=p.get("end_line", 0),
            parent_class=p.get("parent_class"),
            decorators=p.get("decorators", []),
            docstring=p.get("docstring"),
            calls=p.get("calls", []),
            is_fallback_split=p.get("is_fallback_split", False),
        ))
    return chunks
