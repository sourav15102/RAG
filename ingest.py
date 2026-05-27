from __future__ import annotations

import argparse
from dotenv import load_dotenv
load_dotenv()

from src.indexing.base import BaseIndexer
from src.sources.local_files import LocalFileSource


def build_indexer(name: str, qdrant_url: str) -> BaseIndexer:
    if name == "chunking":
        from src.indexing.indexer import ChunkingIndexer
        return ChunkingIndexer(qdrant_url=qdrant_url)
    elif name == "pageindex":
        from src.indexing.page_index_indexer import PageIndexIndexer
        return PageIndexIndexer()
    else:
        raise ValueError(f"Unknown indexer: {name!r}. Choose 'chunking' or 'pageindex'.")


def main(source_dir: str, qdrant_url: str, indexer_name: str) -> None:
    source = LocalFileSource(source_dir)
    indexer = build_indexer(indexer_name, qdrant_url)
    indexer.index_source(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--source", default="sample_docs", help="Directory to load documents from")
    parser.add_argument("--qdrant", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument(
        "--indexer",
        default="chunking",
        choices=["chunking", "pageindex"],
        help="Indexing strategy (default: chunking)",
    )
    args = parser.parse_args()
    main(args.source, args.qdrant, args.indexer)
