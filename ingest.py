from __future__ import annotations

import argparse
from dotenv import load_dotenv
load_dotenv()

from src.chunking.parent_child import ChunkingConfig, ParentChildChunker
from src.indexing.indexer import Indexer
from src.sources.local_files import LocalFileSource


def main(source_dir: str, qdrant_url: str) -> None:
    source = LocalFileSource(source_dir)
    chunker = ParentChildChunker(ChunkingConfig(
        parent_tokens=1500,
        child_tokens=300,
        child_overlap_tokens=50,
    ))
    indexer = Indexer(qdrant_url=qdrant_url)

    total_docs = 0
    for doc in source.load():
        print(f"Chunking: {doc.metadata['filename']}")
        parents, children = chunker.chunk(doc)
        print(f"  → {len(parents)} parents, {len(children)} children")
        indexer.add(parents, children)
        total_docs += 1

    print(f"\nFinalizing index for {total_docs} document(s)...")
    indexer.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--source", default="sample_docs", help="Directory to load documents from")
    parser.add_argument("--qdrant", default="http://localhost:6333", help="Qdrant URL")
    args = parser.parse_args()
    main(args.source, args.qdrant)
