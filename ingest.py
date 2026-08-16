#!/usr/bin/env python
"""CLI: chunk a directory of Python files and index them into Qdrant + Elasticsearch."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from code_chunker.code_chunker import CodeChunker
from embedder.code_embedder import CodeEmbedder
from ingester.storage.bm25_store import BM25Store
from ingester.storage.qdrant_store import QdrantStore

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="sample_docs", help="Directory of .py files to index")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--collection", default="code_chunks", help="Qdrant collection / ES index name")
    parser.add_argument("--repo", default="", help="Repo/namespace prefix for chunk IDs")
    args = parser.parse_args()

    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not voyage_key or not deepseek_key:
        sys.exit("VOYAGE_API_KEY and DEEPSEEK_API_KEY must be set (see .env)")

    es = Elasticsearch(args.es_host)
    qdrant = QdrantStore(host=args.qdrant_host, port=args.qdrant_port, collection=args.collection, repo=args.repo)
    bm25 = BM25Store(es=es, index=args.collection, repo=args.repo)
    bm25.ensure_index()

    chunker = CodeChunker(api_key=deepseek_key)
    embedder = CodeEmbedder(api_key=voyage_key)

    files = sorted(Path(args.source).glob("*.py"))
    if not files:
        sys.exit(f"No .py files found in {args.source}")

    all_chunks = []
    for py_file in files:
        chunks = chunker.process(py_file)
        all_chunks.extend(chunks)
        print(f"chunked {py_file}: {len(chunks)} chunks")

    print(f"\nembedding {len(all_chunks)} chunks ...")
    embeddings = embedder.embed(all_chunks)

    print("upserting to Qdrant ...")
    qdrant.upsert(embeddings)

    print("indexing to Elasticsearch (BM25) ...")
    bm25.index(all_chunks)

    print(f"\ndone — {len(all_chunks)} chunks indexed from {len(files)} files")


if __name__ == "__main__":
    main()
