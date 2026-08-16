#!/usr/bin/env python
"""CLI: ask a question against an indexed codebase (hybrid search + grounded answer)."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from embedder.code_embedder import CodeEmbedder
from ingester.step import PipelineContext
from ingester.steps.search_step import SearchStep
from ingester.storage.bm25_store import BM25Store
from ingester.storage.qdrant_store import QdrantStore

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--qdrant-host", default=os.environ.get("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.environ.get("QDRANT_PORT", "6333")))
    parser.add_argument("--es-host", default=os.environ.get("ES_HOST", "http://localhost:9200"))
    parser.add_argument("--collection", default="code_chunks", help="Qdrant collection / ES index name")
    parser.add_argument("--top-k", type=int, default=50, help="Candidates fetched per search arm before RRF")
    parser.add_argument("--top-n", type=int, default=3, help="Chunks kept after cross-encoder re-ranking")
    parser.add_argument("--rewrite", action="store_true", default=False, help="LLM query rewriting before search")
    parser.add_argument("--hyde", action="store_true", default=False, help="HyDE hypothetical-docstring search")
    parser.add_argument("--verbose", action="store_true", help="Print retrieved chunks and claims")
    args = parser.parse_args()

    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not voyage_key or not deepseek_key:
        sys.exit("VOYAGE_API_KEY and DEEPSEEK_API_KEY must be set (see .env)")

    es = Elasticsearch(args.es_host)
    qdrant = QdrantStore(host=args.qdrant_host, port=args.qdrant_port, collection=args.collection)
    bm25 = BM25Store(es=es, index=args.collection)
    embedder = CodeEmbedder(api_key=voyage_key)

    search = SearchStep(
        bm25_store=bm25,
        qdrant_store=qdrant,
        embedder=embedder,
        top_k=args.top_k,
        rerank_top_n=args.top_n,
        llm_api_key=deepseek_key,
        rewrite=args.rewrite,
        use_hyde=args.hyde,
    )

    result = search.execute(PipelineContext(), args.question)

    if args.verbose:
        print("--- retrieved chunks ---")
        for cid in result["chunk_ids"]:
            print(f"  {cid}")
        print()

    print(result["answer"])

    if result["claims"]:
        print("\n--- citations ---")
        for c in result["claims"]:
            print(f"  [{c.confidence}] {c.claim}  ({c.source_chunk}::{c.source_function} L{c.lines})")

    if result["unanswered_parts"]:
        print(f"\n--- not covered by retrieved code ---\n  {result['unanswered_parts']}")


if __name__ == "__main__":
    main()
