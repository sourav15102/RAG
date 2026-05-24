from __future__ import annotations

import argparse
from dotenv import load_dotenv
load_dotenv()

from src.generation.generator import RAGGenerator
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever


def main(
    query: str,
    top_k: int,
    final_k: int,
    top_n: int,
    qdrant_url: str,
    use_hyde: bool,
    use_rerank: bool,
    verbose: bool,
    model: str,
    dry_run: bool,
) -> None:
    flags = f"HyDE={'on' if use_hyde else 'off'}, rerank={'on' if use_rerank else 'off'}"
    print(f"Query: {query!r}  [{flags}]")
    print("-" * 60)

    # --- Retrieval ---
    retriever = HybridRetriever(qdrant_url=qdrant_url)
    candidates = final_k if not use_rerank else max(final_k, top_n * 4)
    results = retriever.retrieve(query, top_k=top_k, final_k=candidates, use_hyde=use_hyde)

    if not results:
        print("No results found in index.")
        return

    if use_hyde and results[0].hypothetical_doc:
        print(f"\nHypothetical doc:\n  {results[0].hypothetical_doc!r}")

    # --- Re-ranking ---
    if use_rerank:
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, results, top_n=top_n)

    # --- Verbose: show retrieved chunks ---
    if verbose:
        print(f"\nRetrieved {len(results)} chunk(s):")
        for i, r in enumerate(results, 1):
            score_str = f"RRF {r.rrf_score:.4f}"
            if r.rerank_score is not None:
                score_str += f"  rerank {r.rerank_score:.4f}"
            print(f"\n  [{i}] {score_str}")
            print(f"       Source: {r.parent.metadata.get('filename', r.parent.id)}")
            print(f"       Child:  {r.child_content[:150].strip()!r}")
        print()

    # --- Generation ---
    generator = RAGGenerator(model=model)

    if dry_run:
        system, prompt = generator.build_prompt(query, results)
        print("=" * 60)
        print("SYSTEM PROMPT:")
        print("=" * 60)
        print(system)
        print("\n" + "=" * 60)
        print("USER MESSAGE:")
        print("=" * 60)
        print(prompt)
        return

    generator.generate(query, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced RAG pipeline.")
    parser.add_argument("query", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=20, help="Candidates per search arm (default 20)")
    parser.add_argument("--final-k", type=int, default=20, help="Results after RRF (default 20)")
    parser.add_argument("--top-n", type=int, default=5, help="Final results after re-ranking (default 5)")
    parser.add_argument("--qdrant", default="http://localhost:6333")
    parser.add_argument("--hyde", action="store_true", help="Enable HyDE")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder re-ranking")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks before answer")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model for generation")
    parser.add_argument("--dry-run", action="store_true", help="Print the LLM prompt instead of calling the API")
    args = parser.parse_args()
    main(
        args.query, args.top_k, args.final_k, args.top_n,
        args.qdrant, args.hyde, args.rerank, args.verbose, args.model, args.dry_run,
    )
