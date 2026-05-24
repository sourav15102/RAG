from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.retrieval.retriever import RetrievalResult

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-ranks retrieval results by scoring (query, child_content) pairs directly.
    Run after hybrid retrieval on a small candidate set (e.g. top-20).
    Scores are raw logits — higher means more relevant.
    """

    def __init__(self):
        print(f"Loading re-ranker model: {MODEL_NAME}")
        self._model = CrossEncoder(MODEL_NAME)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return results

        pairs = [(query, r.child_content) for r in results]
        scores = self._model.predict(pairs)

        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        reranked = sorted(results, key=lambda r: r.rerank_score, reverse=True)
        return reranked[:top_n]
