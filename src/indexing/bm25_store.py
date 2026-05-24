from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models import ChildChunk

DEFAULT_PATH = Path("data/bm25.pkl")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, index_path: Path = DEFAULT_PATH):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index: BM25Okapi | None = None
        # parallel list to the corpus — same order as what BM25 was built on
        self._children: list[dict] = []

    def build(self, children: list[ChildChunk]) -> None:
        self._children = [
            {
                "child_id": c.id,
                "parent_id": c.parent_id,
                "doc_id": c.doc_id,
                "content": c.content,
                "metadata": c.metadata,
            }
            for c in children
        ]
        corpus = [_tokenize(c.content) for c in children]
        self._index = BM25Okapi(corpus)

    def save(self) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump({"index": self._index, "children": self._children}, f)
        print(f"BM25 index saved to {self.index_path} ({len(self._children)} entries).")

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self._index = data["index"]
        self._children = data["children"]

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self._index is None:
            raise RuntimeError("BM25 index not loaded. Call load() first.")
        scores = self._index.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            if score <= 0:
                break
            results.append({**self._children[idx], "score": score})
        return results
