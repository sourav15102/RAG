from __future__ import annotations

import json
from pathlib import Path

from src.models import ParentChunk

DEFAULT_PATH = Path("data/parents.json")


class ParentStore:
    def __init__(self, store_path: Path = DEFAULT_PATH):
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict] = {}

    def add(self, parents: list[ParentChunk]) -> None:
        for p in parents:
            self._store[p.id] = {
                "id": p.id,
                "doc_id": p.doc_id,
                "content": p.content,
                "metadata": p.metadata,
            }

    def save(self) -> None:
        with open(self.store_path, "w") as f:
            json.dump(self._store, f)
        print(f"Parent store saved to {self.store_path} ({len(self._store)} parents).")

    def load(self) -> None:
        with open(self.store_path) as f:
            self._store = json.load(f)

    def get(self, parent_id: str) -> ParentChunk | None:
        d = self._store.get(parent_id)
        if d is None:
            return None
        return ParentChunk(id=d["id"], doc_id=d["doc_id"], content=d["content"], metadata=d["metadata"])

    def get_many(self, parent_ids: list[str]) -> list[ParentChunk]:
        seen = set()
        results = []
        for pid in parent_ids:
            if pid in seen:
                continue
            seen.add(pid)
            p = self.get(pid)
            if p:
                results.append(p)
        return results
