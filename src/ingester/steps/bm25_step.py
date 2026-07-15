from typing import Any

from ingester.step import PipelineContext, Step
from ingester.storage.bm25_store import BM25Store


class BM25IndexStep(Step):
    name = "bm25_index"

    def __init__(self, store: BM25Store):
        self._store = store

    def execute(self, ctx: PipelineContext, data: Any) -> Any:
        self._store.index(data)
        return data
