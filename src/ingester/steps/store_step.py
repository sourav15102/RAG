from typing import Any

from ingester.step import PipelineContext, Step
from ingester.storage.vector_store import VectorStore


class StoreStep(Step):
    name = "store"

    def __init__(self, store: VectorStore):
        self._store = store

    def execute(self, ctx: PipelineContext, data: Any) -> Any:
        self._store.upsert(data)
        return data
