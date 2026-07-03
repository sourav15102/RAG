from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ingester.pipeline import Pipeline
from ingester.step import PipelineContext


class Ingester:
    def __init__(self, pipelines: list[Pipeline]):
        self.pipelines = pipelines

    def ingest(self, document: str, ctx: PipelineContext | None = None) -> dict[str, Any]:
        if ctx is None:
            ctx = PipelineContext()

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(p.run, document, ctx): p.name
                for p in self.pipelines
            }
            results: dict[str, Any] = {}
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
            return results
