from typing import Optional

from embedder.code_embedder import CodeEmbedder

from ingester.step import PipelineContext, Step


class EmbedStep(Step):
    name = "embed"

    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-4"):
        self._embedder = CodeEmbedder(api_key=api_key, model=model)

    def execute(self, ctx: PipelineContext, data: list) -> list:
        return self._embedder.embed(data)
