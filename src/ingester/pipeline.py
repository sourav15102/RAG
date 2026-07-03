from dataclasses import dataclass, field
from typing import Any

from .step import PipelineContext, Step


@dataclass
class PipelineConfig:
    name: str = ""
    steps: list[Step] = field(default_factory=list)


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.name = config.name
        self.steps = config.steps

    def run(self, document: str, ctx: PipelineContext | None = None) -> Any:
        if ctx is None:
            ctx = PipelineContext()
        data: Any = document
        for step in self.steps:
            data = step.execute(ctx, data)
        return data
