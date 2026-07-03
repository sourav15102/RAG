from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    document_path: str = ""
    document_source: str = ""
    config: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)


class Step(ABC):
    name: str = "step"

    @abstractmethod
    def execute(self, ctx: PipelineContext, data: Any) -> Any:
        ...
