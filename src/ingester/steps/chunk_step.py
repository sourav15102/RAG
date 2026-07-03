from pathlib import Path
from typing import Optional

from code_chunker.code_chunker import CodeChunker

from ingester.step import PipelineContext, Step


class ChunkStep(Step):
    name = "chunk"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_lines: int = 100,
        max_summary_chars: int = 200,
    ):
        self._chunker = CodeChunker(
            api_key=api_key,
            model=model,
            max_lines=max_lines,
            max_summary_chars=max_summary_chars,
        )

    def execute(self, ctx: PipelineContext, data: str) -> list:
        path = ctx.document_path or data
        source: str | Path
        file_path: str

        if Path(data).exists():
            source = Path(data)
            file_path = data
        else:
            source = data
            file_path = path

        ctx.state["file_path"] = file_path
        return self._chunker.process(source, file_path=file_path)
