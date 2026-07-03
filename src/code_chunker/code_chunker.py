from pathlib import Path
from typing import Optional

from .ast_chunker import CodeChunk, chunk_python_file
from .docstring_backfiller import DocstringBackfiller


class CodeChunker:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_lines: int = 100,
        max_summary_chars: int = 200,
        timeout: int = 30,
    ):
        self.max_lines = max_lines
        self._backfiller = DocstringBackfiller(
            api_key=api_key,
            model=model,
            max_summary_chars=max_summary_chars,
            timeout=timeout,
        )

    def process(self, source: str | Path, file_path: Optional[str] = None) -> list[CodeChunk]:
        resolved_path: str
        if isinstance(source, Path):
            resolved_path = str(source)
        elif file_path is not None:
            resolved_path = file_path
        else:
            resolved_path = "<string>"

        chunks = chunk_python_file(source, file_path=resolved_path, max_lines=self.max_lines)
        chunks = self._backfiller.backfill(chunks)
        return chunks
