from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterator

from src.models import RawDocument
from src.sources.base import DocumentSource

SUPPORTED_EXTENSIONS = {".txt", ".md"}


class LocalFileSource(DocumentSource):
    """Loads plain text and markdown files from a local directory tree."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

    @property
    def source_id(self) -> str:
        return f"local:{self.directory.resolve()}"

    def load(self) -> Iterator[RawDocument]:
        for path in sorted(self.directory.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            yield RawDocument(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve()))),
                content=content,
                metadata={
                    "filename": path.name,
                    "path": str(path.resolve()),
                    "extension": path.suffix.lower(),
                    "source": "local",
                },
            )
