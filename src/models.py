from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentChunk:
    id: str
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildChunk:
    id: str
    parent_id: str
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
