from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from src.indexing.base import BaseIndexer
from src.models import RawDocument

THIRD_PARTY_DIR = Path("third_party")
PAGEINDEX_REPO = THIRD_PARTY_DIR / "PageIndex"
PAGEINDEX_URL = "https://github.com/VectifyAI/PageIndex.git"

DATA_DIR = Path("data/page_index")
DOCS_DIR = DATA_DIR / "docs"
TREES_DIR = DATA_DIR / "trees"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Use Claude via LiteLLM — no OpenAI key needed
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _ensure_pageindex() -> None:
    """Clone PageIndex repo if not already present and add to sys.path."""
    if not PAGEINDEX_REPO.exists():
        print(f"Cloning PageIndex into {PAGEINDEX_REPO}...")
        THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", PAGEINDEX_URL, str(PAGEINDEX_REPO)],
            check=True,
        )
        # Install PageIndex dependencies into our venv
        req = PAGEINDEX_REPO / "requirements.txt"
        if req.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
                check=True,
            )

    repo_str = str(PAGEINDEX_REPO.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


class PageIndexIndexer(BaseIndexer):
    """
    Indexes documents using VectifyAI/PageIndex — builds a hierarchical
    table-of-contents tree via LLM reasoning instead of embedding-based chunking.

    Retrieval later uses LLM tree navigation (see pageindex.retrieve) rather
    than vector similarity or BM25.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        _ensure_pageindex()
        self._model = model
        self._queued: list[RawDocument] = []
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        TREES_DIR.mkdir(parents=True, exist_ok=True)

    def add_document(self, doc: RawDocument) -> None:
        self._queued.append(doc)

    def finalize(self) -> None:
        if not self._queued:
            print("Nothing to index.")
            return

        from pageindex.page_index_md import md_to_tree  # noqa: PLC0415

        manifest: dict[str, dict] = {}

        for doc in self._queued:
            print(f"  Building PageIndex tree for: {doc.metadata.get('filename', doc.id)}")

            # Write content to a temp markdown file
            md_path = DOCS_DIR / f"{doc.id}.md"
            md_path.write_text(doc.content, encoding="utf-8")

            # Build tree (async API)
            tree = asyncio.run(md_to_tree(
                md_path=str(md_path),
                if_thinning=True,
                min_token_threshold=500,
                if_add_node_summary=True,
                summary_token_threshold=200,
                model=self._model,
                if_add_doc_description=True,
                if_add_node_text=True,
                if_add_node_id=True,
            ))

            # Persist the tree
            tree_path = TREES_DIR / f"{doc.id}.json"
            tree_path.write_text(json.dumps(tree, indent=2), encoding="utf-8")

            manifest[doc.id] = {
                "doc_id": doc.id,
                "md_path": str(md_path),
                "tree_path": str(tree_path),
                "metadata": doc.metadata,
                "type": "md",
            }
            print(f"  Tree saved → {tree_path}")

        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nPageIndex manifest saved → {MANIFEST_PATH} ({len(manifest)} documents).")
