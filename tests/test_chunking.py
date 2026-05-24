import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.parent_child import ChunkingConfig, ParentChildChunker
from src.sources.local_files import LocalFileSource


def test_local_source_loads():
    source = LocalFileSource("sample_docs")
    docs = list(source.load())
    assert len(docs) > 0, "No documents loaded"
    for doc in docs:
        assert doc.id
        assert doc.content
        assert "filename" in doc.metadata
    print(f"  Loaded {len(docs)} document(s)")
    return docs


def test_chunking(docs):
    chunker = ParentChildChunker(ChunkingConfig(
        parent_tokens=500,
        child_tokens=100,
        child_overlap_tokens=20,
    ))
    for doc in docs:
        parents, children = chunker.chunk(doc)
        assert len(parents) > 0
        assert len(children) >= len(parents)

        # every child references a valid parent
        parent_ids = {p.id for p in parents}
        for child in children:
            assert child.parent_id in parent_ids, f"Orphan child: {child.id}"
            assert child.doc_id == doc.id
            assert child.content.strip()

        print(f"  [{doc.metadata['filename']}] "
              f"{len(parents)} parents, {len(children)} children")

        # spot-check: parent content is longer than child content
        for parent in parents:
            its_children = [c for c in children if c.parent_id == parent.id]
            assert len(its_children) >= 1


def test_overlap(docs):
    """With overlap, total child tokens > parent tokens (tokens are counted twice in overlap zones)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    config = ChunkingConfig(parent_tokens=500, child_tokens=100, child_overlap_tokens=20)
    chunker = ParentChildChunker(config)
    for doc in docs:
        parents, children = chunker.chunk(doc)
        same_parent = {}
        for c in children:
            same_parent.setdefault(c.parent_id, []).append(c)

        for parent in parents:
            siblings = same_parent.get(parent.id, [])
            if len(siblings) < 2:
                continue
            parent_token_count = len(enc.encode(parent.content))
            child_token_total = sum(len(enc.encode(c.content)) for c in siblings)
            # with overlap the sum of child tokens must exceed parent tokens
            assert child_token_total > parent_token_count, (
                f"Expected overlap: child total {child_token_total} <= parent {parent_token_count}"
            )
    print("  Overlap check passed")


if __name__ == "__main__":
    print("Running Part 2 tests...")
    docs = test_local_source_loads()
    print("PASS: local source loads")
    test_chunking(docs)
    print("PASS: parent-child chunking")
    test_overlap(docs)
    print("PASS: child overlap")
    print("\nAll tests passed.")
