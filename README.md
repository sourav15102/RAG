# Advanced RAG Pipeline

A production-grade Retrieval Augmented Generation system built from scratch in Python. Connects to any document source and answers questions using a full advanced RAG pipeline: parent-child chunking, hybrid search with Reciprocal Rank Fusion, HyDE, cross-encoder re-ranking, and LLM answer generation.

## Architecture

```
Document Source (pluggable)
        ↓
  BaseIndexer (pluggable)
        ├── ChunkingIndexer  →  Parent-Child Chunks  →  Qdrant (semantic) + BM25 (keyword)
        └── PageIndexIndexer →  LLM Tree Structure   →  JSON manifest (vectorless)
                                        ↓
                              HybridRetriever
                                  ├── Semantic search (nomic-embed-text-v1.5 + Qdrant)
                                  └── BM25 search (rank_bm25)
                                        ↓
                              Reciprocal Rank Fusion (RRF)
                                        ↓
                              [Optional] HyDE
                                        ↓
                              [Optional] CrossEncoderReranker
                                        ↓
                              Parent chunk lookup
                                        ↓
                              RAGGenerator (Claude API)
                                        ↓
                                    Answer
```

## Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` | Local, free, 8192 token context window |
| Vector store | Qdrant (Docker) | Production-grade, persistent, fast ANN search |
| Keyword search | `rank_bm25` | BM25Okapi, persisted as pickle |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Accurate pairwise scoring, fast on CPU |
| LLM | Claude (Anthropic API) | Answer generation + HyDE hypothetical docs |
| Token counting | `tiktoken` | Accurate chunk sizing without loading embedding model |

## Project Structure

```
src/
  models.py                   # RawDocument, ParentChunk, ChildChunk
  sources/
    base.py                   # DocumentSource ABC — plug any source here
    local_files.py            # LocalFileSource (.txt, .md)
  chunking/
    parent_child.py           # ParentChildChunker + ChunkingConfig
  indexing/
    base.py                   # BaseIndexer ABC — plug any indexing strategy
    indexer.py                # ChunkingIndexer (Qdrant + BM25)
    page_index_indexer.py     # PageIndexIndexer (LLM tree, vectorless)
    embedder.py               # NomicEmbedder (handles task prefixes)
    qdrant_store.py           # QdrantVectorStore
    bm25_store.py             # BM25Store
    parent_store.py           # ParentStore (JSON)
  retrieval/
    retriever.py              # HybridRetriever + RRF + RetrievalResult
    hyde.py                   # HyDEGenerator
    reranker.py               # CrossEncoderReranker
  generation/
    generator.py              # RAGGenerator (streaming)
ingest.py                     # CLI: index documents
query.py                      # CLI: query the pipeline
```

## Setup

**Prerequisites:** Python 3.9+, Docker

**1. Clone and create virtual environment**
```bash
git clone https://github.com/sourav15102/RAG.git
cd RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Start Qdrant**
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant qdrant/qdrant
```

**3. Set your API key**
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

**4. Index your documents**
```bash
python ingest.py --source your_docs/
```

**5. Ask questions**
```bash
python query.py "What is retrieval augmented generation?" --rerank
```

## CLI Reference

### `ingest.py`

```bash
python ingest.py [--source DIR] [--qdrant URL] [--indexer STRATEGY]
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `sample_docs` | Directory of `.txt` / `.md` files to index |
| `--qdrant` | `http://localhost:6333` | Qdrant server URL |
| `--indexer` | `chunking` | `chunking` (Qdrant+BM25) or `pageindex` (LLM tree) |

### `query.py`

```bash
python query.py "your question" [flags]
```

| Flag | Default | Description |
|---|---|---|
| `--top-k` | `20` | Candidates fetched per search arm (semantic + BM25 each) |
| `--final-k` | `20` | Results kept after RRF merge |
| `--top-n` | `5` | Final results after cross-encoder re-ranking |
| `--rerank` | off | Enable cross-encoder re-ranking |
| `--hyde` | off | Enable HyDE (requires `ANTHROPIC_API_KEY`) |
| `--verbose` | off | Print retrieved chunks and scores before the answer |
| `--dry-run` | off | Print the full LLM prompt without calling the API |
| `--model` | `claude-haiku-4-5-20251001` | Claude model for answer generation |
| `--qdrant` | `http://localhost:6333` | Qdrant server URL |

### Examples

```bash
# Inspect what gets sent to the LLM (no API key needed)
python query.py "How does attention work?" --rerank --dry-run

# Full pipeline with verbose retrieval output
python query.py "What is RAG?" --rerank --verbose

# Full pipeline including HyDE
python query.py "What is RAG?" --rerank --hyde

# Use PageIndex (LLM tree) instead of chunking
python ingest.py --indexer pageindex
```

## Plugging in a New Document Source

Subclass `DocumentSource` and implement two methods:

```python
from src.sources.base import DocumentSource
from src.models import RawDocument

class MySource(DocumentSource):
    @property
    def source_id(self) -> str:
        return "my_source"

    def load(self):
        # yield RawDocument objects
        yield RawDocument(id="...", content="...", metadata={})
```

Pass it to any indexer:
```python
indexer.index_source(MySource())
```

## Plugging in a New Indexer

Subclass `BaseIndexer`:

```python
from src.indexing.base import BaseIndexer
from src.models import RawDocument

class MyIndexer(BaseIndexer):
    def add_document(self, doc: RawDocument) -> None: ...
    def finalize(self) -> None: ...
```

---

## Learnings

### Parent-Child Chunking
The core insight: **embed small, retrieve large**. Child chunks (~300 tokens) are embedded and searched because they are focused and produce precise vector matches. But when a child is retrieved, its parent (~1500 tokens) is what gets passed to the LLM — giving it full surrounding context. If you embed the parent directly, the embedding gets diluted across too much content and retrieval precision drops.

### nomic-embed-text-v1.5 Task Prefixes
This model requires explicit task prefixes or quality degrades significantly:
- `search_document: <text>` — for text being stored in the index
- `search_query: <text>` — for query text at search time

In HyDE, the hypothetical document gets `search_document:` (not `search_query:`) because it's meant to behave like a real indexed document in vector space.

### Reciprocal Rank Fusion (RRF)
The formula is `score(d) = Σ 1 / (k + rank(d))` where `k=60` is a standard constant that dampens the impact of very high ranks. The key property: **you never need to normalize scores across systems**. Semantic search returns cosine similarities (0–1), BM25 returns raw term frequencies — completely different scales. RRF only uses the rank position, so the scales don't matter. A document appearing in both lists gets boosted regardless of what the individual scores were.

### HyDE (Hypothetical Document Embeddings)
Short queries live in a different part of embedding space than long documents. HyDE bridges this gap: ask the LLM "write a passage that would answer this question", then embed that passage as if it were a real document. The hypothetical passage is longer, uses domain vocabulary, and matches the style of indexed content — so it lands closer to the right documents in vector space. The original query still drives BM25 since keyword matching doesn't have this semantic gap problem.

### Cross-Encoder vs Bi-Encoder
Bi-encoders (like nomic) encode query and document **separately** then compare vectors. This is fast but loses the interaction signal between query and document. Cross-encoders take `(query, document)` as a **single input** and score the pair directly — much more accurate, but O(n) inference on the candidate set. This is why cross-encoders are only run on a small set of candidates (top-20) after the fast bi-encoder pass, not on the full corpus.

Cross-encoder scores are raw logits — they can be negative. Higher is more relevant, but the absolute values have no fixed meaning. Only the relative ordering matters.

### Embedder Ownership
Initially `QdrantVectorStore` owned a `NomicEmbedder` instance in its constructor. This caused the model to load twice when `Indexer` also created one. The fix: `QdrantVectorStore` only handles vector I/O — it accepts pre-computed vectors in `upsert()` and returns raw payloads in `search()`. The caller (`ChunkingIndexer` or `HybridRetriever`) owns the embedder and decides when to load it.

### `brew install docker` vs Docker Desktop
`brew install docker` installs the CLI only — no daemon. Running `docker run ...` fails with "cannot connect to docker socket". You need either Docker Desktop (GUI app) or Colima (`brew install colima && colima start`) to actually run containers.

### Python 3.9 Union Type Syntax
`X | None` union syntax was introduced in Python 3.10. On Python 3.9 it raises `TypeError: unsupported operand type(s) for |`. Fix: add `from __future__ import annotations` at the top of every file — this enables deferred annotation evaluation and makes the new syntax work transparently on 3.9.

### BM25 Overlap Verification
Testing token overlap between adjacent chunks at the character level is unreliable — `tiktoken` encodes at the subword level so character boundaries don't align with token boundaries. The correct test: sum all child token counts for a parent and verify the total exceeds the parent's own token count. If there's overlap, children collectively contain more tokens than the parent alone.

### PageIndex vs Chunking — Fundamentally Different Paradigms
Standard RAG uses embeddings to find semantically similar chunks. PageIndex skips vectors entirely — it uses an LLM to build a hierarchical table-of-contents tree from the document, then uses another LLM to reason through the tree to find relevant sections. This is better for structured documents with clear hierarchy (reports, manuals, books) but costs LLM tokens at index time. Standard chunking is better for unstructured text at scale.

### `--dry-run` for Debugging RAG
Before spending API credits, always inspect the prompt. Adding a `--dry-run` flag that prints the system prompt and full user message (context + question) without calling the API lets you verify: Is the right context being retrieved? Is the context too long? Is the question framed correctly? This is the fastest way to debug retrieval quality before touching generation.
