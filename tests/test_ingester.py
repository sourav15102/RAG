from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from code_chunker.ast_chunker import CodeChunk
from embedder.code_embedder import CodeEmbedding
from ingester.ingester import Ingester
from ingester.pipeline import Pipeline, PipelineConfig
from ingester.step import PipelineContext, Step
from ingester.steps.chunk_step import ChunkStep
from ingester.steps.embed_step import EmbedStep
from ingester.steps.store_step import StoreStep
from ingester.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_code():
    return "def foo():\n    return 42\n"


@pytest.fixture
def sample_chunks():
    return [
        CodeChunk(
            content="def foo():\n    return 42\n",
            chunk_type="function",
            name="foo",
            file_path="test.py",
            start_line=1,
            end_line=2,
        )
    ]


@pytest.fixture
def sample_embeddings(sample_chunks):
    return [
        CodeEmbedding(chunk=sample_chunks[0], embedding=[0.1, 0.2, 0.3])
    ]


# ---------------------------------------------------------------------------
# Step + PipelineContext
# ---------------------------------------------------------------------------

class TestPipelineContext:
    def test_defaults(self):
        ctx = PipelineContext()
        assert ctx.document_path == ""
        assert ctx.document_source == ""
        assert ctx.config == {}
        assert ctx.state == {}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_runs_steps_in_order(self):
        calls = []

        class LogStep(Step):
            name = "log"
            def __init__(self, label):
                self.label = label
            def execute(self, ctx, data):
                calls.append(self.label)
                return data

        pipeline = Pipeline(PipelineConfig(
            name="test",
            steps=[LogStep("a"), LogStep("b"), LogStep("c")],
        ))
        result = pipeline.run("hello")
        assert calls == ["a", "b", "c"]
        assert result == "hello"

    def test_passes_data_between_steps(self):
        class DoubleStep(Step):
            name = "double"
            def execute(self, ctx, data):
                return data * 2

        pipeline = Pipeline(PipelineConfig(
            name="double",
            steps=[DoubleStep(), DoubleStep()],
        ))
        assert pipeline.run(3) == 12

    def test_creates_default_context(self):
        class CaptureCtx(Step):
            name = "capture"
            def execute(self, ctx, data):
                self._ctx = ctx
                return data

        step = CaptureCtx()
        pipeline = Pipeline(PipelineConfig(name="t", steps=[step]))
        pipeline.run("x")
        assert isinstance(step._ctx, PipelineContext)


# ---------------------------------------------------------------------------
# Ingester
# ---------------------------------------------------------------------------

class TestIngester:
    def test_runs_single_pipeline(self):
        class ConstStep(Step):
            name = "c"
            def execute(self, ctx, data):
                return "result"

        ingester = Ingester(pipelines=[
            Pipeline(PipelineConfig(name="pipe", steps=[ConstStep()])),
        ])
        results = ingester.ingest("input")
        assert results == {"pipe": "result"}

    def test_runs_multiple_pipelines_in_parallel(self):
        import time

        class SlowStep(Step):
            name = "slow"
            def __init__(self, label, delay=0.2):
                self.label = label
                self.delay = delay
            def execute(self, ctx, data):
                time.sleep(self.delay)
                return self.label

        ingester = Ingester(pipelines=[
            Pipeline(PipelineConfig(name="a", steps=[SlowStep("a", 0.3)])),
            Pipeline(PipelineConfig(name="b", steps=[SlowStep("b", 0.1)])),
        ])
        start = time.time()
        results = ingester.ingest("x")
        elapsed = time.time() - start
        assert results == {"a": "a", "b": "b"}
        assert elapsed < 0.45  # parallel, not sequential


# ---------------------------------------------------------------------------
# VectorStore (mock implementation)
# ---------------------------------------------------------------------------

class MockVectorStore(VectorStore):
    def __init__(self):
        self.records = []

    def upsert(self, records):
        self.records.extend(records)

    def search(self, query_vector, top_k=10, vector_name="code"):
        return self.records[:top_k]


# ---------------------------------------------------------------------------
# ChunkStep
# ---------------------------------------------------------------------------

class TestChunkStep:
    def test_uses_path_from_ctx(self, sample_code, sample_chunks):
        step = ChunkStep(api_key="sk-test")
        step._chunker = MagicMock(return_value=sample_chunks)
        step._chunker.process = MagicMock(return_value=sample_chunks)

        ctx = PipelineContext(document_path="mod.py")
        result = step.execute(ctx, sample_code)

        step._chunker.process.assert_called_once_with(sample_code, file_path="mod.py")
        assert ctx.state["file_path"] == "mod.py"
        assert result == sample_chunks

    def test_uses_data_as_path_when_file_exists(self, tmp_path, sample_chunks):
        f = tmp_path / "mod.py"
        f.write_text("def foo(): pass")

        step = ChunkStep(api_key="sk-test")
        step._chunker = MagicMock()
        step._chunker.process = MagicMock(return_value=sample_chunks)

        ctx = PipelineContext()
        result = step.execute(ctx, str(f))

        step._chunker.process.assert_called_once()
        assert ctx.state["file_path"] == str(f)
        assert result == sample_chunks


# ---------------------------------------------------------------------------
# EmbedStep
# ---------------------------------------------------------------------------

class TestEmbedStep:
    def test_embeds_chunks(self, sample_chunks, sample_embeddings):
        step = EmbedStep(api_key="vo-test")
        step._embedder = MagicMock()
        step._embedder.embed = MagicMock(return_value=sample_embeddings)

        ctx = PipelineContext()
        result = step.execute(ctx, sample_chunks)

        step._embedder.embed.assert_called_once_with(sample_chunks)
        assert result == sample_embeddings


# ---------------------------------------------------------------------------
# StoreStep
# ---------------------------------------------------------------------------

class TestStoreStep:
    def test_stores_embeddings(self, sample_embeddings):
        store = MockVectorStore()
        step = StoreStep(store)

        ctx = PipelineContext()
        result = step.execute(ctx, sample_embeddings)

        assert store.records == sample_embeddings
        assert result == sample_embeddings  # passes through

    def test_upsert_called_with_correct_data(self, sample_embeddings):
        mock_store = MagicMock(spec=VectorStore)
        step = StoreStep(mock_store)

        ctx = PipelineContext()
        step.execute(ctx, sample_embeddings)

        mock_store.upsert.assert_called_once_with(sample_embeddings)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestCodeEmbeddingPipeline:
    def test_full_flow_with_mocked_steps(self, sample_code, sample_chunks, sample_embeddings):
        mock_store = MagicMock(spec=VectorStore)

        pipeline = Pipeline(PipelineConfig(
            name="code_embedding",
            steps=[
                ChunkStep(api_key="sk-test"),
                EmbedStep(api_key="vo-test"),
                StoreStep(mock_store),
            ],
        ))

        with (
            patch.object(pipeline.steps[0], "_chunker") as mock_chunker,
            patch.object(pipeline.steps[1], "_embedder") as mock_embedder,
        ):
            mock_chunker.process = MagicMock(return_value=sample_chunks)
            mock_embedder.embed = MagicMock(return_value=sample_embeddings)

            result = pipeline.run(sample_code, PipelineContext(document_path="test.py"))

        mock_chunker.process.assert_called_once_with(sample_code, file_path="test.py")
        mock_embedder.embed.assert_called_once_with(sample_chunks)
        mock_store.upsert.assert_called_once_with(sample_embeddings)
        assert result == sample_embeddings


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

class TestRRF:
    def test_fuses_two_lists(self):
        from search.rrf import rrf_fuse
        bm25 = ["a", "b", "c"]
        vector = ["b", "c", "a"]
        result = rrf_fuse([bm25, vector], k=60)
        assert result[0] == "b"  # rank 1 in vector, rank 2 in BM25
        assert set(result) == {"a", "b", "c"}

    def test_preserves_order_within_same_rank(self):
        from search.rrf import rrf_fuse
        result = rrf_fuse([["x", "y"], ["x", "y"]], k=60)
        assert result[0] == "x"
        assert result[1] == "y"

    def test_single_list(self):
        from search.rrf import rrf_fuse
        assert rrf_fuse([["a", "b"]]) == ["a", "b"]

    def test_empty_input(self):
        from search.rrf import rrf_fuse
        assert rrf_fuse([]) == []


# ---------------------------------------------------------------------------
# ChunkFetcher
# ---------------------------------------------------------------------------

class TestChunkFetcher:
    def test_fetches_chunks_from_qdrant(self):
        from search.fetcher import fetch_chunks

        fake_point = MagicMock()
        fake_point.payload = {
            "chunk_id": "repo/x.py::foo",
            "content": "def foo(): pass",
            "chunk_type": "function",
            "name": "foo",
            "file_path": "x.py",
            "start_line": 1,
            "end_line": 2,
            "parent_class": None,
            "decorators": [],
            "docstring": None,
            "calls": [],
            "is_fallback_split": False,
            "line_count": 1,
        }

        client = MagicMock()
        client.retrieve.return_value = [fake_point]

        store = MagicMock()
        store._client = client

        chunks = fetch_chunks(store, ["repo/x.py::foo"])
        assert len(chunks) == 1
        assert chunks[0].name == "foo"
        assert chunks[0].content == "def foo(): pass"

    def test_skips_missing_ids(self):
        from search.fetcher import fetch_chunks

        client = MagicMock()
        client.retrieve.return_value = []

        store = MagicMock()
        store._client = client

        chunks = fetch_chunks(store, ["missing_id"])
        assert chunks == []


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_builds_context(self):
        from search.generator import _build_context
        chunk = CodeChunk(
            content="def foo(): pass",
            chunk_type="function",
            name="foo",
            file_path="test.py",
            start_line=1,
            end_line=5,
            docstring="Does something.",
        )
        ctx = _build_context([chunk])
        assert "function: foo" in ctx
        assert "test.py:1-5" in ctx
        assert "Does something." in ctx
        assert "def foo(): pass" in ctx

    def test_no_api_key_returns_error(self):
        from search.generator import GenerationResult, generate
        result = generate("query", [], api_key="")
        assert isinstance(result, GenerationResult)
        assert "no API key" in result.answer

    def test_parses_valid_json_response(self):
        import json
        from unittest.mock import patch
        from search.generator import Claim, generate

        fake_response = json.dumps({
            "answer": "process_payment validates amount.",
            "claims": [
                {
                    "claim": "validates amount",
                    "source_chunk": "payments/service.py",
                    "source_function": "PaymentService.process",
                    "lines": "45-48",
                    "confidence": "high",
                }
            ],
            "unanswered_parts": "gateway selection",
        })

        with patch("search.generator._llm_complete", return_value=fake_response):
            result = generate("how does payment work?", [], api_key="sk-test")

        assert result.answer == "process_payment validates amount."
        assert len(result.claims) == 1
        assert isinstance(result.claims[0], Claim)
        assert result.claims[0].confidence == "high"
        assert result.claims[0].source_chunk == "payments/service.py"
        assert result.unanswered_parts == "gateway selection"

    def test_fallback_on_invalid_json(self):
        from unittest.mock import patch
        from search.generator import GenerationResult, generate

        with patch("search.generator._llm_complete", return_value="not json"):
            result = generate("query", [], api_key="sk-test")

        assert isinstance(result, GenerationResult)
        assert result.answer == "not json"
        assert result.claims == []

    def test_parse_claims_defaults_medium_on_invalid_confidence(self):
        from search.generator import _parse_claims
        claims = _parse_claims([{
            "claim": "x",
            "source_chunk": "a.py",
            "source_function": "f",
            "lines": "1-2",
            "confidence": "unknown_value",
        }])
        assert claims[0].confidence == "medium"


# ---------------------------------------------------------------------------
# SearchStep
# ---------------------------------------------------------------------------

class TestSearchStep:
    def test_search_orchestrates_full_pipeline(self):
        from ingester.steps.search_step import SearchStep
        from search.generator import generate

        bm25 = MagicMock()
        bm25.search.return_value = ["id_a", "id_b", "id_c"]

        qdrant = MagicMock()
        qdrant.search.return_value = [("id_b", 0.9), ("id_c", 0.8), ("id_a", 0.7)]

        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1, 0.2, 0.3]

        fake_point_a = MagicMock()
        fake_point_a.payload = {"chunk_id": "id_a", "content": "def a(): pass",
            "chunk_type": "function", "name": "a", "file_path": "x.py",
            "start_line": 1, "end_line": 1}
        fake_point_b = MagicMock()
        fake_point_b.payload = {"chunk_id": "id_b", "content": "def b(): pass",
            "chunk_type": "function", "name": "b", "file_path": "x.py",
            "start_line": 2, "end_line": 2}

        qdrant._client.retrieve.return_value = [fake_point_a, fake_point_b]

        step = SearchStep(bm25, qdrant, embedder, llm_api_key="")
        ctx = PipelineContext()
        result = step.execute(ctx, "how does foo work?")

        assert "chunk_ids" in result
        assert "chunks" in result
        assert "answer" in result
        bm25.search.assert_called_once()
        qdrant.search.assert_called_once()
        embedder.embed_query.assert_called_once_with("how does foo work?")


class TestIngesterWithPipeline:
    def test_ingester_orchestrates_code_pipeline(self, sample_code, sample_chunks, sample_embeddings):
        mock_store = MagicMock(spec=VectorStore)

        pipeline = Pipeline(PipelineConfig(
            name="code_embedding",
            steps=[
                ChunkStep(api_key="sk-test"),
                EmbedStep(api_key="vo-test"),
                StoreStep(mock_store),
            ],
        ))

        ingester = Ingester(pipelines=[pipeline])

        with (
            patch.object(pipeline.steps[0], "_chunker") as mock_chunker,
            patch.object(pipeline.steps[1], "_embedder") as mock_embedder,
        ):
            mock_chunker.process = MagicMock(return_value=sample_chunks)
            mock_embedder.embed = MagicMock(return_value=sample_embeddings)

            results = ingester.ingest(sample_code, PipelineContext(document_path="test.py"))

        assert "code_embedding" in results
        assert results["code_embedding"] == sample_embeddings
        mock_store.upsert.assert_called_once_with(sample_embeddings)
