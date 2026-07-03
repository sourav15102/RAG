import json
import urllib.request
from unittest.mock import patch

import pytest

from code_chunker.ast_chunker import CodeChunk
from code_chunker.docstring_backfiller import DEFAULT_MAX_CHARS, DocstringBackfiller


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunk_with_docstring():
    return CodeChunk(
        content="def foo():\n    pass\n",
        chunk_type="function",
        name="foo",
        file_path="test.py",
        start_line=1,
        end_line=2,
        docstring="Already documented.",
    )


@pytest.fixture
def chunk_without_docstring():
    return CodeChunk(
        content="def bar():\n    return 42\n",
        chunk_type="function",
        name="bar",
        file_path="test.py",
        start_line=1,
        end_line=2,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key is required"):
            DocstringBackfiller(api_key="")

    def test_accepts_explicit_key(self):
        bf = DocstringBackfiller(api_key="sk-test123")
        assert bf.api_key == "sk-test123"

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-key")
        bf = DocstringBackfiller()
        assert bf.api_key == "sk-env-key"

    def test_defaults(self):
        bf = DocstringBackfiller(api_key="sk-test")
        assert bf.model == "deepseek-chat"
        assert bf.max_summary_chars == DEFAULT_MAX_CHARS
        assert bf.base_url == "https://api.deepseek.com"


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

class TestBackfill:
    def test_skips_chunks_with_docstring(self, chunk_with_docstring):
        bf = DocstringBackfiller(api_key="sk-test")
        result = bf.backfill([chunk_with_docstring])
        assert len(result) == 1
        assert result[0].docstring == "Already documented."

    def test_empty_input(self):
        bf = DocstringBackfiller(api_key="sk-test")
        assert bf.backfill([]) == []

    def test_preserves_other_fields(self, chunk_without_docstring):
        bf = DocstringBackfiller(api_key="sk-test")

        def fake_summarize(_c):
            return "Returns 42."

        bf._summarize = fake_summarize
        result = bf.backfill([chunk_without_docstring])
        c = result[0]
        assert c.content == chunk_without_docstring.content
        assert c.chunk_type == chunk_without_docstring.chunk_type
        assert c.name == chunk_without_docstring.name
        assert c.file_path == chunk_without_docstring.file_path
        assert c.start_line == chunk_without_docstring.start_line
        assert c.end_line == chunk_without_docstring.end_line
        assert c.parent_class is None
        assert c.decorators == []
        assert c.calls == []
        assert c.is_fallback_split is False

    def test_returns_new_objects(self, chunk_without_docstring):
        bf = DocstringBackfiller(api_key="sk-test")
        bf._summarize = lambda c: "summary"
        result = bf.backfill([chunk_without_docstring])
        assert result[0] is not chunk_without_docstring


# ---------------------------------------------------------------------------
# _summarize (with mocked HTTP)
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, text: str):
        self._data = json.dumps({
            "choices": [{"message": {"content": text}}]
        }).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _mock_response(text: str) -> bytes:
    return _MockResponse(text)


class TestSummarize:
    def test_calls_api_and_returns_text(self):
        bf = DocstringBackfiller(api_key="sk-test")

        def fake_urlopen(req, **kw):
            return _mock_response("Computes the answer to everything.")

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            result = bf._summarize(
                CodeChunk(
                    content="def answer(): return 42",
                    chunk_type="function",
                    name="answer",
                    file_path="x.py",
                    start_line=1,
                    end_line=1,
                )
            )
        assert result == "Computes the answer to everything."

    def test_truncates_overlong_response(self):
        bf = DocstringBackfiller(api_key="sk-test", max_summary_chars=20)

        def fake_urlopen(req, **kw):
            return _mock_response("A" * 50)

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            result = bf._summarize(
                CodeChunk(
                    content="x = 1", chunk_type="module_body",
                    name="mod", file_path="x.py",
                    start_line=1, end_line=1,
                )
            )
        assert len(result) <= 24  # 20 chars + "…" after last space

    def test_returns_error_message_on_failure(self):
        bf = DocstringBackfiller(api_key="sk-test")

        def fake_urlopen(req, **kw):
            raise ConnectionError("API unreachable")

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            result = bf._summarize(
                CodeChunk(
                    content="x = 1", chunk_type="module_body",
                    name="mod", file_path="x.py",
                    start_line=1, end_line=1,
                )
            )
        assert "summary failed" in result
        assert "API unreachable" in result

    def test_sends_expected_payload(self):
        bf = DocstringBackfiller(api_key="sk-test")

        sent = {}

        def capture(req, **kw):
            sent["body"] = json.loads(req.data)
            sent["url"] = req.full_url
            sent["headers"] = {k: v for k, v in req.headers.items()}
            return _mock_response("ok")

        with patch.object(urllib.request, "urlopen", capture):
            bf._summarize(
                CodeChunk(
                    content="def f(): pass",
                    chunk_type="function",
                    name="f",
                    file_path="x.py",
                    start_line=1,
                    end_line=1,
                )
            )

        assert sent["url"] == "https://api.deepseek.com/v1/chat/completions"
        assert sent["headers"]["Authorization"] == "Bearer sk-test"
        assert sent["body"]["model"] == "deepseek-chat"
        assert sent["body"]["messages"][0]["role"] == "system"
        assert sent["body"]["messages"][1]["role"] == "user"
        assert "f(): pass" in sent["body"]["messages"][1]["content"]


# ---------------------------------------------------------------------------
# integration-style: backfill calls _summarize for missing docstrings
# ---------------------------------------------------------------------------

class TestBackfillIntegration:
    def test_backfill_fills_empty_docstrings(self):
        bf = DocstringBackfiller(api_key="sk-test")

        with patch.object(bf, "_summarize", return_value="Does something."):
            chunks = bf.backfill([
                CodeChunk(
                    content="def a(): pass",
                    chunk_type="function",
                    name="a",
                    file_path="x.py",
                    start_line=1,
                    end_line=1,
                    docstring=None,
                ),
                CodeChunk(
                    content="def b(): pass",
                    chunk_type="function",
                    name="b",
                    file_path="x.py",
                    start_line=2,
                    end_line=2,
                    docstring="I have one.",
                ),
            ])

        assert chunks[0].docstring == "Does something."
        assert chunks[1].docstring == "I have one."
