import ast
import textwrap
from pathlib import Path

import pytest

from ast_chunker import (
    CodeChunk,
    _chunk_class,
    _chunk_function,
    _fallback_split,
    _get_calls,
    _get_decorators,
    _get_docstring,
    _source_segment,
    chunk_python_file,
)


# ---------------------------------------------------------------------------
# CodeChunk data model
# ---------------------------------------------------------------------------

class TestCodeChunk:
    def test_basic_creation(self):
        chunk = CodeChunk(
            content="def foo():\n    pass\n",
            chunk_type="function",
            name="foo",
            file_path="test.py",
            start_line=1,
            end_line=2,
        )
        assert chunk.content == "def foo():\n    pass\n"
        assert chunk.chunk_type == "function"
        assert chunk.name == "foo"
        assert chunk.file_path == "test.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 2

    def test_line_count_property(self):
        chunk = CodeChunk(
            content="a\nb\nc\n",
            chunk_type="module_body",
            name="mod",
            file_path="x.py",
            start_line=5,
            end_line=7,
        )
        assert chunk.line_count == 3

    def test_line_count_single_line(self):
        chunk = CodeChunk(
            content="x = 1\n",
            chunk_type="module_body",
            name="mod",
            file_path="x.py",
            start_line=10,
            end_line=10,
        )
        assert chunk.line_count == 1

    def test_to_dict(self):
        chunk = CodeChunk(
            content="content",
            chunk_type="function",
            name="foo",
            file_path="f.py",
            start_line=1,
            end_line=2,
            parent_class="Bar",
            decorators=["staticmethod"],
            docstring="doc",
            calls=["baz"],
            is_fallback_split=True,
        )
        d = chunk.to_dict()
        assert d["content"] == "content"
        assert d["chunk_type"] == "function"
        assert d["name"] == "foo"
        assert d["file_path"] == "f.py"
        assert d["start_line"] == 1
        assert d["end_line"] == 2
        assert d["parent_class"] == "Bar"
        assert d["decorators"] == ["staticmethod"]
        assert d["docstring"] == "doc"
        assert d["calls"] == ["baz"]
        assert d["line_count"] == 2
        assert d["is_fallback_split"] is True

    def test_to_dict_defaults(self):
        chunk = CodeChunk(
            content="content",
            chunk_type="function",
            name="f",
            file_path="f.py",
            start_line=1,
            end_line=1,
        )
        d = chunk.to_dict()
        assert d["parent_class"] is None
        assert d["decorators"] == []
        assert d["docstring"] is None
        assert d["calls"] == []
        assert d["is_fallback_split"] is False

    def test_default_false_is_not_included_as_true(self):
        chunk = CodeChunk(
            content="content",
            chunk_type="function",
            name="f",
            file_path="f.py",
            start_line=1,
            end_line=1,
        )
        assert chunk.is_fallback_split is False


# ---------------------------------------------------------------------------
# _source_segment
# ---------------------------------------------------------------------------

class TestSourceSegment:
    def test_extracts_source_lines(self):
        code = "x = 1\ny = 2\n"
        source_lines = code.splitlines(keepends=True)
        node = ast.parse(code).body[1]
        result = _source_segment(source_lines, node)
        assert result == "y = 2\n"

    def test_multi_line_node(self):
        code = "if True:\n    pass\n"
        tree = ast.parse(code)
        result = _source_segment(code.splitlines(keepends=True), tree.body[0])
        assert result == code


# ---------------------------------------------------------------------------
# _get_decorators
# ---------------------------------------------------------------------------

class TestGetDecorators:
    def test_simple_name_decorator(self):
        code = "@staticmethod\ndef foo(): pass"
        tree = ast.parse(textwrap.dedent(code))
        dec = _get_decorators(tree.body[0])
        assert dec == ["staticmethod"]

    def test_attribute_decorator(self):
        code = "@app.route\ndef foo(): pass"
        tree = ast.parse(textwrap.dedent(code))
        dec = _get_decorators(tree.body[0])
        assert dec == ["app.route"]

    def test_call_decorator(self):
        code = "@decorator(arg=True)\ndef foo(): pass"
        tree = ast.parse(textwrap.dedent(code))
        dec = _get_decorators(tree.body[0])
        assert dec == ["decorator(arg=True)"]

    def test_multiple_decorators(self):
        code = "@staticmethod\n@property\ndef foo(): pass"
        tree = ast.parse(textwrap.dedent(code))
        dec = _get_decorators(tree.body[0])
        assert dec == ["staticmethod", "property"]

    def test_no_decorators(self):
        code = "def foo(): pass"
        tree = ast.parse(textwrap.dedent(code))
        dec = _get_decorators(tree.body[0])
        assert dec == []


# ---------------------------------------------------------------------------
# _get_docstring
# ---------------------------------------------------------------------------

class TestGetDocstring:
    def test_function_docstring(self):
        code = 'def foo():\n    """My doc."""\n    pass\n'
        tree = ast.parse(code)
        ds = _get_docstring(tree.body[0])
        assert ds == "My doc."

    def test_class_docstring(self):
        code = 'class Bar:\n    """Class doc."""\n    pass\n'
        tree = ast.parse(code)
        ds = _get_docstring(tree.body[0])
        assert ds == "Class doc."

    def test_module_docstring(self):
        code = '"""Module doc."""\nimport os\n'
        tree = ast.parse(code)
        ds = _get_docstring(tree)
        assert ds == "Module doc."

    def test_no_docstring(self):
        code = "def foo():\n    pass\n"
        tree = ast.parse(code)
        ds = _get_docstring(tree.body[0])
        assert ds is None

    def test_non_docstring_string(self):
        code = 'def foo():\n    "not a docstring"\n    pass\n'
        tree = ast.parse(code)
        ds = _get_docstring(tree.body[0])
        assert ds == "not a docstring"

    def test_returns_none_for_non_docstring_node(self):
        code = "x = 1\n"
        tree = ast.parse(code)
        ds = _get_docstring(tree.body[0])
        assert ds is None

    def test_docstring_with_blank_lines(self):
        code = 'def foo():\n    """Doc with\n\n    blank lines."""\n    pass\n'
        tree = ast.parse(code)
        ds = _get_docstring(tree.body[0])
        assert "blank lines." in ds


# ---------------------------------------------------------------------------
# _get_calls
# ---------------------------------------------------------------------------

class TestGetCalls:
    def test_direct_calls(self):
        code = "def foo():\n    bar()\n    baz()\n"
        tree = ast.parse(code)
        calls = _get_calls(tree.body[0])
        assert calls == ["bar", "baz"]

    def test_method_calls(self):
        code = "def foo():\n    self.bar()\n    obj.baz()\n"
        tree = ast.parse(code)
        calls = _get_calls(tree.body[0])
        assert calls == ["bar", "baz"]

    def test_no_calls(self):
        code = "def foo():\n    x = 1\n"
        tree = ast.parse(code)
        calls = _get_calls(tree.body[0])
        assert calls == []

    def test_unique_sorted(self):
        code = "def foo():\n    bar()\n    bar()\n    baz()\n"
        tree = ast.parse(code)
        calls = _get_calls(tree.body[0])
        assert calls == ["bar", "baz"]

    def test_nested_calls(self):
        code = "def foo():\n    bar(baz())\n"
        tree = ast.parse(code)
        calls = _get_calls(tree.body[0])
        assert "bar" in calls
        assert "baz" in calls


# ---------------------------------------------------------------------------
# _fallback_split
# ---------------------------------------------------------------------------

class TestFallbackSplit:
    def test_splits_into_parts(self):
        content = "line1\nline2\nline3\nline4\nline5\n"
        chunks = _fallback_split(
            content=content,
            chunk_type="function",
            name="foo",
            file_path="test.py",
            base_start_line=10,
            parent_class=None,
            max_lines=2,
        )
        assert len(chunks) == 3
        assert chunks[0].content == "line1\nline2\n"
        assert chunks[1].content == "line3\nline4\n"
        assert chunks[2].content == "line5\n"

    def test_sets_fallback_flag(self):
        content = "a\nb\nc\nd\n"
        chunks = _fallback_split(
            content=content, chunk_type="function", name="f",
            file_path="x.py", base_start_line=1, parent_class=None, max_lines=2,
        )
        assert all(c.is_fallback_split for c in chunks)

    def test_names_include_part_suffix(self):
        content = "a\nb\nc\n"
        chunks = _fallback_split(
            content=content, chunk_type="function", name="helper",
            file_path="x.py", base_start_line=1, parent_class="Foo", max_lines=1,
        )
        assert chunks[0].name == "helper__part0"
        assert chunks[1].name == "helper__part1"
        assert chunks[2].name == "helper__part2"

    def test_line_numbers(self):
        content = "a\nb\nc\nd\n"
        chunks = _fallback_split(
            content=content, chunk_type="function", name="f",
            file_path="x.py", base_start_line=5, parent_class=None, max_lines=2,
        )
        assert chunks[0].start_line == 5
        assert chunks[0].end_line == 6
        assert chunks[1].start_line == 7
        assert chunks[1].end_line == 8

    def test_single_chunk_when_under_limit(self):
        content = "a\nb\n"
        chunks = _fallback_split(
            content=content, chunk_type="function", name="f",
            file_path="x.py", base_start_line=1, parent_class=None, max_lines=10,
        )
        assert len(chunks) == 1

    def test_parent_class_preserved(self):
        content = "a\nb\n"
        chunks = _fallback_split(
            content=content, chunk_type="method", name="Foo.m",
            file_path="x.py", base_start_line=1, parent_class="Foo", max_lines=1,
        )
        assert all(c.parent_class == "Foo" for c in chunks)


# ---------------------------------------------------------------------------
# _chunk_function
# ---------------------------------------------------------------------------

class TestChunkFunction:
    def test_basic_function(self):
        code = "def foo():\n    pass\n"
        tree = ast.parse(code)
        chunks = _chunk_function(
            tree.body[0], code.splitlines(keepends=True),
            file_path="test.py", parent_class=None, max_lines=100,
        )
        assert len(chunks) == 1
        c = chunks[0]
        assert c.chunk_type == "function"
        assert c.name == "foo"
        assert c.parent_class is None
        assert c.content == code

    def test_async_function(self):
        code = "async def foo():\n    pass\n"
        tree = ast.parse(code)
        chunks = _chunk_function(
            tree.body[0], code.splitlines(keepends=True),
            file_path="test.py", parent_class=None, max_lines=100,
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "async_function"

    def test_method(self):
        code = "def method(self):\n    pass\n"
        tree = ast.parse(code)
        chunks = _chunk_function(
            tree.body[0], code.splitlines(keepends=True),
            file_path="test.py", parent_class="MyClass", max_lines=100,
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "method"
        assert chunks[0].name == "MyClass.method"
        assert chunks[0].parent_class == "MyClass"

    def test_async_method(self):
        code = "async def method(self):\n    pass\n"
        tree = ast.parse(code)
        chunks = _chunk_function(
            tree.body[0], code.splitlines(keepends=True),
            file_path="test.py", parent_class="MyClass", max_lines=100,
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "async_method"

    def test_fallback_split_when_oversized(self):
        lines = [f"x = {i}\n" for i in range(10)]
        code = "def foo():\n    " + "    ".join(lines)
        tree = ast.parse("def foo():\n    " + "    ".join(lines))
        source_lines = code.splitlines(keepends=True)
        chunks = _chunk_function(
            tree.body[0], source_lines,
            file_path="test.py", parent_class=None, max_lines=5,
        )
        assert len(chunks) > 1
        assert chunks[0].is_fallback_split

    def test_decorators_captured(self):
        code = "@staticmethod\n@decorator(arg=1)\ndef foo():\n    pass\n"
        tree = ast.parse(code)
        chunks = _chunk_function(
            tree.body[0], code.splitlines(keepends=True),
            file_path="test.py", parent_class=None, max_lines=100,
        )
        assert "staticmethod" in chunks[0].decorators
        assert any("decorator" in d for d in chunks[0].decorators)

    def test_calls_detected(self):
        code = "def foo():\n    bar()\n    self.baz()\n"
        tree = ast.parse(code)
        chunks = _chunk_function(
            tree.body[0], code.splitlines(keepends=True),
            file_path="test.py", parent_class=None, max_lines=100,
        )
        assert chunks[0].calls == ["bar", "baz"]


# ---------------------------------------------------------------------------
# _chunk_class
# ---------------------------------------------------------------------------

class TestChunkClass:
    def test_class_header_and_methods(self):
        code = textwrap.dedent("""\
            class MyClass:
                \"\"\"My doc.\"\"\"
                def method_a(self):
                    pass
                def method_b(self):
                    pass
        """)
        tree = ast.parse(code)
        source_lines = code.splitlines(keepends=True)
        chunks = _chunk_class(
            tree.body[0], source_lines,
            file_path="test.py", max_lines=100,
        )
        assert len(chunks) == 3
        assert chunks[0].chunk_type == "class"
        assert chunks[0].name == "MyClass"
        assert chunks[1].chunk_type == "method"
        assert chunks[1].name == "MyClass.method_a"
        assert chunks[2].chunk_type == "method"
        assert chunks[2].name == "MyClass.method_b"

    def test_class_variables_in_header(self):
        code = textwrap.dedent("""\
            class MyClass:
                x: int = 1
                y = 2
                def method(self):
                    pass
        """)
        tree = ast.parse(code)
        source_lines = code.splitlines(keepends=True)
        chunks = _chunk_class(
            tree.body[0], source_lines,
            file_path="test.py", max_lines=100,
        )
        header = chunks[0].content
        assert "x: int = 1" in header
        assert "y = 2" in header

    def test_nested_class(self):
        code = textwrap.dedent("""\
            class Outer:
                class Inner:
                    def method(self):
                        pass
        """)
        tree = ast.parse(code)
        source_lines = code.splitlines(keepends=True)
        chunks = _chunk_class(
            tree.body[0], source_lines,
            file_path="test.py", max_lines=100,
        )
        names = [c.name for c in chunks]
        assert "Outer" in names
        assert "Outer.Inner" in names
        assert "Outer.Inner.method" in names

    def test_decorated_class(self):
        code = textwrap.dedent("""\
            @dataclass
            class MyClass:
                def method(self):
                    pass
        """)
        tree = ast.parse(code)
        source_lines = code.splitlines(keepends=True)
        chunks = _chunk_class(
            tree.body[0], source_lines,
            file_path="test.py", max_lines=100,
        )
        assert chunks[0].decorators == ["dataclass"]

    def test_docstring_in_header(self):
        code = textwrap.dedent("""\
            class MyClass:
                \"\"\"Class doc.\"\"\"
                def method(self):
                    pass
        """)
        tree = ast.parse(code)
        source_lines = code.splitlines(keepends=True)
        chunks = _chunk_class(
            tree.body[0], source_lines,
            file_path="test.py", max_lines=100,
        )
        assert chunks[0].docstring == "Class doc."


# ---------------------------------------------------------------------------
# chunk_python_file (integration)
# ---------------------------------------------------------------------------

class TestChunkPythonFile:
    def test_sample_code(self, sample_source):
        chunks = chunk_python_file(sample_source, file_path="services/payment.py")
        assert len(chunks) > 0

    def test_expected_chunk_types(self, sample_source):
        chunks = chunk_python_file(sample_source, file_path="services/payment.py")
        types = {c.chunk_type for c in chunks}
        assert "class" in types
        assert "method" in types
        assert "async_method" in types
        assert "function" in types

    def test_class_has_methods_as_separate_chunks(self, sample_source):
        chunks = chunk_python_file(sample_source)
        method_names = {c.name for c in chunks if c.chunk_type in ("method", "async_method")}
        assert "PaymentService.__init__" in method_names
        assert "PaymentService.process_payment" in method_names
        assert "PaymentService._validate_order" in method_names
        assert "PaymentService._calculate_charge" in method_names
        assert "PaymentService._complete_payment" in method_names

    def test_top_level_function(self, sample_source):
        chunks = chunk_python_file(sample_source)
        funcs = [c for c in chunks if c.chunk_type == "function"]
        assert any(c.name == "format_receipt" for c in funcs)

    def test_empty_source(self):
        chunks = chunk_python_file("")
        assert chunks == []

    def test_only_imports(self):
        chunks = chunk_python_file("import os\nimport sys\n")
        assert chunks == []

    def test_only_constants(self):
        chunks = chunk_python_file("X = 1\nY = 2\n")
        assert chunks == []

    def test_from_path(self, tmp_path):
        f = tmp_path / "test_mod.py"
        f.write_text("def foo():\n    pass\n")
        chunks = chunk_python_file(f)
        assert len(chunks) == 1
        assert chunks[0].name == "foo"
        assert chunks[0].file_path == str(f)

    def test_metadata_populated(self, sample_source):
        chunks = chunk_python_file(sample_source, file_path="services/payment.py")
        for c in chunks:
            assert c.file_path == "services/payment.py"
            assert c.start_line >= 1
            assert c.end_line >= c.start_line
            assert isinstance(c.content, str)
            assert len(c.content) > 0

    def test_max_lines_triggers_fallback(self):
        code = textwrap.dedent("""\
            def foo():
                pass
                pass
                pass
                pass
        """)
        chunks = chunk_python_file(code, max_lines=2)
        assert len(chunks) >= 2
        assert all(c.is_fallback_split for c in chunks[1:]) or chunks[0].is_fallback_split

    def test_syntax_error_raises(self):
        with pytest.raises(SyntaxError):
            chunk_python_file("def foo(:\n    pass\n")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_deeply_nested_classes(self):
        code = textwrap.dedent("""\
            class A:
                class B:
                    class C:
                        def method(self):
                            pass
        """)
        chunks = chunk_python_file(code)
        names = {c.name for c in chunks}
        assert "A" in names
        assert "A.B" in names
        assert "A.B.C" in names
        assert "A.B.C.method" in names

    def test_async_top_level_function(self):
        code = "async def fetch():\n    pass\n"
        chunks = chunk_python_file(code)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "async_function"

    def test_class_with_only_class_vars(self):
        code = textwrap.dedent("""\
            class Config:
                DEBUG = True
                PORT = 8080
        """)
        chunks = chunk_python_file(code)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "class"

    def test_class_with_no_body(self):
        code = "class Empty:\n    pass\n"
        chunks = chunk_python_file(code)
        assert len(chunks) == 1

    def test_docstring_not_consumed_as_method(self):
        code = textwrap.dedent("""\
            class MyClass:
                \"\"\"Docstring.\"\"\"
                pass
        """)
        chunks = chunk_python_file(code)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "class"
