import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    content: str                          # Raw source of this chunk
    chunk_type: str                       # "function" | "async_function" | "class" | "method" | "async_method" | "module_body"
    name: str                             # Qualified name, e.g. "PaymentService.process"
    file_path: str                        # Source file
    start_line: int                       # 1-indexed, inclusive
    end_line: int                         # 1-indexed, inclusive
    parent_class: Optional[str] = None   # Set for methods
    decorators: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    calls: list[str] = field(default_factory=list)   # Names of functions called inside
    is_fallback_split: bool = False       # True if chunk was produced by line fallback

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_class": self.parent_class,
            "decorators": self.decorators,
            "docstring": self.docstring,
            "calls": self.calls,
            "line_count": self.line_count,
            "is_fallback_split": self.is_fallback_split,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_segment(source_lines: list[str], node: ast.AST) -> str:
    """Extract raw source for an AST node using its line number attributes."""
    start = node.lineno - 1        # ast is 1-indexed
    end = node.end_lineno          # end_lineno is inclusive, slice is exclusive
    return "".join(source_lines[start:end])


def _get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Return decorator names as strings."""
    decorators = []
    for d in node.decorator_list:
        if isinstance(d, ast.Name):
            decorators.append(d.id)
        elif isinstance(d, ast.Attribute):
            decorators.append(f"{ast.unparse(d)}")
        else:
            decorators.append(ast.unparse(d))
    return decorators


def _get_docstring(node: ast.AST) -> Optional[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        return None
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value.strip()
    return None


def _get_calls(node: ast.AST) -> list[str]:
    calls = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.add(child.func.attr)
    return sorted(calls)


def _fallback_split(
    content: str,
    chunk_type: str,
    name: str,
    file_path: str,
    base_start_line: int,
    parent_class: Optional[str],
    max_lines: int,
) -> list[CodeChunk]:
    """
    When a single node exceeds max_lines, split it into line-based sub-chunks.
    This is a last resort — the chunks won't be semantically perfect but they'll
    be retrievable and bounded in size.
    """
    lines = content.splitlines(keepends=True)
    chunks = []
    i = 0
    part = 0
    while i < len(lines):
        slice_lines = lines[i : i + max_lines]
        slice_content = "".join(slice_lines)
        abs_start = base_start_line + i
        abs_end = abs_start + len(slice_lines) - 1
        chunks.append(CodeChunk(
            content=slice_content,
            chunk_type=chunk_type,
            name=f"{name}__part{part}",
            file_path=file_path,
            start_line=abs_start,
            end_line=abs_end,
            parent_class=parent_class,
            is_fallback_split=True,
        ))
        i += max_lines
        part += 1
    return chunks

def _chunk_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    file_path: str,
    parent_class: Optional[str],
    max_lines: int,
) -> list[CodeChunk]:
    is_async = isinstance(node, ast.AsyncFunctionDef)
    if parent_class:
        chunk_type = "async_method" if is_async else "method"
        qualified_name = f"{parent_class}.{node.name}"
    else:
        chunk_type = "async_function" if is_async else "function"
        qualified_name = node.name

    content = _source_segment(source_lines, node)
    line_count = node.end_lineno - node.lineno + 1

    base_chunk = CodeChunk(
        content=content,
        chunk_type=chunk_type,
        name=qualified_name,
        file_path=file_path,
        start_line=node.lineno,
        end_line=node.end_lineno,
        parent_class=parent_class,
        decorators=_get_decorators(node),
        docstring=_get_docstring(node),
        calls=_get_calls(node),
    )

    if line_count > max_lines:
        # Fallback: split this oversized function into line-bounded parts
        return _fallback_split(
            content=content,
            chunk_type=chunk_type,
            name=qualified_name,
            file_path=file_path,
            base_start_line=node.lineno,
            parent_class=parent_class,
            max_lines=max_lines,
        )

    return [base_chunk]


def _chunk_class(
    node: ast.ClassDef,
    source_lines: list[str],
    file_path: str,
    max_lines: int,
) -> list[CodeChunk]:
    chunks = []
    class_name = node.name

    # --- Class header chunk ---
    # Reconstruct just the class definition lines (decorators + class line + docstring)
    header_lines = []
    for decorator in node.decorator_list:
        header_lines.extend(
            source_lines[decorator.lineno - 1 : decorator.end_lineno]
        )
    header_lines.append(source_lines[node.lineno - 1])  # the `class Foo:` line

    docstring = _get_docstring(node)
    if docstring and node.body and isinstance(node.body[0], ast.Expr):
        docstring_node = node.body[0]
        header_lines.extend(
            source_lines[docstring_node.lineno - 1 : docstring_node.end_lineno]
        )

    # Also include class-level assignments (e.g. class variables)
    class_var_lines = []
    for child in node.body:
        if isinstance(child, (ast.Assign, ast.AnnAssign)):
            class_var_lines.extend(
                source_lines[child.lineno - 1 : child.end_lineno]
            )

    class_header_content = "".join(header_lines + class_var_lines)
    chunks.append(CodeChunk(
        content=class_header_content,
        chunk_type="class",
        name=class_name,
        file_path=file_path,
        start_line=node.lineno,
        end_line=node.end_lineno,  # logical end of class
        parent_class=None,
        decorators=_get_decorators(node),
        docstring=docstring,
        calls=[],  # class header itself doesn't call things
    ))

    # --- Method chunks (recurse) ---
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(_chunk_function(
                node=child,
                source_lines=source_lines,
                file_path=file_path,
                parent_class=class_name,
                max_lines=max_lines,
            ))
        elif isinstance(child, ast.ClassDef):
            # Nested class — recurse with prefixed name
            nested_chunks = _chunk_class(child, source_lines, file_path, max_lines)
            for c in nested_chunks:
                # Prefix the nested class name with the outer class
                c.name = f"{class_name}.{c.name}"
                if c.parent_class:
                    c.parent_class = f"{class_name}.{c.parent_class}"
            chunks.extend(nested_chunks)

    return chunks


def chunk_python_file(
    source: str | Path,
    file_path: str = "<string>",
    max_lines: int = 100,
) -> list[CodeChunk]:
    if isinstance(source, Path):
        file_path = str(source)
        source = source.read_text(encoding="utf-8")

    tree = ast.parse(source)
    source_lines = source.splitlines(keepends=True)
    chunks: list[CodeChunk] = []

    # Walk only the top-level nodes of the module
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(_chunk_function(
                node=node,
                source_lines=source_lines,
                file_path=file_path,
                parent_class=None,
                max_lines=max_lines,
            ))
        elif isinstance(node, ast.ClassDef):
            chunks.extend(_chunk_class(
                node=node,
                source_lines=source_lines,
                file_path=file_path,
                max_lines=max_lines,
            ))

    return chunks