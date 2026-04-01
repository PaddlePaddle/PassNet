import importlib.util
import sys
from pathlib import Path
from unittest import mock

_mod_path = Path(__file__).resolve().parents[1] / "ai4c_agent/tools/file_editor.py"
_spec = importlib.util.spec_from_file_location("file_editor", _mod_path)
_mod = importlib.util.module_from_spec(_spec)

# file_editor.py replaces sys.stdout at module level with io.TextIOWrapper.
# Suppress that replacement so pytest's capture infrastructure is not disrupted.
with mock.patch("io.TextIOWrapper", side_effect=lambda *a, **kw: sys.stdout):
    _spec.loader.exec_module(_mod)

StrReplaceEditor = _mod.StrReplaceEditor
EditorError = _mod.EditorError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_editor():
    return StrReplaceEditor(file_history={})


# ---------------------------------------------------------------------------
# view – file
# ---------------------------------------------------------------------------

def test_view_py_file(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("x = 1\ny = 2\n", encoding="utf-8")
    editor = make_editor()
    result = editor.run("view", str(f))
    assert result.error == ""
    assert "x = 1" in result.output
    assert "y = 2" in result.output


def test_view_non_py_file_python_only_false(tmp_path):
    f = tmp_path / "readme.txt"
    f.write_text("hello world\n", encoding="utf-8")
    editor = make_editor()
    result = editor.run("view", str(f), python_only=False)
    assert result.error == ""
    assert "hello world" in result.output


def test_view_non_py_file_python_only_true_returns_error(tmp_path):
    f = tmp_path / "readme.txt"
    f.write_text("hello world\n", encoding="utf-8")
    editor = make_editor()
    result = editor.run("view", str(f), python_only=True)
    assert result.error != ""
    assert "not a .py file" in result.error


def test_view_with_range(tmp_path):
    f = tmp_path / "sample.py"
    lines = "\n".join(f"line{i}" for i in range(1, 11))
    f.write_text(lines, encoding="utf-8")
    editor = make_editor()
    result = editor.run("view", str(f), view_range=[3, 5])
    assert result.error == ""
    assert "line3" in result.output
    assert "line5" in result.output
    assert "line1" not in result.output
    assert "line6" not in result.output


def test_view_invalid_range_returns_error(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    editor = make_editor()
    result = editor.run("view", str(f), view_range=[10, 20])
    assert result.error != ""


# ---------------------------------------------------------------------------
# view – directory
# ---------------------------------------------------------------------------

def test_view_directory_python_only_false(tmp_path):
    (tmp_path / "a.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("text\n", encoding="utf-8")
    editor = make_editor()
    result = editor.run("view", str(tmp_path), python_only=False)
    assert result.error == ""
    assert "b.txt" in result.output


def test_view_directory_python_only_true_excludes_non_py(tmp_path):
    (tmp_path / "a.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("text\n", encoding="utf-8")
    editor = make_editor()
    result = editor.run("view", str(tmp_path), python_only=True)
    assert result.error == ""
    assert "a.py" in result.output
    assert "b.txt" not in result.output


# ---------------------------------------------------------------------------
# python_only defaults to False
# ---------------------------------------------------------------------------

def test_python_only_default_is_false(tmp_path):
    """run() default python_only=False: non-.py files must be viewable."""
    f = tmp_path / "data.json"
    f.write_text('{"key": "value"}\n', encoding="utf-8")
    editor = make_editor()
    # Call without specifying python_only – relies on the default
    result = editor.run("view", str(f))
    assert result.error == ""
    assert "key" in result.output


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

def test_create_new_file(tmp_path):
    f = tmp_path / "new.py"
    editor = make_editor()
    result = editor.run("create", str(f), file_text="print('hi')\n")
    assert result.error == ""
    assert f.exists()
    assert f.read_text(encoding="utf-8") == "print('hi')\n"


def test_create_existing_file_raises(tmp_path):
    f = tmp_path / "existing.py"
    f.write_text("old\n", encoding="utf-8")
    editor = make_editor()
    try:
        editor.run("create", str(f), file_text="new\n")
        assert False, "Expected EditorError"
    except EditorError as e:
        assert "already exists" in str(e)


# ---------------------------------------------------------------------------
# str_replace
# ---------------------------------------------------------------------------

def test_str_replace_basic(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("foo = 1\nbar = 2\n", encoding="utf-8")
    editor = make_editor()
    result = editor.run("str_replace", str(f), old_str="foo = 1", new_str="foo = 99")
    assert result.error == ""
    assert "foo = 99" in f.read_text(encoding="utf-8")


def test_str_replace_not_found_raises(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("foo = 1\n", encoding="utf-8")
    editor = make_editor()
    try:
        editor.run("str_replace", str(f), old_str="does_not_exist", new_str="x")
        assert False, "Expected EditorError"
    except EditorError as e:
        assert "No occurrences" in str(e)


def test_str_replace_multiple_occurrences_raises(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\nx = 1\n", encoding="utf-8")
    editor = make_editor()
    try:
        editor.run("str_replace", str(f), old_str="x = 1", new_str="x = 2")
        assert False, "Expected EditorError"
    except EditorError as e:
        assert "Multiple occurrences" in str(e)


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------

def test_insert_line(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("line1\nline3\n", encoding="utf-8")
    editor = make_editor()
    result = editor.run("insert", str(f), insert_line=1, new_str="line2")
    assert result.error == ""
    content = f.read_text(encoding="utf-8")
    assert "line2" in content
    lines = content.splitlines()
    assert lines[1] == "line2"


def test_insert_invalid_line_raises(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("a\n", encoding="utf-8")
    editor = make_editor()
    try:
        editor.run("insert", str(f), insert_line=999, new_str="x")
        assert False, "Expected EditorError"
    except EditorError as e:
        assert "Invalid insert_line" in str(e)


# ---------------------------------------------------------------------------
# undo_edit
# ---------------------------------------------------------------------------

def test_undo_edit(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("original\n", encoding="utf-8")
    editor = make_editor()
    editor.run("str_replace", str(f), old_str="original", new_str="changed")
    assert "changed" in f.read_text(encoding="utf-8")
    editor.run("undo_edit", str(f))
    assert "original" in f.read_text(encoding="utf-8")


def test_undo_edit_no_history_raises(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x\n", encoding="utf-8")
    editor = make_editor()
    try:
        editor.run("undo_edit", str(f))
        assert False, "Expected EditorError"
    except EditorError as e:
        assert "No previous edits" in str(e)


# ---------------------------------------------------------------------------
# validate_path
# ---------------------------------------------------------------------------

def test_view_nonexistent_path_raises(tmp_path):
    editor = make_editor()
    try:
        editor.run("view", str(tmp_path / "ghost.py"))
        assert False, "Expected EditorError"
    except EditorError as e:
        assert "does not exist" in str(e)


def test_str_replace_on_directory_raises(tmp_path):
    editor = make_editor()
    try:
        editor.run("str_replace", str(tmp_path), old_str="x", new_str="y")
        assert False, "Expected EditorError"
    except EditorError as e:
        assert "directory" in str(e)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v", "--capture=no"]))
