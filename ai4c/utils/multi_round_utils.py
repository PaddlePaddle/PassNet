from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from typing import Optional, Union


def truncate_text(s: Optional[str], max_chars: int, *, suffix: str = "\n... <truncated> ...") -> str:
    """
    Truncate a string to at most max_chars characters.
    - Returns "" if s is None.
    - Keeps behavior stable across callers by centralizing truncation.
    """
    if s is None:
        return ""
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + suffix


def read_text(path: Union[str, Path], *, encoding: str = "utf-8") -> str:
    """Read a UTF-8 text file (small helper to avoid repeating open(...))."""
    p = Path(path)
    return p.read_text(encoding=encoding)


def sh_quote(s: str) -> str:
    """Single-quote a string for safe usage in a POSIX shell command."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def parse_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def tail_lines(text: str, *, max_lines: int = 200) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def read_text_if_exists(path: Optional[str], *, max_chars: int = 20000) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    s = p.read_text(errors="replace")
    if len(s) > max_chars:
        s = s[:max_chars] + "\n... <truncated> ..."
    return s


def parse_rectified_speedup(score_path: Optional[str]) -> Optional[float]:
    if not score_path:
        return None
    p = Path(score_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(errors="replace"))
    except Exception:
        return None
    v = data.get("score")
    return parse_float(v) if not isinstance(v, (int, float)) else float(v)


