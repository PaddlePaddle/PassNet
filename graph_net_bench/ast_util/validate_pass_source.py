import ast
from typing import Optional

from .extract_all_import_alias import extract_all_import_alias
from .extract_all_call_targets import extract_all_call_targets

BLOCKED_IMPORT_PREFIXES = [
    "torch.nn",
    "torch.nn.functional",
    "torch.ops",
    "torch.autograd",
]

BLOCKED_CALL_PREFIXES = [
    "torch"
]

ALLOWED_TORCH_CALLS = [
    "torch.empty",
    "torch.empty_like",
    "torch.zeros",
    "torch.zeros_like",
    "torch.ones",
    "torch.ones_like",
    "torch.full",
    "torch.full_like",
    "torch.as_tensor"
]

# Functions whose bodies are allowed to use blocked APIs
EXEMPT_FUNCTION_NAMES = {"pattern", "replacement_args"}


def validate_pass_source(source: str) -> list[str]:
    """
    Viba:
    list[$violation_message str]
      <- $source str
      <- ($tree ast.AST <- $source)
      <- ($import_violations list[str] <- $tree)
      <- ($call_violations list[str] <- $tree)
    """
    tree = ast.parse(source)
    violations = []

    # --- Import blocklist ---
    violations.extend(_check_imports(tree))

    # --- Call-target analysis on non-exempt code ---
    alias_map = _build_alias_map(tree)
    non_exempt_nodes = _extract_non_exempt_nodes(tree)
    violations.extend(_check_call_targets(non_exempt_nodes, alias_map))

    return violations


def _matches_blocked_prefix(name: str) -> bool:
    return any(
        name == p or name.startswith(p + ".")
        for p in BLOCKED_IMPORT_PREFIXES
    )


def _check_imports(tree: ast.AST) -> list[str]:
    """Check for blocked import statements."""
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches_blocked_prefix(alias.name):
                    violations.append(
                        f"Line {node.lineno}: blocked import '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _matches_blocked_prefix(node.module):
                violations.append(
                    f"Line {node.lineno}: blocked from-import "
                    f"'from {node.module} import ...'"
                )
        # Check submodule rename assignments: F = torch.nn.functional
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Attribute):
            continue
        dotted = _resolve_dotted(node.value)
        if dotted and _matches_blocked_prefix(dotted):
            violations.append(
                f"Line {node.lineno}: blocked submodule alias '{dotted}'"
            )
    return violations


def _build_alias_map(tree: ast.AST) -> dict[str, str]:
    """Build a mapping from alias names to their resolved module paths.

    E.g. `import torch.nn.functional as F` -> {"F": "torch.nn.functional"}
         `F = torch.nn.functional`         -> {"F": "torch.nn.functional"}
    """
    alias_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    alias_map[alias.asname] = alias.name
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                key = alias.asname if alias.asname else alias.name
                alias_map[key] = f"{node.module}.{alias.name}"
            continue
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Attribute):
            continue
        dotted = _resolve_dotted(node.value)
        if not dotted:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                alias_map[target.id] = dotted
    return alias_map


def _resolve_dotted(node: ast.expr) -> Optional[str]:
    """Resolve an attribute chain to a dotted string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = [node.attr]
        curr = node.value
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
            return ".".join(reversed(parts))
    return None


def _extract_non_exempt_nodes(tree: ast.AST) -> list[ast.AST]:
    """Extract top-level nodes that are NOT exempt functions (pattern, replacement_args)."""
    nodes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name in EXEMPT_FUNCTION_NAMES:
            continue
        nodes.append(node)
    return nodes


def _is_blocked_call(resolved: str) -> bool:
    if resolved in ALLOWED_TORCH_CALLS:
        return False
    return any(resolved.startswith(p) for p in BLOCKED_CALL_PREFIXES)


def _check_call_targets(
    nodes: list[ast.AST], alias_map: dict[str, str]
) -> list[str]:
    """Check call targets in non-exempt nodes against blocked prefixes."""
    violations = []
    for node in nodes:
        for target, lineno in extract_all_call_targets(node):
            resolved = _resolve_call_target(target, alias_map)
            if _is_blocked_call(resolved):
                violations.append(
                    f"Line {lineno}: blocked call '{target}' "
                    f"(resolves to '{resolved}')"
                )
    return violations


def _resolve_call_target(target: str, alias_map: dict[str, str]) -> str:
    """Resolve a call target using the alias map.

    E.g. if alias_map has F -> torch.nn.functional,
    then F.gelu -> torch.nn.functional.gelu
    """
    root = target.split(".")[0]
    if root in alias_map:
        rest = target[len(root):]
        return alias_map[root] + rest
    return target


def test_main():
    # Case 1: Blocked import
    src1 = "import torch.nn.functional as F"
    res1 = validate_pass_source(src1)
    assert any("blocked import" in v for v in res1), f"Case 1 failed: {res1}"
    print(f"Case 1 passed: {res1}")

    # Case 2: Blocked from-import
    src2 = "from torch.nn.functional import relu"
    res2 = validate_pass_source(src2)
    assert any("blocked from-import" in v for v in res2), f"Case 2 failed: {res2}"
    print(f"Case 2 passed: {res2}")

    # Case 3: Blocked submodule alias
    src3 = """import torch
F = torch.nn.functional
"""
    res3 = validate_pass_source(src3)
    assert any("blocked submodule alias" in v for v in res3), f"Case 3 failed: {res3}"
    print(f"Case 3 passed: {res3}")

    # Case 4: Blocked call in replacement_func via alias
    src4 = """import torch.nn.functional as F

def pattern(x):
    return F.gelu(x)

def replacement_func(x):
    return F.gelu(x)
"""
    res4 = validate_pass_source(src4)
    # Should have import violation + call violation in replacement_func
    call_violations = [v for v in res4 if "blocked call" in v]
    assert len(call_violations) >= 1, f"Case 4 failed: {res4}"
    # The call in pattern() should NOT generate a call violation
    pattern_call_violations = [v for v in call_violations if "Line 4" in v]
    assert len(pattern_call_violations) == 0, f"Case 4 pattern leak: {res4}"
    print(f"Case 4 passed: {res4}")

    # Case 5: Allowed patterns (torch.empty_like, triton, tl)
    src5 = """import torch

def replacement_func(x):
    y = torch.empty_like(x)
    z = triton_add(x, y)
    return z
"""
    res5 = validate_pass_source(src5)
    assert len(res5) == 0, f"Case 5 failed (should be clean): {res5}"
    print(f"Case 5 passed: {res5}")

    # Case 6: Direct torch.nn.functional.relu call (no alias)
    src6 = """import torch

def replacement_func(x):
    return torch.nn.functional.relu(x)
"""
    res6 = validate_pass_source(src6)
    call_violations6 = [v for v in res6 if "blocked call" in v]
    assert len(call_violations6) >= 1, f"Case 6 failed: {res6}"
    print(f"Case 6 passed: {res6}")

    # Case 7: torch.ops call
    src7 = """import torch

def replacement_func(x, y):
    return torch.ops.aten.add(x, y)
"""
    res7 = validate_pass_source(src7)
    call_violations7 = [v for v in res7 if "blocked call" in v]
    assert len(call_violations7) >= 1, f"Case 7 failed: {res7}"
    print(f"Case 7 passed: {res7}")

    # Case 8: Demo pass file should be clean (torch.add is allowed)
    src8 = """import torch

def pattern(x, w):
    return torch.nn.functional.relu(torch.nn.functional.linear(x, w))

def replacement_args(x, w):
    return (x, w)

def replacement_func():
    return torch.add
"""
    res8 = validate_pass_source(src8)
    assert len(res8) == 0, f"Case 8 failed (demo pass should be clean): {res8}"
    print(f"Case 8 passed: {res8}")

    print("All validate_pass_source tests passed.")


if __name__ == "__main__":
    test_main()
