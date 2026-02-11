import ast
from typing import Optional


def extract_all_call_targets(tree: ast.AST) -> list[tuple[str, int]]:
    """
    Viba:
    CallTargetAndLineno := list[$dotted_call_target str * $lineno int]

    extract_all_call_targets :=
      CallTargetAndLineno
      <- $tree ast.AST
      <- ($all_call_nodes list[ast.Call] <- $tree)
      <- ($resolve_callee str <- $call_node ast.Call)
    """

    def resolve_callee(node: ast.expr) -> Optional[str]:
        # Resolve a call's func node to a dotted string
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

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = resolve_callee(node.func)
        if target is not None:
            results.append((target, node.lineno))
    return results


def test_main():
    # Case 1: Simple function call
    src1 = "relu(x)"
    tree1 = ast.parse(src1)
    res1 = extract_all_call_targets(tree1)
    assert res1 == [("relu", 1)], f"Case 1 failed: {res1}"
    print(f"Case 1 passed: {res1}")

    # Case 2: Single-level attribute call
    src2 = "F.gelu(x)"
    tree2 = ast.parse(src2)
    res2 = extract_all_call_targets(tree2)
    assert res2 == [("F.gelu", 1)], f"Case 2 failed: {res2}"
    print(f"Case 2 passed: {res2}")

    # Case 3: Deep dotted call
    src3 = "torch.nn.functional.relu(x)"
    tree3 = ast.parse(src3)
    res3 = extract_all_call_targets(tree3)
    assert res3 == [("torch.nn.functional.relu", 1)], f"Case 3 failed: {res3}"
    print(f"Case 3 passed: {res3}")

    # Case 4: Multiple calls with line numbers
    src4 = """y = torch.relu(x)
z = my_kernel(y)
w = torch.ops.aten.add(y, z)
"""
    tree4 = ast.parse(src4)
    res4 = extract_all_call_targets(tree4)
    targets = [(t, l) for t, l in res4]
    assert ("torch.relu", 1) in targets, f"Case 4a failed: {targets}"
    assert ("my_kernel", 2) in targets, f"Case 4b failed: {targets}"
    assert ("torch.ops.aten.add", 3) in targets, f"Case 4c failed: {targets}"
    print(f"Case 4 passed: {res4}")

    # Case 5: Nested calls
    src5 = "torch.relu(F.gelu(x))"
    tree5 = ast.parse(src5)
    res5 = extract_all_call_targets(tree5)
    targets5 = [t for t, _ in res5]
    assert "torch.relu" in targets5
    assert "F.gelu" in targets5
    print(f"Case 5 passed: {res5}")

    print("All extract_all_call_targets tests passed.")


if __name__ == "__main__":
    test_main()
