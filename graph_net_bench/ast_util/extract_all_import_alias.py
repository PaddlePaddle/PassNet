import ast


def extract_all_import_alias(tree: ast.AST) -> list[tuple[str, int]]:
    """
    Viba:
    AliasedModuleAndLineno := list[$python_aliased_module_name str * $lineno_first_occur int]

    extract_all_import_alias :=
      AliasedModuleAndLineno
      <- $tree ast.AST
      <- ($python_module_name_in_from_xxx_import_yyy AliasedModuleAndLineno <- $tree)
      <- ($python_module_name_in_import_xxx_as_yyy AliasedModuleAndLineno <- $tree)
      <- ($python_module_name_for_submodule_rename_assign AliasedModuleAndLineno <- $tree)
      <- ($merged <- $list_a <- $list_b <- $list_c)
    """

    def get_from_root_modules(t: ast.AST) -> list[tuple[str, int]]:
        # 'from xxx.yyy import zzz' -> 'xxx'
        return [
            (node.module.split(".")[0], node.lineno)
            for node in ast.walk(t)
            if isinstance(node, ast.ImportFrom) and node.module
        ]

    def get_aliased_imports(t: ast.AST) -> list[tuple[str, int]]:
        # 'import numpy as np' -> 'numpy'
        return [
            (alias.name, node.lineno)
            for node in ast.walk(t)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.asname
        ]

    def get_submodule_rename_assigns(t: ast.AST) -> list[tuple[str, int]]:
        # 'F = torch.nn.functional' -> 'torch'
        results = []
        for node in ast.walk(t):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
                curr = node.value
                while isinstance(curr, ast.Attribute):
                    curr = curr.value
                if isinstance(curr, ast.Name):
                    results.append((curr.id, node.lineno))
        return results

    def merge_first_occurrences(*lists: list) -> list[tuple[str, int]]:
        # Layer 2: Merge and ensure uniqueness based on first lineno
        registry = {}
        for sublist in lists:
            for name, line in sublist:
                if name and name not in registry:
                    registry[name] = line
        return sorted(registry.items(), key=lambda x: x[1])

    # Resolve Exponent logic (<-)
    from_list = get_from_root_modules(tree)
    alias_list = get_aliased_imports(tree)
    assign_list = get_submodule_rename_assigns(tree)

    return merge_first_occurrences(from_list, alias_list, assign_list)


def test_main():
    # Case 1: Root module from dotted 'from' import
    src1 = "from os.path import join"  # Expected: [('os', 1)]

    # Case 2: Aliased import returns the original module name
    src2 = "import numpy as np"  # Expected: [('numpy', 1)]

    # Case 3: Submodule rename assignment
    src3 = "import torch; F = torch.nn.functional"  # Expected: [('torch', 2)]

    # Case 4: Ignored standard imports and Duplicates
    src4 = """
import os
from math.constants import pi
import numpy as np
G = torch.nn.utils
    """  # Expected: [('math', 3), ('numpy', 4), ('torch', 5)]

    def run_check(source, expected_keys, ignored_keys, label):
        t = ast.parse(source)
        res = extract_all_import_alias(t)
        found = [item[0] for item in res]

        for key in expected_keys:
            assert key in found, f"[{label}] Expected {key} missing from {found}"
        for key in ignored_keys:
            assert key not in found, f"[{label}] {key} should have been ignored"

        print(f"{label} passed: {res}")

    print("Executing Viba test suite...")
    run_check(src1, ["os"], [], "From-Import Root")
    run_check(src2, ["numpy"], ["np"], "Aliased Import Original Name")
    run_check(src3, ["torch"], ["F"], "Submodule Rename")
    run_check(src4, ["math", "numpy", "torch"], ["os"], "Ignored and Mixed Patterns")


if __name__ == "__main__":
    test_main()
