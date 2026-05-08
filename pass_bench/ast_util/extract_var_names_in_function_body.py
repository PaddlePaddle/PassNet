import ast


def extract_var_names_in_function_body(
    tree: ast.AST, ignored_global_function_name: list[str]
) -> list[tuple[str, int]]:
    """
    Viba:
    list[$var_name str * $linno_first_occur int]
      <- $tree ast.AST
      <- $ignored_global_function_name list[str]
      <- ($global_function_def_nodes list[ast.FunctionDef] <- $tree <- $ignored_global_function_name)
      <- ($all_ast_name_node_in_function_body list[ast.Name] <- $global_function_def_nodes)
    """

    def select_functions(t: ast.AST, ignores: list[str]) -> list[ast.FunctionDef]:
        # Filter top-level functions not in the ignore list
        return [
            n
            for n in ast.iter_child_nodes(t)
            if isinstance(n, ast.FunctionDef) and n.name not in ignores
        ]

    def extract_from_nodes(nodes: list[ast.FunctionDef]) -> list[tuple[str, int]]:
        # Layer 2: Extract unique names with their first line occurrence
        registry = {}  # name -> linno

        for fn in nodes:
            # Step 1: Map parameters to exclude them from the body-only results
            params = {a.arg for a in fn.args.args}

            # Step 2: Iterate body nodes and filter
            process_body(fn.body, params, registry)

        return list(registry.items())

    def process_body(body: list[ast.stmt], params: set[str], registry: dict):
        # Flattened walker to minimize indentation depth
        for node in (n for stmt in body for n in ast.walk(stmt)):
            if not isinstance(node, ast.Name):
                continue

            name = node.id
            # Exclude params, ignored globals, and duplicates (keep first occur)
            if name in params or name in ignored_global_function_name:
                continue
            if name not in registry:
                registry[name] = node.lineno

    # Logic execution flow matching AdtExpr <-
    target_nodes = select_functions(tree, ignored_global_function_name)

    if not target_nodes:
        return []

    return extract_from_nodes(target_nodes)


def test_main():
    # Case 1: Standard extraction with parameters exclusion
    src1 = """
def calculate(factor):
    base = 10
    total = base * factor
    return total
    """
    # Case 2: Multiple functions and global ignore
    src2 = """
def helper():
    temp = 1
    return temp

def main_task(x):
    y = x + external_val
    return y
    """
    # Case 3: Redefined variables (should only capture first line)
    src3 = """
def redefinition():
    val = 1
    val = 2
    return val
    """

    # Test execution
    def run_test(source, ignore, expected_names):
        tree = ast.parse(source)
        results = extract_var_names_in_function_body(tree, ignore)
        names = [pair[0] for pair in results]
        for name in expected_names:
            assert name in names, f"Expected {name} in {names}"
        print(f"Passed: {names}")

    print("Running Test Case 1...")
    # 'factor' is a param, 'calculate' is the target. 'base', 'total' are vars.
    run_test(src1, [], ["base", "total"])

    print("Running Test Case 2...")
    # Ignore 'helper' function and 'external_val' global.
    # Should pick up 'y' from main_task. 'x' is param.
    run_test(src2, ["helper", "external_val"], ["y"])

    print("Running Test Case 3...")
    tree3 = ast.parse(src3)
    res3 = extract_var_names_in_function_body(tree3, [])
    # 'val' appears on line 3 and 4. registry should keep line 3.
    assert len(res3) == 1
    assert res3[0][0] == "val"
    print(f"Passed redefinition: {res3}")


if __name__ == "__main__":
    test_main()
