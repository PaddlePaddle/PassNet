import torch
import inspect
from torch.fx import GraphModule, Graph, Node
from torch.fx.subgraph_rewriter import ReplacedPatterns, _replace_attributes
from torch.fx.passes.utils.matcher_utils import InternalMatch
from typing import Callable, Union, List, Optional, Dict, Set


class ForceArgsTracer(torch.fx.Tracer):
    """
    Custom Tracer that normalizes all function call arguments
    to be purely positional (args) with defaults filled in, eliminating kwargs.
    """

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):

        def _force_empty_kwargs(target, args, kwargs):
            sig = inspect.signature(target)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            normalized_args = tuple(bound_args.args)
            normalized_kwargs = {}  # force empty kwargs
            return normalized_args, normalized_kwargs

        if kind == "call_function" and callable(target):
            try:
                normalized_args, normalized_kwargs = _force_empty_kwargs(
                    target, args, kwargs
                )
                return super().create_node(
                    kind, target, normalized_args, normalized_kwargs, name, type_expr
                )
            except (ValueError, TypeError):
                # backoff to default behavior to prevent crash
                pass

        return super().create_node(kind, target, args, kwargs, name, type_expr)


def force_args_symbolic_trace(root):
    tracer = ForceArgsTracer()
    graph = tracer.trace(root)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return torch.fx.GraphModule(tracer.root, graph, name)


def _replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, Graph, GraphModule],
    replacement: Union[Callable, Graph, GraphModule],
    match_filters: Optional[
        List[Callable[["InternalMatch", Graph, Graph], bool]]
    ] = None,
    ignore_literals: bool = False,
) -> List[ReplacedPatterns]:
    """modified from torch.fx._replace_pattern function to support custom matching."""
    from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch

    if match_filters is None:
        match_filters = []

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph: Graph = gm.graph

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        pattern_graph = force_args_symbolic_trace(pattern).graph

    if isinstance(replacement, GraphModule):
        replacement_graph = replacement.graph
    elif isinstance(replacement, Graph):
        replacement_graph = replacement
    else:
        replacement_graph = force_args_symbolic_trace(replacement).graph

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=ignore_literals,
    )
    _matches: List[InternalMatch] = matcher.match(original_graph)

    # Filter out matches that don't match the filter
    _matches = [
        m
        for m in _matches
        if all(
            match_filter(m, original_graph, pattern_graph)
            for match_filter in match_filters
        )
    ]

    replacement_placeholders = [
        n for n in replacement_graph.nodes if n.op == "placeholder"
    ]

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: Dict[Node, Node] = {}

    match_and_replacements = []
    for match in _matches:

        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
                if gn != val_map[rn]:
                    # Update match.placeholder_nodes and match.nodes_map with the node that replaced gn
                    gn_ind = match.placeholder_nodes.index(gn)
                    match.placeholder_nodes[gn_ind] = match_changed_node[gn]
                    map_key = list(match.nodes_map.keys())[
                        list(match.nodes_map.values()).index(gn)
                    ]
                    match.nodes_map[map_key] = match_changed_node[gn]
            else:
                val_map[rn] = gn

        # Copy the replacement graph over
        user_nodes: Set[Node] = set()
        for n in match.returning_nodes:
            user_nodes.update(n.users)
        assert user_nodes, "The returning_nodes should have at least one user node"

        if len(user_nodes) == 1:
            first_user_node = next(iter(user_nodes))
        else:
            # If there are multiple user nodes, we need to find the first user node
            # in the current execution order of the `original_graph`
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break

        with original_graph.inserting_before(first_user_node):  # type: ignore[possibly-undefined]
            copied_returning_nodes = original_graph.graph_copy(
                replacement_graph, val_map
            )

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes,)

        # Get a list of nodes that have been replaced into the graph
        replacement_nodes: List[Node] = [
            v for v in val_map.values() if v not in match.placeholder_nodes
        ]

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        assert len(match.returning_nodes) == len(copied_returning_nodes)  # type: ignore[arg-type]
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):  # type: ignore[arg-type]
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node
        # Remove the original nodes
        for node in reversed(pattern_graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)

        match_and_replacements.append(
            ReplacedPatterns(
                anchor=match.anchors[0],
                nodes_map=match.nodes_map,
                replacements=replacement_nodes,
            )
        )

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)

    return match_and_replacements
