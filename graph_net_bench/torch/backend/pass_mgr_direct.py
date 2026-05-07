import inspect
import torch
from graph_net_bench.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net_bench.torch.backend.pass_mgr_backend import (
    PassMgrBackend,
    with_dispatch_wrapper_run,
)

import graph_net_bench.torch.backend.pass_mgr_backend as _pass_mgr_backend


def _reorder_placeholders(gm, sample_inputs, param_names, original_input_tensors):
    """Reorder GM placeholders to match the original calling order.

    Dynamo may reorder and rename placeholders (e.g., L_in_3_, L_in_1_).
    This function uses id(tensor) to map sample_inputs back to their
    original calling order and renames them back to original names
    (in_0, in_1, ...) so that gm(*args) works directly with no
    per-call overhead.

    Args:
        gm: FX GraphModule from dynamo
        sample_inputs: the inputs list dynamo passed to the backend.
                       NOTE: this list may be in dynamo's (reordered)
                       order, not the original forward call order.
                       id(tensor) is used to recover the mapping.
    """
    ph_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(ph_nodes) != len(sample_inputs):
        return  # can't reorder if counts don't match

    # Determine desired order: for each param_name, find its tensor in
    # sample_inputs via the original_input_tensors list (which preserves
    # the original forward calling order), then find its placeholder.
    # original_input_tensors[i] corresponds to param_names[i].
    id_to_ph = {id(t): ph for t, ph in zip(sample_inputs, ph_nodes)}
    reordered = [id_to_ph[id(t)] for t in original_input_tensors if id(t) in id_to_ph]
    if len(reordered) != len(ph_nodes):
        return  # can't reorder if some tensors not found

    # Insert new placeholders in the correct order at the beginning
    first_non_ph = next(n for n in gm.graph.nodes if n.op != "placeholder")
    with gm.graph.inserting_before(first_non_ph):
        for name, old_ph in zip(param_names, reordered):
            new_node = gm.graph.placeholder(name)
            old_ph.replace_all_uses_with(new_node)
            gm.graph.erase_node(old_ph)

    gm.recompile()


class PassMgrDirectBackend(GraphCompilerBackend):
    """Backend that applies passes via torch.compile, then returns the
    captured GraphModule directly — no wrapper, no dynamo guard overhead.

    The approach:
      1. torch.compile captures FX graph via dynamo on first call
      2. Apply pattern-replacement passes (via PassMgrBackend)
      3. After dynamo is done, replace with_dispatch_wrapper_run node targets
         with g_replacement_func in the FX graph, then reorder placeholders
      4. From the second call onward, GM is called directly — no dynamo overhead

    Note: The first forward call still goes through dynamo to capture the graph.
    From the second call onward, dynamo is bypassed entirely because the
    returned object IS the GraphModule, not a dynamo-compiled function.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._pass_mgr = PassMgrBackend(config)
        self._optimized_gm = None
        self._sample_inputs = None
        self._param_names = None
        self._original_input_tensors = None

    def __call__(self, model):
        self._optimized_gm = None
        self._sample_inputs = None
        self._param_names = None
        self._original_input_tensors = None
        return _CompileOnceWrapper(self, model)

    def _torch_compile_backend(self, gm: torch.fx.GraphModule, sample_inputs: list):
        pass_result = self._pass_mgr.pass_manager(gm)

        optimized_gm = pass_result.graph_module

        # Save sample_inputs for placeholder reordering (done after dynamo finishes)
        self._sample_inputs = list(sample_inputs)
        self._optimized_gm = optimized_gm
        return optimized_gm

    def _finalize_gm(self):
        """Reorder placeholders after dynamo is done with the GM."""
        if self._optimized_gm is not None and self._sample_inputs is not None:
            # Replace dispatch wrapper targets in FX graph before recompile,
            # so that gm.recompile() inside _reorder_placeholders preserves
            # the replacement (unlike swapping __globals__ which gets overwritten).
            replacement_func = _pass_mgr_backend.g_replacement_func
            if replacement_func is not None:
                for node in self._optimized_gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target is with_dispatch_wrapper_run
                    ):
                        node.target = replacement_func
            _reorder_placeholders(
                self._optimized_gm,
                self._sample_inputs,
                self._param_names,
                self._original_input_tensors,
            )
            self._sample_inputs = None  # only once

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class _CompileOnceWrapper(torch.nn.Module):
    """Triggers dynamo compilation on first call, then hot-swaps forward
    to the bare GM's forward — zero wrapper overhead from the second call on.

    After the first call, self.forward IS gm.forward, so subsequent calls
    go straight to gm.forward(*args) with no branch check or double
    nn.Module.__call__ dispatch.
    """

    def __init__(self, backend, model):
        super().__init__()
        self._backend = backend
        self._backend._param_names = [
            name
            for name, param in inspect.signature(model.forward).parameters.items()
            if name != "self"
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        self._compiled = torch.compile(model, backend=backend._torch_compile_backend)

    def forward(self, *args, **kwargs):
        # First call only: trigger compilation via dynamo.
        # Save original input order for placeholder reordering.
        if self._backend._original_input_tensors is None:
            if args:
                self._backend._original_input_tensors = list(args)
            elif kwargs and self._backend._param_names:
                self._backend._original_input_tensors = [
                    kwargs[k] for k in self._backend._param_names if k in kwargs
                ]
        result = self._compiled(*args, **kwargs)

        if self._backend._optimized_gm is not None:
            # Dynamo is done. Reorder placeholders so gm(*args) works.
            self._backend._finalize_gm()
            gm = self._backend._optimized_gm
            # Hot-swap: replace forward with GM's forward directly.
            # Subsequent calls go to gm.forward(*args) with zero overhead —
            # no if-branch, no double nn.Module.__call__.
            self.__dict__["forward"] = gm.forward
            # Release dynamo-compiled object — no longer needed.
            del self._compiled

        return result
