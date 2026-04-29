import inspect
import torch
from graph_net_bench.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net_bench.torch.backend.pass_mgr_backend import (
    PassMgrBackend,
    with_dispatch_wrapper_run,
)

import graph_net_bench.torch.backend.pass_mgr_backend as _pass_mgr_backend


def _reorder_placeholders(gm, sample_inputs, param_names):
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
    ph_nodes = [n for n in gm.graph.nodes if n.op == 'placeholder']
    if len(ph_nodes) != len(sample_inputs):
        return  # can't reorder if counts don't match

    # Build id(tensor) → placeholder node mapping
    id_to_ph = {id(t): ph for t, ph in zip(sample_inputs, ph_nodes)}

    # Determine desired order: for each sample_input, find its placeholder
    reordered = [id_to_ph[id(t)] for t in sample_inputs if id(t) in id_to_ph]
    if len(reordered) != len(ph_nodes):
        return  # can't reorder if some tensors not found

    # Insert new placeholders in the correct order at the beginning
    first_non_ph = next(n for n in gm.graph.nodes if n.op != 'placeholder')
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
      3. Swap with_dispatch_wrapper_run → replacement_func in GM's forward globals
      4. After dynamo is done, reorder GM placeholders to match positional args order
      5. From the second call onward, GM is called directly — no dynamo overhead

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

    def __call__(self, model):
        self._optimized_gm = None
        self._sample_inputs = None
        self._param_names = None
        return _CompileOnceWrapper(self, model)

    def _torch_compile_backend(self, gm: torch.fx.GraphModule, sample_inputs: list):
        pass_result = self._pass_mgr.pass_manager(gm)
        if not pass_result.modified:
            raise RuntimeError("[PassMgrDirectBackend] No passes modified the graph.")

        optimized_gm = pass_result.graph_module

        # Replace dispatch wrapper with the real kernel in GM's forward globals.
        replacement_func = _pass_mgr_backend.g_replacement_func
        if replacement_func is not None:
            fwd_globals = optimized_gm.forward.__globals__
            for name, obj in list(fwd_globals.items()):
                if obj is with_dispatch_wrapper_run:
                    fwd_globals[name] = replacement_func
                    break

        # Save sample_inputs for placeholder reordering (done after dynamo finishes)
        self._sample_inputs = list(sample_inputs)
        self._optimized_gm = optimized_gm
        return optimized_gm

    def _finalize_gm(self):
        """Reorder placeholders after dynamo is done with the GM."""
        if self._optimized_gm is not None and self._sample_inputs is not None:
            _reorder_placeholders(self._optimized_gm, self._sample_inputs, self._param_names)
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
            name for name, param in inspect.signature(model.forward).parameters.items()
            if name != 'self' and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        self._compiled = torch.compile(model, backend=backend._torch_compile_backend)

    def forward(self, *args, **kwargs):
        # First call only: trigger compilation via dynamo.
        result = self._compiled(*args, **kwargs)

        if self._backend._optimized_gm is not None:
            # Dynamo is done. Reorder placeholders so gm(*args) works.
            self._backend._finalize_gm()
            gm = self._backend._optimized_gm
            # Hot-swap: replace forward with GM's forward directly.
            # Subsequent calls go to gm.forward(*args) with zero overhead —
            # no if-branch, no double nn.Module.__call__.
            self.__dict__['forward'] = gm.forward
            # Release dynamo-compiled object — no longer needed.
            del self._compiled

        return result
