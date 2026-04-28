import torch
from graph_net_bench.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net_bench.torch.backend.pass_mgr_backend import PassMgrBackend


class _PassMgrFXSerializeWrapper(torch.nn.Module):
    """Wrapper that auto-swaps from torch.compile to direct GraphModule call after warmup."""

    def __init__(self, backend, model):
        super().__init__()
        self._backend = backend
        self._compiled = torch.compile(model, backend=backend._torch_compile_backend)
        self._input_keys = None

    def forward(self, *args, **kwargs):
        if self._backend._optimized_gm is not None:
            if args:
                return self._backend._optimized_gm(*args)
            if self._input_keys is not None:
                positional_inputs = [kwargs[k] for k in self._input_keys]
                return self._backend._optimized_gm(*positional_inputs)
            return self._backend._optimized_gm(**kwargs)
        if not args and kwargs:
            self._input_keys = list(kwargs.keys())
        return self._compiled(*args, **kwargs)


class PassMgrFXSerializeBackend(GraphCompilerBackend):
    """Backend that applies passes via torch.compile, then bypasses wrapper overhead."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._pass_mgr = PassMgrBackend(config)
        self._optimized_gm = None

    def __call__(self, model):
        self._optimized_gm = None
        return _PassMgrFXSerializeWrapper(self, model)

    def _torch_compile_backend(self, gm: torch.fx.GraphModule, sample_inputs: list):
        pass_result = self._pass_mgr.pass_manager(gm)
        if not pass_result.modified:
            raise RuntimeError("[PassMgrFXSerializeBackend] No passes modified the graph.")
        new_gm = self._build_optimized_graph(gm)
        gm_to_return = new_gm if new_gm is not None else pass_result.graph_module
        self._optimized_gm = gm_to_return
        return gm_to_return

    @staticmethod
    def _build_optimized_graph(gm: torch.fx.GraphModule):
        from graph_net_bench.torch.backend.pass_mgr_backend import (
            g_replacement_func,
            with_dispatch_wrapper_run,
        )

        if g_replacement_func is None:
            return None

        dispatch_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is with_dispatch_wrapper_run
        ]
        if not dispatch_nodes:
            return None

        for node in dispatch_nodes:
            node.target = g_replacement_func

        gm.recompile()
        return gm

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
