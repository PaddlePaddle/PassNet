import torch
import torch.fx as fx
from graph_net_bench.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net_bench.torch.backend.pass_mgr_backend import (
    PassMgrBackend,
    with_dispatch_wrapper_run,
)

# NOTE: g_replacement_func is a module-level mutable singleton in pass_mgr_backend.
# We must access it dynamically via the module object because
#   from pass_mgr_backend import g_replacement_func
# creates a snapshot binding; mutations inside pass_mgr_backend are NOT visible here.
import graph_net_bench.torch.backend.pass_mgr_backend as _pass_mgr_backend


class _PassMgrFXSerializeWrapper(torch.nn.Module):
    """Wrapper that delegates to torch.compile on the first call, then switches
    to the serialized standalone module for all subsequent calls.

    On the first forward call, dynamo captures the FX graph and invokes
    _torch_compile_backend, which applies passes and produces a serialized
    nn.Module. From the second call onward, forward goes directly to the
    serialized module, completely bypassing dynamo/torch.compile overhead.
    """

    def __init__(self, backend, model):
        super().__init__()
        self._backend = backend
        self._compiled = torch.compile(model, backend=backend._torch_compile_backend)
        self._serialized = None

    def forward(self, *args, **kwargs):
        if self._serialized is not None:
            return self._serialized(*args, **kwargs)

        # First call: dynamo captures graph, backend serializes it.
        result = self._compiled(*args, **kwargs)

        # After backend runs, _optimized_gm holds the serialized module.
        # Switch to it for all future calls (bypass dynamo entirely).
        if self._backend._optimized_gm is not None:
            self._serialized = self._backend._optimized_gm

        return result


class PassMgrFXSerializeBackend(GraphCompilerBackend):
    """Backend that applies passes via torch.compile, then serializes optimized FX graph to a standalone module.

    The serialization process:
      1. Run pattern-replacement passes on the FX graph (via PassMgrBackend)
      2. Replace dispatch wrapper nodes with the real kernel function
      3. Export the optimized GraphModule's forward code via gm.graph.python_code()
      4. Use exec() with pc.globals (same mechanism FX uses internally for
         GraphModule.recompile) to create a standalone nn.Module class
      5. Restore state_dict (parameters and buffers)

    After the first torch.compile call, subsequent invocations go directly
    to the serialized module, completely bypassing dynamo guard overhead.
    """

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

        # Replace dispatch wrapper with direct function calls before serialization
        # so the serialized module has no runtime dependency on global state.
        optimized_gm = pass_result.graph_module
        replacement_func = _pass_mgr_backend.g_replacement_func
        if replacement_func is not None:
            for node in optimized_gm.graph.nodes:
                if node.op == "call_function" and node.target is with_dispatch_wrapper_run:
                    node.target = replacement_func
            optimized_gm.recompile()

        serialized_module = self._serialize_graph_module(optimized_gm)
        self._optimized_gm = serialized_module
        return serialized_module

    # ============================================================
    # Serialization helpers
    # ============================================================

    @staticmethod
    def _serialize_graph_module(gm: torch.fx.GraphModule) -> torch.nn.Module:
        """Serialize an FX GraphModule to a standalone torch.nn.Module.

        Uses the same mechanism as FX's own GraphModule.recompile():
        python_code() returns (src, globals), then exec() compiles the forward
        method with the globals dict providing all external references (torch
        builtins, custom kernels, etc.). No import statement generation or
        .py file writing needed.

        Returns:
            A new torch.nn.Module instance that executes the same forward logic
            without any FX or dynamo dependencies.
        """
        # 1. Save state_dict for parameters/buffers
        state_dict = gm.state_dict()

        # 2. Get Python source and auto-collected globals from FX
        pc = gm.graph.python_code(root_module="self")

        # 3. Build exec globals from pc.globals (same as FX's _exec_with_source)
        globs = dict(pc.globals)

        # 4. Handle double-underscore-prefixed global names.
        #    Python name mangling inside class methods turns __xxx into
        #    _ClassName__xxx, breaking global references. Replace with safe aliases.
        alias_map = {}
        for name in list(globs.keys()):
            if name.startswith("__"):
                safe_name = "_fxw_" + name.lstrip("_")
                alias_map[name] = safe_name
                globs[safe_name] = globs.pop(name)

        # 5. Prepare forward source: strip FX wrap declarations, apply alias map
        forward_src = pc.src
        lines = []
        for line in forward_src.split("\n"):
            if line.strip().startswith("torch.fx._symbolic_trace.wrap("):
                continue
            lines.append(line)
        clean_src = "\n".join(lines)

        if alias_map:
            for old_name, safe_name in alias_map.items():
                clean_src = clean_src.replace(old_name, safe_name)

        # 6. Build class source and exec
        class_src = "class _SerializedGraphModule(torch.nn.Module):\n"
        for line in clean_src.split("\n"):
            if line.strip() == "":
                class_src += "\n"
            else:
                class_src += "    " + line + "\n"

        exec(compile(class_src, "<fx_serialize>", "exec"), globs)
        cls = globs["_SerializedGraphModule"]

        # 7. Instantiate and restore state_dict
        model = cls()
        for key, value in state_dict.items():
            is_param = isinstance(value, torch.nn.Parameter)
            if not is_param and hasattr(value, "requires_grad") and value.requires_grad:
                is_param = True
            if is_param:
                model.register_parameter(key, torch.nn.Parameter(torch.empty_like(value)))
            else:
                model.register_buffer(key, torch.empty_like(value))
        model.load_state_dict(state_dict)

        return model

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
