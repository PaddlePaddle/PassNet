import torch
import torch.fx as fx
import tempfile
import os
import sys
import importlib.util
from pathlib import Path
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
    """Wrapper that delegates to torch.compile; the backend returns the serialized module directly.

    Dynamo caches the returned serialized_module after the first compilation,
    so subsequent calls go straight to the standalone module with only minimal
    guard overhead (~5us).
    """

    def __init__(self, backend, model):
        super().__init__()
        self._compiled = torch.compile(model, backend=backend._torch_compile_backend)

    def forward(self, *args, **kwargs):
        return self._compiled(*args, **kwargs)


class PassMgrFXSerializeBackend(GraphCompilerBackend):
    """Backend that applies passes via torch.compile, then serializes optimized FX graph to a standalone module.

    The serialization process:
      1. Run pattern-replacement passes on the FX graph
      2. Replace dispatch wrapper nodes with the real kernel function
      3. Collect custom call_function targets from the graph nodes directly
      4. Export the optimized GraphModule's forward code via print_readable()
      5. Generate a standalone Python module with safe aliases for injected kernels
      6. Save/restore state_dict (parameters and buffers)
      7. Return the deserialized standalone module for direct invocation

    This completely bypasses dynamo/torch.compile overhead after the first call.
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

        # Collect all custom call_function targets that the serialized module will need.
        # We walk the FX graph nodes directly, so there is no need to parse string
        # qualnames or scan FX wrap declarations.
        injected_functions = PassMgrFXSerializeBackend._collect_graph_targets(optimized_gm)

        serialized_module = self._serialize_graph_module(optimized_gm, injected_functions)
        self._optimized_gm = serialized_module
        return serialized_module

    # ============================================================
    # Serialization helpers
    # ============================================================

    @staticmethod
    def _get_fx_qualname(target) -> str:
        """Compute the qualname string that FX print_readable uses for a callable.

        FX joins __module__ and __qualname__ with underscores, e.g.:
            add_fuse.fused_add  ->  "add_fuse_fused_add"
        """
        mod = getattr(target, "__module__", "")
        name = getattr(target, "__qualname__", getattr(target, "__name__", ""))
        return (mod + "." + name).replace(".", "_")

    @staticmethod
    def _collect_graph_targets(gm: torch.fx.GraphModule) -> dict:
        """Collect all custom call_function targets from the graph.

        Returns a mapping:
            { "safe_alias": <callable>, ... }
        """
        injected = {}
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if not callable(target):
                continue
            mod = getattr(target, "__module__", "")
            # Skip torch builtins and standard library
            if mod in ("", "builtins", "torch") or mod.startswith("torch."):
                continue
            qualname = PassMgrFXSerializeBackend._get_fx_qualname(target)
            safe_alias = PassMgrFXSerializeBackend._to_safe_alias(qualname)
            injected[safe_alias] = target
        return injected

    @staticmethod
    def _serialize_graph_module(
        gm: torch.fx.GraphModule, injected_functions: dict = None
    ) -> torch.nn.Module:
        """Serialize an FX GraphModule to a standalone torch.nn.Module via source code generation.

        Args:
            gm: The optimized FX GraphModule to serialize.
            injected_functions: Mapping of safe_alias -> callable to inject.

        Returns:
            A new torch.nn.Module instance that executes the same forward logic
            without any FX or dynamo dependencies.
        """
        # 1. Save state_dict for parameters/buffers
        state_dict = gm.state_dict()

        # 2. Get readable Python source from the FX graph
        readable = gm.print_readable(print_output=False)

        # 3. Determine custom callables that must be injected into the deserialized module.
        funcs_to_inject = injected_functions if injected_functions is not None else {}
        if not funcs_to_inject:
            funcs_to_inject = PassMgrFXSerializeBackend._collect_graph_targets(gm)

        # 4. Write to a temporary Python file
        tmpdir = tempfile.mkdtemp()
        module_path = os.path.join(tmpdir, "_fx_serialized_module.py")
        state_path = os.path.join(tmpdir, "_fx_state_dict.pt")

        torch.save(state_dict, state_path)

        # Build alias map: every injected function gets a safe Python identifier
        alias_map = {
            qualname: PassMgrFXSerializeBackend._to_safe_alias(qualname)
            for qualname in funcs_to_inject.keys()
        }

        # Build source: rename class and replace qualnames with safe aliases
        full_source = "import torch\n\n"
        full_source += PassMgrFXSerializeBackend._rename_class(readable, alias_map)
        full_source += (
            "\n"
            f"_STATE_PATH = {repr(state_path)}\n"
            "\n"
            "def _load_state_and_create():\n"
            "    model = SerializedGraphModule()\n"
            "    state = torch.load(_STATE_PATH)\n"
            "    for key, value in state.items():\n"
            "        is_param = isinstance(value, torch.nn.Parameter)\n"
            "        if not is_param and hasattr(value, 'requires_grad'):\n"
            "            is_param = value.requires_grad\n"
            "        if is_param:\n"
            "            model.register_parameter(key, torch.nn.Parameter(torch.empty_like(value)))\n"
            "        else:\n"
            "            model.register_buffer(key, torch.empty_like(value))\n"
            "    model.load_state_dict(state)\n"
            "    return model\n"
        )

        with open(module_path, "w") as f:
            f.write(full_source)

        # 5. Dynamically import the module
        module_name = "_fx_serialized_module_" + os.path.basename(tmpdir)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 6. Inject custom callable objects directly into module namespace
        for safe_alias, func_obj in funcs_to_inject.items():
            module.__dict__[safe_alias] = func_obj

        # 7. Instantiate and restore state
        serialized_model = module._load_state_and_create()
        return serialized_model

    @staticmethod
    def _to_safe_alias(qualname: str) -> str:
        """Convert a qualname to a safe Python identifier avoiding name mangling."""
        if qualname.startswith("__"):
            return "_fxw_" + qualname[2:]
        return qualname

    @staticmethod
    def _rename_class(readable: str, alias_map: dict = None) -> str:
        """Rename the class and replace qualnames with safe aliases.

        Strips torch.fx._symbolic_trace.wrap() declarations and renames the class.
        Also replaces all FX-wrapped function references with safe aliases to avoid
        Python's double-underscore name mangling inside class methods.
        """
        lines = readable.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            # Skip FX wrap declarations — they raise NotImplementedError when executed
            if stripped.startswith("torch.fx._symbolic_trace.wrap("):
                continue
            if stripped.startswith("class ") and "(torch.nn.Module)" in stripped:
                result.append("class SerializedGraphModule(torch.nn.Module):")
            else:
                result.append(line)

        source = "\n".join(result)
        if alias_map:
            for qualname, safe_alias in alias_map.items():
                source = source.replace(qualname, safe_alias)
        return source

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
