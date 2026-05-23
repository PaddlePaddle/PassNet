import os
import random
import string
import inspect
import json
import torch
from torch.fx.passes.infra.pass_manager import PassManager, PassResult
from collections import OrderedDict
from pathlib import Path
import importlib.util as imp
from pass_bench import imp_util
from pass_bench.torch.backend.graph_compiler_backend import GraphCompilerBackend
from pass_bench.torch.custom_replacement import _replace_pattern
from pass_bench.torch.posion_dispatch_tensor import wrap_args, unwrap_args, unwrap_tensor
from pass_bench.torch.override_dispatch_flag import get_global_override_dispatch


# ---------------------------------------------------------------------------
# Module-level dispatch infrastructure (mirrors pass_mgr_backend.py)
# ---------------------------------------------------------------------------

g_replacement_func = None


def set_g_replacement_func(f):
    global g_replacement_func
    if g_replacement_func is not None:
        assert g_replacement_func is f
    else:
        g_replacement_func = f


@torch.fx.wrap
def with_dispatch_wrapper_run(*args):
    if get_global_override_dispatch():
        args = wrap_args(args)
        outs = g_replacement_func(*args)
        outs = unwrap_args(outs) if isinstance(outs, (tuple, list)) else unwrap_tensor(outs)
    else:
        outs = g_replacement_func(*args)
    return outs


def replacement_core_decorator():
    def func(*args):
        return with_dispatch_wrapper_run(*args)
    return func


# ---------------------------------------------------------------------------
# Pass source validation (mirrors pass_mgr_backend.py)
# ---------------------------------------------------------------------------

def is_pass_source_valid(path):
    from pass_bench.ast_util import validate_pass_source
    with open(path, "r") as f:
        source = f.read()
    violations = validate_pass_source(source)
    if violations:
        print(f"[PassMgrDirect] Detected hacking behavior, forbidden torch API usage in replacement_func", flush=True)
        print(f"[PassMgrDirect] Pass source validation failed for {path}:", flush=True)
        for v in violations:
            print(f"  - {v}", flush=True)
        print(f"[PassMgrDirect] Skipping loading of {path} due to validation failures.", flush=True)
        return False
    return True


def is_pass_source_valid_by_customized_checker(path):
    with open(path, "r") as f:
        source = f.read()
    pass_source_checker_paths = os.environ.get("AI4C_CUSTOM_PASS_SOURCE_CHECKER_PATH")
    if pass_source_checker_paths is None:
        return True
    for checker_path in pass_source_checker_paths.split(':'):
        if not Path(checker_path).is_file():
            continue
        module = imp_util.load_module(checker_path)
        violations = module.validate_pass_source(source)
        if violations:
            print(f"[PassMgrDirect] Detected hacking behavior, forbidden torch API usage in replacement_func", flush=True)
            print(f"[PassMgrDirect] Pass source validation failed for {path}:", flush=True)
            for v in violations:
                print(f"  - {v}", flush=True)
            print(f"[PassMgrDirect] Skipping loading of {path} due to validation failures.", flush=True)
            return False
    return True


def load_py_module(path, name='unamed'):
    if not is_pass_source_valid(path):
        return None
    if not is_pass_source_valid_by_customized_checker(path):
        return None
    import sys
    sys.path.insert(0, str(Path(path).parent.parent))
    spec = imp.spec_from_file_location(name, path)
    module = imp.module_from_spec(spec)
    module.__file__ = path
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Placeholder reordering
# ---------------------------------------------------------------------------

def _reorder_placeholders(gm, sample_inputs, param_names, original_input_tensors):
    """Reorder GM placeholders to match the original calling order.

    Dynamo may reorder and rename placeholders (e.g., L_in_3_, L_in_1_).
    This function uses id(tensor) to map sample_inputs back to their
    original calling order and renames them back to original names
    (in_0, in_1, ...) so that gm(*args) works directly with no
    per-call overhead.
    """
    ph_nodes = [n for n in gm.graph.nodes if n.op == 'placeholder']
    if len(ph_nodes) != len(sample_inputs):
        return  # can't reorder if counts don't match

    id_to_ph = {id(t): ph for t, ph in zip(sample_inputs, ph_nodes)}
    reordered = [id_to_ph[id(t)] for t in original_input_tensors if id(t) in id_to_ph]
    if len(reordered) != len(ph_nodes):
        return  # can't reorder if some tensors not found

    first_non_ph = next(n for n in gm.graph.nodes if n.op != 'placeholder')
    with gm.graph.inserting_before(first_non_ph):
        for name, old_ph in zip(param_names, reordered):
            new_node = gm.graph.placeholder(name)
            old_ph.replace_all_uses_with(new_node)
            gm.graph.erase_node(old_ph)

    gm.recompile()


# ---------------------------------------------------------------------------
# FixedPatternReplacementPass — handles multi-output patterns correctly
# ---------------------------------------------------------------------------

def _count_pattern_outputs(pattern):
    """Trace the pattern function and count the number of returning nodes."""
    try:
        from pass_bench.torch.custom_replacement import force_args_symbolic_trace
        pattern_gm = force_args_symbolic_trace(pattern)
    except Exception:
        return 1
    output_node = list(pattern_gm.graph.nodes)[-1]
    if output_node.op != 'output' or not output_node.args:
        return 1
    output_val = output_node.args[0]
    if isinstance(output_val, tuple):
        return len(output_val)
    return 1


def _reset_func_arg_names(arg_names):
    args_str = ", ".join(arg_names)
    func_name = "dynamic_func_" + "".join(random.choices(string.ascii_lowercase, k=5))
    source = f"""
def {func_name}(f):
    def func({args_str}):
        return f({args_str})
    return func
"""
    namespace = {}
    exec(source, globals(), namespace)
    return namespace[func_name]


class FixedPatternReplacementPass:
    """Handles both single-output and multi-output pattern replacement.

    For multi-output patterns, destructures the dispatch wrapper result
    with getitem so the FX tracer creates individual output nodes.
    """

    def __init__(self, pass_rule, pass_name="unnamed_pass"):
        arg_names = list(inspect.signature(pass_rule.pattern).parameters.keys())
        set_g_replacement_func(pass_rule.replacement_func())
        f = replacement_core_decorator()

        n_outputs = _count_pattern_outputs(pass_rule.pattern)

        if n_outputs > 1:
            replacement = self._make_multi_output_replacement(
                arg_names, n_outputs, f, pass_rule.replacement_args
            )
        else:
            @_reset_func_arg_names(arg_names)
            def replacement(*args):
                outs = f(*pass_rule.replacement_args(*args))
                return outs

        self.pattern = pass_rule.pattern
        self.replacement = replacement
        self.pass_name = pass_name

    @staticmethod
    def _make_multi_output_replacement(arg_names, n_outputs, f, replacement_args):
        """Build a replacement function that destructures the wrapper result
        with getitem, so the FX tracer creates individual output nodes."""
        args_str = ", ".join(arg_names)
        func_name = "dynamic_func_" + "".join(random.choices(string.ascii_lowercase, k=5))
        getitem_lines = ", ".join(f"outs[{i}]" for i in range(n_outputs))

        source = f"""
def {func_name}(f, ra):
    def func({args_str}):
        outs = f(*ra({args_str}))
        return ({getitem_lines},)
    return func
"""
        namespace = {}
        exec(source, globals(), namespace)
        return namespace[func_name](f, replacement_args)

    def __call__(self, gm: torch.fx.GraphModule):
        try:
            matches = _replace_pattern(gm, self.pattern, self.replacement)
        except Exception as e:
            print(f"[PassMgrDirect] Pass {self.pass_name} CRASHED with error: {e}", flush=True)
            raise e

        modified = len(matches) > 0
        if modified:
            gm.recompile()
            print(f"[PassMgrDirect] Applied {len(matches)} replacements with {self.pass_name}.", flush=True)
        else:
            print(f"[PassMgrDirect] Pass {self.pass_name} failed to match.", flush=True)

        return PassResult(gm, modified)


def _create_pass(pass_name, pass_rule):
    gm_pass = FixedPatternReplacementPass(pass_rule, pass_name)
    def func(gm):
        return gm_pass(gm)
    func.__name__ = pass_name
    func.__qualname__ = pass_name
    return func


# ---------------------------------------------------------------------------
# PassMgrDirectBackend
# ---------------------------------------------------------------------------

class PassMgrDirectBackend(GraphCompilerBackend):
    """Backend that applies passes via torch.compile, then returns the
    captured GraphModule directly — no wrapper, no dynamo guard overhead.

    Uses FixedPatternReplacementPass which correctly handles multi-output
    patterns. After compilation, the GM's forward is hot-swapped in,
    eliminating all dynamo guard/dispatch overhead.

    The approach:
      1. torch.compile captures FX graph via dynamo on first call
      2. Apply pattern-replacement passes with FixedPatternReplacementPass
      3. After dynamo is done, replace dispatch wrapper node targets
         with the actual replacement function, then reorder placeholders
      4. From the second call onward, GM is called directly — no dynamo overhead
    """

    def __init__(self, config: dict):
        assert isinstance(config, dict)
        super().__init__(self._make_config(**config))
        self.pass_manager = self._make_pass_manager()
        self._optimized_gm = None
        self._sample_inputs = None
        self._param_names = None
        self._original_input_tensors = None

    # -- Config construction (mirrors PassMgrBackend) --

    def _make_config(
        self,
        input_pass_rule_dir: str,
        output_pass_rule_dir: str,
        output_pass_pattern_limit: int,
        output_pass_replacement_func_limit: int,
        pass_match_result_file_path: str = None,
        **kwargs,
    ):
        sorted_input_pass_rule_names = self._get_sorted_input_pass_rule_names(
            input_pass_rule_dir, output_pass_rule_dir
        )
        sorted_output_pass_rule_names = self._get_sorted_output_pass_rule_names(
            output_pass_rule_dir
        )
        return {
            'input_pass_rule_dir': input_pass_rule_dir,
            'output_pass_rule_dir': output_pass_rule_dir,
            'output_pass_pattern_limit': output_pass_pattern_limit,
            'output_pass_replacement_func_limit': output_pass_replacement_func_limit,
            'sorted_input_pass_rule_names': sorted_input_pass_rule_names,
            'sorted_output_pass_rule_names': sorted_output_pass_rule_names,
            'pass_match_result_file_path': pass_match_result_file_path,
        }

    def _get_sorted_output_pass_rule_names(self, output_pass_rule_dir):
        output_pass_file_path = Path(output_pass_rule_dir) / "sorted_output_pass_rule_names.json"
        if not output_pass_file_path.exists():
            return []
        with open(output_pass_file_path) as f:
            rule_names = json.load(f)
        assert isinstance(rule_names, list)
        return rule_names

    def _get_sorted_input_pass_rule_names(self, input_pass_rule_dir, output_pass_rule_dir):
        input_pass_file_path = Path(input_pass_rule_dir) / "sorted_input_pass_rule_names.json"
        if input_pass_file_path.exists():
            with open(input_pass_file_path) as f:
                default_input_rule_names = json.load(f)
        else:
            default_input_rule_names = []
        assert isinstance(default_input_rule_names, list)
        customized_input_pass_file_path = Path(output_pass_rule_dir) / "sorted_input_pass_rule_names.json"
        if not customized_input_pass_file_path.exists():
            return default_input_rule_names
        with open(customized_input_pass_file_path) as f:
            customized_input_rule_names = json.load(f)
        assert set(default_input_rule_names) == set(customized_input_rule_names)
        return customized_input_rule_names

    # -- Pass loading (mirrors PassMgrBackend) --

    def _make_pass_manager(self):
        return PassManager(passes=self._get_passes())

    def _get_passes(self):
        passes = [
            _create_pass(
                pass_name=pass_name,
                pass_rule=pass_rule,
            )
            for pass_name, pass_rule in self._get_named_pass_rules()
        ]
        print(f"[PassMgrDirect] Loaded {len(passes)} passes: {[p.__name__ for p in passes]}", flush=True)
        return passes

    def _get_named_pass_rules(self):
        name2output_pass_rules = OrderedDict(
            (Path(inspect.getfile(rule)).stem, rule)
            for rule in self._get_output_pass_rules()
        )
        name2input_pass_rules = OrderedDict(
            (Path(inspect.getfile(rule)).stem, rule)
            for rule in self._get_input_pass_rules()
        )
        for name in name2input_pass_rules.keys():
            if name not in name2output_pass_rules:
                continue
            name2input_pass_rules[name] = name2output_pass_rules[name]
            del name2output_pass_rules[name]
        return [
            *name2input_pass_rules.items(),
            *name2output_pass_rules.items()
        ]

    def _get_input_pass_rules(self):
        input_pass_rule_dir = self.config['input_pass_rule_dir']
        sorted_input_pass_rule_names = self.config['sorted_input_pass_rule_names']
        return [
            rule
            for name in sorted_input_pass_rule_names
            if (rule := self._find_rule(dir_path=input_pass_rule_dir, name=name))
            is not None
        ]

    def _get_output_pass_rules(self):
        output_pass_rule_dir = self.config['output_pass_rule_dir']
        sorted_output_pass_rule_names = self.config['sorted_output_pass_rule_names']
        rules = [
            rule
            for name in sorted_output_pass_rule_names
            if (rule := self._find_rule(dir_path=output_pass_rule_dir, name=name))
            is not None
        ]
        rules = self._bound_by_replacement_func_limit(rules)
        rules = self._bound_by_pattern_limit(rules)
        return rules

    def _bound_by_replacement_func_limit(self, rules):
        allowed_replacement_funcs = self._get_allowed_replacement_funcs(rules)
        return [
            rule
            for rule in rules
            if rule.replacement_func() in allowed_replacement_funcs
        ]

    def _get_allowed_replacement_funcs(self, rules):
        replacement_func_limit = self.config['output_pass_replacement_func_limit']
        replacement_func2none = OrderedDict([])
        unstable_rules = []
        for rule in rules:
            func = rule.replacement_func()
            if func is not rule.replacement_func():
                rule_file = getattr(rule, '__file__', 'unknown')
                rule_name = getattr(rule, '__name__', repr(rule))
                unstable_rules.append((rule_name, rule_file))
                continue
            replacement_func2none[func] = None
        if unstable_rules:
            error_msg = "The following pass rules have unstable replacement_func():\n"
            for name, path in unstable_rules:
                error_msg += f"  - {name} ({path})\n"
            error_msg += (
                "\nFix: Return a module-level function, not a nested def/lambda. "
                "Example: define 'def f(x): return x' at top level, then return f."
            )
            raise RuntimeError(error_msg)
        replacement_funcs = list(replacement_func2none.keys())
        if not replacement_funcs:
            raise RuntimeError(
                f"No replacement functions available after filtering {len(rules)} rules."
            )
        if len(replacement_funcs) <= replacement_func_limit:
            return set(replacement_funcs)
        indices = random.sample(range(len(replacement_funcs)), replacement_func_limit)
        indices.sort()
        return set(replacement_funcs[i] for i in indices)

    def _bound_by_pattern_limit(self, rules):
        pattern_limit = self.config['output_pass_pattern_limit']
        if len(rules) <= pattern_limit:
            return rules
        indices = random.sample(range(len(rules)), pattern_limit)
        indices.sort()
        return [rules[i] for i in indices]

    def _find_rule(self, dir_path, name):
        return load_py_module(os.path.join(dir_path, f"{name}.py"), name=name)

    # -- Backend entry point --

    def __call__(self, model):
        self._optimized_gm = None
        self._sample_inputs = None
        self._param_names = None
        self._original_input_tensors = None
        return _CompileOnceWrapper(self, model)

    def _torch_compile_backend(self, gm: torch.fx.GraphModule, sample_inputs: list):
        pass_result = self.pass_manager(gm)

        if self.config.get('pass_match_result_file_path') is not None:
            tmp_file = Path(self.config['pass_match_result_file_path'])
            tmp_file.write_text(str(pass_result.modified))

        if not pass_result.modified:
            raise RuntimeError("[PassMgrDirect] No passes modified the graph.")

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
            replacement_func = g_replacement_func
            if replacement_func is not None:
                for node in self._optimized_gm.graph.nodes:
                    if node.op == 'call_function' and node.target is with_dispatch_wrapper_run:
                        node.target = replacement_func
            if self._original_input_tensors is not None:
                _reorder_placeholders(self._optimized_gm, self._sample_inputs, self._param_names, self._original_input_tensors)
            self._sample_inputs = None  # only once

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class _CompileOnceWrapper(torch.nn.Module):
    """Triggers dynamo compilation on first call, then hot-swaps forward
    to the bare GM's forward — zero wrapper overhead from the second call on.
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
        # Save original input order for placeholder reordering.
        if self._backend._original_input_tensors is None:
            if args:
                self._backend._original_input_tensors = list(args)
            elif kwargs and self._backend._param_names:
                self._backend._original_input_tensors = [
                    kwargs[k] for k in self._backend._param_names if k in kwargs
                ]
            elif kwargs:
                # model(**input_dict) with *args/**kwargs signature:
                # _param_names is empty, fall back to kwargs values order
                self._backend._original_input_tensors = list(kwargs.values())
        result = self._compiled(*args, **kwargs)

        if self._backend._optimized_gm is not None:
            # Dynamo is done. Reorder placeholders so gm(*args) works.
            self._backend._finalize_gm()
            gm = self._backend._optimized_gm
            # Hot-swap: replace forward with GM's forward directly.
            self.__dict__['forward'] = gm.forward
            # Release dynamo-compiled object — no longer needed.
            del self._compiled

        return result
