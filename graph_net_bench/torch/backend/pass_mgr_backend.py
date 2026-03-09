import os
import torch
import torch.fx
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.fx.passes.infra.pass_manager import PassManager, PassResult
import random
import string
import inspect
import json
from collections import OrderedDict
from pathlib import Path
import importlib.util as imp
from .graph_compiler_backend import GraphCompilerBackend
from dataclasses import dataclass
from typing import Any, List, Optional, Dict
from enum import Enum, auto
from ..custom_replacement import _replace_pattern

class FailureType(Enum):
    OP_MISMATCH = auto()
    TARGET_MISMATCH = auto()
    ATTR_MISMATCH = auto()
    NOT_CONTAINED = auto()

@dataclass
class MatchFailure:
    pattern_node: torch.fx.Node
    target_node: torch.fx.Node
    failure_type: FailureType
    expected: Any = None
    actual: Any = None

    def __repr__(self):
        return (f"MatchFailure(type={self.failure_type.name}, "
                f"p={self.pattern_node.name}, t={self.target_node.name}, "
                f"exp={self.expected}, act={self.actual})")

class DiagnosticMatcher(SubgraphMatcher):
    def __init__(self, pattern_graph: torch.fx.Graph):
        super().__init__(pattern_graph)
        self.failures: List[MatchFailure] = []
        self._recording = False
        self._current_failures: List[MatchFailure] = []
        self._best_failures: Optional[List[MatchFailure]] = None
        self._best_match_size = -1
        self._match_depth = 0
        self._containment_failures: List[MatchFailure] = []

    def _record_failure(self, p_node, t_node, failure_type, expected=None, actual=None):
        if self._recording:
            self._current_failures.append(
                MatchFailure(p_node, t_node, failure_type, expected, actual)
            )
        return False

    def _match_attributes(self, pn, gn):
        assert isinstance(pn.target, str), f"pn.target {pn.target} must be a string."
        assert isinstance(gn.target, str), f"gn.target {gn.target} must be a string."
        pn_value = torch.fx.graph_module._get_attr(pn.graph.owning_module, pn.target)
        gn_value = torch.fx.graph_module._get_attr(gn.graph.owning_module, gn.target)
        if type(pn_value) is not type(gn_value):
            return self._record_failure(pn, gn, FailureType.ATTR_MISMATCH,
                                        type(pn_value).__name__, type(gn_value).__name__)
        if isinstance(pn_value, torch.Tensor):
            return True
        raise RuntimeError(f"Unsupported type {pn_value} when matching attributes")

    def _nodes_are_equal(self, pn, gn, node_name_match=""):
        if not self.match_placeholder and pn.op == "placeholder":
            return True
        if node_name_match and node_name_match in gn.name:
            return True
        if pn.op != gn.op:
            return self._record_failure(pn, gn, FailureType.OP_MISMATCH, pn.op, gn.op)
        if pn.op in ("placeholder", "output"):
            return True
        if pn.op == "get_attr":
            return self._match_attributes(pn, gn)
        if pn.target != gn.target:
            if pn.op == "call_function":
                p_name = getattr(pn.target, "__name__", str(pn.target))
                t_name = getattr(gn.target, "__name__", str(gn.target))
                return self._record_failure(pn, gn, FailureType.TARGET_MISMATCH, p_name, t_name)
            return self._record_failure(pn, gn, FailureType.TARGET_MISMATCH, pn.target, gn.target)
        return True

    def _match_nodes(self, pn, gn, match, node_name_match=""):
        self._match_depth += 1
        is_top_level = self._match_depth == 1
        if is_top_level:
            self._recording = True
            self._current_failures = []
        try:
            result = super()._match_nodes(pn, gn, match)
            if is_top_level:
                self._recording = False
                if not result:
                    match_size = len(match.nodes_map)
                    if match_size > self._best_match_size:
                        self._best_match_size = match_size
                        self._best_failures = self._current_failures[:]
            return result
        finally:
            self._match_depth -= 1
            if is_top_level:
                self._recording = False

    def _is_contained(self, nodes_map):
        result = super()._is_contained(nodes_map)
        if result:
            return True
        lookup = {gn: pn for pn, gn in nodes_map.items() if pn.op != "placeholder"}
        for gn, pn in lookup.items():
            if pn in self.pattern_returning_nodes:
                continue
            for user in gn.users:
                if user not in lookup:
                    self._containment_failures.append(
                        MatchFailure(pn, gn, FailureType.NOT_CONTAINED,
                                     "internal node", f"leaks to {user.name}")
                    )
        return False

    def match(self, graph, node_name_match=""):
        self._best_failures = None
        self._best_match_size = -1
        self._containment_failures = []
        self.failures = []
        result = super().match(graph)
        if not result:
            if self._containment_failures:
                self.failures = self._containment_failures
            elif self._best_failures is not None:
                self.failures = self._best_failures
        return result


class PassMgrBackend(GraphCompilerBackend):
    def __init__(self, config: dict):
        assert isinstance(config, dict)
        super().__init__(self._make_config(**config))
        self.pass_manager = self.make_pass_manager()

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

    def __call__(self, model):
        return torch.compile(model, backend=self.torch_compile_backend)

    def torch_compile_backend(self, gm: torch.fx.GraphModule, sample_inputs: list):
        pass_result = self.pass_manager(gm)
        if self.config['pass_match_result_file_path'] is not None: 
            tmp_file = Path(self.config['pass_match_result_file_path'])
            tmp_file.write_text(str(pass_result.modified))
        if not pass_result.modified:
            print("[PassMgrBackend] Warning: No passes modified the graph. Returning original.")
            # exit(-1)  <-- Removed to allow continued execution or fallback
        return pass_result.graph_module

    def make_pass_manager(self):
        return PassManager(passes=self.get_passes())

    def get_passes(self):
        passes = [
            create_pass(
                pass_name=pass_name,
                pass_rule=pass_rule
            )
            for pass_name, pass_rule in self._get_named_pass_rules()
        ]
        print(f"[PassMgrBackend] Loaded {len(passes)} passes: {[p.__name__ for p in passes]}")
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
            if rule.replacement_func in allowed_replacement_funcs
        ]

    def _get_allowed_replacement_funcs(self, rules):
        replacement_func_limit = self.config['output_pass_replacement_func_limit']
        replacement_func2none = OrderedDict([])
        for rule in rules:
            replacement_func2none[rule.replacement_func] = None
        replacement_funcs = list(replacement_func2none.keys())
        if len(replacement_funcs) <= replacement_func_limit:
            return set(replacement_funcs)
        indices = random.sample(range(len(replacement_funcs)), replacement_func_limit)
        indices.sort()
        return set(
            replacement_funcs[index]
            for index in indices
        )

    def _bound_by_pattern_limit(self, rules):
        pattern_limit = self.config['output_pass_pattern_limit']
        if len(rules) <= pattern_limit:
            return rules
        indices = random.sample(range(len(rules)), pattern_limit)
        indices.sort()
        return [rules[i] for i in indices]

    def _find_rule(self, dir_path, name):
        return load_py_module(os.path.join(dir_path, f"{name}.py"), name=name)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class PatternReplacementPass:
    def __init__(self, pass_rule, pass_name="unnamed_pass"):
        arg_names = list(inspect.signature(pass_rule.pattern).parameters.keys())
        @self.reset_func_arg_names(arg_names)
        def replacement(*args):
            return pass_rule.replacement_func()(*pass_rule.replacement_args(*args))
            
        self.pattern = pass_rule.pattern
        self.replacement = replacement
        self.pass_name = pass_name

    def _print_diagnostic_report(self, gm: torch.fx.GraphModule) -> None:
        try:
            if isinstance(self.pattern, torch.fx.GraphModule):
                pattern_graph = self.pattern.graph
            elif isinstance(self.pattern, torch.fx.Graph):
                pattern_graph = self.pattern
            else:
                pattern_graph = torch.fx.symbolic_trace(self.pattern).graph
            matcher = DiagnosticMatcher(pattern_graph)
            matcher.match(gm.graph)
            if not matcher.failures:
                print(f"[PassMgrBackend] No specific node failures recorded for {self.pass_name}.")
                return
            print(f"[PassMgrBackend] Diagnostic for {self.pass_name} (best-attempt):")
            non_op = [f for f in matcher.failures if f.failure_type != FailureType.OP_MISMATCH]
            for f in (non_op or matcher.failures)[:10]:
                print(f"  - {f}")
        except Exception as e:
            print(f"[PassMgrBackend] Diagnostic failed: {e}")
    
    @classmethod
    def reset_func_arg_names(cls, arg_names):
        # arg_names is a list like ['x', 'y', 'z']
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

    def __call__(self, gm: torch.fx.GraphModule):
        try:
            matches = _replace_pattern(gm, self.pattern, self.replacement)
        except Exception as e:
            print(f"[PassMgrBackend] Pass {self.pass_name} CRASHED with error: {e}")
            raise e
        
        # Determine if the graph actually changed
        modified = len(matches) > 0
        
        if modified:
            gm.recompile()
            print(f"Applied {len(matches)} replacements.")
        else:
            # Diagnose pattern matching failure
            print(f"[PassMgrBackend] Pass {self.pass_name} failed to match.")
            self._print_diagnostic_report(gm)

        # Return the PassResult object
        return PassResult(gm, modified)

def create_pass(pass_name, pass_rule):
    gm_pass = PatternReplacementPass(pass_rule, pass_name)
    def func(gm):
        return gm_pass(gm)
    func.__name__ = pass_name
    func.__qualname__ = pass_name
    return func

def load_py_module(path, name='unamed'):
    from graph_net_bench.ast_util import validate_pass_source
    with open(path, "r") as f:
        source = f.read()
    violations = validate_pass_source(source)
    if violations:
        print(f"[PassMgrBackend] Detected hacking behavior, forbidden torch API usage in replacement_func")
        print(f"[PassMgrBackend] Pass source validation failed for {path}:")
        for v in violations:
            print(f"  - {v}")
        print(f"[PassMgrBackend] Skipping loading of {name} due to validation failures.")
        return None

    spec = imp.spec_from_file_location(name, path)
    module = imp.module_from_spec(spec)
    module.__file__ = path
    spec.loader.exec_module(module)
    return module
