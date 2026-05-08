import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pass_bench.torch.backend.pass_mgr_backend import (
    DiagnosticMatcher,
    FailureType,
    PatternReplacementPass,
)


def _trace(fn):
    return torch.fx.symbolic_trace(fn).graph


def test_issue1_anchor_scan_noise_is_suppressed():
    def pattern(x):
        return torch.add(x, 1)

    def target(x):
        return torch.mul(x, 2)

    matcher = DiagnosticMatcher(_trace(pattern))
    matches = matcher.match(_trace(target))

    assert matches == []
    assert matcher.failures == []


def test_issue3_reports_not_contained_failure():
    def pattern(x):
        y = torch.neg(x)
        z = torch.relu(y)
        return z

    def target(x):
        y = torch.neg(x)
        z = torch.relu(y)
        w = torch.add(y, 1)
        return torch.add(z, w)

    matcher = DiagnosticMatcher(_trace(pattern))
    matches = matcher.match(_trace(target))

    assert matches == []
    assert matcher.failures
    assert any(f.failure_type == FailureType.NOT_CONTAINED for f in matcher.failures)


def test_pass_mismatch_prints_diagnostic_report(capsys):
    def pattern(x):
        y = torch.neg(x)
        z = torch.relu(y)
        return torch.add(z, 1)

    def replacement(x):
        return torch.neg(x)

    def target(x):
        y = torch.neg(x)
        z = torch.sigmoid(y)
        return torch.add(z, 1)

    pass_rule = SimpleNamespace(
        pattern=pattern,
        replacement_func=lambda: replacement,
        replacement_args=lambda *args: args,
    )

    gm_pass = PatternReplacementPass(pass_rule, pass_name="diag_demo_pass")
    gm = torch.fx.symbolic_trace(target)
    result = gm_pass(gm)

    out = capsys.readouterr().out
    print(out, end="")
    assert result.modified is False
    assert "[PassMgrBackend] Pass diag_demo_pass failed to match." in out
    assert "[PassMgrBackend] Diagnostic for diag_demo_pass (best-attempt):" in out
    assert "MatchFailure(" in out


def test_complex_branch_mismatch_prints_actionable_failure(capsys):
    def pattern(x):
        base = torch.neg(x)
        left = torch.relu(base)
        right = torch.sigmoid(base)
        merged = torch.add(left, right)
        return merged

    def replacement(x):
        return torch.conv2d(x, torch.ones(1, 1, 3, 3))

    def target(x):
        base = torch.neg(x)
        left = torch.relu(base)
        right = torch.tanh(base)
        merged = torch.add(left, right)
        return merged

    pass_rule = SimpleNamespace(
        pattern=pattern,
        replacement_func=lambda: replacement,
        replacement_args=lambda *args: args,
    )

    gm_pass = PatternReplacementPass(pass_rule, pass_name="diag_complex_branch")
    gm = torch.fx.symbolic_trace(target)
    result = gm_pass(gm)

    out = capsys.readouterr().out
    print(out, end="")
    assert result.modified is False
    assert "[PassMgrBackend] Pass diag_complex_branch failed to match." in out
    assert "[PassMgrBackend] Diagnostic for diag_complex_branch (best-attempt):" in out
    assert "MatchFailure(type=TARGET_MISMATCH" in out
    assert "exp=sigmoid" in out
    assert "act=tanh" in out


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([str(Path(__file__).resolve())]))
