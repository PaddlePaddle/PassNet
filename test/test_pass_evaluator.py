import importlib.util
import subprocess
import sys
from pathlib import Path

_mod_path = Path(__file__).resolve().parents[1] / "pass_agent/tools/pass_evaluator.py"
_spec = importlib.util.spec_from_file_location("pass_evaluator", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_with_timeout = _mod.run_with_timeout


def test_normal_exit():
    returncode, stdout, stderr = run_with_timeout(
        ["bash", "-c", "echo hello"], cwd="/tmp", timeout=5
    )
    assert returncode == 0
    assert "hello" in stdout
    assert stderr == ""


def test_nonzero_exit():
    returncode, stdout, stderr = run_with_timeout(
        ["bash", "-c", "exit 42"], cwd="/tmp", timeout=5
    )
    assert returncode == 42


def test_stderr_collected():
    returncode, stdout, stderr = run_with_timeout(
        ["bash", "-c", "echo err >&2"], cwd="/tmp", timeout=5
    )
    assert returncode == 0
    assert "err" in stderr


def test_timeout_flushes_output():
    # Run in a subprocess to isolate sys.exit called on timeout
    script = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('pass_evaluator', r'{_mod_path}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "mod.run_with_timeout(['bash','-c','echo before_timeout; sleep 10'], cwd='/tmp', timeout=2)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "before_timeout" in result.stdout


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v", "--capture=no"]))
