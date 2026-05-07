#!/root/.venv/bin/python
"""
Description: PassNet Pass Evaluator - Runs evaluation on optimized pass and returns performance metrics.

This tool executes the entry.sh script in the PassNet problem directory to validate and benchmark
the optimized pass code. The score is extracted from the evaluation output.

The tool expects to be run from the problem directory (where entry.sh is located).

Parameters:
  None - uses current working directory
"""

import argparse
import subprocess
import select
import time
import sys
import os
import json
import io
from pathlib import Path

# Ensure stdout/stderr can handle UTF-8 encoding issues
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')


def _remove_useless_lines(result_stdout, result_stderr):
    """Remove noisy lines from stdout/stderr based on fixed markers."""
    remove_markers = (
        "[Correctness][all_close",
        "Trial",
        "[Profiling]",
    )

    def _filter_lines(text):
        kept = []
        for line in text.splitlines():
            if any(marker in line for marker in remove_markers):
                continue
            kept.append(line)
        return kept

    filtered_stdout = _filter_lines(result_stdout)
    filtered_stderr = _filter_lines(result_stderr)
    return filtered_stdout, filtered_stderr


def run_with_timeout(cmd, cwd, timeout):
    """Run a command, collecting stdout/stderr. On timeout, flush collected output before exit."""
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='ignore',
    )

    stdout_lines, stderr_lines = [], []
    start_time = time.time()
    readable_fds = [proc.stdout, proc.stderr]

    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            proc.terminate()
            proc.wait()
            print(''.join(stdout_lines), end='')
            print(''.join(stderr_lines), end='', file=sys.stderr)
            print("\n❌ ERROR: Evaluation timed out after 10 minutes")
            sys.exit(1)

        readable, _, _ = select.select(readable_fds, [], [], timeout - elapsed)

        if not readable:
            if proc.poll() is not None:
                break
            continue

        for stream in readable:
            line = stream.readline()
            if line:
                (stdout_lines if stream is proc.stdout else stderr_lines).append(line)
            else:
                readable_fds.remove(stream)

        if not readable_fds:
            break

    proc.wait()
    return proc.returncode, ''.join(stdout_lines), ''.join(stderr_lines)


def run_evaluation():
    """
    Run the PassNet evaluation script (entry.sh) and parse results.

    Uses the current working directory as the problem directory.

    Returns:
        int: exit code (0 for success)
    """
    # Use current working directory
    problem_path = os.getcwd()
    sample_path = Path(problem_path)

    # Validate that problem directory exists
    if not sample_path.exists():
        print(f"ERROR: Problem directory does not exist: {problem_path}")
        sys.exit(1)

    # Validate that entry.sh exists
    entry_script = sample_path / "entry.sh"
    if not entry_script.exists():
        print(f"ERROR: entry.sh not found in {problem_path}")
        sys.exit(1)

    # Run entry.sh
    print(f"Running evaluation for problem: {problem_path}")

    try:
        returncode, stdout_text, stderr_text = run_with_timeout(
            ["bash", str(entry_script)], cwd=str(sample_path), timeout=600
        )
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    filtered_stdout, filtered_stderr = _remove_useless_lines(stdout_text, stderr_text)
    print("[STDOUT]")
    for line in filtered_stdout:
        print(line)
    for line in filtered_stderr:
        print(line)

    # Parse the aggregated score
    output_path = Path("/tmp/workspace_graph_net_bench_test")
    score_file = output_path / "aggregated_score.json"

    if score_file.exists():
        with open(score_file, 'r') as f:
            score_data = json.load(f)

        print("\n✅ Evaluation completed successfully!")
        print(f"Score data: {json.dumps(score_data, indent=2)}")
    else:
        print(f"\n⚠️  Warning: Score file not found at {score_file}")
        print(f"Check if evaluation completed successfully.")

    # Check if pass matched (look for "Has Any pass matched?" in output)
    if "Has Any pass matched?" in stdout_text:
        if "[False]" in stdout_text:
            print("\n❌ FAIL: Pass did not match any pattern")
            print("This means the pass optimization pattern was not triggered.")
            sys.exit(1)
        elif "[True]" in stdout_text:
            print("\n✅ SUCCESS: Pass matched and was applied")

    # Return code check
    if returncode != 0:
        print(f"\n❌ Evaluation failed with return code: {returncode}")
        sys.exit(returncode)

    return returncode


def main():
    parser = argparse.ArgumentParser(
        description="PassNet Pass Evaluator: Run evaluation on optimized pass code."
    )
    # No arguments needed - uses current working directory

    args = parser.parse_args()

    # Run evaluation
    run_evaluation()


if __name__ == "__main__":
    main()
