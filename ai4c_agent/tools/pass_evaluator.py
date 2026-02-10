#!/root/.venv/bin/python
"""
Description: AI4C Pass Evaluator - Runs evaluation on optimized pass and returns performance metrics.

This tool executes the entry.sh script in the AI4C problem directory to validate and benchmark
the optimized pass code. It returns performance metrics including speedup and correctness.

The tool expects to be run from the problem directory (where entry.sh is located).

Parameters:
  None - uses current working directory
"""

import argparse
import subprocess
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


def run_evaluation():
    """
    Run the AI4C evaluation script (entry.sh) and parse results.

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
    print(f"Evaluation script: {entry_script}")
    print("-" * 80)

    try:
        # Execute entry.sh
        result = subprocess.run(
            ["bash", str(entry_script)],
            cwd=str(sample_path),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=600  # 10 minute timeout
        )

        print("[STDOUT]")
        print(result.stdout)
        print("\n[STDERR]")
        print(result.stderr)
        print("-" * 80)

        # Parse the aggregated score
        output_path = Path("/tmp/workspace_graph_net_bench_test")
        score_file = output_path / "aggregated_score.json"

        if score_file.exists():
            with open(score_file, 'r') as f:
                score_data = json.load(f)

            print("\n✅ Evaluation completed successfully!")
            print(f"Score data: {json.dumps(score_data, indent=2)}")

            # Extract key metrics
            if isinstance(score_data, dict):
                speedup = score_data.get('speedup', 'N/A')
                correctness = score_data.get('correctness', 'N/A')
                print(f"\n📊 Performance Metrics:")
                print(f"  - Speedup: {speedup}")
                print(f"  - Correctness: {correctness}")
        else:
            print(f"\n⚠️  Warning: Score file not found at {score_file}")
            print(f"Check if evaluation completed successfully.")

        # Check if pass matched (look for "Has Any pass matched?" in output)
        if "Has Any pass matched?" in result.stdout:
            if "[False]" in result.stdout:
                print("\n❌ FAIL: Pass did not match any pattern")
                print("This means the pass optimization pattern was not triggered.")
                sys.exit(1)
            elif "[True]" in result.stdout:
                print("\n✅ SUCCESS: Pass matched and was applied")

        # Return code check
        if result.returncode != 0:
            print(f"\n❌ Evaluation failed with return code: {result.returncode}")
            sys.exit(result.returncode)

        return result.returncode

    except subprocess.TimeoutExpired:
        print("\n❌ ERROR: Evaluation timed out after 10 minutes")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="AI4C Pass Evaluator: Run evaluation on optimized pass code."
    )
    # No arguments needed - uses current working directory

    args = parser.parse_args()

    # Run evaluation
    run_evaluation()


if __name__ == "__main__":
    main()
