import os
import argparse
import hashlib
import shutil
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict


def get_ai4c_root():
    return Path(__file__).resolve().parent.parent


def load_model_list(filepath: Path):
    """
    Load mapping from sample_uid to model_path.
    Expected format per line:
        <uid> <model_path>
    """
    sample_uid2model_path = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample_uid, model_path = line.split(maxsplit=1)
            sample_uid2model_path[sample_uid] = model_path
    return sample_uid2model_path


def compute_hash(sample_uids):
    """
    Compute SHA256 hash of sorted uids.
    Sorting ensures order-independent hashing.
    """
    sorted_sample_uids = sorted(sample_uids)
    data = "\n".join(sorted_sample_uids).encode()
    return hashlib.sha256(data).hexdigest()


def safe_relative_symlink(src: Path, dst: Path):
    """
    Create or replace a symbolic link.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    # Compute relative path from dst directory to src
    relative_src = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(relative_src)


def generate_sample(sample_uids: List[str], sample_uid2model_path: Dict[str, str], output_path: Path, graphs_path_in_ai4c: Path) -> str:
    """
    Generate a AI4C sample.
    """
    # Compute group hash
    hash_value = compute_hash(sample_uids)

    # Construct directory hierarchy: xx/xx/fullhash
    dir1 = hash_value[0:2]
    dir2 = hash_value[2:4]
    sample_output_path = output_path / dir1 / dir2 / hash_value

    # Idempotent behavior
    if sample_output_path.exists():
        print(f"Sample already generated at {sample_output_path}.")
        return None

    ai4c_root = get_ai4c_root()

    # Create required pass_dir directory and create .ignore
    (sample_output_path / "pass_dir").mkdir(parents=True, exist_ok=True)
    (sample_output_path / "pass_dir" / ".ignore").touch()

    # Create symbolic links
    safe_relative_symlink(ai4c_root / "graphs", sample_output_path / "graphs")
    safe_relative_symlink(
        ai4c_root / "entry_scripts/entry.sh",
        sample_output_path / "entry.sh",
    )
    safe_relative_symlink(
        ai4c_root / "graph_net_bench",
        sample_output_path / "graph_net_bench",
    )

    # Write sample_uids.txt
    with open(sample_output_path / "sample_uids.txt", "w") as f:
        f.write(",".join(sample_uids))

    # Write graph_list.txt
    with open(sample_output_path / "graph_list.txt", "w") as f:
        for uid in sample_uids:
            if uid in sample_uid2model_path:
                rel_model_path = graphs_path_in_ai4c / sample_uid2model_path[uid]
                f.write(str(rel_model_path) + "\n")
            else:
                raise ValueError(f"{uid} is missing from model_path.txt")

    return str(sample_output_path)


def evaluate_sample(sample_path: str) -> bool:
    """
    Evaluate a AI4C sample.
    """
    validation_log_path = Path("/tmp/workspace_graph_net_bench_test/validation.log")
    aggregated_score_path = Path("/tmp/workspace_graph_net_bench_test/aggregated_score.json")

    sample_dir = Path(sample_path).resolve()
    entry_script = sample_dir / "entry.sh"

    if not entry_script.exists():
        print(f"{entry_script} not found.")
        return False

    # Clean old outputs before execution
    for path in (validation_log_path, aggregated_score_path):
        if path.exists() or path.is_symlink():
            path.unlink()

    try:
        # Execute entry.sh inside sample_dir
        proc = subprocess.run(
            ["bash", str(entry_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1800,
            text=True,
        )
        if proc.returncode != 0:
            print(f"Run {entry_script} failed with return code {proc.returncode}.")
            return False

        # Validate log file existence
        if not validation_log_path.exists():
            print(f"{validation_log_path} not found for {sample_dir}.")
            return False

        # Extract rectified_speedup from log
        log_content = proc.stdout
        match = re.search(r"rectified_speedup=([0-9.eE+-]+)", log_content)
        if not match:
            print(f"rectified_speedup not found in log {log_path} for {sample_dir}.")
            return False

        rectified_speedup = float(match.group(1))

        # Validate aggregated_score.json
        if not aggregated_score_path.exists():
            print(f"{aggregated_score_path} not found for {sample_dir}")
            return False

        with open(aggregated_score_path, "r") as f:
            score = json.load(f)

        if score["score"] != rectified_speedup:
            print(f"Score is not equal to rectified_speedup ({score['score']} != {rectified_speedup}).")
            return False

        return True
    except subprocess.TimeoutExpired:
        print(f"Execution timeout for {sample_dir}.")
        return False
    except Exception as e:
        print(f"Execution error: {str(e)}")
        return False


def main(args):
    ai4c_root = get_ai4c_root()
    group_sample_uids_path = Path(args.grouped_sample_uids_list)
    model_path = Path(args.model_path_list)
    graphs_path_in_ai4c = Path(os.path.relpath(Path(args.graphs_path_in_ai4c).resolve(), start=ai4c_root.resolve()))

    output_path = Path(args.output_path)

    sample_uid2model_path = load_model_list(model_path)
    output_path.mkdir(parents=True, exist_ok=True)

    num_successed = 0
    generated_sample_list = []
    with open(group_sample_uids_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            sample_uids = line.split(",")
            print(f"- [{idx}] Generate AI4C sample for uids {sample_uids}")
            sample_output_path = generate_sample(sample_uids, sample_uid2model_path, output_path, graphs_path_in_ai4c)
            if sample_output_path is None:
                continue

            if args.do_eval:
                eval_stat = evaluate_sample(sample_output_path)
                if not eval_stat:
                    shutil.rmtree(Path(sample_output_path))
                    continue

            sample_output_path = os.path.relpath(Path(sample_output_path).resolve(), start=ai4c_root.resolve())
            generated_sample_list.append(sample_output_path)
            num_successed += 1

    output_sample_list_path = Path(args.output_sample_list)
    output_sample_list_path.write_text("\n".join(generated_sample_list))
    print(f"Generated {num_successed} samples, written to {output_sample_list_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI4C samples for sample groups in GraphNet format."
    )
    parser.add_argument(
        "--grouped-sample-uids-list",
        required=True,
        help="Path to grouped_sample_uids.txt",
    )
    parser.add_argument(
        "--model-path-list",
        required=True,
        help="Path to model_path.txt",
    )
    parser.add_argument(
        "--graphs-path-in-ai4c",
        required=True,
        help="Graphs root path in ai4c repo",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output root directory",
    )
    parser.add_argument(
        "--output-sample-list",
        required=True,
        help="Output sample list",
    )
    parser.add_argument(
        "--do-eval",
        action="store_true",
        help="Run evaluation after generating samples"
    )
    args = parser.parse_args()
    main(args)
