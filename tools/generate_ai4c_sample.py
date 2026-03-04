import os
import argparse
import hashlib
from pathlib import Path


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


def generate_sample(sample_uids, sample_uid2model_path, output_path, graphs_path_in_ai4c):
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
        for line in f:
            line = line.strip()
            if not line:
                continue

            sample_uids = line.split(",")
            print(sample_uids)
            sample_output_path = generate_sample(sample_uids, sample_uid2model_path, output_path, graphs_path_in_ai4c)
            if sample_output_path:
                sample_output_path = os.path.relpath(Path(sample_output_path).resolve(), start=ai4c_root.resolve())
                generated_sample_list.append(sample_output_path)
                num_successed += 1
            break

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
    args = parser.parse_args()
    main(args)
