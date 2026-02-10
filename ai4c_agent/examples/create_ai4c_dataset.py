#!/usr/bin/env python3
"""
Script to create a JSONL dataset from AI4C sample lists.
"""

import json
import os
from pathlib import Path

def create_jsonl_dataset(sample_list_path: str, output_path: str, docker_image: str = "ai4c:20260126"):
    """
    Create a JSONL dataset from a sample list file.

    Args:
        sample_list_path: Path to the text file containing sample directories (one per line)
        output_path: Path to the output JSONL file
        docker_image: Docker image to use for the dataset
    """
    dataset_entries = []

    with open(sample_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Extract instance_id (last component of the path)
            instance_id = os.path.basename(line)

            # Create dataset entry
            entry = {
                "swe_type": "ai4c",
                "sample_dir": line,
                "repo": "ai4c",
                "docker_image": docker_image,
                "instance_id": instance_id
            }

            dataset_entries.append(entry)

    # Write JSONL file
    with open(output_path, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Created dataset with {len(dataset_entries)} entries")
    print(f"📁 Output: {output_path}")

    # Print first few entries as preview
    print(f"\n📋 Preview (first 3 entries):")
    for i, entry in enumerate(dataset_entries[:3]):
        print(f"\n{i+1}. {entry['instance_id']}")
        print(f"   Sample: {entry['sample_dir']}")

    return dataset_entries

if __name__ == "__main__":
    # Paths
    sample_list_path = "/ssd1/hesijun/ai4c/sample_lists/demo_fusible_samples.txt"
    output_path = "/ssd1/hesijun/baidu/personal-code/R2E-Gym/ai4c_demo_dataset.jsonl"

    # Create dataset
    entries = create_jsonl_dataset(sample_list_path, output_path)

    print(f"\n✅ Dataset creation complete!")
    print(f"   Total entries: {len(entries)}")
