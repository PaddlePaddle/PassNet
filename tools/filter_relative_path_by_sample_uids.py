#!/usr/bin/env python3
"""
filter_relative_path_by_sample_uids.py

Reads a mapping file (TSV: uuid<TAB>relative_path) and a list of selected UUIDs,
and outputs the relative paths corresponding to UUIDs that appear in the selected list.

Usage:
    python filter_relative_path_by_sample_uids.py <mapping_file> <selected_uuid_file>

Input format:
    mapping_file: tab-separated, each line: uuid<tab>relative_path
    selected_uuid_file: one UUID per line

Output:
    For each matching UUID (in order of appearance in mapping_file),
    the relative_path is printed on a separate line to stdout.
"""

import argparse
import sys
from typing import Set, List

def read_selected_uuids(file_path: str) -> Set[str]:
    """Read UUIDs from a file (one per line) and return a set."""
    selected = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            selected.add(line)
    return selected

def filter_relative_paths(mapping_file: str, selected_uuids: Set[str]) -> List[str]:
    """
    Read mapping_file (TSV: uuid<TAB>relative_path) and return a list of
    relative_paths whose uuid is in selected_uuids.
    """
    result = []
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: {mapping_file}:{line_num} has invalid format, skipped: {line}",
                      file=sys.stderr)
                continue
            uuid, rel_path = parts
            if uuid in selected_uuids:
                result.append(rel_path)
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Filter relative paths by a list of selected UUIDs."
    )
    parser.add_argument('mapping_file',
                        help='TSV file with two columns: uuid and relative_path (tab-separated)')
    parser.add_argument('selected_uuid_file',
                        help='File containing one selected UUID per line')
    args = parser.parse_args()

    selected = read_selected_uuids(args.selected_uuid_file)
    output_paths = filter_relative_paths(args.mapping_file, selected)

    # Print each relative path on its own line
    for path in output_paths:
        print(path)

if __name__ == '__main__':
    main()
