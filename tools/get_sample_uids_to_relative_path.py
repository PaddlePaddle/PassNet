#!/usr/bin/env python3
"""
get_sample_uids_to_relative_path.py

For each 'sample_uids.txt' file found recursively under a given root directory,
output a line containing the entire file content followed by a tab and the
relative path of the file's parent directory (relative to the root, without a
leading './').

Note: If a file's content contains newlines, those newlines become part of the
output line, which may affect parsing. This script does not escape them.
"""

import argparse
from pathlib import Path
from typing import List

def get_sample_uids_to_relative_path(root_dir: str) -> List[str]:
    """
    Core logic: yield lines formatted as "content\trelative_path" for each
    sample_uids.txt under root_dir.
    """
    root_path = Path(root_dir).expanduser().resolve()
    lines = []

    # Find all sample_uids.txt files recursively
    file_paths = list(root_path.rglob('sample_uids.txt'))

    for file_path in file_paths:
        try:
            # Read entire file content as a string (preserving newlines)
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            # Skip unreadable files
            continue

        # Compute relative path of parent directory
        try:
            rel_dir = file_path.parent.relative_to(root_path)
        except ValueError:
            # Fallback (should not happen) – use absolute path
            rel_dir = file_path.parent

        rel_str = str(rel_dir)
        # Remove possible leading './' (though relative_to should not produce it)
        if rel_str.startswith('./'):
            rel_str = rel_str[2:]

        # Append formatted line
        lines.append(f"{content}\t{rel_str}")

    return lines

def main():
    parser = argparse.ArgumentParser(
        description="Map contents of sample_uids.txt files to relative paths."
    )
    parser.add_argument('root_dir', help='Root directory to search recursively')
    args = parser.parse_args()

    output_lines = get_sample_uids_to_relative_path(args.root_dir)
    for line in output_lines:
        print(line)

if __name__ == '__main__':
    main()
