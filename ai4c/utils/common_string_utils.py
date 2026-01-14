import os


def read_file(file_path: str, encoding="utf-8") -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    content = None
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()
    return content
