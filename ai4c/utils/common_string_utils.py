import os
import re


def write_file(file_path: str, content: str, mode="w", encoding="utf-8") -> None:
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)
    with open(file_path, mode, encoding=encoding) as f:
        f.write(content)


def read_file(file_path: str, encoding="utf-8") -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    content = None
    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()
    return content


def extract_code_blocks(text, code_language_types: list[str]) -> str:
    """
    Extract all code blocks from text, combine them to return as a single string
    """
    pattern = r"```.*?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type) :].strip()
        combined_code.append(code)

    return " \n ".join(combined_code) if combined_code else ""
