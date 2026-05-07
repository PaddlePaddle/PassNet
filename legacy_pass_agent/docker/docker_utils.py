"""
Modify from https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/docker_utils.py
"""

import re
import sys
import docker
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def setup_logger(
    instance_id: str,
    log_file: Path,
    mode="w",
    add_stdout: bool = False,
    encoding="utf-8",
):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.

    If `add_stdout` is True, logs will also be sent to stdout, which can be used for
    streaming ephemeral output from Modal containers.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode, encoding=encoding)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    if add_stdout:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"%(asctime)s - {instance_id} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    # To avoid too many open files
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def list_images(client: docker.DockerClient):
    """
    List all images from the Docker client.
    NOTE: don't use this in multi-threaded context
    """
    return {tag for i in client.images.list(all=True) for tag in i.tags}


def run_sequential(func, payloads):
    """
    Run a function with a list of payloads sequentially.

    Args:
        func: Function to run for each payload
        payloads: List of payloads to process

    Returns:
        tuple: (succeeded, failed) lists of payloads
    """
    succeeded, failed = [], []
    for payload in payloads:
        try:
            func(*payload)
            succeeded.append(payload)
        except Exception:
            traceback.print_exc()
            failed.append(payload)
    return succeeded, failed


def run_threadpool(func, payloads, max_workers):
    """
    Run a function with a list of payloads using ThreadPoolExecutor.

    Args:
        func: Function to run for each payload
        payloads: List of payloads to process
        max_workers: Maximum number of worker threads

    Returns:
        tuple: (succeeded, failed) lists of payloads
    """
    if max_workers <= 0:
        return run_sequential(func, payloads)
    succeeded, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for running each instance
        futures = {executor.submit(func, *payload): payload for payload in payloads}
        # Wait for each future to complete
        for future in as_completed(futures):
            try:
                # Check if instance ran successfully
                future.result()
                succeeded.append(futures[future])
            except Exception as e:
                print(f"{type(e)}: {e}")
                traceback.print_exc()
                failed.append(futures[future])
    return succeeded, failed


def ansi_escape(text: str) -> str:
    """
    Remove ANSI escape sequences from text
    """
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)
