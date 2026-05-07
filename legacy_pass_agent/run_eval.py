import json
import logging
from pathlib import Path
import shutil
from typing import Dict
import docker
import platform
import argparse

if platform.system() == "Linux":
    import resource
from docker.types import DeviceRequest 
from legacy_pass_agent.docker.docker_utils import setup_logger, close_logger
from legacy_pass_agent.docker.docker_builder import build_image


BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
WORKDIR = "/workspace"


class LocalTestSpec:
    """ Local Test Specification """
    def __init__(self, data: Dict, dataset_dir: Path):
        self.repo = data["repo"]
        self.instance_id = data["instance_id"]
        self.entry_point = data["entry"]
        self.graph_list = data["graph_list"]
        self.dataset_dir = dataset_dir
        
        self.image_tag = f"{self.repo}:{self.instance_id}".replace("/", "_").lower()
        
        self.local_repo_path = self.dataset_dir / self.repo / self.instance_id
        if not self.local_repo_path.exists():
            raise FileNotFoundError(f"Repository not found at {self.local_repo_path}")

    @property
    def dockerfile(self) -> str:
        """ Generate Dockerfile for local code injection """
        return f"""
FROM {BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1


WORKDIR {WORKDIR}

# Copy local code to container (during build, code will be copied to the repo directory in the build context)
COPY repo/ .

RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
"""


def prepare_build_context(spec: LocalTestSpec, build_dir: Path):
    """ prepare build context """
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # copy source code to build_dir/repo
    dest_repo_path = build_dir / "repo"
    shutil.copytree(spec.local_repo_path, dest_repo_path)


def run_evaluation(client: docker.DockerClient, spec: LocalTestSpec, output_dir: Path, logger: logging.Logger):
    """ Run Container to perform evaluation """

    container = None
    try:
        logger.info(f"Starting container for {spec.image_tag}...")

        gpu_requests = [
            DeviceRequest(count=-1, capabilities=[["gpu"]])
        ]
        
        # start container
        container = client.containers.run(
            spec.image_tag,
            command="tail -f /dev/null", # keep container running
            detach=True,
            device_requests=gpu_requests,
            working_dir=WORKDIR
        )

        # run entry.sh
        eval_cmd = f"./{spec.entry_point}"
        logger.info(f"Running evaluation command: {eval_cmd}")

        # execute command
        exec_result = container.exec_run(eval_cmd)
        exit_code = exec_result.exit_code
        output = exec_result.output.decode("utf-8", errors="replace")

        logger.info(f"Execution finished with exit code: {exit_code}")
        if exit_code == 0:
            logger.info("Batch execution successful.")
            logger.info(f"Output:\n{output}")
        else:
            logger.error(f"Batch execution failed (code {exit_code}). Output:\n{output}")

        results = {
            "instance_id": spec.instance_id,
            "cmd": eval_cmd,
            "graphs_evaluated": spec.graph_list,
            "exit_code": exit_code,
            "raw_output": output
        }

        result_file = output_dir / f"{spec.instance_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Full execution logs saved to {result_file}")

    except Exception as e:
        logger.error(f"Error running evaluation for {spec.instance_id}: {e}")
        raise e
    finally:
        if container:
            logger.info("Stopping and removing container...")
            container.stop()
            container.remove()


def main(args):
    assert args.dataset_dir is not None
    assert args.output_dir is not None

    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_NOFILE, (args.open_file_limit, args.open_file_limit))
    else:
        print("Warning: Resource limits not set (non-Linux system).")
    
    # Load dataset
    data_jsonl_file = Path(args.dataset_dir) / "data.jsonl"
    dataset = [json.loads(line) for line in Path(data_jsonl_file).read_text().splitlines()]

    # foreach instance to eval
    client = docker.from_env()
    for item in dataset:
        spec = LocalTestSpec(item, dataset_path)
        log_file = output_path / f"{spec.instance_id}_build.log"
        logger = setup_logger(spec.instance_id, log_file, add_stdout=True)
        logger.info(f"Processing instance: {spec.instance_id}")

        # build image
        build_context_dir = output_path / "build_temp" / spec.instance_id
        prepare_build_context(spec, build_context_dir)

        build_image(
            image_name=spec.image_tag,
            setup_scripts={},
            dockerfile=spec.dockerfile,
            platform=None,
            client=client,
            build_dir=build_context_dir,
            nocache=args.clean
        )

        run_evaluation(client, spec, output_path, logger)
        close_logger(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-d",
        "--dataset-dir",
        required=False,
        default=None,
        type=str,
        help="Path to the dataset directory (include data.jsonl file and repos)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        type=str,
        help="Path to the passes directory",
    )
    parser.add_argument(
        "--cache-level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="instance",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", action="store_true", help="Clean images above cache level"
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )

    args = parser.parse_args()
    main(args=args)
    