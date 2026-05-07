"""
Modify from https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/docker_build.py
"""
import docker
from pathlib import Path
from .docker_utils import run_threadpool, setup_logger, ansi_escape, close_logger


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def build_image(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    client: docker.DockerClient,
    build_dir: Path,
    nocache: bool = False,
):
    """
    Builds a docker image with the given name, setup scripts, dockerfile, and platform.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform to build the image for
        client (docker.DockerClient): Docker client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """
    # Create a logger for the build process
    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    try:
        # Write the setup scripts to the build directory
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        # Write the dockerfile to the build directory
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build the image
        logger.info(
            f"Building docker image {image_name} in {build_dir} with platform {platform}"
        )
        response = client.api.build(
            path=str(build_dir),
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform=platform,
            nocache=nocache,
        )

        # Log the build process continuously
        buildlog = ""
        for chunk in response:
            if "stream" in chunk:
                # Remove ANSI escape sequences from the log
                chunk_stream = ansi_escape(chunk["stream"])
                logger.info(chunk_stream.strip())
                buildlog += chunk_stream
            elif "errorDetail" in chunk:
                # Decode error message, raise BuildError
                logger.error(f"Error: {ansi_escape(chunk['errorDetail']['message'])}")
                raise docker.errors.BuildError(
                    chunk["errorDetail"]["message"], buildlog
                )
        logger.info("Image built successfully!")
    except docker.errors.BuildError as e:
        logger.error(f"docker.errors.BuildError during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)  # functions that create loggers should close them
