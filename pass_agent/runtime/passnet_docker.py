"""
PassNetDocker Runtime - extends R2E-Gym DockerRuntime for PassNet tasks.

This runtime handles PassAgent-specific setup including:
- Mounting PassNet workspace directory
- Installing pass_evaluator tool
- Setting problem path environment
- Loading target graph information
- Calculating rewards based on speedup and correctness
"""

import json
import re
import docker
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub import CMD_TIMEOUT


class PassNetDocker(DockerRuntime):
    """
    Docker runtime for PassNet pass optimization tasks.

    Tasks involve optimizing compiler passes by implementing high-performance
    custom kernels in the replacement_func. The agent should:
    1. View and understand the existing pass code structure
    2. Implement optimized replacement functions
    3. Validate the changes maintain semantic equivalence
    4. Run evaluations to verify performance improvements
    """

    def __init__(
        self,
        ds,  # dataset entry containing PassBench sample info
        repo_path: str = "/workspace",  # PassNet workspace path
        alt_path: str = "/root",
        docker_image: str = None,
        command: list = ["/bin/bash", "-l"],
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        """
        Initialize PassNet docker runtime.

        Args:
            ds: Dataset entry with fields like 'sample_dir', 'dsl', 'device', etc.
            repo_path: Path where PassNet workspace is mounted
            docker_image: PassNet docker image name
            command: Shell command to run in container
            logger: Logger instance
            backend: Runtime backend (docker/kubernetes/rle)
            **docker_kwargs: Additional docker arguments
        """
        # Store sample directory for this specific problem
        self.sample_dir = ds.get("sample_dir", "")
        if not self.sample_dir:
            raise ValueError("PassBench dataset must contain 'sample_dir' field")

        # Set the problem-specific working directory
        # This will be the directory where the agent works
        self.problem_path = f"/workspace/passnet/{self.sample_dir}"

        # Track speedup history across the trajectory
        self.speedup_history = []
        self.time_cost = {}

        # Add GPU support
        if "device_requests" not in docker_kwargs and backend == "docker":
            docker_kwargs["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        # Add privileged mode (required for proper GPU operation)
        if "privileged" not in docker_kwargs and backend == "docker":
            docker_kwargs["privileged"] = True

        # Add dummy fields required by r2e-gym's DockerRuntime but not needed for PassNet
        # The parent DockerRuntime expects these fields for SWE-bench tasks
        if "parsed_commit_content" not in ds:
            # Create a valid ParsedCommit mock with all required fields
            import json
            from datetime import datetime

            mock_parsed_commit = {
                "file_diffs": [],  # Empty list of file diffs
                "old_commit_hash": "0000000000000000000000000000000000000000",  # Dummy hash
                "new_commit_hash": "0000000000000000000000000000000000000000",  # Dummy hash
                "commit_message": "PassNet task - no commit",  # Placeholder message
                "commit_date": datetime.now().isoformat(),  # Current timestamp
                "metadata": {},  # Empty metadata
            }
            ds["parsed_commit_content"] = json.dumps(mock_parsed_commit)

        # Pass privileged mode and other docker_kwargs to parent

        super().__init__(
            ds=ds,
            repo_path=self.problem_path,  # Set repo_path to problem-specific directory
            alt_path=alt_path,
            docker_image=docker_image,
            command=command,
            logger=logger,
            backend=backend,
            **docker_kwargs,
        )

    def setup_env(self):
        """
        Setup the PassAgent environment by:
        1. Installing pass_evaluator tool
        2. Setting PYTHONPATH
        3. Verifying problem structure
        """
        try:
            self.run("mkdir -p /root/.venv/bin/")
            python_interpreter_path, exit_code = self.run("which python")
            self.run(f"ln -s {python_interpreter_path.strip()} /root/.venv/bin/python")

            # Verify problem directory exists
            output, code = self.run(f"ls -la {self.problem_path}")
            if code != "0":
                self.logger.error(f"Problem directory not found: {self.problem_path}")
                raise RuntimeError(
                    f"Problem directory does not exist: {self.problem_path}"
                )
            else:
                self.logger.info(f"✅ Problem directory verified: {self.problem_path}")

            # Verify entry.sh exists
            entry_sh = f"{self.problem_path}/entry.sh"
            output, code = self.run(
                f"test -f {entry_sh} && echo 'exists' || echo 'not found'"
            )
            if "exists" in output:
                self.logger.info(f"✅ entry.sh verified: {entry_sh}")
            else:
                self.logger.warning(f"⚠️  entry.sh not found at: {entry_sh}")

            # Verify pass_dir exists
            pass_dir = f"{self.problem_path}/pass_dir"
            output, code = self.run(
                f"test -d {pass_dir} && echo 'exists' || echo 'not found'"
            )
            if "exists" in output:
                self.logger.info(f"✅ pass_dir verified: {pass_dir}")

                # List pass files for debugging
                output, code = self.run(f"ls {pass_dir}/*.py 2>/dev/null | head -5")
                if code == "0" and output.strip():
                    self.logger.info(f"Pass files found:\n{output}")
            else:
                self.logger.warning(f"⚠️  pass_dir not found at: {pass_dir}")

            self.logger.info("✅ PassAgent environment setup completed")
            self.logger.info(f"📂 Working directory: {self.problem_path}")
            self.logger.info(f"📁 Pass directory: {self.problem_path}/pass_dir")

        except Exception as e:
            self.logger.error(f"❌ Error setting up PassAgent environment: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

    def run(
        self,
        code,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir=None,
        type: str = None,
    ) -> tuple[str, str]:
        """
        Override run() to set working directory to problem_path by default.

        This way pass_evaluator and all commands run in the problem directory,
        so they can access entry.sh and other files with relative paths.
        """
        # Set workdir to problem_path if not explicitly specified
        if workdir is None:
            workdir = self.problem_path

        return super().run(code, timeout=timeout, args=args, workdir=workdir, type=type)

    def get_task_instruction(self) -> str:
        """
        Get the task instruction for the pass optimization task.

        This method returns only the graph information which will be inserted
        into the {graph_info} placeholder in the scaffold's instance_prompt.

        Returns:
            Graph information string to be inserted into {graph_info} placeholder
        """
        # Try to read graph information from the problem directory
        graph_info = self._get_graph_info()

        if not graph_info:
            self.logger.error(
                "Failed to load target graph information - this is required!"
            )
            # Return error message if graph loading fails
            return "\n**ERROR:** Could not load target graph information. Please check graph_list.txt exists.\n"

        return graph_info

    def _get_graph_info(self) -> str:
        """
        Get target graph information if available.
        Returns a formatted string with model and weight metadata.
        """
        try:
            # Read graph_list.txt to find the graph path(s)
            graph_list_path = f"{self.problem_path}/graph_list.txt"
            output, code = self.run(f"cat {graph_list_path}", timeout=10)

            if code != "0" or not output.strip():
                self.logger.info("graph_list.txt not found or empty")
                return ""

            # graph_path is RELATIVE to problem_path and already includes "graphs/" prefix
            # e.g., "graphs/fusible_subgraphs/samples/timm/..."

            graph_paths = [
                line.strip() for line in output.strip().split("\n") if line.strip()
            ]

            graphs_data = []
            for graph_path in graph_paths:
                # Construct full path: problem_path + graph_path (graph_path already has "graphs/")
                graph_full_path = f"{self.problem_path}/{graph_path}"

                # Try to read model.py
                model_cmd = f"cat {graph_full_path}/model.py 2>/dev/null || echo ''"
                model_output, _ = self.run(model_cmd, timeout=10)

                # Try to read weight_meta.py
                weight_cmd = (
                    f"cat {graph_full_path}/weight_meta.py 2>/dev/null || echo ''"
                )
                weight_output, _ = self.run(weight_cmd, timeout=10)

                if model_output.strip() or weight_output.strip():
                    graphs_data.append(
                        {
                            "name": graph_path,
                            "model_code": model_output.strip(),
                            "weight_meta": weight_output.strip(),
                        }
                    )

            if not graphs_data:
                self.logger.info("No graph information found")
                return ""

            # Format the graph information similar to agent_analysis_pass.j2
            info = "\n**Target Graphs:**\n"
            for graph in graphs_data:
                info += f"\nGraph: {graph['name']}\n"
                info += f"Location: ./{graph['name']}/\n"
                info += f"Files available: model.py, weight_meta.py\n"

                if graph["model_code"]:
                    info += f"\nFile path: ./{graph['name']}/model.py\n"
                    info += "model.py (Computation to optimize):\n```python\n"
                    info += graph["model_code"]
                    info += "\n```\n"

                if graph["weight_meta"]:
                    info += f"\nFile path: ./{graph['name']}/weight_meta.py\n"
                    info += "weight_meta.py (Input shapes and types):\n```python\n"
                    info += graph["weight_meta"]
                    info += "\n```\n"

            self.logger.info(f"Successfully loaded {len(graphs_data)} graph(s)")
            return info

        except Exception as e:
            self.logger.warning(f"Could not load graph info: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())

        return ""

    def _calculate_reward(self, get_test_output=False, timeout: int = 600) -> float:
        """
        Calculate reward by running the evaluation script and parsing results.

        TODO: Implement proper reward calculation based on speedup and correctness metrics.
        For now, returns 1.0 to enable trajectory generation.

        Args:
            get_test_output: If True, return (reward, output) tuple
            timeout: Timeout for evaluation in seconds

        Returns:
            float: Reward value (currently always 1.0)
            or tuple: (reward, output) if get_test_output=True
        """
        # TODO: Implement actual reward calculation
        # For now, return 1.0 to enable trajectory output
        reward = 1.0
        output = "Reward calculation not yet implemented"

        if get_test_output:
            return reward, output
        return reward

        # Below is the original implementation to be completed later:
        """
        try:
            # Run entry.sh evaluation in the problem directory
            # Note: self.repo_path is already set to the problem-specific directory
            entry_sh = f"{self.problem_path}/entry.sh"

            # Run entry.sh evaluation
            self.logger.info(f"Running evaluation: {entry_sh}")
            output, exit_code = self.run(f"bash entry.sh", timeout=timeout, workdir=self.problem_path)

            self.logger.info(f"Evaluation completed with exit code: {exit_code}")

            # Parse the aggregated score
            score_file = "/tmp/workspace_graph_net_bench_test/aggregated_score.json"
            score_output, score_code = self.run(f"cat {score_file}")

            reward = 0.0
            if score_code == "0":
                try:
                    score_data = json.loads(score_output)
                    self.logger.info(f"Score data: {score_data}")

                    # Extract speedup - considers it successful if speedup > 1.0
                    speedup = score_data.get('speedup', 0.0)
                    correctness = score_data.get('correctness', False)

                    # Reward is 1.0 if speedup > 1.0 and correct
                    if speedup > 1.0 and correctness:
                        reward = 1.0
                        self.logger.info(f"✅ SUCCESS: Speedup={speedup}, Correctness={correctness}")
                    else:
                        self.logger.info(f"❌ FAIL: Speedup={speedup}, Correctness={correctness}")

                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse score JSON: {e}")
            else:
                self.logger.warning(f"Score file not found or could not be read")

            # Check for pass matching in output
            if "Has Any pass matched?" in output:
                if "[False]" in output:
                    self.logger.warning("⚠️  Pass did not match - pattern was not triggered")
                    reward = 0.0
                elif "[True]" in output:
                    self.logger.info("✅ Pass matched and was applied")

            if get_test_output:
                return reward, output
            return reward

        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            if get_test_output:
                return 0.0, f"ERROR: {str(e)}"
            return 0.0
        """

    def update_speedup(self, output) -> float:
        """
        Parse speedup from pass_evaluator output and add to history.

        pass_evaluator output contains lines like:
        - "rectified_speedup=0.79333"
        - "graph-net-test-compiler-log [Speedup][e2e]: 0.79333"

        Args:
            output: String output from pass_evaluator

        Returns:
            float: Parsed speedup value (0.0 if not found)
        """
        speedup = 0.0

        # Try to match rectified_speedup first (most reliable)
        match = re.search(r"rectified_speedup=([\d\.]+)", output)
        if match:
            speedup = float(match.group(1))
        else:
            # Fall back to Speedup log line
            match = re.search(r"\[Speedup\]\[e2e\]:\s*([\d\.]+)", output)
            if match:
                speedup = float(match.group(1))

        self.speedup_history.append(speedup)
        return speedup

    def update_time_cost(self, output):
        """
        Parse time costs from pass_evaluator output if available.

        Currently a stub as PassNet doesn't track stage-based time costs like KernelBench.
        Could be extended to track compilation time, evaluation time, etc.

        Args:
            output: String output from pass_evaluator
        """
        # TODO: Could parse timing information if needed
        # For now, just pass - the stub is here for API compatibility
        pass

    def get_speedup(self):
        """
        Get speedup history for trajectory tracking.

        Returns:
            list: List of speedup values tracked from pass_evaluator runs
        """
        return self.speedup_history

    def get_time_cost(self):
        """
        Get time cost breakdown for trajectory tracking.

        Returns:
            dict: Dictionary of time costs by stage (currently empty)
        """
        return self.time_cost
