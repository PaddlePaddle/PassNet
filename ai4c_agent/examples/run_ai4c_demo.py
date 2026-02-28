#!/usr/bin/env python3
"""
Run AI4C agent on demo dataset.

This script demonstrates how to run the AI4C agent using direct imports.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import concurrent.futures

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AI4C agent on demo dataset"
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=os.getenv("LLM_BASE_URL"),
        help="Base URL for LLM API"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (default: from env var)"
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=os.getenv("ANTHROPIC_API_KEY"),
        help="Anthropic API key (default: from env var)"
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default="openai/glm-4.7",
        help="LLM model name (default: openai/glm-4.7)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./datasets/ai4c_demo_dataset.jsonl",
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs",
        help="Path to config directory"
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="./trajectories/ai4c",
        help="Directory to save trajectories"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="ai4c_full_trajectory",
        help="Experiment name"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per task"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of tasks to run (None = all)"
    )
    return parser.parse_args()

# Parse args early so we can set env vars before imports
args = parse_args()

# Set environment variables FIRST (before any r2e-gym imports)
os.environ["LLM_BASE_URL"] = args.llm_base_url
if args.openai_api_key:
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
if args.anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
os.environ["MAX_WORKERS"] = str(args.max_workers)

# Add ai4c_agent directory to path
# sys.path.insert(0, "/ssd1/hesijun/ai4c/ai4c_agent")

# Import AI4C runtime directly from runtime folder
from runtime.ai4c_docker import AI4CDocker

# Monkey-patch: Replace DockerRuntime with AI4CDocker BEFORE importing r2e-gym
import r2egym.agenthub.runtime.docker as docker_module
docker_module.DockerRuntime = AI4CDocker

# Import AI4CAgent
from agent.ai4c_agent import AI4CAgent

# Monkey-patch: Allow custom config path via scaffold parameter
import r2egym.agenthub.run.edit as edit_module
from r2egym.agenthub.agent.agent import AgentArgs
from pathlib import Path

original_runagent = edit_module.runagent

def patched_runagent(ds, scaffold="r2egym", **kwargs):
    """Patched runagent that accepts custom config paths"""
    # Check if scaffold is a custom path (contains "/" or is absolute)
    if "/" in scaffold:
        # Custom config path - load it directly
        import time
        from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
        from r2egym.agenthub.agent.agent import Agent
        from r2egym.logging import setup_logging, INFO
        from datetime import datetime

        exp_name = kwargs.get('exp_name')
        llm_name = kwargs.get('llm_name', 'gpt-4o')
        max_steps = kwargs.get('max_steps', 40)
        use_fn_calling = kwargs.get('use_fn_calling', True)
        backend = kwargs.get('backend', 'docker')
        max_reward_calc_time = kwargs.get('max_reward_calc_time', 300)

        logger = setup_logging(
            name=ds["docker_image"].replace("/", "_"),
            log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}.log",
            console=True,
            level=INFO,
        )

        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize environment
        env_args = EnvArgs(ds=ds)
        env = RepoEnv(env_args, logger=logger, backend=backend)

        # Load our custom config
        config_file = Path(scaffold) / "edit_fn_calling.yaml" if use_fn_calling else Path(scaffold) / "edit_non_fn_calling.yaml"
        agent_args = AgentArgs.from_yaml(config_file)
        agent_args.llm_name = llm_name

        # Initialize agent - use AI4CAgent instead of generic Agent
        agent = AI4CAgent(name="AI4CAgent", args=agent_args, logger=logger)

        # Run agent
        try:
            from r2egym.agenthub.run.edit import run_agent_with_restarts
            trajectory = run_agent_with_restarts(
                agent, env,
                max_steps=max_steps,
                num_restarts=kwargs.get('num_restarts', 1),
                temperature=kwargs.get('temperature', 0),
                max_steps_absolute=kwargs.get('max_steps_absolute', 50),
                use_fn_calling=use_fn_calling,
                max_iterations=kwargs.get('max_iterations', 1),
                scaffold="r2egym",  # Use r2egym scaffold, AI4C tools detected via command_files
                max_tokens=kwargs.get('max_tokens', 65536),
            )
            logger.info(f"Trajectory type: {type(trajectory)}")

            # Get history from agent instance (stored as final_history)
            history = getattr(agent, 'final_history', None)
            logger.info(f"History retrieved: {history is not None}")
        except Exception as e:
            logger.error(f"Error during agent run: {e}")
            import traceback
            logger.error(traceback.format_exc())
            env.close()
            return None

        # Calculate reward
        reward_calc_time = time.time()
        reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
        reward_calc_time = time.time() - reward_calc_time

        env.close()

        trajectory.reward = reward
        trajectory.test_output = test_output
        trajectory.ds = ds
        trajectory.exp_name = exp_name
        trajectory.reward_calc_time = reward_calc_time

        # Return both trajectory and history
        return trajectory.model_dump_json(), history
    else:
        # Use original for standard scaffolds
        return original_runagent(ds=ds, scaffold=scaffold, **kwargs)

edit_module.runagent = patched_runagent

# Now import the patched runagent
from r2egym.agenthub.run.edit import runagent


def run_multiple_ai4c(
    dataset_path: str,
    traj_dir: str,
    exp_name: str,
    config_path: str,
    llm_name: str = "openai/glm-4.7",
    max_steps: int = 100,
    max_steps_absolute: int = 100,
    temperature: float = 1.0,
    max_iterations: int = 1,
    max_tokens: int = 131072,
    start_idx: int = 0,
    k: int = None,
    max_workers: int = 1,
):
    """
    Run AI4C agent on multiple tasks from a JSONL dataset.

    Args:
        dataset_path: Path to JSONL dataset file
        traj_dir: Directory to save trajectories
        exp_name: Experiment name
        config_path: Path to config directory
        llm_name: LLM model name
        max_steps: Maximum steps per task
        max_steps_absolute: Absolute maximum steps
        temperature: Sampling temperature
        max_iterations: Maximum iterations
        max_tokens: Maximum tokens
        start_idx: Starting index in dataset
        k: Number of tasks to run (None = all)
        max_workers: Number of parallel workers
    """
    # Load dataset from JSONL
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # Select subset
    if k is None:
        k = len(dataset)
    ds_selected = dataset[start_idx:start_idx + k]

    print(f"Loaded {len(dataset)} total tasks")
    print(f"Running on {len(ds_selected)} tasks (index {start_idx} to {start_idx + len(ds_selected) - 1})")
    print(f"Max workers: {max_workers}")
    print("-" * 80)

    # Create trajectories directory
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate JSONL output files
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"
    completions_file = traj_dir_path / f"{exp_name}_completions.jsonl"
    print(f"Trajectories will be saved to: {jsonl_file}")
    print(f"Completions (history) will be saved to: {completions_file}")
    print("-" * 80)

    # Run tasks
    results = []

    def run_single_task(ds_entry, idx):
        """Run agent on a single task"""
        print(f"\n[Task {idx+1}/{len(ds_selected)}] Starting: {ds_entry.get('sample_dir', ds_entry.get('instance_id', 'unknown'))}")

        try:
            result = runagent(
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=1,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                use_fn_calling=True,
                backend="docker",
                max_reward_calc_time=600,
                max_iterations=max_iterations,
                scaffold=config_path,  # Just pass the config directory path
                max_tokens=max_tokens,
            )

            if result:
                # Unpack result - it's either (trajectory_json, history) tuple or just trajectory_json
                if isinstance(result, tuple):
                    trajectory_json, history = result
                else:
                    trajectory_json = result
                    history = None

                # Write trajectory to JSONL file immediately
                with open(jsonl_file, 'a') as f:
                    f.write(trajectory_json + '\n')

                # Write history (completions) to separate JSONL file
                if history:
                    import json
                    history_entry = {
                        "sample_dir": ds_entry.get('sample_dir', ds_entry.get('instance_id', 'unknown')),
                        "messages": history,
                    }
                    with open(completions_file, 'a') as f:
                        f.write(json.dumps(history_entry, ensure_ascii=False) + '\n')

                print(f"[Task {idx+1}/{len(ds_selected)}] Completed successfully")
                return trajectory_json
            else:
                print(f"[Task {idx+1}/{len(ds_selected)}] Failed (returned None)")
                return None

        except Exception as e:
            print(f"[Task {idx+1}/{len(ds_selected)}] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Run tasks in parallel or sequential
    if max_workers == 1:
        # Sequential execution
        for idx, ds_entry in enumerate(ds_selected):
            result = run_single_task(ds_entry, idx)
            results.append(result)
    else:
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_task, ds_entry, idx): (ds_entry, idx)
                for idx, ds_entry in enumerate(ds_selected)
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

    print("\n" + "=" * 80)
    print(f"Completed {len([r for r in results if r])} / {len(ds_selected)} tasks successfully")
    print(f"Trajectories saved to: {jsonl_file}")
    print(f"Completions saved to: {completions_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    print(f"Running AI4C agent on dataset: {args.dataset}")
    print(f"Trajectories will be saved to: {args.traj_dir}")
    print(f"Using config from: {args.config}")
    print(f"LLM: {args.llm_name}")
    print("-" * 80)

    # Run AI4C agent
    results = run_multiple_ai4c(
        dataset_path=args.dataset,
        traj_dir=args.traj_dir,
        exp_name=args.exp_name,
        config_path=args.config,
        llm_name=args.llm_name,
        max_steps=args.max_steps,
        max_steps_absolute=args.max_steps,
        temperature=args.temperature,
        max_iterations=1,
        max_tokens=131072,
        start_idx=args.start_idx,
        k=args.k,
        max_workers=args.max_workers,
    )

    print("\n" + "=" * 80)
    print(f"All tasks completed!")
