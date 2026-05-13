# PassAgent

R2E-Gym extension for training agents on PassNet (AI for Compiler) optimization tasks.

## Overview

PassAgent provides the runtime, tools, and configurations needed to train reinforcement learning agents on compiler optimization tasks using the R2E-Gym framework. The agents learn to implement high-performance GPU kernels using Triton to optimize compiler passes.

## Architecture

This package extends R2E-Gym with PassNet-specific components:

- **Runtime** (`pass_agent/runtime/`): PassNetDocker runtime that handles environment setup
- **Agent** (`pass_agent/agent/`): PassAgent class extending R2E-Gym Agent
- **Configs** (`pass_agent/configs/`): Agent scaffold configurations
- **Examples** (`pass_agent/examples/`): Example scripts for running agents
- **Datasets** (`pass_agent/datasets/`): PassBench dataset files

## Installation

### Prerequisites

See [project root README](../README.md#quick-start) for environment requirements (Python, PyTorch, CUDA, Docker) and Docker image build instructions.

Additionally, PassBench sample data must be available (in `samples/` directory).

### Install PassAgent

```bash
cd pass_agent
pip install -r requirements.txt
```

This will install r2e-gym from GitHub, which is the only code dependency.

**Note on PassNet Dependency:**
PassAgent does NOT have a Python code dependency on PassNet. Instead:
1. **Docker Image**: Uses Docker image built from `../Dockerfile.nvidia`
2. **Data**: Mounts PassBench sample directories as volumes at runtime
3. **No imports**: Does not import any PassNet Python modules directly

## Usage

### Quick Start

```bash
cd pass_agent

# Run on demo dataset (10 tasks)
python examples/run_pass_agent_demo.py \
    --llm-name openai/glm-4.7 \
    --llm-base-url <your-llm-base-url> \
    --openai-api-key <your-api-key> \
    --dataset datasets/passbench_demo_dataset.jsonl \
    --max-steps 50 \
    --k 10
```

### Command Line Arguments

The `run_pass_agent_demo.py` script supports the following arguments:

- `--llm-name`: LLM model name (default: `openai/glm-4.7`)
- `--llm-base-url`: Base URL for LLM API (default: from `LLM_BASE_URL` env var)
- `--openai-api-key`: OpenAI API key (default: from `OPENAI_API_KEY` env var)
- `--anthropic-api-key`: Anthropic API key (default: from `ANTHROPIC_API_KEY` env var)
- `--dataset`: Path to dataset JSONL file (default: `datasets/passbench_demo_dataset.jsonl`)
- `--config`: Path to config directory (default: `configs/`)
- `--traj-dir`: Directory to save trajectories (default: `trajectories/pass_agent`)
- `--exp-name`: Experiment name (default: `pass_agent_full_trajectory`)
- `--max-steps`: Maximum steps per task (default: 100)
- `--temperature`: Sampling temperature (default: 1.0)
- `--max-workers`: Number of parallel workers (default: 1)
- `--start-idx`: Starting index in dataset (default: 0)
- `--k`: Number of tasks to run (default: None = all tasks)

### Output Files

The agent saves two JSONL files for each run:

1. **Trajectories** (`trajectories/pass_agent/{exp_name}.jsonl`):
   - Contains complete trajectory data for each task
   - Includes: trajectory_steps, problem_statement, exit_reason, reward, speedup metrics

2. **Completions** (`trajectories/pass_agent/{exp_name}_completions.jsonl`):
   - Contains full message history (LLM conversations) for each task
   - Format: `{"sample_dir": "...", "messages": [...]}`

### Environment Variables

```bash
export LLM_BASE_URL=<your-llm-base-url>
export OPENAI_API_KEY=<your-api-key>
export MAX_WORKERS=1

python examples/run_pass_agent_demo.py \
    --llm-name openai/glm-4.7 \
    --dataset datasets/passbench_demo_dataset.jsonl \
    --max-steps 50
```

## Directory Structure

```
pass_agent/
├── __init__.py
├── runtime/
│   ├── __init__.py
│   └── passnet_docker.py       # PassNetDocker runtime class
├── agent/
│   └── pass_agent.py           # PassAgent class
├── configs/
│   └── edit_fn_calling.yaml    # Agent scaffold config
├── datasets/
│   ├── passbench_demo_dataset.jsonl      # 10 sample tasks
│   └── passbench_demo_dataset_200.jsonl  # 200 sample tasks
├── examples/
│   ├── run_pass_agent_demo.py
│   └── create_passbench_dataset.py
├── requirements.txt
└── README.md
```

## How It Works

### PassNetDocker Runtime

The `PassNetDocker` class extends R2E-Gym's `DockerRuntime` to:

1. Mount PassNet workspace directory
2. Set up problem-specific working directory
3. Load target graph information
4. Track speedup history across iterations
5. Calculate rewards based on performance metrics

### Pass Evaluator Tool

The `pass_evaluator` tool (defined as a function-calling tool in PassAgent):

- Executes `entry.sh` in the problem directory
- Validates pass matching and correctness
- Reports performance metrics (speedup, correctness)
- Parses `aggregated_score.json` for results

### Agent Workflow

See [PassBench Evaluation Pipeline](../pass_bench/README.md) for the overall optimization flow. Within that pipeline, the agent interacts via two tools:

- **file_editor**: Creates/modifies pass files in `pass_dir/`
- **pass_evaluator**: Executes `entry.sh` and reports metrics back to the agent

## Dependencies

### Code Dependencies
- **r2e-gym**: Agent framework and orchestration (installed from GitHub: https://github.com/R2E-Gym/R2E-Gym)

### Runtime Dependencies (not Python packages)
- **PassNet Docker Image**: Built from `../Dockerfile.nvidia` (see [root README](../README.md#docker-usage))
- **PassBench Sample Data**: Located in `../samples/` directory, mounted at runtime
- **GPU**: NVIDIA GPU with CUDA support

### Dependency Model
```
pass_agent/
├── Code: imports r2e-gym only
├── Runtime: uses Docker image from ../Dockerfile.nvidia
└── Data: mounts ../samples/ as volume
```

PassAgent and PassNet are **deployment-coupled** but **code-decoupled**:
- They share the same repository for convenience
- Agent uses Docker image (built from `../Dockerfile.nvidia`) and data produced by PassNet
- No Python imports between them
