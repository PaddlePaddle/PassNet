# AI4C Agent

R2E-Gym extension for training agents on AI4C (AI for Compiler) optimization tasks.

## Overview

AI4C Agent provides the runtime, tools, and configurations needed to train reinforcement learning agents on compiler optimization tasks using the R2E-Gym framework. The agents learn to implement high-performance GPU kernels using Triton to optimize compiler passes.

## Architecture

This package extends R2E-Gym with AI4C-specific components:

- **Runtime** (`pass_agent/runtime/`): AI4CDocker runtime that handles AI4C environment setup
- **Tools** (`pass_agent/tools/`): pass_evaluator tool for running evaluations
- **Configs** (`pass_agent/configs/`): Agent scaffold configurations
- **Examples** (`pass_agent/examples/`): Example scripts for running agents
- **Tests** (`pass_agent/tests/`): Test files for development
- **Docs** (`pass_agent/docs/`): Detailed documentation

## Installation

### Prerequisites

- Python 3.10+
- Docker with GPU support
- AI4C Docker image built (from parent AI4C project's `Dockerfile.nvidia`)
- AI4C sample data available (in parent `../samples/` directory)

### Install AI4C Agent

```bash
cd ai4c/pass_agent
pip install -r requirements.txt
```

This will install r2e-gym from GitHub, which is the only code dependency.

**Note on AI4C Dependency:**
AI4C-agent does NOT have a Python code dependency on AI4C. Instead:
1. **Docker Image**: Uses Docker image built from `../Dockerfile.nvidia` in parent AI4C project
2. **Data**: Mounts AI4C sample directories as volumes at runtime
3. **No imports**: Does not import any AI4C Python modules directly

### Build AI4C Docker Image

Before using ai4c-agent, build the Docker image from the parent AI4C project:

```bash
# From the parent ai4c directory (not pass_agent)
cd ai4c
docker build -f Dockerfile.nvidia -t ai4c:latest .

# This creates the Docker image that ai4c-agent will use at runtime
```

## Usage

### Quick Start

Run the AI4C agent with a simple command:

```bash
cd pass_agent

# Run on demo dataset (10 tasks)
python examples/run_ai4c_demo.py \
    --llm-name openai/glm-4.7 \
    --llm-base-url http://127.0.0.0:8000/v1 \
    --openai-api-key sk-YOUR_API_KEY \
    --dataset datasets/ai4c_demo_dataset.jsonl \
    --max-steps 50 \
    --k 10

# Run on a single task for testing
python examples/run_ai4c_demo.py \
    --llm-name openai/glm-4.7 \
    --llm-base-url http://127.0.0.0:8000/v1 \
    --openai-api-key sk-YOUR_API_KEY \
    --dataset datasets/ai4c_single.jsonl \
    --max-steps 5 \
    --k 1
```

### Command Line Arguments

The `run_ai4c_demo.py` script supports the following arguments:

- `--llm-name`: LLM model name (default: `openai/glm-4.7`)
  - Format: `openai/model-name` for OpenAI-compatible APIs
  - Examples: `openai/glm-4.7`, `anthropic/claude-3-opus`, `gpt-4o`

- `--llm-base-url`: Base URL for LLM API (default: from `LLM_BASE_URL` env var)
  - Example: `http://127.0.0.0:8000/v1`

- `--openai-api-key`: OpenAI API key (default: from `OPENAI_API_KEY` env var)
  - Required for authentication

- `--anthropic-api-key`: Anthropic API key (default: from `ANTHROPIC_API_KEY` env var)
  - Use if running Claude models

- `--dataset`: Path to dataset JSONL file
  - Default: `datasets/ai4c_demo_dataset.jsonl`
  - Single task: `datasets/ai4c_single.jsonl`

- `--config`: Path to config directory (default: `configs/`)
  - Contains agent scaffold configuration (edit_fn_calling.yaml)

- `--traj-dir`: Directory to save trajectories (default: `trajectories/ai4c`)

- `--exp-name`: Experiment name (default: `ai4c_full_trajectory`)
  - Used for trajectory and completion filenames

- `--max-steps`: Maximum steps per task (default: 100)

- `--temperature`: Sampling temperature (default: 1.0)

- `--max-workers`: Number of parallel workers (default: 1)
  - Set to > 1 for parallel execution

- `--start-idx`: Starting index in dataset (default: 0)

- `--k`: Number of tasks to run (default: None = all tasks)

### Output Files

The agent saves two JSONL files for each run:

1. **Trajectories** (`trajectories/ai4c/{exp_name}.jsonl`):
   - Contains complete trajectory data for each task
   - Includes: trajectory_steps, problem_statement, exit_reason, reward, speedup metrics
   - Format: One JSON object per line (one task per line)

2. **Completions** (`trajectories/ai4c/{exp_name}_completions.jsonl`):
   - Contains full message history (LLM conversations) for each task
   - Includes: all system/user/assistant/tool messages
   - Format: `{"sample_dir": "...", "messages": [...]}`
   - Useful for debugging and analyzing LLM behavior

### Environment Variables

You can also set these as environment variables instead of command-line arguments:

```bash
export LLM_BASE_URL=http://127.0.0.0:8000/v1
export OPENAI_API_KEY=sk-YOUR_API_KEY
export MAX_WORKERS=1

python examples/run_ai4c_demo.py \
    --llm-name openai/glm-4.7 \
    --dataset datasets/ai4c_demo_dataset.jsonl \
    --max-steps 50
```

**Security Note:** Never commit API keys to version control! Use environment variables or secure credential management systems in production.

### Example: Running on Different Datasets

```bash
# Demo dataset (10 tasks)
python examples/run_ai4c_demo.py \
    --llm-name openai/glm-4.7 \
    --llm-base-url your-base-url \
    --openai-api-key sk-YOUR_API_KEY \
    --dataset datasets/ai4c_demo_dataset.jsonl \
    --max-steps 50 \
    --k 10

# Single task for testing (fast)
python examples/run_ai4c_demo.py \
    --llm-name openai/glm-4.7 \
    --llm-base-url http://127.0.0.0:8000/v1 \
    --openai-api-key sk-YOUR_API_KEY \
    --dataset datasets/ai4c_single.jsonl \
    --max-steps 5 \
    --k 1

# Parallel execution (4 workers)
python examples/run_ai4c_demo.py \
    --llm-name openai/glm-4.7 \
    --llm-base-url http://127.0.0.0:8000/v1 \
    --openai-api-key sk-YOUR_API_KEY \
    --dataset datasets/ai4c_demo_dataset.jsonl \
    --max-steps 50 \
    --max-workers 4 \
    --k 10
```

## Directory Structure

```
pass_agent/
├── __init__.py
├── runtime/
│   ├── __init__.py
│   └── ai4c_docker.py       # AI4CDocker runtime class
├── tools/
│   ├── __init__.py
│   └── pass_evaluator.py    # Pass evaluation tool
├── configs/
│   └── edit_fn_calling.yaml # Agent scaffold config
├── datasets/
│   ├── README.md            # Dataset documentation
│   └── ai4c_demo_dataset.jsonl  # 10 sample tasks
├── examples/
│   └── create_ai4c_dataset.py
├── scripts/
│   └── run_ai4c_full.sh
├── requirements.txt
└── README.md
```

## How It Works

### AI4CDocker Runtime

The `AI4CDocker` class extends R2E-Gym's `DockerRuntime` to:

1. Mount AI4C workspace directory
2. Set up problem-specific working directory
3. Inject `AI4C_PROBLEM_PATH` environment variable
4. Load target graph information
5. Track speedup history across iterations
6. Calculate rewards based on performance metrics

### Pass Evaluator Tool

The `pass_evaluator.py` tool:

- Executes `entry.sh` in the problem directory
- Validates pass matching and correctness
- Reports performance metrics (speedup, correctness)
- Parses `aggregated_score.json` for results

### Agent Workflow

1. **Analyze target computation**: Study graph info (model.py, weight_meta.py)
2. **Design optimization passes**: Create pass files with pattern matching
3. **Implement optimized kernels**: Write high-performance Triton kernels
4. **Run evaluation**: Use pass_evaluator tool
5. **Iterate for better performance**: Adjust implementation based on results

## Dependencies

### Code Dependencies
- **r2e-gym**: Agent framework and orchestration (installed from GitHub: https://github.com/R2E-Gym/R2E-Gym)

### Runtime Dependencies (not Python packages)
- **AI4C Docker Image**: Built from `../Dockerfile.nvidia`, contains torch/triton/evaluation scripts
- **AI4C Sample Data**: Located in parent `../samples/` directory, mounted at runtime
- **GPU**: NVIDIA GPU with CUDA support

### Dependency Model
```
ai4c-agent/
├── Code: imports r2e-gym only
├── Runtime: uses Docker image from ../Dockerfile.nvidia
└── Data: mounts ../samples/ as volume
```

AI4C-agent and AI4C are **deployment-coupled** but **code-decoupled**:
- They share the same repository for convenience
- Agent uses Docker image (built from `../Dockerfile.nvidia`) and data produced by AI4C
- No Python imports between them

## Contributing

This is part of the AI4C project. See the parent AI4C repository for contribution guidelines.

## License

See the parent AI4C repository for license information.
