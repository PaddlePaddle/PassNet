# AI4C Agent Examples

This directory contains example scripts for running the AI4C agent.

## Available Scripts

### `run_ai4c_demo.py`
Run the agent on the 10-task demo dataset.

**Usage:**
```bash
cd examples/
python run_ai4c_demo.py
```

**Configuration:**
- Dataset: `../datasets/ai4c_demo_dataset.jsonl` (10 tasks)
- Max steps: 100
- LLM: openai/glm-4.7 (configurable via env var)
- Output: `../trajectories/ai4c/`

### `run_ai4c_single.py`
Run the agent on a single task for quick testing.

**Usage:**
```bash
cd examples/
python run_ai4c_single.py
```

**Configuration:**
- Dataset: `../datasets/ai4c_single.jsonl` (1 task)
- Max steps: 50
- LLM: openai/gpt-4 (configurable via env var)
- Output: `../trajectories/single/`

### `create_ai4c_dataset.py`
Create custom AI4C datasets from sample directories.

**Usage:**
```bash
cd examples/
python create_ai4c_dataset.py
```

## Environment Variables

All scripts support these environment variables:

- `LLM_BASE_URL`: Base URL for LLM API (default: http://yy.dbh.baidu-int.com/v1)
- `OPENAI_API_KEY`: API key for OpenAI/compatible API
- `ANTHROPIC_API_KEY`: API key for Anthropic API
- `MAX_WORKERS`: Maximum parallel workers (default: 1)

**Example:**
```bash
export OPENAI_API_KEY="your-key-here"
export LLM_BASE_URL="https://api.openai.com/v1"
python run_ai4c_single.py
```

## How These Scripts Work

1. **Import AI4C Runtime**: Scripts import `AI4CDocker` from `ai4c_agent.runtime`
   ```python
   from ai4c_agent.runtime import AI4CDocker
   ```

2. **Load Dataset**: Load JSONL dataset using HuggingFace datasets library

3. **Run Agent**: Call r2e-gym's `runagent_multiple` with `runtime_class=AI4CDocker`
   ```python
   results = runagent_multiple(
       dataset=dataset,
       runtime_class=AI4CDocker,  # Use AI4C runtime directly
       scaffold="configs/",
       ...
   )
   ```

4. **Save Trajectories**: Results are saved to `trajectories/` directory

**Note:** We don't need to register the runtime with a map since we only care about AI4C in this repo. We just pass `runtime_class=AI4CDocker` directly.

## Customization

To customize the agent behavior, edit the scripts and modify:

- `--max_steps`: Maximum number of agent steps
- `--llm_name`: LLM model to use
- `--temperature`: Sampling temperature
- `--backend`: Runtime backend (docker/kubernetes/rle)
- `--scaffold`: Path to config directory (defaults to `../configs`)

## Shell Wrapper

For convenience, you can also use the bash wrapper:

```bash
# From ai4c_agent root directory
bash scripts/run_ai4c.sh
```

This wrapper calls `examples/run_ai4c_demo.py` with proper environment setup.
