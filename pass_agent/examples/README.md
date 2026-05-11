# PassAgent Examples

This directory contains example scripts for running the PassAgent.

## Available Scripts

### `run_pass_agent_demo.py`
Run the agent on the 10-task demo dataset.

**Usage:**
```bash
cd examples/
python run_pass_agent_demo.py
```

**Configuration:**
- Dataset: `../datasets/passbench_demo_dataset.jsonl` (10 tasks)
- Max steps: 100
- LLM: openai/glm-4.7 (configurable via env var)
- Output: `../trajectories/pass_agent/`

### `create_passbench_dataset.py`
Create custom PassBench datasets from sample directories.

**Usage:**
```bash
cd examples/
python create_passbench_dataset.py
```

## Environment Variables

All scripts support these environment variables:

- `LLM_BASE_URL`: Base URL for LLM API
- `OPENAI_API_KEY`: API key for OpenAI/compatible API
- `ANTHROPIC_API_KEY`: API key for Anthropic API
- `MAX_WORKERS`: Maximum parallel workers (default: 1)

**Example:**
```bash
export OPENAI_API_KEY="your-key-here"
export LLM_BASE_URL="https://api.openai.com/v1"
python run_pass_agent_demo.py
```

## How These Scripts Work

1. **Import PassNet Runtime**: Scripts import `PassNetDocker` from `runtime`
   ```python
   from runtime.passnet_docker import PassNetDocker
   ```

2. **Monkey-patch DockerRuntime**: Replace r2e-gym's `DockerRuntime` with `PassNetDocker`
   ```python
   import r2egym.agenthub.runtime.docker as docker_module
   docker_module.DockerRuntime = PassNetDocker
   ```

3. **Load Dataset**: Load JSONL dataset (PassBench format)

4. **Run Agent**: Call patched `runagent` with PassAgent and custom config path

5. **Save Trajectories**: Results are saved to `trajectories/pass_agent/` directory

## Customization

To customize the agent behavior, modify CLI arguments:

- `--max-steps`: Maximum number of agent steps
- `--llm-name`: LLM model to use
- `--temperature`: Sampling temperature
- `--config`: Path to config directory (defaults to `../configs`)
- `--max-workers`: Number of parallel workers
