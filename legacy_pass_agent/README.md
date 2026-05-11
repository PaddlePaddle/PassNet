# Legacy PassAgent

> This is the original single-script pass generation agent. For the current R2E-Gym based agent, see [pass_agent/](../pass_agent/).

See [project root README](../README.md#quick-start) for environment setup (Python, PyTorch, Docker image build).

## Usage

### Environment Variables

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/passnet/repo/root/dir
export PASSNET_BASE_URL='your llm base url'
export PASSNET_API_KEY='your llm api key'
export PASSNET_API_MODEL_NAME='your llm api model name'
```

### Run (Local)

```bash
cd /path/to/passnet/repo/root/dir

# Generate pass for a sample
python3 -m legacy_pass_agent.naive_pass_generate_agents --max-turn 1 --dsl triton \
    --model-dir samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4

# Verify generated result
bash samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4/entry.sh
```

### Run (Docker)

```bash
docker run --gpus all --privileged \
    -v <path-to-local-passnet-project>:/workspace \
    -w /workspace \
    -e PASSNET_BASE_URL=<your-llm-base-url> \
    -e PASSNET_API_KEY=<your-llm-api-key> \
    -e PASSNET_API_MODEL_NAME=<your-llm-model-name> \
    passnet:latest \
    python3 -m legacy_pass_agent.naive_pass_generate_agents --max-turn 1 --dsl triton \
        --model-dir samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4
```
