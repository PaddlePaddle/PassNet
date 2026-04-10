# ai4c



### Demo

Requirements:

1. python>=3.10
2. torch>=2.9.0

```bash
cd /path/to/ai4c/repo/root/dir

export PYTHONPATH=$PYTHONPATH:/path/to/ai4c/repo/root/dir
export AI4C_BASE_URL='your llm base url'
export AI4C_API_KEY='your llm api key'
export AI4C_API_MODEL_NAME=' your llm api model name'

# 运行实例样本
python3 -m ai4c.naive_pass_generate_agents --max-turn 1 --dsl triton  --model-dir samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4

# 验证生成结果
samples/fusible_subgraphs/bat_resnext26ts.ch_in1k/bat_resnext26ts.ch_in1k_0_start11_end15_2/entry.sh
```

### Docker

#### Building the Docker Image

```bash
docker build . -t ai4c:latest -f Dockerfile.nvidia
```

#### Run evaluation with Docker

```bash
docker run --gpus all --privileged \
          -v <path-to-local-ai4c-project>:/workspace \
          -w /workspace \
          -e AI4C_BASE_URL=<your-llm-base-url> \
          -e AI4C_API_KEY=<your-llm-api-key> \
          -e AI4C_API_MODEL_NAME=<your-llm-model-name> \
          ai4c:latest \
          python3 -m ai4c.naive_pass_generate_agents --max-turn 1 --dsl triton --model-dir samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4


docker run --gpus all --privileged \
          -v <path-to-local-ai4c-project>:/workspace \
          -w /workspace \
          ai4c:latest \
          bash samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4/entry.sh
```