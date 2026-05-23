# PassAgent

基于 R2E-Gym 框架，面向 PassNet（AI for Compiler）编译器优化任务的 Agent 评估框架。

## 概述

PassAgent 提供了在 R2E-Gym 框架上训练强化学习 Agent 所需的运行时、工具和配置。Agent 通过学习使用 Triton 实现高性能 GPU 算子来完成编译器 Pass 优化任务。

## 架构

本包在 R2E-Gym 基础上扩展了 PassNet 专属组件：

- **Runtime** (`pass_agent/runtime/`)：PassNetDocker 运行时，负责环境配置
- **Agent** (`pass_agent/agent/`)：继承自 R2E-Gym Agent 的 PassAgent 类
- **Configs** (`pass_agent/configs/`)：Agent 脚手架配置文件
- **Examples** (`pass_agent/examples/`)：运行 Agent 的示例脚本
- **Datasets** (`pass_agent/datasets/`)：PassBench 数据集文件

## 安装

### 前置条件

环境要求（Python、PyTorch、CUDA、Docker）及 Docker 镜像构建方式见[项目根目录 README](../README_cn.md#快速开始)。

同时需要确保 PassBench 样本数据已就位（位于 `samples/` 目录）。

### 安装 PassAgent

```bash
cd pass_agent
pip install -r requirements.txt
```

此命令会从 GitHub 安装 r2e-gym，这是唯一的代码依赖。

**关于 PassNet 依赖的说明：**
PassAgent **不依赖** PassNet 的 Python 代码，而是通过以下方式集成：
1. **Docker 镜像**：使用由 `../Dockerfile.nvidia` 构建的 Docker 镜像
2. **数据**：运行时将 PassBench 样本目录以 volume 方式挂载
3. **无直接导入**：不直接导入任何 PassNet Python 模块

## 使用方法

### 快速开始

```bash
cd pass_agent

# 在演示数据集（10 个任务）上运行
python examples/run_pass_agent_demo.py \
    --llm-name openai/glm-4.7 \
    --llm-base-url <your-llm-base-url> \
    --openai-api-key <your-api-key> \
    --dataset datasets/passbench_demo_dataset.jsonl \
    --max-steps 50 \
    --k 10
```

### 命令行参数

`run_pass_agent_demo.py` 支持以下参数：

- `--llm-name`：LLM 模型名称（默认：`openai/glm-4.7`）
- `--llm-base-url`：LLM API 的 Base URL（默认：从环境变量 `LLM_BASE_URL` 读取）
- `--openai-api-key`：OpenAI API Key（默认：从环境变量 `OPENAI_API_KEY` 读取）
- `--anthropic-api-key`：Anthropic API Key（默认：从环境变量 `ANTHROPIC_API_KEY` 读取）
- `--dataset`：数据集 JSONL 文件路径（默认：`datasets/passbench_demo_dataset.jsonl`）
- `--config`：配置目录路径（默认：`configs/`）
- `--traj-dir`：轨迹保存目录（默认：`trajectories/pass_agent`）
- `--exp-name`：实验名称（默认：`pass_agent_full_trajectory`）
- `--max-steps`：每个任务的最大步数（默认：100）
- `--temperature`：采样温度（默认：1.0）
- `--max-workers`：并行 Worker 数量（默认：1）
- `--start-idx`：数据集起始索引（默认：0）
- `--k`：运行的任务数量（默认：None，即全部任务）

### 输出文件

每次运行生成两个 JSONL 文件：

1. **轨迹文件** (`trajectories/pass_agent/{exp_name}.jsonl`)：
   - 包含每个任务的完整轨迹数据
   - 内容包括：trajectory_steps、problem_statement、exit_reason、reward、speedup 指标

2. **对话记录** (`trajectories/pass_agent/{exp_name}_completions.jsonl`)：
   - 包含每个任务的完整消息历史（LLM 对话）
   - 格式：`{"sample_dir": "...", "messages": [...]}`

### 环境变量

```bash
export LLM_BASE_URL=<your-llm-base-url>
export OPENAI_API_KEY=<your-api-key>
export MAX_WORKERS=1

python examples/run_pass_agent_demo.py \
    --llm-name openai/glm-4.7 \
    --dataset datasets/passbench_demo_dataset.jsonl \
    --max-steps 50
```

## 目录结构

```
pass_agent/
├── __init__.py
├── runtime/
│   ├── __init__.py
│   └── passnet_docker.py       # PassNetDocker 运行时类
├── agent/
│   └── pass_agent.py           # PassAgent 类
├── configs/
│   └── edit_fn_calling.yaml    # Agent 脚手架配置
├── datasets/
│   ├── passbench_demo_dataset.jsonl      # 10 个示例任务
│   └── passbench_demo_dataset_200.jsonl  # 200 个示例任务
├── examples/
│   ├── run_pass_agent_demo.py
│   └── create_passbench_dataset.py
├── requirements.txt
└── README.md
```

## 工作原理

### PassNetDocker 运行时

`PassNetDocker` 类继承自 R2E-Gym 的 `DockerRuntime`，扩展了以下功能：

1. 挂载 PassNet 工作区目录
2. 配置问题级别的工作目录
3. 加载目标计算图信息
4. 跨迭代追踪加速比历史
5. 基于性能指标计算奖励值

### PassBench 评测

`pass_evaluator` 工具通过在样本目录中执行 `entry.sh` 触发完整的 PassBench 评测流程：

- **Pass 匹配**：从 `pass_dir/` 加载 Pass 文件，通过 `pass_mgr` 对 FX 图执行匹配与替换
- **正确性验证**：在各 dtype 对应的容忍度阈值下比较 eager 和编译后模型的输出
- **性能评测**：100 次试运行统计加速比，聚合计算 ES(t) 分数
- **结果上报**：解析 `aggregated_score.json`，将加速比和正确性结果返回给 Agent

完整评测流程说明见 [PassBench 评测流程](../pass_bench/README_cn.md)。

### Agent 工作流

Agent 按如下循环迭代：

1. 读取 `model.py` 和 `weight_meta.py`，理解目标子图结构
2. 使用 **file_editor** 在 `pass_dir/` 中编写或更新 Pass 文件
3. 使用 **pass_evaluator** 触发 PassBench，获取反馈（加速比、正确性）
4. 重复上述过程，直到 Pass 匹配成功并达到目标加速比，或达到最大步数为止

## 依赖说明

### 代码依赖
- **r2e-gym**：Agent 框架与编排（从 GitHub 安装：https://github.com/R2E-Gym/R2E-Gym）

### 运行时依赖（非 Python 包）
- **PassNet Docker 镜像**：由 `../Dockerfile.nvidia` 构建（见[根目录 README](../README_cn.md#docker-使用)）
- **PassBench 样本数据**：位于 `../samples/` 目录，运行时挂载
- **GPU**：支持 CUDA 的 NVIDIA GPU

### 依赖关系

```
pass_agent/
├── 代码：仅依赖 r2e-gym
├── 运行时：使用由 ../Dockerfile.nvidia 构建的 Docker 镜像
└── 数据：将 ../samples/ 以 volume 方式挂载
```

PassAgent 与 PassNet **部署耦合，代码解耦**：
- 共用同一代码仓库，便于协作
- Agent 使用 PassNet 生成的 Docker 镜像和数据
- 两者之间无 Python 模块互相导入
