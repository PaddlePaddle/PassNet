# PassNet

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue)](https://huggingface.co/datasets/PassNet/PassNet)

PassNet 是一个面向编译器优化的 AI 系统，利用大语言模型（LLM）驱动的 Agent 自动生成高性能 GPU 算子，通过编译器 Pass 机制实现计算图优化。PassNet 包含完整的优化工具链、评测基准 PassBench 以及 Agent 评估框架 PassAgent。

**[English](README.md)** | **中文**

## 目录

- [项目结构](#项目结构)
- [架构概览](#架构概览)
- [核心组件](#核心组件)
- [数据与样本](#数据与样本)
- [快速开始](#快速开始)
- [评测流程](#评测流程)
- [Agent 评估](#agent-评估)
- [许可证](#许可证)

## 项目结构

```
PassNet/
├── pass_bench/               # PassBench编译器评测框架：算子编译、正确性验证、性能评测
├── pass_agent/               # PassAgent评估框架
├── samples/                  # PassBench样本数据
├── sample_lists/             # PassBench样本列表文件（评测/训练划分）
├── entry_scripts/            # 评测入口脚本
├── graphs/                   # 子图数据
├── graph_lists/              # 子图列表与分组信息
├── test/                     # 单元测试
├── Dockerfile.nvidia         # Docker镜像定义
└── requirements.txt          # Python依赖
```

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             PassAgent                                   │
│                    (LLM-driven Pass Generation)                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │◄───┐
│ │  Multi-step Iterative Solving  ·  k-attempts  ·  R2E-Gym Framework  │ │    │
│ └─────────────────────────────────────────────────────────────────────┘ │    │
└────────────────┬───────────────────────────────────────┬────────────────┘    │
      read data  │                        generated pass │                     │
                 ▼                                       ▼                     │
┌───────────────────────────────────┐    ┌───────────────────────────────┐     │
│             DataSet               │    │          PassBench            │     │
│  ┌─────────────────────────────┐  │    │  ┌──────────────────────────┐ │     │
│  │ graphs/                     │  │    │  │ 1. Execution & Eval      │ │     │
│  │  sole_op  (5,939)           │  │    │  │    Eager Execution       │ │     │
│  │  fusible  (22,870)          │  │    │  │    pass_mgr Execution    │ │     │
│  │  typical  (25,151)          │  │    │  └────────────┬─────────────┘ │     │
│  └─────────────────────────────┘  │    │               │               │     │
│  ┌─────────────────────────────┐  │    │               ▼               │  feedback
│  │ samples/                    │  │    │  ┌──────────────────────────┐ │     │
│  │  sole_op  (1,029)           │  │    │  │ 2. Result Checking       │ │     │
│  │  fusible  (4,676)           │  │    │  │    Correctness & Speedup │ │     │
│  │  typical  (4,278)           │  │    │  └────────────┬─────────────┘ │     │
│  └─────────────────────────────┘  │    │               │               │     │
│  ┌─────────────────────────────┐  │    │               ▼               │     │
│  │ sample_lists/               │  │    │  ┌──────────────────────────┐ │     │
│  │  train/                     │  │    │  │ 3. Score Aggregation     │ │     │
│  │  eval/                      │  │    │  │    ES(t) & AS Met        │ │     │
│  └─────────────────────────────┘  │    │  └──────────────────────────┘ │     │
└───────────────────────────────────┘    └───────────────────────────────┘     │
                                                         └─────────────────────┘
```

## 核心组件

### [PassBench](pass_bench/) — 编译器评测框架

提供算子编译、正确性验证与性能评测能力：

- **算子编译**：通过 `pass_mgr` 编译器方法执行 Pass 匹配与替换
- **正确性验证**：基于容差机制验证优化后算子的数值正确性
- **性能评测**：统计加速比等性能指标，输出 `aggregated_score.json`
- **评分聚合**：`aggregate_es_scores.py` 汇总多次评测结果

### [PassAgent](pass_agent/) — R2E-Gym Agent 评估框架

基于 R2E-Gym 框架评估 Agent 的编译器优化能力。详见 [pass_agent/README.md](pass_agent/README.md)。

## 数据集

### graphs — 子图原始数据

存放从深度学习模型中提取的原始计算子图，作为 PassBench 样本的生成来源：

- **fusible_subgraphs/**：少量示例可融合子图（1,456 个），包含多算子融合优化的计算图
- **hf_subgraphs/**（Legacy）：旧版子图样本，包含单算子（1,410 个）、可融合（4,167 个）、经典（6,157 个）三类
- **hf_subgraphs_v2/**：HuggingFace 模型子图，分为三类：
  - `sole_op_subgraphs`：单算子子图（5,939 个）
  - `fusible_subgraphs`：可融合子图（22,870 个）
  - `typical_subgraphs`：经典子图（25,151 个）

### graph_lists — 子图列表与分组

存放子图的路径列表、UID 分组等信息，用于样本筛选和分组管理：

**子图路径列表**（每行格式：`子图UID\t子图相对路径`）

| 文件 | 子图数 | 说明 |
|------|--------|------|
| [`fusible_subgraphs.txt`](graph_lists/fusible_subgraphs.txt) | 1,455 | 示例可融合子图路径 |
| [`hf_sole_op_subgraphs.txt`](graph_lists/hf_sole_op_subgraphs.txt) | 1,410 | 旧版单算子子图路径 |
| [`hf_fusible_subgraphs.txt`](graph_lists/hf_fusible_subgraphs.txt) | 4,166 | 旧版可融合子图路径 |
| [`hf_typical_subgraphs.txt`](graph_lists/hf_typical_subgraphs.txt) | 6,157 | 旧版经典子图路径 |
| [`hf_sole_op_subgraphs_v2.txt`](graph_lists/hf_sole_op_subgraphs_v2.txt) | 5,939 | v2 单算子子图路径 |
| [`hf_fusible_subgraphs_v2.txt`](graph_lists/hf_fusible_subgraphs_v2.txt) | 22,870 | v2 可融合子图路径 |
| [`hf_typical_subgraphs_v2.txt`](graph_lists/hf_typical_subgraphs_v2.txt) | 25,151 | v2 经典子图路径 |

### samples — PassBench 评测样本

从 `graphs/` 中加工生成的评测样本，每个样本为独立可执行的评测单元：

- **fusible_subgraphs/**：少量示例样本，来自 TIMM 模型的可融合子图，按 `模型名/子图编号` 组织
- **hf_subgraphs/**（Legacy）：旧版子图样本，包含单算子（590 个）、可融合（2,489 个）、典型（3,382 个）三类
- **hf_subgraphs_v2/**：v2 版子图样本，扩展了多种数据类型的支持，包含单算子（1,029 个）、可融合（4,676 个）、典型（4,278 个）三类，按哈希路径 `xx/yy/hash/` 组织，子图数据集发布于 [PassNet/PassNet](https://huggingface.co/datasets/PassNet/PassNet)

每个样本目录包含：

| 文件 | 说明 |
|------|------|
| `entry.sh` | 评测入口脚本，执行编译、验证与性能统计 |
| `graph_list.txt` | 该样本包含的计算图列表 |
| `graphs/` | 计算图定义文件（model.py、weight_meta.py 等） |
| `pass_dir/` | 生成的优化 Pass 输出目录 |
| `pass_bench/` | 评测框架副本（用于 Docker 容器内独立运行） |
| `sample_uids.txt` | 样本唯一标识（仅 hf_subgraphs_v2） |

### sample_lists — 评测/训练样本划分

存放用于评测和训练的样本路径列表，按用途和子图类型组织，同时提供 txt 和 csv 格式：

**train/**（训练集）

| 文件 | 样本数 | 说明 |
|------|--------|------|
| [`hf_sole_op_train_samples_v2.txt`](sample_lists/train/hf_sole_op_train_samples_v2.txt) | 1,028 | 单算子子图训练样本 |
| [`hf_fusible_train_samples_v2.txt`](sample_lists/train/hf_fusible_train_samples_v2.txt) | 4,476 | 可融合子图训练样本 |
| [`hf_typical_train_samples_v2.txt`](sample_lists/train/hf_typical_train_samples_v2.txt) | 4,078 | 经典子图训练样本 |
| [`hf_sole_op_train_samples.txt`](sample_lists/train/hf_sole_op_train_samples.txt)（Legacy） | 589 | 旧版单算子子图训练样本 |
| [`hf_fusible_train_samples.txt`](sample_lists/train/hf_fusible_train_samples.txt)（Legacy） | 2,289 | 旧版可融合子图训练样本 |
| [`hf_typical_train_samples.txt`](sample_lists/train/hf_typical_train_samples.txt)（Legacy） | 3,182 | 旧版经典子图训练样本 |

**eval/**（评估集）

| 文件 | 样本数 | 说明 |
|------|--------|------|
| [`hf_fusible_eval_samples_v2.txt`](sample_lists/eval/hf_fusible_eval_samples_v2.txt) | 200 | 可融合子图评估样本 |
| [`hf_typical_eval_samples_v2.txt`](sample_lists/eval/hf_typical_eval_samples_v2.txt) | 200 | 经典子图评估样本 |
| [`hf_fusible_eval_samples.txt`](sample_lists/eval/hf_fusible_eval_samples.txt)（Legacy） | 200 | 旧版可融合子图评估样本 |
| [`hf_typical_eval_samples.txt`](sample_lists/eval/hf_typical_eval_samples.txt)（Legacy） | 200 | 旧版经典子图评估样本 |

## 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.9+（CUDA 12.8）
- NVIDIA GPU（CUDA 支持）
- Docker（可选，用于容器化评测）

### 安装

```bash
cd /path/to/passnet

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:/path/to/passnet
export PASSNET_BASE_URL='your_llm_base_url'
export PASSNET_API_KEY='your_llm_api_key'
export PASSNET_API_MODEL_NAME='your_llm_api_model_name'
```

### 运行示例

```bash
# 验证样本评测
bash samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4/entry.sh
```

### Docker 使用

#### 构建镜像

```bash
docker build . -t passnet:latest -f Dockerfile.nvidia
```

#### 容器内验证单个样本执行

```bash
docker run --gpus all --privileged \
    -v <path-to-passnet-project>:/workspace \
    -w /workspace \
    passnet:latest \
    bash samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4/entry.sh
```

## 评测流程

PassNet 的评测流程如下：

1. **分析计算图**：读取目标子图的 `model.py` 和 `weight_meta.py`
2. **生成优化 Pass**：Agent 创建匹配模式与替换函数
3. **Pass 匹配与替换**：`pass_mgr` 编译器执行 Pass 应用
4. **正确性验证**：验证优化后算子与原始算子的数值一致性
5. **性能评测**：统计加速比，输出评测结果

## Agent 评估

使用 PassAgent 框架评估 Agent：

```bash
cd pass_agent
pip install -r requirements.txt

python examples/run_pass_agent_demo.py \
    --llm-name openai/glm-4.7 \
    --llm-base-url <your-llm-base-url> \
    --openai-api-key <your-api-key> \
    --dataset datasets/passbench_demo_dataset.jsonl \
    --max-steps 50 \
    --k 10
```

详见 [pass_agent/README.md](pass_agent/README.md)。

## 许可证

请参阅项目根目录下的许可证文件。
