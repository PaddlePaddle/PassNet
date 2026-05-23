# PassNet

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue)](https://huggingface.co/datasets/PassNet/PassNet)

PassNet is an AI system for compiler optimization that leverages LLM-driven agents to automatically generate high-performance GPU kernels through compiler pass mechanisms for computation graph optimization. PassNet includes a complete optimization toolchain, the PassBench evaluation benchmark, and the PassAgent agent evaluation framework.

**English** | **[中文](README_cn.md)**

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [DataSet](#dataset)
- [Quick Start](#quick-start)
- [PassBench Evaluation Pipeline](#passbench-evaluation-pipeline)
- [PassAgent Evaluation](#passagent-evaluation)
- [License](#license)

## Project Structure

```
PassNet/
├── pass_bench/               # PassBench compiler evaluation framework: kernel compilation, correctness verification, performance benchmarking
├── pass_agent/               # PassAgent evaluation framework
├── samples/                  # PassBench sample data
├── sample_lists/             # PassBench sample list files (eval/train splits)
├── entry_scripts/            # Evaluation entry scripts
├── graphs/                   # Subgraph data
├── graph_lists/              # Subgraph lists and grouping info
├── test/                     # Unit tests
├── Dockerfile.nvidia         # Docker image definition
└── requirements.txt          # Python dependencies
```

## Architecture Overview

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

## Core Components

### [PassBench](pass_bench/) — Compiler Evaluation Framework

Provides kernel compilation, correctness verification, and performance benchmarking. It serves as both a standalone evaluation tool and the backend evaluation framework invoked by PassAgent:

- **Kernel Compilation**: Executes pass matching and replacement via the `pass_mgr` compiler method
- **Correctness Verification**: Validates numerical correctness of optimized kernels against dtype-specific tolerance thresholds (float32 / float16 / bfloat16)
- **Performance Benchmarking**: Measures speedup over 100 trials and outputs `aggregated_score.json`
- **Score Aggregation**: `aggregate_es_scores.py` computes ES(t) scores across all graphs in a sample

### [PassAgent](pass_agent/) — R2E-Gym Agent Evaluation Framework

Evaluates agent capabilities for compiler optimization using the R2E-Gym framework. See [pass_agent/README.md](pass_agent/README.md) for details.

## DataSet

### graphs — Raw Subgraph Data

Stores raw computation subgraphs extracted from deep learning models, serving as the source for PassBench samples:

- **fusible_subgraphs/**: A small set of example fusible subgraphs (1,456), containing computation graphs with multi-operator fusion opportunities
- **hf_subgraphs/** (Legacy): Previous version subgraph data, containing sole op (1,410), fusible (4,167), and typical (6,157) categories
- **hf_subgraphs_v2/**: HuggingFace model subgraphs, organized into three categories:
  - `sole_op_subgraphs`: Single-operator subgraphs (5,939)
  - `fusible_subgraphs`: Fusible subgraphs (22,870)
  - `typical_subgraphs`: Typical subgraphs (25,151)

### graph_lists — Subgraph Lists and Grouping

Stores subgraph path lists, UID groupings, and other information for sample filtering and group management:

**Subgraph Path Lists** (line format: `subgraph_UID\tsubgraph_relative_path`)

| File | Subgraphs | Description |
|------|-----------|-------------|
| [`fusible_subgraphs.txt`](graph_lists/fusible_subgraphs.txt) | 1,455 | Example fusible subgraph paths |
| [`hf_sole_op_subgraphs.txt`](graph_lists/hf_sole_op_subgraphs.txt) | 1,410 | Legacy sole op subgraph paths |
| [`hf_fusible_subgraphs.txt`](graph_lists/hf_fusible_subgraphs.txt) | 4,166 | Legacy fusible subgraph paths |
| [`hf_typical_subgraphs.txt`](graph_lists/hf_typical_subgraphs.txt) | 6,157 | Legacy typical subgraph paths |
| [`hf_sole_op_subgraphs_v2.txt`](graph_lists/hf_sole_op_subgraphs_v2.txt) | 5,939 | v2 sole op subgraph paths |
| [`hf_fusible_subgraphs_v2.txt`](graph_lists/hf_fusible_subgraphs_v2.txt) | 22,870 | v2 fusible subgraph paths |
| [`hf_typical_subgraphs_v2.txt`](graph_lists/hf_typical_subgraphs_v2.txt) | 25,151 | v2 typical subgraph paths |

### samples — PassBench Evaluation Samples

Evaluation samples generated from `graphs/`, each serving as an independently executable evaluation unit:

- **fusible_subgraphs/**: A small set of example samples from TIMM models' fusible subgraphs, organized by `model_name/subgraph_index`
- **hf_subgraphs/** (Legacy): Previous version subgraph samples, containing sole op (590), fusible (2,489), and typical (3,382) categories
- **hf_subgraphs_v2/**: v2 subgraph samples with extended multi-dtype support, containing sole op (1,029), fusible (4,676), and typical (4,278) categories, organized by hash path `xx/yy/hash/`, dataset published at [PassNet/PassNet](https://huggingface.co/datasets/PassNet/PassNet)

Each sample directory contains:

| File | Description |
|------|-------------|
| `entry.sh` | Evaluation entry script that executes compilation, verification, and performance statistics |
| `graph_list.txt` | List of computation graphs included in the sample |
| `graphs/` | Computation graph definitions (model.py, weight_meta.py, etc.) |
| `pass_dir/` | Output directory for generated optimization passes |
| `pass_bench/` | Copy of the evaluation framework (for standalone execution within Docker containers) |
| `sample_uids.txt` | Unique sample identifier (hf_subgraphs_v2 only) |

### sample_lists — Eval/Train Sample Splits

Stores sample path lists for evaluation and training, organized by purpose and subgraph type, available in both txt and csv formats:

**train/** (Training Set)

| File | Samples | Description |
|------|---------|-------------|
| [`hf_sole_op_train_samples_v2.txt`](sample_lists/train/hf_sole_op_train_samples_v2.txt) | 1,028 | Sole op subgraph training samples |
| [`hf_fusible_train_samples_v2.txt`](sample_lists/train/hf_fusible_train_samples_v2.txt) | 4,476 | Fusible subgraph training samples |
| [`hf_typical_train_samples_v2.txt`](sample_lists/train/hf_typical_train_samples_v2.txt) | 4,078 | Typical subgraph training samples |
| [`hf_sole_op_train_samples.txt`](sample_lists/train/hf_sole_op_train_samples.txt) (Legacy) | 589 | Legacy sole op subgraph training samples |
| [`hf_fusible_train_samples.txt`](sample_lists/train/hf_fusible_train_samples.txt) (Legacy) | 2,289 | Legacy fusible subgraph training samples |
| [`hf_typical_train_samples.txt`](sample_lists/train/hf_typical_train_samples.txt) (Legacy) | 3,182 | Legacy typical subgraph training samples |

**eval/** (Evaluation Set)

| File | Samples | Description |
|------|---------|-------------|
| [`hf_fusible_eval_samples_v2.txt`](sample_lists/eval/hf_fusible_eval_samples_v2.txt) | 200 | Fusible subgraph evaluation samples |
| [`hf_typical_eval_samples_v2.txt`](sample_lists/eval/hf_typical_eval_samples_v2.txt) | 200 | Typical subgraph evaluation samples |
| [`hf_fusible_eval_samples.txt`](sample_lists/eval/hf_fusible_eval_samples.txt) (Legacy) | 200 | Legacy fusible subgraph evaluation samples |
| [`hf_typical_eval_samples.txt`](sample_lists/eval/hf_typical_eval_samples.txt) (Legacy) | 200 | Legacy typical subgraph evaluation samples |

## Quick Start

### Requirements

- Python 3.12+
- PyTorch 2.9+ (CUDA 12.8)
- NVIDIA GPU (CUDA support)
- Docker (optional, for containerized evaluation)

### Installation

```bash
cd /path/to/passnet

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/path/to/passnet
```

### Run Example

```bash
# Verify sample evaluation
bash samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4/entry.sh
```

### Docker Usage

#### Build Image

```bash
docker build . -t passnet:latest -f Dockerfile.nvidia
```

#### Verify Single Sample Execution in Container

```bash
docker run --gpus all --privileged \
    -v <path-to-passnet-project>:/workspace \
    -w /workspace \
    passnet:latest \
    bash samples/fusible_subgraphs/crossvit_15_dagger_240.in1k/crossvit_15_dagger_240.in1k_0_start14_end16_4/entry.sh
```

## PassBench Evaluation Pipeline

The PassNet evaluation pipeline works as follows:

1. **Analyze computation graph**: Read `model.py` and `weight_meta.py` to understand the target subgraph's operators, tensor shapes, and dtypes
2. **Generate optimization pass**: LLM agent generates a pass file and places it in `pass_dir/`
3. **Pass matching and replacement**: `pass_mgr` matches the pattern in the FX graph and replaces it with the optimized kernel
4. **Correctness verification**: Compare eager and compiled outputs using dtype-specific tolerance thresholds
5. **Performance benchmarking**: Measure speedup and compute ES(t), output `aggregated_score.json`

```bash
# place your pass file
cp MyPass.py samples/<type>/<hash>/pass_dir/
echo '["MyPass"]' > samples/<type>/<hash>/pass_dir/sorted_output_pass_rule_names.json

# run evaluation for a single sample
bash samples/<type>/<hash>/entry.sh
```

See **[pass_bench/README.md](pass_bench/README.md)** for pass file format and batch evaluation.

## PassAgent Evaluation

Evaluate agents using the PassAgent framework:

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

See [pass_agent/README.md](pass_agent/README.md) for details.

## License

Please refer to the license file in the project root directory.
