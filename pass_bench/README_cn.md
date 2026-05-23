# PassBench 评测流程

PassNet 评测流程：

1. [分析计算图](#1-分析计算图)
2. [生成优化 Pass](#2-生成优化-pass)
3. [Pass 匹配与替换](#3-pass-匹配与替换)
4. [正确性验证](#4-正确性验证)
5. [性能评测](#5-性能评测)

---

## 1. 分析计算图

每个样本目录在 `graphs/` 下包含一个或多个计算图。每个计算图由以下文件定义：

| 文件 | 说明 |
|------|------|
| `model.py` | PyTorch FX 图定义——算子及其连接关系 |
| `weight_meta.py` | 权重张量的形状、数据类型和设备信息 |
| `input_meta.py` | 输入张量的形状和数据类型 |
| `input_tensor_constraints.py` | 输入值域约束 |
| `graph_net.json` | 序列化的图结构 |

一个样本可能包含同一子图在不同数据类型（如 `float32`、`float16`、`bfloat16`）和不同 batch size 下的多个变体，评测会对所有变体运行 Pass。

```
sample/
└── graphs/
    └── hf_subgraphs_v2/fusible_subgraphs/
        ├── float16/1/.../<subgraph_name>/
        │   ├── model.py
        │   ├── weight_meta.py
        │   ├── input_meta.py
        │   └── ...
        ├── float32/1/.../<subgraph_name>/
        └── bfloat16/1/.../<subgraph_name>/
```

此步骤的目标是识别目标计算模式——哪些算子出现、以何种顺序连接、以及对应的张量形状和数据类型。

---

## 2. 生成优化 Pass

Pass 文件是放置在 `pass_dir/` 中的 Python 模块，用于告知 `pass_mgr` 需要匹配哪个子图模式，以及用什么优化算子替换它。

### Pass 文件格式

Pass 文件必须暴露三个模块级函数：

| 函数 | 说明 |
|------|------|
| `pattern(*args)` | 用 PyTorch 算子描述目标子图，`pass_mgr` 以此作为匹配模板与 FX 图进行匹配 |
| `replacement_args(*args)` | 将匹配到的模式输入映射为转发给替换算子的参数 |
| `replacement_func()` | 返回优化后的算子包装函数——必须返回稳定的模块级函数，不能是嵌套 `def` 或 `lambda` |

典型 Pass 文件结构：

```
MyPass.py
├── def pattern(...)           # 待匹配的子图（PyTorch 算子）
├── def replacement_args(...)  # 参数重映射
├── @triton.jit kernel         # 优化后的 Triton 算子实现
├── @torch.fx.wrap wrapper     # 可在 FX 图中调用的算子包装函数
└── def replacement_func()     # 返回上述包装函数
```

### 放置 Pass 文件

将 Pass 文件放入样本的 `pass_dir/`，同时创建 `sorted_output_pass_rule_names.json` 声明加载顺序（文件名去掉 `.py` 后缀）：

```
sample/
└── pass_dir/
    ├── MyPass.py
    └── sorted_output_pass_rule_names.json   # ["MyPass"]
```

支持多个 Pass，按优先级顺序列出：

```json
["PassA", "PassB"]
```

---

## 3. Pass 匹配与替换

`entry.sh` 调用 `pass_bench.torch.test_compiler`，使用 `--compiler pass_mgr`。`PassMgrBackend` 从 `pass_dir/` 加载所有 Pass 文件，然后对每个计算图：

1. 通过 `torch.compile` 对模型进行 trace，得到 FX 图
2. 使用 `SubgraphMatcher` 在 FX 图中查找所有匹配 `pattern` 的子图
3. 将每处匹配替换为 `replacement_func()` 返回的算子包装函数
4. 重新编译修改后的图

匹配成功时的日志输出：

```
[PassMgrBackend] Loaded 1 passes: ['MyPass']
[PassMgrBackend] Applied 1 replacements with MyPass.
```

若模式未能匹配任何子图，`pass_mgr` 报错并提前退出：

```
[PassMgrBackend] Pass MyPass failed to match.
Has Any pass matched? [False]
Pass testing early exits on pass mismatch.
```

Pass 未匹配时，该样本在所有 tolerance 级别下均获得最低分（`ES(t) = 0.1`）。

---

## 4. 正确性验证

优化图编译完成后，`test_compiler` 对同一输入分别运行原始 eager 模型和编译后模型，在多组容忍度级别下使用 `torch.allclose` 逐一比较输出。

### 不同 dtype 的精度阈值

每种 dtype 有固定的 `(rtol, atol)` 精度阈值，作为基准正确性判据：

| dtype | rtol | atol |
|-------|------|------|
| float32 | 1.3E-06 | 1.00E-05 |
| float16 | 1.00E-03 | 1.00E-05 |
| bfloat16 | 1.60E-02 | 1.00E-05 |

日志中每条正确性检查的 key 格式为 `[all_close_atol_<atol>_rtol_<rtol>]`，各 dtype 基准检查对应的日志行如下：

```
[Correctness][all_close_atol_1.00E-05_rtol_1.30E-06]: 1   # float32
[Correctness][all_close_atol_1.00E-05_rtol_1.00E-03]: 1   # float16
[Correctness][all_close_atol_1.00E-05_rtol_1.60E-02]: 1   # bfloat16
...
[Correctness][max_diff]: 0.0001220703125
[Correctness][mean_diff]: 1.862645149230957e-09
```

同时还会验证数据类型一致性：

```
[Datatype][eager]: bfloat16
[Datatype][compiled]: bfloat16
[DataType] eager:['bfloat16'] compiled:['bfloat16'] match:True
```

只有数据类型和正确性检查均通过，该计算图才标记为 `success`：

```
[Result] status: success
```

否则标记为 `failed`。

---

## 5. 性能评测

对每个通过正确性验证的计算图，`test_compiler` 对 eager 和编译后模型分别进行 benchmark：

- **预热**：25 次（不计入统计）
- **测试**：100 次计时运行，每次记录端到端（`e2e`）和纯 GPU（`gpu`）耗时

```
[Profiling] Using device: cuda NVIDIA A30, warm up 25, trials 100
Trial 1: e2e=0.314 ms, gpu=0.260 ms
Trial 2: e2e=0.365 ms, gpu=0.313 ms
...
[Performance][eager]:    {"e2e": {"median": 0.264, ...}, "gpu": {"median": 0.230, ...}}
[Performance][compiled]: {"e2e": {"median": 0.297, ...}, "gpu": {"median": 0.259, ...}}
[Speedup][e2e]: 0.889
[Speedup][gpu]: 0.888
```

加速比 = `eager 中位数 / compiled 中位数`，加速比 > 1 表示优化后更快。

### 分数聚合

样本内所有计算图评测完成后，`pass_bench.aggregate_es_scores` 对每个 tolerance 级别 `t` 计算核心指标 **ES(t)**：

- 对每个计算图计算*修正加速比*：
  - 在 `t=1` 时正确且加速比为 `s`：修正加速比 = `s`（若 `s ≥ 1`）或 `s²`（对减速进行惩罚）
  - 正确性验证失败或 Pass 未匹配：修正加速比 = `b = 0.1`（基准惩罚值）
  - 在 `t=1` 时失败，但该失败类型在 tolerance `t` 下被容忍：修正加速比 = 1
- **ES(t)** = 样本内所有计算图修正加速比的几何平均值

```
  - ESt=0.100 for tolerance=-10.
  - ESt=0.100 for tolerance=-5.
  - ESt=0.912 for tolerance=1.
  - ESt=0.912 for tolerance=2.
  ...
aggregated_speedup=0.912
Result is saved to /tmp/workspace_pass_bench_test/aggregated_score.json
```

最终结果写入 `aggregated_score.json`。

---

## 运行评测

```bash
# 单样本评测
bash samples/<type>/<hash>/entry.sh

# 批量评测
SAMPLE_LIST="sample_lists/eval/hf_fusible_eval_samples_v2.txt"
LOG_FILE="eval.log"
> "$LOG_FILE"
idx=0; total=$(grep -c . "$SAMPLE_LIST")

while IFS= read -r sample_path; do
    [ -z "$sample_path" ] && continue
    idx=$((idx + 1))
    echo "===== [$idx/$total] $(basename "$sample_path") =====" | tee -a "$LOG_FILE"
    bash "$sample_path/entry.sh" >> "$LOG_FILE" 2>&1 && \
        echo "OK" | tee -a "$LOG_FILE" || \
        echo "FAILED" | tee -a "$LOG_FILE"
done < "$SAMPLE_LIST"

echo "===== Total: $total =====" | tee -a "$LOG_FILE"
```
