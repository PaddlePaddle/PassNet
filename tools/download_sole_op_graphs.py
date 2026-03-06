#!/usr/bin/env python3
"""
下载 sole_op_graph 数据到 ai4c 仓库
只下载 hf_sole_op_subgraphs.txt 中列出的模型
"""
import os
import sys
from pathlib import Path

AI4C_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI4C_ROOT))

from huggingface_hub import login

# 登录 HuggingFace
# 设置方式: export HF_TOKEN="your_token"
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

from datasets import load_dataset

REPO_ID = "PaddlePaddle/GraphNet"
REVISION = "20260224"

# 读取需要下载的样本列表 - 只需要模型名
SAMPLE_LIST = AI4C_ROOT / "graph_lists" / "hf_sole_op_subgraphs.txt"
needed_models = set()
with open(SAMPLE_LIST) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 1)
        if len(parts) == 2:
            model_path = parts[1]
            # 提取模型名: samples/xxx/yyy/_decomposed/zzz -> samples/xxx/yyy
            if "/_decomposed/" in model_path:
                model_name = model_path.split("/_decomposed/")[0]
                needed_models.add(model_name)

print(f"需要下载的模型数量: {len(needed_models)}")

# 加载数据集
print(f"正在加载 GraphNet 数据集...")
ds = load_dataset(REPO_ID, split="GraphNet", revision=REVISION)
ds_list = list(ds)
print(f"数据集大小: {len(ds_list)}")

# 下载需要的模型
OUTPUT_ROOT = AI4C_ROOT / "graphs" / "hf_subgraphs" / "sole_op_subgraphs"

count = 0
model_count = 0
for item in ds_list:
    path = item["path"]
    # 只处理 sole_op_graph 的文件
    if not path.startswith("sole_op_graph/samples/"):
        continue

    # 提取模型名
    relative_path = path.replace("sole_op_graph/", "")
    if "/_decomposed/" not in relative_path:
        continue

    model_name = relative_path.split("/_decomposed/")[0]

    # 只下载需要的模型
    if model_name not in needed_models:
        continue

    # 保存文件
    full_path = OUTPUT_ROOT / relative_path
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(item["content"])

    model_count += 1
    if model_count % 100 == 0:
        print(f"已下载 {model_count} 个文件...")

print(f"\n完成！已下载 {model_count} 个文件到 {OUTPUT_ROOT}")
