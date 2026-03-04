#!/bin/bash 

AI4C_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

rm -rf ${AI4C_ROOT}/samples/hf_subgraphs/fusible_subgraphs
python ${AI4C_ROOT}/tools/generate_ai4c_sample.py \
    --grouped-sample-uids-list "${AI4C_ROOT}/graph_lists/hf_grouped_fusible_subgraph_uids.txt" \
    --model-path-list "${AI4C_ROOT}/graph_lists/hf_fusible_subgraphs.txt" \
    --graphs-path-in-ai4c "${AI4C_ROOT}/graphs/hf_subgraphs/fusible_subgraphs" \
    --output-path "${AI4C_ROOT}/samples/hf_subgraphs/fusible_subgraphs" \
    --output-sample-list "${AI4C_ROOT}/sample_lists/hf_fusible_samples.txt" \
    --do-eval
