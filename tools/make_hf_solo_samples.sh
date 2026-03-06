#!/bin/bash

# Generate AI4C solo op samples (single operator samples)
# This script generates samples for each individual operator (solo op)

AI4C_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

python3 ${AI4C_ROOT}/tools/generate_ai4c_sample.py \
    --grouped-sample-uids-list "${AI4C_ROOT}/graph_lists/hf_grouped_sole_op_subgraph_uids.txt" \
    --model-path-list "${AI4C_ROOT}/graph_lists/hf_sole_op_subgraphs.txt" \
    --graphs-path-in-ai4c "${AI4C_ROOT}/graphs/hf_subgraphs/sole_op_subgraphs" \
    --output-path "${AI4C_ROOT}/samples/hf_subgraphs/sole_op_subgraphs" \
    --output-sample-list "${AI4C_ROOT}/sample_lists/hf_sole_op_samples.txt"
