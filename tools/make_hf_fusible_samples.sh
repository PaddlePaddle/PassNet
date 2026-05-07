#!/bin/bash

PASSNET_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

python ${PASSNET_ROOT}/tools/generate_passbench_sample.py \
    --grouped-sample-uids-list "${PASSNET_ROOT}/graph_lists/hf_grouped_fusible_subgraph_uids.txt" \
    --model-path-list "${PASSNET_ROOT}/graph_lists/hf_fusible_subgraphs.txt" \
    --graphs-path-in-passnet "${PASSNET_ROOT}/graphs/hf_subgraphs/fusible_subgraphs" \
    --output-path "${PASSNET_ROOT}/samples/hf_subgraphs/fusible_subgraphs" \
    --output-sample-list "${PASSNET_ROOT}/sample_lists/hf_fusible_samples.txt" \
    --do-eval
