#!/bin/bash

SAMPLE_ROOT=$(dirname $0)
OUTPUT_PATH=/tmp/workspace_graph_net_bench_test

mkdir -p "$OUTPUT_PATH"
model_list="$SAMPLE_ROOT/graph_list.txt"

python3 -m graph_net_bench.torch.test_compiler \
    --model-path-prefix $SAMPLE_ROOT \
    --allow-list $model_list \
    --compiler pass_mgr \
    --device cuda \
    --config $(base64 -w 0 <<EOF
{
    "input_pass_rule_dir": "$SAMPLE_ROOT/const_pass_dir",
    "output_pass_rule_dir": "$SAMPLE_ROOT/pass_dir",
    "output_pass_pattern_limit": 100,
    "output_pass_replacement_func_limit": 1
}
EOF
) 2>&1 | tee "$OUTPUT_PATH/validation.log"

python3 -m graph_net_bench.aggregate_es_scores \
    --benchmark-path "$OUTPUT_PATH/validation.log" \
    --sample-id 1 \
    --output-json-file-path "$OUTPUT_PATH/aggregated_score.json"
