#!/bin/bash

AI4C_ROOT=$(python3 -c "import graph_net_lite; import os; print(os.path.dirname(os.path.dirname(graph_net_lite.__file__)))")
OUTPUT_PATH=/tmp/workspace_graph_net_lite_test

mkdir -p "$OUTPUT_PATH"
model_list="$AI4C_ROOT/test/workspace_graph_net_lite_test/small10_torch_samples_list.txt"

python3 -m graph_net_lite.torch.test_compiler \
    --model-path-prefix $AI4C_ROOT/test/workspace_graph_net_lite_test \
    --allow-list $model_list \
    --compiler pass_mgr \
    --device cuda \
    --config $(base64 -w 0 <<EOF
{
    "input_pass_rule_dir": "$AI4C_ROOT/test/workspace_graph_net_lite_test/example_input_pass_rule_dir",
    "output_pass_rule_dir": "$AI4C_ROOT/test/workspace_graph_net_lite_test/example_output_pass_rule_dir",
    "output_pass_pattern_limit": 1,
    "output_pass_replacement_func_limit": 1
}
EOF
) 2>&1 | tee "$OUTPUT_PATH/validation.log"

python3 -m graph_net_lite.aggregate_es_scores \
    --benchmark-path "$OUTPUT_PATH/validation.log" \
    --sample-id 1 \
    --output-json-file-path "$OUTPUT_PATH/aggregated_score.json"
