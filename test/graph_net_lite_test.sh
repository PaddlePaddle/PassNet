#!/bin/bash

AI4C_ROOT=$(python3 -c "import graph_net_lite; import os; print(os.path.dirname(os.path.dirname(graph_net_lite.__file__)))")
OUTPUT_PATH=/tmp/workspace_graph_net_lite_test

mkdir -p "$OUTPUT_PATH"
model_list="$AI4C_ROOT/test/workspace_graph_net_lite_test/small10_torch_samples_list.txt"

python3 -m graph_net_lite.torch.test_compiler \
    --model-path-prefix $AI4C_ROOT/test/workspace_graph_net_lite_test \
    --allow-list $model_list \
    --compiler nope \
    --device cuda \
    --config $(base64 -w 0 <<EOF
{
    "model_path_prefix": "$AI4C_ROOT",
    "sample_root": "$AI4C_ROOT"
}
EOF
) 2>&1 | tee "$OUTPUT_PATH/validation.log"

