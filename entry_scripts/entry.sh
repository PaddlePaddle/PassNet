#!/bin/bash

ai4c_repo_root=$(realpath $0 | xargs -I{} dirname {} | xargs -I{} dirname {})
if [ ! -f "$ai4c_repo_root/graph_net_bench/__init__.py" ]; then
    echo Python module graph_net_bench not found
    exit -1
fi
export PYTHONPATH=$ai4c_repo_root:$PYTHONPATH

SAMPLE_ROOT=$(dirname $0)
OUTPUT_PATH=/tmp/workspace_graph_net_bench_test

mkdir -p "$OUTPUT_PATH"
model_list="$SAMPLE_ROOT/graph_list.txt"

pass_match_result_file_path=$(mktemp)

compiler_method=pass_mgr

python3 -m graph_net_bench.torch.test_compiler \
    --model-path-prefix $SAMPLE_ROOT \
    --allow-list $model_list \
    --compiler $compiler_method \
    --device cuda \
    --warmup 25 \
    --trials 100 \
    --config $(base64 -w 0 <<EOF
{
    "pass_match_result_file_path": "$pass_match_result_file_path",
    "input_pass_rule_dir": "$SAMPLE_ROOT/const_pass_dir",
    "output_pass_rule_dir": "$SAMPLE_ROOT/pass_dir",
    "output_pass_pattern_limit": 100,
    "output_pass_replacement_func_limit": 1
}
EOF
) 2>&1 | tee "$OUTPUT_PATH/validation.log"

pass_match_result=$(cat $pass_match_result_file_path)
unlink $pass_match_result_file_path
echo Has Any pass matched? [$pass_match_result]
if [[ $compiler_method == "pass_mgr" && $pass_match_result == "False" ]]; then
    echo Pass testing early exits on pass mismatch.
fi

python3 -m graph_net_bench.aggregate_es_scores \
    --benchmark-path "$OUTPUT_PATH/validation.log" \
    --sample-id 1 \
    --disable-aggregation-mode \
    --fpdb 0.001 \
    --output-json-file-path "$OUTPUT_PATH/aggregated_score.json"
