#!/bin/bash

BASE_DIR="/path/to/ai4c/repo/"

INPUT_LISTS=(
    "${BASE_DIR}/sample_lists/hf_typical_samples_v2.txt"
    "${BASE_DIR}/sample_lists/hf_sole_op_samples_v2.txt"
    "${BASE_DIR}/sample_lists/hf_fusible_samples_v2.txt"
)

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: base directory $BASE_DIR not found"
    exit 1
fi

for INPUT_LIST in "${INPUT_LISTS[@]}"; do
    BASENAME=$(basename "$INPUT_LIST" .txt)
    OUTPUT_FILE="${BASENAME}_all_expanded.txt"
    > "$OUTPUT_FILE"

    echo "Processing $INPUT_LIST ..."

    count=0
    while IFS= read -r rel_path || [ -n "$rel_path" ]; do
        clean_rel_path=$(echo "$rel_path" | tr -d '\r' | xargs)
        [ -z "$clean_rel_path" ] && continue

        TARGET_FILE="${BASE_DIR}/${clean_rel_path}/graph_list.txt"

        if [ -f "$TARGET_FILE" ]; then
            cat "$TARGET_FILE" >> "$OUTPUT_FILE"
            ((count++))
        else
            echo "Skipped: $TARGET_FILE not found"
        fi
    done < "$INPUT_LIST"

    echo "Done: $count directories processed -> $(pwd)/$OUTPUT_FILE"
done

echo "All tasks completed."