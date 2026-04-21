#!/bin/bash
set -u

# Pipeline: Run torch compiler on graph datasets, then extract triton kernels
# into a clean output directory containing (subgraph, triton_kernel) pairs.
#
# Supports two data sources:
#   list - Read samples from .txt list files (one relative graph path per line)
#   hf   - Scan GraphNet_hf directory structure for model.py files
#
# Step 1 (compilation) is parallelized across GPUs; Steps 2-5 run sequentially.
# With a single GPU, this behaves identically to a sequential script.
#
# For each dataset:
#   Step 1: Run compiler to extract triton code, saving inductor debug output to cache dir
#   Step 2: Filter samples by kernel speedup: keep (speedup >= 1) vs discard (others)
#   Step 3: Clean temp files (__pycache__, *.pyc, *.pyo) from kept samples
#   Step 4: Extract triton kernels and copy only needed files to output dir
#   Step 5: Clean output samples that have original_graph but no triton_kernel
#
# Usage:
#   bash extract_triton_kernels.sh <source> [gpu_ids]
#
# Args:
#   source   (required): "list" or "hf"
#   gpu_ids  (optional): comma-separated GPU IDs, e.g. "0,2,5,7"
#
# Examples:
#   bash extract_triton_kernels.sh list            # list source, auto-detect GPUs
#   bash extract_triton_kernels.sh hf 0,2,5,7      # hf source, specified GPUs

# ============================================================
# Arguments
# ============================================================

SOURCE="${1:?Usage: bash extract_triton_kernels.sh <source> [gpu_ids]  (source: list | hf)}"

if [[ "$SOURCE" != "list" && "$SOURCE" != "hf" ]]; then
    echo "ERROR: Invalid source '$SOURCE'. Must be 'list' or 'hf'."
    exit 1
fi

GPU_ARG="${2:-}"

# ============================================================
# Configuration
# ============================================================

DATASET_BASE_DIR="/path/to/ai4c_dataset"
GRAPHNET_DIR="/path/to/GraphNet/GitHub/repo/"
AI4C_BASE="/path/to/ai4c/repo"
GRAPHNET_HF_DIR="/path/to/GraphNet/Huggingface/repo/"

export PYTHONPATH="$GRAPHNET_DIR:$PYTHONPATH"

if [[ "$SOURCE" == "list" ]]; then
    DATASET_DIR="${DATASET_BASE_DIR}/GitHubV2"
    GRAPH_LIST_FILES=(
        "${DATASET_BASE_DIR}/hf_sole_op_samples_v2_all_expanded.txt"
        "${DATASET_BASE_DIR}/hf_fusible_samples_v2_all_expanded.txt"
        "${DATASET_BASE_DIR}/hf_typical_samples_v2_all_expanded.txt"
    )
elif [[ "$SOURCE" == "hf" ]]; then
    DATASET_DIR="${DATASET_BASE_DIR}/Huggingface"
    HF_SUBDIRS=("sole_op_graph" "fusible_graph" "typical_graph")
else
    echo "ERROR: Invalid source '$SOURCE'. Must be 'list' or 'hf'."
    exit 1
fi

DATASET_NAMES=(
    "sole_op_graph"
    "fusible_graph"
    "typical_graph"
)
CACHE_DIRS=(
    "${DATASET_DIR}/sole_op_graph_inductor_dump"
    "${DATASET_DIR}/fusible_graph_inductor_dump"
    "${DATASET_DIR}/typical_graph_inductor_dump"
)

NUM_DATASETS=${#DATASET_NAMES[@]}

# ============================================================
# GPU detection
# ============================================================

# Priority: $2 arg > $CUDA_VISIBLE_DEVICES env > auto-detect all
if [ -n "$GPU_ARG" ]; then
    IFS=',' read -ra GPU_IDS <<< "$GPU_ARG"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    # Auto-detect: parse "GPU 0:", "GPU 1:", ... from nvidia-smi
    mapfile -t GPU_IDS < <(nvidia-smi -L 2>/dev/null | grep -oP 'GPU \K\d+')
    if [ ${#GPU_IDS[@]} -eq 0 ]; then
        echo "ERROR: No GPUs detected. Pass GPU IDs as argument or check nvidia-smi."
        exit 1
    fi
fi

# Unset so workers get their own isolated CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

NUM_GPUS=${#GPU_IDS[@]}
echo "Source: $SOURCE"
echo "Using $NUM_GPUS GPU(s): ${GPU_IDS[*]}"

# ============================================================
# Enumerate HF samples (hf source only)
# ============================================================

# Args: $1 = hf_subdir (e.g. "sole_op_graph")
enumerate_hf_samples() {
    local hf_subdir="$1"
    local base_dir="$GRAPHNET_HF_DIR/$hf_subdir"

    if [ ! -d "$base_dir" ]; then
        echo "ERROR: HF dataset directory not found: $base_dir" >&2
        return 1
    fi

    find "$base_dir" -name "model.py" -type f -printf '%h\n' | sort
}

# ============================================================
# Worker: compile a chunk of graphs on a single GPU
# ============================================================
# Args: $1 = chunk_file, $2 = cache_dir, $3 = gpu_id
compile_worker() {
    local chunk_file="$1"
    local cache_dir="$2"
    local gpu_id="$3"

    export CUDA_VISIBLE_DEVICES="$gpu_id"

    local total
    total=$(grep -cve '^\s*$' "$chunk_file")
    local current=0 test_count=0 skip_count=0 fail_count=0

    while IFS= read -r sample_path; do
        [[ -z "$sample_path" ]] && continue
        current=$((current + 1))

        # Compute unique_dir and full model path based on source
        local unique_dir full_model_path
        if [[ "$SOURCE" == "list" ]]; then
            unique_dir="${sample_path//\//_}"
            full_model_path="$AI4C_BASE/$sample_path"
        else
            local rel_path="${sample_path#$GRAPHNET_HF_DIR/}"
            unique_dir="${rel_path//\//_}"
            full_model_path="$sample_path"
        fi

        local graph_cache_dir="$cache_dir/$unique_dir"
        local log_file="$graph_cache_dir/test_compiler_log.log"

        # Resume: skip if already completed (check original path + kept/ + discarded/)
        if   { [ -f "$log_file" ] && grep -q '\[Speedup\]\[kernel\]:' "$log_file"; } \
          || [ -d "$cache_dir/kept/$unique_dir" ] \
          || [ -d "$cache_dir/discarded/$unique_dir" ]; then
            echo "[GPU$gpu_id $current/$total] SKIP: $sample_path"
            skip_count=$((skip_count + 1))
            continue
        fi

        # Clean up incomplete cache from previous interrupted run before retrying
        rm -rf "$graph_cache_dir"
        mkdir -p "$graph_cache_dir"

        echo "[GPU$gpu_id $current/$total] Compiling: $full_model_path"

        export TORCH_COMPILE_DEBUG=1
        export TORCHINDUCTOR_CACHE_DIR="$graph_cache_dir"

        if ! python3 -m graph_net_bench.torch.test_compiler --model-path "$full_model_path" --kernel-time \
                > "$log_file" 2>&1; then
            fail_count=$((fail_count + 1))
        fi

        local original_graph_dir="$graph_cache_dir/original_graph"
        mkdir -p "$original_graph_dir"
        cp -r "$full_model_path"/* "$original_graph_dir/"

        test_count=$((test_count + 1))
    done < "$chunk_file"

    echo "[GPU$gpu_id] Done: $test_count compiled, $skip_count skipped, $fail_count failed (total: $total)"
}

# ============================================================
# Post-processing functions
# ============================================================

clean_temp_files() {
    local dir="$1"
    echo "Cleaning temp files from $dir ..."
    find "$dir" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find "$dir" -type f -name "*.pyc" -delete 2>/dev/null
    find "$dir" -type f -name "*.pyo" -delete 2>/dev/null
}

filter_samples_by_speedup() {
    local cache_dir="$1"
    local kept_dir="$cache_dir/kept"
    local discarded_dir="$cache_dir/discarded"
    mkdir -p "$kept_dir" "$discarded_dir"

    local total=0
    for d in "$cache_dir"/*/; do
        [ ! -d "$d" ] && continue
        local n; n=$(basename "$d")
        [[ "$n" == "kept" || "$n" == "discarded" || "$n" == _chunk_* || "$n" == _worker_* ]] && continue
        total=$((total + 1))
    done

    local current=0 kept_count=0 discarded_count=0 skip_count=0

    for graph_dir in "$cache_dir"/*/; do
        [ ! -d "$graph_dir" ] && continue
        local graph_name; graph_name=$(basename "$graph_dir")
        [[ "$graph_name" == "kept" || "$graph_name" == "discarded" || "$graph_name" == _chunk_* || "$graph_name" == _worker_* ]] && continue
        current=$((current + 1))

        if [ -d "$kept_dir/$graph_name" ] || [ -d "$discarded_dir/$graph_name" ]; then
            skip_count=$((skip_count + 1)); continue
        fi

        local log_file="$graph_dir/test_compiler_log.log"
        local should_keep=false
        if [ -f "$log_file" ]; then
            local speedup
            speedup=$(grep -oP '\[Speedup\]\[kernel\]:\s*\K[\d.]+' "$log_file" | tail -1)
            if [ -n "$speedup" ] && awk "BEGIN {exit !($speedup >= 1)}"; then
                should_keep=true
            fi
        fi

        if $should_keep; then
            mv "$graph_dir" "$kept_dir/$graph_name"
            kept_count=$((kept_count + 1))
        else
            mv "$graph_dir" "$discarded_dir/$graph_name"
            discarded_count=$((discarded_count + 1))
        fi
        echo "[$current/$total] $(if $should_keep; then echo KEPT; else echo DISCARDED; fi): $graph_name"
    done
    echo "Filter: $kept_count kept, $discarded_count discarded, $skip_count skipped (total: $total)"
}

extract_triton_kernels() {
    local cache_dir="$1"
    local output_dir="$2"
    local kept_dir="$cache_dir/kept"

    if [ ! -d "$kept_dir" ]; then
        echo "ERROR: Kept directory does not exist: $kept_dir"
        return 1
    fi

    mkdir -p "$output_dir"

    # Clean up incomplete .tmp dirs from previous interrupted runs
    find "$output_dir" -maxdepth 1 -name "*.tmp" -type d -exec rm -rf {} +

    local total=0
    for d in "$kept_dir"/*/; do
        [ ! -d "$d" ] && continue
        [ ! -f "${d}/original_graph/graph_hash.txt" ] && continue
        total=$((total + 1))
    done

    local current=0 total_kernels=0 processed_files=0 copied_graphs=0 skip_count=0

    for graph_dir in "$kept_dir"/*/; do
        [ ! -d "$graph_dir" ] && continue
        [ ! -f "${graph_dir}/original_graph/graph_hash.txt" ] && continue
        current=$((current + 1))

        local graph_name; graph_name=$(basename "$graph_dir")
        local dest_graph_dir="$output_dir/$graph_name"

        if [ -d "$dest_graph_dir" ]; then
            skip_count=$((skip_count + 1)); continue
        fi

        echo "[$current/$total] Extracting: $graph_name"

        # Write to .tmp dir first, then atomic rename on completion
        local tmp_dir="${dest_graph_dir}.tmp"
        rm -rf "$tmp_dir"
        mkdir -p "$tmp_dir"

        if [ -f "${graph_dir}original_graph/model.py" ]; then
            mkdir -p "$tmp_dir/original_graph"
            cp "${graph_dir}original_graph/model.py" "$tmp_dir/original_graph/model.py"
        fi

        while IFS= read -r output_code_file; do
            processed_files=$((processed_files + 1))
            local extracted
            extracted=$(perl -0777 -ne '
                while (/async_compile\.triton\(\x27([^\x27]+)\x27,\s*\x27\x27\x27(.*?)\x27\x27\x27/gs) {
                    print "===KERNEL_NAME===$1\n$2\n===KERNEL_END===\n";
                }
            ' "$output_code_file")
            [ -z "$extracted" ] && continue

            local triton_kernel_dir="$tmp_dir/triton_kernel"
            mkdir -p "$triton_kernel_dir"

            local current_name="" current_code=""
            while IFS= read -r line; do
                if [[ "$line" == ===KERNEL_NAME===* ]]; then
                    current_name="${line#===KERNEL_NAME===}"; current_code=""
                elif [[ "$line" == "===KERNEL_END===" ]]; then
                    if [ -n "$current_name" ]; then
                        printf '%s\n' "$current_code" > "$triton_kernel_dir/${current_name}.py"
                        total_kernels=$((total_kernels + 1))
                    fi
                    current_name=""; current_code=""
                else
                    if [ -z "$current_code" ]; then current_code="$line"
                    else current_code="$current_code
$line"
                    fi
                fi
            done <<< "$extracted"
        done < <(find "$graph_dir" -name "output_code.py" -type f)

        # Atomic: rename .tmp to final destination (same filesystem = rename())
        mv "$tmp_dir" "$dest_graph_dir"
        copied_graphs=$((copied_graphs + 1))
    done

    echo "Extraction: $processed_files files, $total_kernels kernels, $copied_graphs graphs, $skip_count skipped (total: $total)"
    echo "Output:     $output_dir"
}

# Remove output samples that have original_graph/ but no triton_kernel/ directory.
#
# Args: $1 = output_dir
clean_empty_kernel_samples() {
    local output_dir="$1"

    if [ ! -d "$output_dir" ]; then
        echo "WARNING: Output directory does not exist: $output_dir"
        return
    fi

    local total=0
    local removed=0

    for sample_dir in "$output_dir"/*/; do
        [ ! -d "$sample_dir" ] && continue
        total=$((total + 1))

        if [ -d "${sample_dir}original_graph" ] && [ ! -d "${sample_dir}triton_kernel" ]; then
            echo "  Removing (no triton_kernel): $(basename "$sample_dir")"
            rm -rf "$sample_dir"
            removed=$((removed + 1))
        fi
    done

    echo "Cleanup: $removed removed (no triton_kernel), $((total - removed)) kept (total: $total)"
}

# ============================================================
# Main
# ============================================================

# Kill all workers on Ctrl+C
WORKER_PIDS=()
cleanup() {
    echo ""
    echo "Interrupted. Killing workers..."
    for pid in "${WORKER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    exit 1
}
trap cleanup INT TERM

for i in "${!DATASET_NAMES[@]}"; do
    cache_dir="${CACHE_DIRS[$i]}"
    output_dir="${cache_dir}_subgraph_triton_kernel_pair"
    dataset_idx=$((i + 1))

    echo ""
    echo "======================================================"
    echo " Dataset [$dataset_idx/$NUM_DATASETS]: ${DATASET_NAMES[$i]}"

    # Build sample list based on source
    mkdir -p "$cache_dir"
    sample_list_file="$cache_dir/_sample_list.txt"

    if [[ "$SOURCE" == "list" ]]; then
        graph_list_file="${GRAPH_LIST_FILES[$i]}"
        echo " Graph list: $graph_list_file"

        if [ ! -f "$graph_list_file" ]; then
            echo " ERROR: Graph list file not found: $graph_list_file"
            echo "======================================================"
            continue
        fi
        cp "$graph_list_file" "$sample_list_file"
    else
        hf_subdir="${HF_SUBDIRS[$i]}"
        echo " HF dir:     $GRAPHNET_HF_DIR/$hf_subdir"
        enumerate_hf_samples "$hf_subdir" > "$sample_list_file"
    fi

    total_samples=$(grep -cve '^\s*$' "$sample_list_file" || echo 0)
    echo " Samples:    $total_samples"
    echo " Cache dir:  $cache_dir"
    echo " Output dir: $output_dir"
    echo " GPUs:       ${GPU_IDS[*]}"
    echo "======================================================"

    if [ "$total_samples" -eq 0 ]; then
        echo "ERROR: No samples found for dataset: ${DATASET_NAMES[$i]}"
        continue
    fi

    # --- Step 1: Split sample list and compile in parallel ---
    echo ""
    echo "=== Step 1: Parallel compilation ($NUM_GPUS GPUs) ==="

    # Split into chunks (round-robin by GPU index to balance workload)
    chunk_dir="$cache_dir/_chunk_files"
    mkdir -p "$chunk_dir"
    rm -f "$chunk_dir"/chunk_*.txt

    local_idx=0
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        gpu_id="${GPU_IDS[$((local_idx % NUM_GPUS))]}"
        echo "$line" >> "$chunk_dir/chunk_gpu${gpu_id}.txt"
        local_idx=$((local_idx + 1))
    done < "$sample_list_file"

    # Launch one worker per GPU
    WORKER_PIDS=()
    for gpu_id in "${GPU_IDS[@]}"; do
        chunk_file="$chunk_dir/chunk_gpu${gpu_id}.txt"
        [ ! -f "$chunk_file" ] && continue

        worker_log="$cache_dir/_worker_gpu${gpu_id}.log"
        compile_worker "$chunk_file" "$cache_dir" "$gpu_id" > "$worker_log" 2>&1 &
        WORKER_PIDS+=($!)
        echo "  Launched worker GPU $gpu_id (PID ${WORKER_PIDS[-1]}, log: $worker_log)"
    done

    # Wait for all workers
    echo "  Waiting for ${#WORKER_PIDS[@]} workers..."
    all_ok=true
    for pid in "${WORKER_PIDS[@]}"; do
        if ! wait "$pid"; then
            all_ok=false
        fi
    done

    # Print worker summaries
    for gpu_id in "${GPU_IDS[@]}"; do
        worker_log="$cache_dir/_worker_gpu${gpu_id}.log"
        [ -f "$worker_log" ] && tail -1 "$worker_log"
    done

    if ! $all_ok; then
        echo "WARNING: Some workers had errors. Check _worker_gpu*.log files."
    fi

    # Cleanup chunk files
    rm -rf "$chunk_dir"

    # --- Steps 2-5: Sequential post-processing ---
    echo ""
    echo "=== Step 2: Filter by speedup ==="
    filter_samples_by_speedup "$cache_dir"

    echo ""
    echo "=== Step 3: Clean temp files ==="
    clean_temp_files "$cache_dir/kept"

    echo ""
    echo "=== Step 4: Extract triton kernels ==="
    extract_triton_kernels "$cache_dir" "$output_dir"

    echo ""
    echo "=== Step 5: Clean samples without triton kernels ==="
    clean_empty_kernel_samples "$output_dir"
done

echo ""
echo "All datasets processed."
