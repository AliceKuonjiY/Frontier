#!/bin/bash
# Qwen3-4B PD-AF simulation with the filled profiling package.
#
# Data layout expected under data/profiling:
#   compute/hpp2/Qwen3-4B/linear_op_kernel_only.csv
#   compute/h200/Qwen3-4B/{linear_op.csv,linear_op_kernel_only.csv,
#                          attention.csv,attention_kernel_only.csv}
#   cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only.csv
#   network/a2f_h200_to_hpp2/m2n_transfer_lut.csv
#   network/hpp2_16rank/{all_reduce.csv,weight_stream_lut.csv}
#   network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv
#
# The analytical M2N/CC settings and KV LUT contract below match those tables:
#   M2N HostBounce (segmented): IB 366 Gbps + per-layer HPP2 PCIe 23 GB/s
#     - the IB segment is priced by the analytical M2N bandwidth
#     - the PCIe segment is priced by the per-layer HPP2 CPU-overhead table
#     - the two segments are serial; do not also use the 122.44 Gbps
#       HostBounce single-segment effective bandwidth
#   KV Cache Transfer (LUT-driven by PD node placement):
#     - <=8 H200 total (same_node):  NVLink_IntraNode (measured LUT, ~3172.5 Gbps / 0.0208 ms)
#     - >8 H200 total (cross_node):  RDMA_IB_400G_Direct (400 Gbps SOL, ~1.51 ms @ 512 tok)
#   HPP2 ring:  2984 Gbps intra-node (373 GB/s), 0 latency

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-qwen3_4b_pd_af_baseline}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/qwen3-4b-pd-af}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
PREFILL_TOKENS="${PREFILL_TOKENS:-512}"
DECODE_TOKENS="${DECODE_TOKENS:-64}"
ARRIVAL_PATTERN="${ARRIVAL_PATTERN:-burst}"
PREFILL_REPLICAS="${PREFILL_REPLICAS:-1}"
PREFILL_TP="${PREFILL_TP:-8}"
DECODE_ATTN_REPLICAS="${DECODE_ATTN_REPLICAS:-1}"
DECODE_ATTN_TP="${DECODE_ATTN_TP:-8}"
DECODE_FFN_REPLICAS="${DECODE_FFN_REPLICAS:-1}"
WEIGHT_STREAM_LAYERS="${WEIGHT_STREAM_LAYERS:-2}"
HPP2_USABLE_MEMORY_MB="${HPP2_USABLE_MEMORY_MB:-170}"
HPP2_PLANNING_OVERHEAD_MB="${HPP2_PLANNING_OVERHEAD_MB:-10}"
AF_MICRO_BATCH="${AF_MICRO_BATCH:-1}"
DECODE_ATTN_MICRO_BATCH_SIZE="${DECODE_ATTN_MICRO_BATCH_SIZE:-8}"
KV_LUT_PATH="${KV_LUT_PATH-data/profiling/network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv}"
KV_PD_PLACEMENT="${KV_PD_PLACEMENT:-auto}"
KV_LINK_TYPE="${KV_LINK_TYPE:-}"
KV_LUT_PRICING_MODE="${KV_LUT_PRICING_MODE:-piecewise_linear}"
KV_EXTRA_LATENCY_MS="${KV_EXTRA_LATENCY_MS:-0.0}"
KV_ALLOW_INFEASIBLE_PLACEMENT="${KV_ALLOW_INFEASIBLE_PLACEMENT:-0}"

H200_PREFILL_CARDS=$(( PREFILL_REPLICAS * PREFILL_TP ))
H200_DECODE_CARDS=$(( DECODE_ATTN_REPLICAS * DECODE_ATTN_TP ))
H200_TOTAL_CARDS=$(( H200_PREFILL_CARDS + H200_DECODE_CARDS ))

if [ "$KV_PD_PLACEMENT" = "same_node" ] && [ "$H200_TOTAL_CARDS" -gt 8 ] && [ "$#" -eq 0 ]; then
  if [ "$KV_ALLOW_INFEASIBLE_PLACEMENT" != "1" ]; then
    echo "ERROR: KV_PD_PLACEMENT=same_node is infeasible for ${H200_PREFILL_CARDS}P + ${H200_DECODE_CARDS}D = ${H200_TOTAL_CARDS} H200 GPUs (> 8/node)." >&2
    echo "       Use KV_PD_PLACEMENT=auto/cross_node for RDMA_IB_400G_Direct, or set KV_ALLOW_INFEASIBLE_PLACEMENT=1 for sensitivity." >&2
    exit 2
  fi
  echo "WARNING: Running infeasible KV_PD_PLACEMENT=same_node on ${H200_TOTAL_CARDS} H200 GPUs with KV_ALLOW_INFEASIBLE_PLACEMENT=1." >&2
fi

cd "$REPO_ROOT"

MODEL_JSON="data/config/models/Qwen3-4B.json"
REQUIRED=(
  "$MODEL_JSON"
  "data/profiling/compute/hpp2/Qwen3-4B/linear_op_kernel_only.csv"
  "data/profiling/compute/h200/Qwen3-4B/linear_op.csv"
  "data/profiling/compute/h200/Qwen3-4B/linear_op_kernel_only.csv"
  "data/profiling/compute/h200/Qwen3-4B/attention.csv"
  "data/profiling/compute/h200/Qwen3-4B/attention_kernel_only.csv"
  "data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only.csv"
  "data/profiling/network/a2f_h200_to_hpp2/m2n_transfer_lut.csv"
  "data/profiling/network/hpp2_16rank/all_reduce.csv"
  "data/profiling/network/hpp2_16rank/weight_stream_lut.csv"
  "data/profiling/network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv"
)
for path in "${REQUIRED[@]}"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: missing required profiling file: $path" >&2
    exit 2
  fi
done

# The per-layer HPP2 CPU overhead is the HostBounce PCIe segment.
# Warn if a replacement table no longer declares that source.
HPP2_CPU_OVERHEAD_TABLE="data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only.csv"
if ! grep -qi "hostbounce" "$HPP2_CPU_OVERHEAD_TABLE"; then
  echo "WARNING: $HPP2_CPU_OVERHEAD_TABLE does not declare a hostbounce source;" >&2
  echo "         per-layer prepare/process may no longer be the HostBounce PCIe segment." >&2
fi

case "$WEIGHT_STREAM_LAYERS" in
  0) CPU_OVERHEAD_TEMPLATE="" ;;
  1|2|3)
    STREAM_CPU_OVERHEAD_FILE="data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only_stream${WEIGHT_STREAM_LAYERS}.csv"
    if [ ! -f "$STREAM_CPU_OVERHEAD_FILE" ]; then
      echo "ERROR: missing stream CPU-overhead table: $STREAM_CPU_OVERHEAD_FILE" >&2
      exit 2
    fi
    # Keep {NETWORK_DEVICE}/{MODEL} placeholders so only hpp2_16rank resolves
    # the HPP2 streaming table while h200_dgx stays unpopulated (host term = 0).
    CPU_OVERHEAD_TEMPLATE="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads_kernel_only_stream${WEIGHT_STREAM_LAYERS}.csv"
    ;;
  *)
    echo "ERROR: WEIGHT_STREAM_LAYERS must be 0, 1, 2, or 3; got $WEIGHT_STREAM_LAYERS" >&2
    exit 2
    ;;
esac

CLI_OVERRIDES_FFN_BUDGET=0
for arg in "$@"; do
  case "$arg" in
    --cluster_config_decode_ffn_replica_config_streamed_ffn_layers|--cluster_config_decode_ffn_replica_config_streamed_ffn_layers=*)
      echo "ERROR: Do not pass $arg via trailing script arguments; set WEIGHT_STREAM_LAYERS=<0|1|2|3> so the matching stream CPU-overhead table and SRAM guard stay in sync." >&2
      exit 2
      ;;
    --cluster_config_decode_ffn_replica_config_usable_memory_mb|--cluster_config_decode_ffn_replica_config_usable_memory_mb=*|\
    --cluster_config_decode_ffn_replica_config_planning_overhead_mb|--cluster_config_decode_ffn_replica_config_planning_overhead_mb=*|\
    --cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size|--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size=*)
      CLI_OVERRIDES_FFN_BUDGET=1
      ;;
  esac
done

# Validate HPP2 SRAM budget env inputs before applying the TP16 guard.
if ! awk -v u="$HPP2_USABLE_MEMORY_MB" -v o="$HPP2_PLANNING_OVERHEAD_MB" \
  'BEGIN { num = "^[0-9]+(\\.[0-9]+)?$"; exit !(u ~ num && o ~ num && u > 0 && u <= 192 && o >= 0 && o < u) }'; then
  echo "ERROR: Invalid HPP2 SRAM budget (HPP2_USABLE_MEMORY_MB=${HPP2_USABLE_MEMORY_MB}, HPP2_PLANNING_OVERHEAD_MB=${HPP2_PLANNING_OVERHEAD_MB}); require numeric 0 < usable <= 192 MiB and 0 <= overhead < usable MiB." >&2
  exit 2
fi

# Strict binary MiB HPP2 SRAM guard: 170 MiB usable - 10 MiB overhead = 160 MiB.
# At TP16, stream 0/1 needs 36 * 4,669,440 B = 160.3125 MiB on-chip; fail fast
# unless the numeric net budget (usable - overhead) is explicitly relaxed above 160.3125 MiB
# or trailing CLI flags override decode-FFN budget/TP for MemoryPlanner to validate.
if [ "$WEIGHT_STREAM_LAYERS" = "0" ] || [ "$WEIGHT_STREAM_LAYERS" = "1" ]; then
  if [ "$CLI_OVERRIDES_FFN_BUDGET" = "1" ]; then
    echo "WARNING: Deferring WEIGHT_STREAM_LAYERS=$WEIGHT_STREAM_LAYERS HPP2 SRAM check to MemoryPlanner due to trailing CLI decode-FFN budget/TP override." >&2
  elif awk -v u="$HPP2_USABLE_MEMORY_MB" -v o="$HPP2_PLANNING_OVERHEAD_MB" \
    'BEGIN { exit !((u - o) > 160.3125) }'; then
    echo "WARNING: Running WEIGHT_STREAM_LAYERS=$WEIGHT_STREAM_LAYERS with relaxed HPP2 SRAM budget (usable=${HPP2_USABLE_MEMORY_MB} MiB, overhead=${HPP2_PLANNING_OVERHEAD_MB} MiB; default 170/10 MiB is infeasible)." >&2
  else
    echo "ERROR: WEIGHT_STREAM_LAYERS=$WEIGHT_STREAM_LAYERS requires 160.31 MiB + ${HPP2_PLANNING_OVERHEAD_MB} MiB > ${HPP2_USABLE_MEMORY_MB} MiB usable SRAM (default 170.00 MiB usable / 10.00 MiB overhead, nominal 192 MiB)." >&2
    echo "       Use WEIGHT_STREAM_LAYERS=2 (155.86 + 10.00 = 165.86 MiB) or 3 (151.41 + 10.00 = 161.41 MiB), or override HPP2_USABLE_MEMORY_MB / HPP2_PLANNING_OVERHEAD_MB for diagnostics." >&2
    exit 2
  fi
fi

case "$ARRIVAL_PATTERN" in
  burst)
    ARRIVAL_ARGS=(--interval_generator_config_type static)
    ARRIVAL_DESCRIPTION="all requests arrive at time 0"
    if [ -n "${QPS:-}" ]; then
      echo "Note: QPS is ignored for a burst; set ARRIVAL_PATTERN=poisson to use QPS." >&2
    fi
    ;;
  poisson)
    QPS="${QPS:-1.0}"
    ARRIVAL_ARGS=(--interval_generator_config_type poisson
                  --poisson_request_interval_generator_config_qps "$QPS"
                  --offline_use_generated_request_arrivals)
    ARRIVAL_DESCRIPTION="Poisson arrivals, offered QPS=$QPS"
    ;;
  *)
    echo "ERROR: ARRIVAL_PATTERN must be burst or poisson; got $ARRIVAL_PATTERN" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON_BIN" -m frontier.main
  --simulation_mode offline
  --sys_arch pd-af-disaggregation
  --no-enable_parallel_clusters

  --cluster_config_prefill_cluster_num_replicas "$PREFILL_REPLICAS"
  --cluster_config_decode_attn_cluster_num_replicas "$DECODE_ATTN_REPLICAS"
  --cluster_config_decode_ffn_cluster_num_replicas "$DECODE_FFN_REPLICAS"

  --cluster_config_decode_attn_af_pipeline_num_micro_batch "$AF_MICRO_BATCH"
  --cluster_config_decode_ffn_af_pipeline_num_micro_batch "$AF_MICRO_BATCH"
  --cluster_config_decode_attn_micro_batch_size "$DECODE_ATTN_MICRO_BATCH_SIZE"

  # PREFILL: H200, dense attention TP=PREFILL_TP.  The dense FFN follows attn TP;
  # moe_tp only satisfies the shared role contract and is unused by a dense FFN.
  --cluster_config_prefill_replica_config_num_pipeline_stages 1
  --cluster_config_prefill_replica_config_attn_tensor_parallel_size "$PREFILL_TP"
  --cluster_config_prefill_replica_config_moe_tensor_parallel_size "$PREFILL_TP"
  --cluster_config_prefill_replica_config_moe_expert_parallel_size 1
  --cluster_config_prefill_replica_config_total_expert_num 1
  --cluster_config_prefill_replica_config_router_topk 1
  --cluster_config_prefill_replica_config_device h200
  --cluster_config_prefill_replica_config_network_device h200_dgx
  --cluster_config_prefill_replica_config_memory_margin_fraction 0.2

  # DECODE_ATTN: H200, dense attention TP=DECODE_ATTN_TP.
  --cluster_config_decode_attn_replica_config_num_pipeline_stages 1
  --cluster_config_decode_attn_replica_config_attn_tensor_parallel_size "$DECODE_ATTN_TP"
  --cluster_config_decode_attn_replica_config_device h200
  --cluster_config_decode_attn_replica_config_network_device h200_dgx
  --cluster_config_decode_attn_replica_config_memory_margin_fraction 0.2

  # DECODE_FFN: 8x HPP2 = 16 dies/ranks, dense FFN TP16.
  # The CPU-overhead loader selects moe_tensor_parallel_size for DECODE_FFN,
  # so attn_tensor_parallel_size stays at its default.
  --cluster_config_decode_ffn_replica_config_num_pipeline_stages 1
  --cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size 16
  --cluster_config_decode_ffn_replica_config_moe_expert_parallel_size 1
  --cluster_config_decode_ffn_replica_config_total_expert_num 1
  --cluster_config_decode_ffn_replica_config_router_topk 1
  --cluster_config_decode_ffn_replica_config_device hpp2
  --cluster_config_decode_ffn_replica_config_streamed_ffn_layers "$WEIGHT_STREAM_LAYERS"
  --cluster_config_decode_ffn_replica_config_usable_memory_mb "$HPP2_USABLE_MEMORY_MB"
  --cluster_config_decode_ffn_replica_config_planning_overhead_mb "$HPP2_PLANNING_OVERHEAD_MB"
  --cluster_config_decode_ffn_replica_config_network_device hpp2_16rank
  --cluster_config_decode_ffn_replica_config_memory_margin_fraction 0.0

  --cluster_config_prefill_replica_scheduler_config_type vllm_v1
  --cluster_config_decode_attn_replica_scheduler_config_type vllm_v1
  --cluster_config_decode_ffn_replica_scheduler_config_type orca

  --cc_backend_config_type analytical
  --m2n_transfer_config_type analytical
  --kv_cache_transfer_config_type analytical

  # Fitted from the filled M2N / all-reduce LUTs; KV transfer is priced directly
  # from kv_cache_transfer_lut.csv according to PD node placement (auto: <=8 H200
  # -> NVLink_IntraNode, >8 H200 -> RDMA_IB_400G_Direct).
  # M2N HostBounce is segmented: 366 Gbps is the IB segment; the per-layer
  # PCIe segment is carried by the HPP2 CPU-overhead table (23 GB/s).
  --analytical_m2_n_transfer_config_p2p_protocol host_bounce
  --analytical_m2_n_transfer_config_memory_bandwidth_gbps 366.0
  --analytical_m2_n_transfer_config_network_latency_ms 0.0
  --no-analytical_m2_n_transfer_config_enable_p2p_optimization
  --analytical_kv_cache_transfer_config_pd_placement "$KV_PD_PLACEMENT"
  --analytical_kv_cache_transfer_config_lut_pricing_mode "$KV_LUT_PRICING_MODE"
  --analytical_kv_cache_transfer_config_extra_latency_ms "$KV_EXTRA_LATENCY_MS"
  --analytical_cc_backend_config_intra_node_bandwidth_gbps 2984.0
  --analytical_cc_backend_config_network_bandwidth_gbps 292.8
  --analytical_cc_backend_config_network_latency_us 0.0

  --replica_config_model_name Qwen3-4B
  --replica_config_moe_routing_distribution_type balanced
  --replica_config_moe_routing_seed 42

  --vllm_v1_scheduler_config_max_tokens_in_batch 8192
  --vllm_v1_scheduler_config_long_prefill_token_threshold 64
  --vllm_v1_scheduler_config_block_size 16
  --vllm_v1_scheduler_config_num_blocks 4096
  --vllm_v1_scheduler_config_enable_chunked_prefill

  --request_generator_config_type synthetic
  --synthetic_request_generator_config_num_requests "$NUM_REQUESTS"
  --length_generator_config_type fixed
  --fixed_request_length_generator_config_prefill_tokens "$PREFILL_TOKENS"
  --fixed_request_length_generator_config_decode_tokens "$DECODE_TOKENS"
  "${ARRIVAL_ARGS[@]}"

  # Use the filled HPP2 CPU-overhead table.  H200 clusters have no table and
  # will log a warning and price their host terms as 0.
  --no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling
  --random_forrest_execution_time_predictor_config_k_fold_cv_splits 3

  --metrics_config_output_dir "$OUT_DIR"
  --metrics_config_run_id "$RUN_ID"
  --metrics_config_write_metrics
  --metrics_config_store_request_metrics
  --metrics_config_store_batch_metrics
  --metrics_config_store_token_completion_metrics
  --metrics_config_store_utilization_metrics
  --no-metrics_config_store_plots
  --no-metrics_config_enable_chrome_trace
  --no-metrics_config_write_json_trace
)

if [ -n "$KV_LUT_PATH" ]; then
  if [ ! -f "$KV_LUT_PATH" ]; then
    echo "ERROR: missing KV cache transfer LUT file: $KV_LUT_PATH" >&2
    exit 2
  fi
  CMD+=(
    --analytical_kv_cache_transfer_config_lut_path "$KV_LUT_PATH"
  )
fi

if [ -n "$KV_LINK_TYPE" ]; then
  CMD+=(
    --analytical_kv_cache_transfer_config_link_type "$KV_LINK_TYPE"
  )
fi

if [ "$KV_ALLOW_INFEASIBLE_PLACEMENT" = "1" ]; then
  CMD+=(
    --analytical_kv_cache_transfer_config_allow_infeasible_placement_override
  )
fi

if [ -n "$CPU_OVERHEAD_TEMPLATE" ]; then
  CMD+=(
    --random_forrest_execution_time_predictor_config_cpu_overhead_kernel_only_input_file
    "$CPU_OVERHEAD_TEMPLATE"
  )
fi

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

echo "========================================================="
echo " Qwen3-4B PD-AF"
echo " model: Qwen3-4B"
echo " PREFILL:    ${PREFILL_REPLICAS}x h200 TP${PREFILL_TP} (${H200_PREFILL_CARDS} GPU)"
echo " DECODE_ATTN:${DECODE_ATTN_REPLICAS}x h200 TP${DECODE_ATTN_TP} (${H200_DECODE_CARDS} GPU)"
echo " DECODE_FFN: ${DECODE_FFN_REPLICAS}x hpp2 TP16 (${HPP2_USABLE_MEMORY_MB} MiB usable, ${HPP2_PLANNING_OVERHEAD_MB} MiB overhead, 192 MiB nominal)"
echo " KV link:    placement=${KV_PD_PLACEMENT} (${H200_TOTAL_CARDS} H200), lut=${KV_LUT_PATH:-none}, mode=${KV_LUT_PRICING_MODE}, extra_lat=${KV_EXTRA_LATENCY_MS} ms"
echo " requests:   $NUM_REQUESTS, prefill=$PREFILL_TOKENS, decode=$DECODE_TOKENS; $ARRIVAL_DESCRIPTION"
echo " weight stream: $WEIGHT_STREAM_LAYERS layer(s)"
echo " metrics:    $OUT_DIR/$RUN_ID"
echo "========================================================="
exec "${CMD[@]}"