import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from frontier.config import (
    ClusterConfig,
    MetricsConfig,
    OrcaSchedulerConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
    SyntheticRequestGeneratorConfig,
    global_vars,
)
from frontier.config.device_sku_config import (
    BINARY_MIB_BYTES,
    H200DeviceSKUConfig,
    HPP2DeviceSKUConfig,
)
from frontier.entities.replica import Replica
from frontier.errors import FrontierMemoryOOMError
from frontier.metrics.metrics_store import MetricsStore
from frontier.scheduler.replica_scheduler.orca_replica_scheduler import (
    OrcaReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.scheduler.utils.sram_budget import (
    SRAMProfileContract,
    load_sram_profile_contract,
)
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter

REPO_ROOT = Path(__file__).resolve().parents[2]
HPP2_LINEAR_CSV = (
    REPO_ROOT / "data/profiling/compute/hpp2/Qwen3-4B/linear_op_kernel_only.csv"
)
H200_LINEAR_CSV = (
    REPO_ROOT / "data/profiling/compute/h200/Qwen3-4B/linear_op_kernel_only.csv"
)


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def ffn_replica(streamed_layers, **overrides):
    config = dict(
        model_name="Qwen3-4B",
        cluster_prefix="decode_ffn",
        device="hpp2",
        network_device="hpp2_16rank",
        moe_tensor_parallel_size=16,
        streamed_ffn_layers=streamed_layers,
    )
    return ReplicaConfig(**(config | overrides))


def test_hpp2_sku_defaults_and_non_hpp2_unchanged():
    hpp2 = HPP2DeviceSKUConfig()
    assert hpp2.get_nominal_memory_bytes() == 192 * BINARY_MIB_BYTES
    assert hpp2.get_usable_memory_bytes() == 170 * BINARY_MIB_BYTES
    assert hpp2.get_planning_overhead_bytes() == 10 * BINARY_MIB_BYTES

    h200 = H200DeviceSKUConfig()
    assert h200.get_usable_memory_bytes() == h200.get_nominal_memory_bytes()
    assert h200.get_planning_overhead_bytes() == 0


@pytest.mark.parametrize("streamed_layers", [0, 1, 2, 3, 36])
def test_dense_ffn_streaming_preserves_logical_weights_and_budgets_one_buffer(streamed_layers):
    config = ffn_replica(streamed_layers)
    counter = ParamCounter(config, ClusterType.DECODE_FFN)
    planner = MemoryPlanner(config, replica=None, cluster_type=ClusterType.DECODE_FFN)
    layer_bytes = 3 * 2560 * 9728 // 16
    resident = (36 - streamed_layers) * layer_bytes
    buffer_bytes = layer_bytes if streamed_layers else 0
    assert counter.get_num_mlp_parameters_per_device() == 36 * layer_bytes
    assert counter.get_streamed_ffn_weight_bytes_per_device() == streamed_layers * layer_bytes
    assert counter.get_parameter_memory_per_device_bytes() == resident
    assert counter.get_ffn_streaming_buffer_bytes_per_device() == buffer_bytes
    assert planner.get_parameter_memory_per_device_bytes() == resident + buffer_bytes


@pytest.mark.parametrize("n", [-1, 37, 1.5])
def test_invalid_layer_count(n):
    with pytest.raises(ValueError, match="streamed_ffn_layers"):
        ffn_replica(n)


@pytest.mark.parametrize("overrides", [
    {"cluster_prefix": "decode_attn"},
    {"model_name": "step-moe-noquant"},
    {"num_pipeline_stages": 2},
])
def test_streamed_layers_require_supported_role_and_uniform_dense_weights(overrides):
    with pytest.raises(ValueError, match="streamed_ffn_layers"):
        ffn_replica(1, **overrides)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"usable_memory_mb": 0}, "usable_memory_mb"),
        ({"usable_memory_mb": 193.0}, "cannot exceed nominal"),
        ({"usable_memory_mb": True}, "usable_memory_mb"),
        ({"planning_overhead_mb": -1.0}, "planning_overhead_mb"),
        ({"planning_overhead_mb": 170.0}, "strictly smaller than usable"),
    ],
)
def test_invalid_usable_or_planning_overhead_overrides_fail_fast(overrides, match):
    with pytest.raises(ValueError, match=match):
        ffn_replica(2, **overrides)


def test_simulation_config_cli_parses_decode_ffn_sram_budget_flags():
    cli_args = [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters",
        "--replica_config_model_name", "Qwen3-4B",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", "1",
        "--cluster_config_decode_ffn_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_prefill_replica_config_device", "h200",
        "--cluster_config_prefill_replica_config_network_device", "h200_dgx",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "8",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "8",
        "--cluster_config_decode_attn_replica_config_device", "h200",
        "--cluster_config_decode_attn_replica_config_network_device", "h200_dgx",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", "8",
        "--cluster_config_decode_ffn_replica_config_device", "hpp2",
        "--cluster_config_decode_ffn_replica_config_network_device", "hpp2_16rank",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "16",
        "--cluster_config_decode_ffn_replica_config_streamed_ffn_layers", "2",
        "--cluster_config_decode_ffn_replica_config_usable_memory_mb", "170",
        "--cluster_config_decode_ffn_replica_config_planning_overhead_mb", "10",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.0",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",
        "--cc_backend_config_type", "analytical",
        "--m2n_transfer_config_type", "analytical",
        "--kv_cache_transfer_config_type", "analytical",
    ]
    global_vars.reset_global_vars()
    try:
        with patch.object(sys, "argv", cli_args):
            ffn = SimulationConfig.create_from_cli_args().cluster_config.decode_ffn_replica_config
        assert ffn.usable_memory_mb == 170.0
        assert ffn.planning_overhead_mb == 10.0
        assert not ffn.has_relaxed_memory_budget_override()
    finally:
        global_vars.reset_global_vars()


@pytest.mark.parametrize("tp,streamed_layers,activation_bytes,should_fit", [
    (8, 0, None, False),
    (8, 3, None, False),
    (16, 0, None, False),
    (16, 1, None, False),
    (16, 2, None, True),
    (16, 3, None, True),
    (16, 3, 0, False),
    (16, 3, 1, True),
])
def test_decode_ffn_requires_free_activation_capacity(tp, streamed_layers, activation_bytes, should_fit):
    config = ffn_replica(streamed_layers, moe_tensor_parallel_size=tp)
    rank = SimpleNamespace(total_memory_gb=0.1875, memory_margin_fraction=0)
    planner = MemoryPlanner(config, rank, ClusterType.DECODE_FFN)
    overhead = 0
    if activation_bytes is not None:
        overhead = 160 * BINARY_MIB_BYTES - planner.get_parameter_memory_per_device_bytes() - activation_bytes
    if should_fit:
        assert planner.get_max_batch_size(non_kv_cache_overhead_bytes=overhead) >= 1
    else:
        with pytest.raises(FrontierMemoryOOMError) as error:
            planner.get_max_batch_size(non_kv_cache_overhead_bytes=overhead)
        assert error.value.reason == "insufficient_activation_budget"


def test_profiling_csv_sram_contract_and_diagnostic_override(tmp_path):
    tp16 = load_sram_profile_contract(str(HPP2_LINEAR_CSV), 16)
    assert tp16 is not None
    assert tp16.sram_fit_with_10mb_overhead is False
    assert tp16.weight_streaming_required is True
    assert load_sram_profile_contract(str(H200_LINEAR_CSV), 8) is None

    bad_csv = tmp_path / "conflict.csv"
    bad_csv.write_text(
        "num_tensor_parallel_workers,sram_fit_with_10mb_overhead\n16,False\n16,True\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Conflicting values"):
        load_sram_profile_contract(str(bad_csv), 16)

    # CSV rejects all-resident even when analytical weights fit under default budget.
    synthetic_contract = SRAMProfileContract(
        sram_fit_170mb=True, sram_fit_with_10mb_overhead=False, weight_streaming_required=True
    )
    strict_planner = MemoryPlanner(
        ffn_replica(0, moe_tensor_parallel_size=32, memory_margin_fraction=0.0),
        replica=None,
        cluster_type=ClusterType.DECODE_FFN,
        sram_profile_contract=synthetic_contract,
    )
    with pytest.raises(FrontierMemoryOOMError) as error:
        strict_planner.get_max_batch_size()
    assert error.value.reason == "profiling_sram_budget_exceeded"

    # Explicit relaxed override allows diagnostic stream-0 and logs one WARNING.
    diag_cfg = ffn_replica(0, usable_memory_mb=192.0, planning_overhead_mb=0.0, memory_margin_fraction=0.0)
    diag_planner = MemoryPlanner(
        diag_cfg, replica=None, cluster_type=ClusterType.DECODE_FFN, sram_profile_contract=tp16
    )
    collector = _RecordCollector()
    planner_logger = logging.getLogger("frontier.scheduler.utils.memory_planner")
    planner_logger.addHandler(collector)
    try:
        assert diag_planner.get_max_batch_size() >= 1
        assert diag_planner.get_max_batch_size() >= 1
    finally:
        planner_logger.removeHandler(collector)
    warnings = [m for m in collector.messages if "SRAM_BUDGET_DIAGNOSTIC_OVERRIDE" in m]
    assert len(warnings) == 1
    assert "sram_fit_with_10mb_overhead=False" in warnings[0]


def test_metrics_store_and_scheduler_report_nominal_192_mib_and_usable_170_mib(tmp_path):
    global_vars.reset_global_vars()
    try:
        cluster_cfg = ClusterConfig(
            num_replicas=None,
            replica_config=ReplicaConfig(model_name="Qwen3-4B"),
            prefill_cluster_num_replicas=1,
            decode_attn_cluster_num_replicas=1,
            decode_ffn_cluster_num_replicas=1,
            decode_attn_af_pipeline_num_micro_batch=1,
            decode_ffn_af_pipeline_num_micro_batch=1,
            prefill_replica_config_device="h200",
            prefill_replica_config_network_device="h200_dgx",
            prefill_replica_config_attn_tensor_parallel_size=8,
            prefill_replica_config_moe_tensor_parallel_size=8,
            decode_attn_replica_config_device="h200",
            decode_attn_replica_config_network_device="h200_dgx",
            decode_attn_replica_config_attn_tensor_parallel_size=8,
            decode_ffn_replica_config_device="hpp2",
            decode_ffn_replica_config_network_device="hpp2_16rank",
            decode_ffn_replica_config_moe_tensor_parallel_size=16,
            decode_ffn_replica_config_streamed_ffn_layers=2,
            decode_ffn_replica_config_memory_margin_fraction=0.0,
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=False,
            ),
        )
        sim_cfg = SimulationConfig(
            sys_arch="pd-af-disaggregation",
            enable_parallel_clusters=False,
            cluster_config=cluster_cfg,
            metrics_config=MetricsConfig(
                output_dir=str(tmp_path),
                store_plots=False,
                store_utilization_metrics=False,
                store_operation_metrics=False,
                store_frontier_stage_batch_ledger=False,
            ),
        )
        store = MetricsStore(sim_cfg, sim_cfg.get_clusters())
        store._write_system_metrics()
        payload = json.loads(
            (Path(store._config.output_dir) / "system_metrics.json").read_text(encoding="utf-8")
        )
        ffn = payload["model_weight_memory"]["DECODE_FFN"]
        expected_nominal = 163_430_400 / (192 * BINARY_MIB_BYTES) * 100.0
        expected_usable = 163_430_400 / (170 * BINARY_MIB_BYTES) * 100.0
        assert ffn["memory_utilization_percent"] == pytest.approx(expected_usable)
        assert ffn["usable_memory_utilization_percent"] == pytest.approx(expected_usable)
        assert ffn["nominal_memory_utilization_percent"] == pytest.approx(expected_nominal)
        assert ffn["device_info"]["nominal_device_memory_mib"] == 192.0
        assert ffn["device_info"]["usable_device_memory_mib"] == 170.0
        assert ffn["device_info"]["planning_overhead_mib"] == 10.0
        assert ffn["weight_residency"]["resident_plus_buffer_bytes"] == 163_430_400
        assert ffn["memory_budget"]["usable_headroom_bytes"] == 4_341_760
        assert ffn["profiling_sram_contract"]["sram_fit_with_10mb_overhead"] is False

        prefill = payload["model_weight_memory"]["PREFILL"]
        assert prefill["nominal_memory_utilization_percent"] == pytest.approx(
            prefill["usable_memory_utilization_percent"]
        )
        assert "profiling_sram_contract" not in prefill

        # Verify DECODE_FFN scheduler logs resident+buffer+10 MiB breakdown for stream-3.
        replica_cfg = ffn_replica(3, memory_margin_fraction=0.0)
        generator_cfg = SyntheticRequestGeneratorConfig(num_requests=1)
        collector = _RecordCollector()
        sched_logger = logging.getLogger(
            "frontier.scheduler.replica_scheduler.base_replica_scheduler"
        )
        sched_logger.addHandler(collector)
        try:
            scheduler = OrcaReplicaScheduler(
                replica_config=replica_cfg,
                replica_scheduler_config=OrcaSchedulerConfig(),
                request_generator_config=generator_cfg,
                replica=Replica(replica_cfg, generator_cfg, ClusterType.DECODE_FFN),
                predictor=SimpleNamespace(
                    _config=RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=False)
                ),
                cluster_scheduler=SimpleNamespace(
                    get_stage_execution_context=lambda rid, sid: StageExecutionContext(
                        replica_id=rid, stage_id=sid, ep_size=1
                    )
                ),
                cluster_type=ClusterType.DECODE_FFN,
            )
        finally:
            sched_logger.removeHandler(collector)
        assert scheduler._activation_memory_capacity_bytes == 9_011_200
        cap_logs = [m for m in collector.messages if "[ACTIVATION_MEMORY_CAPACITY]" in m]
        assert len(cap_logs) == 1
        for token in ("192.00 MiB", "170.00 MiB", "10.00 MiB", "158760960B", "169246720B", "9011200B"):
            assert token in cap_logs[0]
    finally:
        global_vars.reset_global_vars()
