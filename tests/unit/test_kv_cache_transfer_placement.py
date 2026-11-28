"""Unit and E2E tests for PD node placement and LUT-backed KV cache transfer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from frontier.config import global_vars
from frontier.config.config import (
    ClusterConfig,
    FixedRequestLengthGeneratorConfig,
    MetricsConfig,
    OrcaSchedulerConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
    StaticRequestIntervalGeneratorConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.entities.batch import Batch
from frontier.entities.replica import Replica
from frontier.entities.request import Request
from frontier.kv_cache_transfer import (
    AnalyticalKVCacheTransferPredictor,
    KVCacheTransferPredictorRegistry,
)
import frontier.kv_cache_transfer.link_profile as link_profile
from frontier.simulator import Simulator
from frontier.types import ClusterType, KVCacheTransferType

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN3_KV_LUT = (
    REPO_ROOT / "data/profiling/network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv"
)
KV_512_BYTES = 36 * 8 * 128 * 2 * 2 * 512  # 75,497,472 B


def _pdaf_cluster(
    p_rep: int, p_tp: int, d_rep: int, d_tp: int, *, d_device: str = "h200", d_net: str = "h200_dgx"
) -> ClusterConfig:
    return ClusterConfig(
        num_replicas=None,
        replica_config=ReplicaConfig(model_name="Qwen3-4B"),
        replica_scheduler_config=VllmV1SchedulerConfig(num_blocks=256),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True, dummy_execution_time_ms=0.5, skip_cpu_overhead_modeling=True
        ),
        prefill_cluster_num_replicas=p_rep,
        decode_attn_cluster_num_replicas=d_rep,
        decode_ffn_cluster_num_replicas=1,
        prefill_replica_config_attn_tensor_parallel_size=p_tp,
        prefill_replica_config_moe_tensor_parallel_size=p_tp,
        prefill_replica_config_device="h200",
        prefill_replica_config_network_device="h200_dgx",
        decode_attn_replica_config_attn_tensor_parallel_size=d_tp,
        decode_attn_replica_config_device=d_device,
        decode_attn_replica_config_network_device=d_net,
        decode_ffn_replica_config_moe_tensor_parallel_size=16,
        decode_ffn_replica_config_device="hpp2",
        decode_ffn_replica_config_network_device="hpp2_16rank",
        decode_ffn_replica_config_streamed_ffn_layers=2,
        decode_ffn_replica_config_memory_margin_fraction=0.0,
        prefill_replica_scheduler_config_type="vllm_v1",
        decode_attn_replica_scheduler_config_type="vllm_v1",
        decode_ffn_replica_scheduler_config_type="orca",
        decode_attn_af_pipeline_num_micro_batch=1,
        decode_ffn_af_pipeline_num_micro_batch=1,
        decode_attn_micro_batch_size=8,
    )


@pytest.mark.parametrize(
    ("p_rep", "p_tp", "d_rep", "d_tp", "expected_same", "expected_auto"),
    [
        (1, 2, 1, 2, True, "same_node"),
        (2, 2, 2, 2, True, "same_node"),
        (2, 2, 4, 2, False, "cross_node"),
        (1, 8, 1, 8, False, "cross_node"),
        (3, 2, 5, 2, False, "cross_node"),
    ],
)
def test_pd_node_placement_auto_and_fail_fast(
    p_rep: int, p_tp: int, d_rep: int, d_tp: int, expected_same: bool, expected_auto: str
) -> None:
    cluster = _pdaf_cluster(p_rep, p_tp, d_rep, d_tp)
    pl = link_profile.evaluate_pd_node_placement(cluster, "pd-af-disaggregation", "auto")
    assert pl.same_node_feasible is expected_same
    assert pl.resolved_placement == expected_auto
    assert pl.total_pd_devices == p_rep * p_tp + d_rep * d_tp

    if not expected_same:
        with pytest.raises(ValueError, match="exceeds single-node capacity"):
            link_profile.evaluate_pd_node_placement(cluster, "pd-af-disaggregation", "same_node")
        ov = link_profile.evaluate_pd_node_placement(
            cluster, "pd-af-disaggregation", "same_node", allow_infeasible_override=True
        )
        assert ov.resolved_placement == "same_node"
        assert ov.placement_feasible is False and ov.diagnostic_override is True


def test_pd_placement_pdd_and_heterogeneous_sku() -> None:
    pdd = ClusterConfig(
        num_replicas=None,
        replica_config=ReplicaConfig(model_name="Qwen3-4B"),
        replica_scheduler_config=VllmV1SchedulerConfig(num_blocks=256),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True),
        prefill_cluster_num_replicas=1,
        decode_cluster_num_replicas=1,
        prefill_replica_config_attn_tensor_parallel_size=8,
        prefill_replica_config_moe_tensor_parallel_size=8,
        prefill_replica_config_device="h200",
        prefill_replica_config_network_device="h200_dgx",
        decode_replica_config_attn_tensor_parallel_size=8,
        decode_replica_config_moe_tensor_parallel_size=8,
        decode_replica_config_device="h200",
        decode_replica_config_network_device="h200_dgx",
    )
    assert link_profile.evaluate_pd_node_placement(pdd, "pd-disaggregation", "auto").resolved_placement == "cross_node"

    hetero = _pdaf_cluster(1, 2, 1, 2, d_device="h100", d_net="h100_dgx")
    ph = link_profile.evaluate_pd_node_placement(hetero, "pd-af-disaggregation", "auto")
    assert ph.same_node_feasible is False and ph.resolved_placement == "cross_node"


def test_kv_lut_pricing_modes_and_validation(tmp_path: Path) -> None:
    nv = link_profile.load_kv_cache_transfer_lut(str(QWEN3_KV_LUT), "NVLink_IntraNode")
    rdma = link_profile.load_kv_cache_transfer_lut(str(QWEN3_KV_LUT), "RDMA_IB_400G_Direct")

    assert nv.predict_time_ms(KV_512_BYTES, "piecewise_linear") == pytest.approx(0.213349)
    assert nv.predict_time_ms(KV_512_BYTES, "linear_fit") == pytest.approx(0.211199, rel=1e-4)
    assert rdma.predict_time_ms(KV_512_BYTES, "piecewise_linear") == pytest.approx(1.50994944)
    assert rdma.predict_time_ms(KV_512_BYTES, "linear_fit") == pytest.approx(1.50994944)
    assert rdma.predict_time_ms(0.0) == pytest.approx(0.0)
    assert rdma.predict_time_ms(KV_512_BYTES / 2.0) == pytest.approx(0.75497472)
    assert rdma.predict_time_ms(6_039_797_760 * 2.0) == pytest.approx(120.7959552 * 2.0)

    with pytest.raises(ValueError, match="not found"):
        link_profile.load_kv_cache_transfer_lut(str(QWEN3_KV_LUT), "Missing_Link")
    bad = tmp_path / "bad.csv"
    bad.write_text("link_type,kv_cache_bytes,time_stats.kv_cache_transfer.median\nA,100,2.0\nA,200,1.0\n")
    with pytest.raises(ValueError, match="Non-monotonic"):
        link_profile.load_kv_cache_transfer_lut(str(bad), "A")


def test_resolve_profile_conflicts_override_and_legacy_compat(tmp_path: Path) -> None:
    c8, c16 = _pdaf_cluster(2, 2, 2, 2), _pdaf_cluster(1, 8, 1, 8)
    cfg_auto = AnalyticalKVCacheTransferConfig(lut_path=str(QWEN3_KV_LUT), pd_placement="auto")
    assert link_profile.resolve_kv_cache_transfer_profile(cfg_auto, c8, "pd-af-disaggregation").link_type == "NVLink_IntraNode"
    assert link_profile.resolve_kv_cache_transfer_profile(cfg_auto, c16, "pd-af-disaggregation").link_type == "RDMA_IB_400G_Direct"

    with pytest.raises(ValueError, match="Cannot combine kv_cache_transfer lut_path"):
        link_profile.resolve_kv_cache_transfer_profile(
            AnalyticalKVCacheTransferConfig(lut_path=str(QWEN3_KV_LUT), network_bandwidth_gbps=3172.5),
            c16, "pd-af-disaggregation",
        )
    with pytest.raises(ValueError, match="requires lut_path"):
        AnalyticalKVCacheTransferConfig(link_type="RDMA_IB_400G_Direct")
    with pytest.raises(ValueError, match="Conflicting KV transfer settings"):
        link_profile.resolve_kv_cache_transfer_profile(
            AnalyticalKVCacheTransferConfig(lut_path=str(QWEN3_KV_LUT), pd_placement="cross_node", link_type="NVLink_IntraNode"),
            c8, "pd-af-disaggregation",
        )
    with pytest.raises(ValueError, match="cannot use transfer_protocol='nvlink'"):
        link_profile.resolve_kv_cache_transfer_profile(
            AnalyticalKVCacheTransferConfig(pd_placement="cross_node", transfer_protocol="nvlink", network_bandwidth_gbps=3172.5),
            c16, "pd-af-disaggregation",
        )

    with patch.object(link_profile.logger, "warning") as warn_spy:
        sens = link_profile.resolve_kv_cache_transfer_profile(
            AnalyticalKVCacheTransferConfig(
                lut_path=str(QWEN3_KV_LUT), pd_placement="same_node", allow_infeasible_placement_override=True
            ),
            c16, "pd-af-disaggregation",
        )
        legacy = link_profile.resolve_kv_cache_transfer_profile(
            AnalyticalKVCacheTransferConfig(transfer_protocol="nvlink", network_bandwidth_gbps=3172.5, network_latency_ms=0.0208),
            c16, "pd-af-disaggregation",
        )
    assert sens.link_type == "NVLink_IntraNode" and sens.placement.placement_feasible is False
    assert legacy.predict_time_ms(KV_512_BYTES) == pytest.approx(0.0208 + KV_512_BYTES / (3172.5e9 / 8000.0))
    msgs = [str(c.args[0]) for c in warn_spy.call_args_list]
    assert any("KV_PLACEMENT_DIAGNOSTIC_OVERRIDE" in m for m in msgs)
    assert any("KV_TRANSFER_PLACEMENT_WARNING" in m for m in msgs)

    pred = KVCacheTransferPredictorRegistry.get(
        KVCacheTransferType.ANALYTICAL,
        AnalyticalKVCacheTransferConfig(
            lut_path=str(QWEN3_KV_LUT), link_type="RDMA_IB_400G_Direct",
            enable_compression=True, compression_ratio=2.0, extra_latency_ms=0.02,
        ),
    )
    assert isinstance(pred, AnalyticalKVCacheTransferPredictor)
    assert pred.get_transfer_time(ClusterType.PREFILL, ClusterType.DECODE_ATTN, None, KV_512_BYTES) == pytest.approx(0.75497472 + 0.02)

    with pytest.raises(ValueError, match="require a disaggregated sys_arch"):
        SimulationConfig(
            simulation_mode="offline", sys_arch="co-location",
            cluster_config=ClusterConfig(
                num_replicas=1, replica_config=ReplicaConfig(model_name="llama2_7b_dense_example"),
                replica_scheduler_config=OrcaSchedulerConfig(),
                execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True),
            ),
            kv_cache_transfer_config=AnalyticalKVCacheTransferConfig(lut_path=str(QWEN3_KV_LUT), pd_placement="auto"),
            metrics_config=MetricsConfig(output_dir=str(tmp_path / "out"), run_id="coloc", write_metrics=False),
        )


def test_cli_parses_kv_placement_flags(tmp_path: Path) -> None:
    global_vars.reset_global_vars()
    argv = [
        "frontier.main", "--simulation_mode", "offline", "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters", "--log_level", "error",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", "1",
        "--cluster_config_decode_ffn_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "8",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "8",
        "--cluster_config_prefill_replica_config_device", "h200",
        "--cluster_config_prefill_replica_config_network_device", "h200_dgx",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", "8",
        "--cluster_config_decode_attn_replica_config_device", "h200",
        "--cluster_config_decode_attn_replica_config_network_device", "h200_dgx",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "16",
        "--cluster_config_decode_ffn_replica_config_device", "hpp2",
        "--cluster_config_decode_ffn_replica_config_network_device", "hpp2_16rank",
        "--cluster_config_decode_ffn_replica_config_streamed_ffn_layers", "2",
        "--replica_config_model_name", "Qwen3-4B",
        "--kv_cache_transfer_config_type", "analytical",
        "--analytical_kv_cache_transfer_config_lut_path", str(QWEN3_KV_LUT),
        "--analytical_kv_cache_transfer_config_pd_placement", "auto",
        "--analytical_kv_cache_transfer_config_extra_latency_ms", "0.015",
        "--cc_backend_config_type", "analytical",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--metrics_config_output_dir", str(tmp_path / "cli"),
        "--metrics_config_run_id", "cli_kv",
        "--no-metrics_config_write_metrics",
    ]
    try:
        with patch("sys.argv", argv):
            cfg = SimulationConfig.create_from_cli_args()
        prof = cfg.resolved_kv_cache_transfer_profile
        assert prof.link_type == "RDMA_IB_400G_Direct" and prof.extra_latency_ms == pytest.approx(0.015)
    finally:
        global_vars.reset_global_vars()


@pytest.mark.parametrize(
    ("run_id", "topo", "placement", "override", "exp_link", "exp_feasible", "exp_ms"),
    [
        ("e2e_8gpu_nvlink", (2, 2, 2, 2), "auto", False, "NVLink_IntraNode", True, 0.213349),
        ("e2e_16gpu_rdma", (1, 8, 1, 8), "auto", False, "RDMA_IB_400G_Direct", True, 1.50994944),
        ("e2e_16gpu_sens", (1, 8, 1, 8), "same_node", True, "NVLink_IntraNode", False, 0.213349),
    ],
)
def test_simulator_records_kv_link_type_in_system_metrics(
    tmp_path: Path, run_id: str, topo: tuple[int, int, int, int], placement: str,
    override: bool, exp_link: str, exp_feasible: bool, exp_ms: float,
) -> None:
    global_vars.reset_global_vars()
    Replica._id = Batch._id = Request._id = -1
    try:
        cfg = SimulationConfig(
            simulation_mode="offline", sys_arch="pd-af-disaggregation",
            enable_parallel_clusters=False, log_level="error",
            cluster_config=_pdaf_cluster(*topo),
            request_generator_config=SyntheticRequestGeneratorConfig(
                num_requests=1,
                length_generator_config=FixedRequestLengthGeneratorConfig(prefill_tokens=512, decode_tokens=4),
                interval_generator_config=StaticRequestIntervalGeneratorConfig(),
            ),
            kv_cache_transfer_config=AnalyticalKVCacheTransferConfig(
                lut_path=str(QWEN3_KV_LUT), pd_placement=placement,
                allow_infeasible_placement_override=override,
            ),
            metrics_config=MetricsConfig(
                output_dir=str(tmp_path / "m"), run_id=run_id, write_metrics=True,
                store_request_metrics=True, store_batch_metrics=False,
                store_token_completion_metrics=False, store_utilization_metrics=False,
                store_plots=False, enable_chrome_trace=False, write_json_trace=False,
            ),
        )
        Simulator(cfg).run()
        payload = json.loads((Path(cfg.metrics_config.output_dir) / "system_metrics.json").read_text())
        kv = payload["kv_cache_transfer_statistics"]
        assert kv["total_transfers"] == 1
        assert kv["total_transfer_time_ms"] == pytest.approx(exp_ms)
        assert kv["link_type"] == exp_link
        assert kv["placement_feasible"] is exp_feasible
        assert payload["system_architecture_info"]["kv_cache_transfer_link"]["link_type"] == exp_link
    finally:
        global_vars.reset_global_vars()
        Replica._id = Batch._id = Request._id = -1
