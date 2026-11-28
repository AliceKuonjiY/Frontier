from dataclasses import dataclass, field
from typing import Optional

from frontier.config.base_poly_config import BasePolyConfig
from frontier.types import KVCacheTransferType

# vLLM KV connectors of a PDD prefill instance whose device work Frontier prices.
KV_CONNECTORS = ("none", "p2p_nccl")

PD_PLACEMENTS = ("auto", "same_node", "cross_node")
LUT_PRICING_MODES = ("piecewise_linear", "linear_fit")
DEFAULT_INTRA_NODE_LINK_TYPE = "NVLink_IntraNode"
DEFAULT_INTER_NODE_LINK_TYPE = "RDMA_IB_400G_Direct"
DEFAULT_NETWORK_BANDWIDTH_GBPS = 100.0
DEFAULT_NETWORK_LATENCY_MS = 0.1


def _require_positive(value: float | int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {value}")


def _require_non_negative(value: float | int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")


def _require_optional_positive(value: Optional[int], field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be > 0 when set, got {value}")


@dataclass
class BaseKVCacheTransferConfig(BasePolyConfig):
    """Base configuration for KV Cache transfer predictors."""
    
    # Network configuration
    network_bandwidth_gbps: float = field(
        default=100.0,
        metadata={"help": "Network bandwidth in Gbps for KV cache transfer."},
    )
    network_latency_ms: float = field(
        default=0.1,
        metadata={"help": "Network latency in milliseconds for KV cache transfer."},
    )
    
    # Transfer optimization configuration
    enable_compression: bool = field(
        default=False,
        metadata={"help": "Whether to enable compression for KV cache transfer."},
    )
    compression_ratio: float = field(
        default=1.0,
        metadata={"help": "Compression ratio for KV cache transfer (1.0 = no compression)."},
    )
    
    # Future extension configuration
    enable_latency_hiding: bool = field(
        default=False,
        metadata={"help": "Whether to enable latency hiding with layer-by-layer transfer."},
    )
    enable_request_transfer: bool = field(
        default=False,
        metadata={"help": "Whether to transfer request information along with KV cache."},
    )
    transfer_protocol: str = field(
        default="rdma",
        metadata={"help": "Transfer protocol: rdma, infiniband, nvlink, host_bounce, tcp."},
    )
    pd_placement: Optional[str] = field(
        default=None,
        metadata={"help": "PD node placement: None (legacy explicit), 'auto', 'same_node', or 'cross_node'."},
    )
    lut_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to kv_cache_transfer_lut.csv for placement-driven LUT pricing."},
    )
    link_type: Optional[str] = field(
        default=None,
        metadata={"help": "Optional explicit link_type row in lut_path (requires lut_path)."},
    )
    lut_pricing_mode: str = field(
        default="piecewise_linear",
        metadata={"help": "LUT pricing mode: 'piecewise_linear' or 'linear_fit'."},
    )
    extra_latency_ms: float = field(
        default=0.0,
        metadata={"help": "Additive fixed latency in ms for sensitivity runs."},
    )
    allow_infeasible_placement_override: bool = field(
        default=False,
        metadata={"help": "Diagnostic opt-in: allow infeasible pd_placement='same_node' with WARNING."},
    )
    kv_connector: str = field(
        default="none",
        metadata={
            "help": "vLLM KV connector of the PDD prefill instance: none, or p2p_nccl "
            "(P2pNcclConnector), whose per-layer KV extract each prefill step pays."
        },
    )
    kv_send_slowdown_per_layer_ms: float = field(
        default=0.0,
        metadata={
            "help": "Measured correction, not a model of the send: device time (ms) added to each layer "
            "of a PDD prefill step that completes a prompt, for the slowdown of the layer's later "
            "operators while the kv_connector sends its KV. Requires a kv_connector; 0 disables it."
        },
    )

    def __post_init__(self) -> None:
        _require_positive(self.network_bandwidth_gbps, "network_bandwidth_gbps")
        _require_non_negative(self.network_latency_ms, "network_latency_ms")
        _require_positive(self.compression_ratio, "compression_ratio")
        _require_non_negative(self.extra_latency_ms, "extra_latency_ms")
        _require_non_negative(self.kv_send_slowdown_per_layer_ms, "kv_send_slowdown_per_layer_ms")
        self.pd_placement = (self.pd_placement or "").strip() or None
        self.lut_path = (self.lut_path or "").strip() or None
        self.link_type = (self.link_type or "").strip() or None
        if self.pd_placement is not None and self.pd_placement not in PD_PLACEMENTS:
            raise ValueError(f"pd_placement must be one of {list(PD_PLACEMENTS)}, got {self.pd_placement!r}")
        if self.link_type is not None and self.lut_path is None:
            raise ValueError(f"link_type={self.link_type!r} requires lut_path to be set.")
        if self.lut_pricing_mode not in LUT_PRICING_MODES:
            raise ValueError(f"lut_pricing_mode must be one of {list(LUT_PRICING_MODES)}, got {self.lut_pricing_mode!r}")
        if self.kv_connector not in KV_CONNECTORS:
            raise ValueError(f"kv_connector must be one of {list(KV_CONNECTORS)}, got {self.kv_connector!r}")
        if self.kv_send_slowdown_per_layer_ms > 0 and self.kv_connector == "none":
            raise ValueError("kv_send_slowdown_per_layer_ms corrects the KV send of a kv_connector; kv_connector is 'none'.")


@dataclass
class AnalyticalKVCacheTransferConfig(BaseKVCacheTransferConfig):
    """Configuration for analytical KV cache transfer predictor.

    Deprecation timeline:
        - kv_cache_dtype_size_bytes is deprecated as of 2026-01.
        - Planned removal no earlier than 2026-07-01.
        - Prefer op-quantization config entries for kv_cache_transfer precision.
    """
    
    # Data type configuration for KV cache size calculation
    kv_cache_dtype_size_bytes: int = field(
        default=2,
        metadata={
            "help": "Size of KV cache data type in bytes (e.g., 2 for fp16, 4 for fp32). Deprecated as of 2026-01; planned removal no earlier than 2026-07-01. Use op quantization config."
        },
    )
    
    # Model-specific overrides (optional)
    override_num_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Override number of layers for KV cache size calculation."},
    )
    override_num_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Override number of KV heads (num_kv_heads) for KV cache size calculation. This should be the total number of key-value heads, not query heads."},
    )
    override_head_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Override head dimension for KV cache size calculation."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_positive(self.kv_cache_dtype_size_bytes, "kv_cache_dtype_size_bytes")
        _require_optional_positive(self.override_num_layers, "override_num_layers")
        _require_optional_positive(self.override_num_heads, "override_num_heads")
        _require_optional_positive(self.override_head_dim, "override_head_dim")

    @classmethod
    def get_type(cls) -> KVCacheTransferType:
        return KVCacheTransferType.ANALYTICAL
    
    @classmethod
    def get_name(cls) -> str:
        return "analytical"
