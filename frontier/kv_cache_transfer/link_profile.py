"""PD node-placement resolution and LUT-backed KV cache transfer profiles."""

from __future__ import annotations

from bisect import bisect_left
import csv
from dataclasses import dataclass
import math
import os
from typing import Any, Optional

from frontier.config.kv_cache_transfer_config import (
    DEFAULT_INTER_NODE_LINK_TYPE,
    DEFAULT_INTRA_NODE_LINK_TYPE,
    DEFAULT_NETWORK_BANDWIDTH_GBPS,
    DEFAULT_NETWORK_LATENCY_MS,
    LUT_PRICING_MODES,
    PD_PLACEMENTS,
)
from frontier.logger import init_logger
from frontier.types import ClusterType, NodeSKUType

logger = init_logger(__name__)

GBPS_TO_BYTES_PER_MS = 1e9 / (8.0 * 1000.0)
_REQUIRED_LUT_COLS = ("link_type", "kv_cache_bytes", "time_stats.kv_cache_transfer.median")
_NVLINK_NODE_SKUS = frozenset(
    {
        NodeSKUType.A40_PAIRWISE_NVLINK,
        NodeSKUType.A100_PAIRWISE_NVLINK,
        NodeSKUType.H100_PAIRWISE_NVLINK,
        NodeSKUType.A100_DGX,
        NodeSKUType.H100_DGX,
        NodeSKUType.H200_DGX,
        NodeSKUType.A800_DGX,
        NodeSKUType.H800_DGX,
        NodeSKUType.H20_DGX,
    }
)
_NVLINK_NODE_SKU_NAMES = frozenset(sku.name.lower() for sku in _NVLINK_NODE_SKUS)


def _sku_key(cfg: Any, nested_attr: str, raw_attr: str) -> Any:
    nested = getattr(cfg, nested_attr, None)
    if hasattr(nested, "get_type"):
        return nested.get_type()
    return str(getattr(cfg, raw_attr, "")).strip().lower()


def _supports_nvlink_intra_node(node_sku_key: Any) -> bool:
    if node_sku_key in _NVLINK_NODE_SKUS:
        return True
    return isinstance(node_sku_key, str) and node_sku_key.strip().lower() in _NVLINK_NODE_SKU_NAMES


def infer_protocol_from_link_type(link_type: str, fallback: str = "rdma") -> str:
    low = link_type.strip().lower()
    if "nvlink" in low or "nvswitch" in low:
        return "nvlink"
    if "host_bounce" in low or "hostbounce" in low:
        return "host_bounce"
    if "rdma" in low or "infiniband" in low or "ib_" in low or "_ib" in low:
        return "rdma"
    if "pcie" in low:
        return "pcie"
    if "tcp" in low or "eth" in low:
        return "tcp"
    return fallback.strip() or "rdma"


def _link_locality(link_type: str) -> Optional[str]:
    low = link_type.strip().lower()
    is_intra = link_type == DEFAULT_INTRA_NODE_LINK_TYPE or any(
        k in low for k in ("intranode", "intra_node", "nvlink", "nvswitch", "xgmi")
    )
    is_inter = link_type == DEFAULT_INTER_NODE_LINK_TYPE or any(
        k in low for k in ("internode", "inter_node", "rdma", "infiniband", "ib_", "_ib", "host_bounce", "hostbounce", "tcp")
    )
    if is_intra and not is_inter:
        return "same_node"
    if is_inter and not is_intra:
        return "cross_node"
    return None


@dataclass(frozen=True)
class PDTopologyPlacement:
    """Evaluated physical placement of PREFILL and decode-side clusters."""

    requested_placement: str
    resolved_placement: str
    same_node_feasible: bool
    placement_feasible: bool
    diagnostic_override: bool
    prefill_devices: int
    decode_devices: int
    devices_per_node: Optional[int]
    infeasibility_reason: Optional[str] = None
    shared_node_sku: Any = None

    @property
    def total_pd_devices(self) -> int:
        return self.prefill_devices + self.decode_devices

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "requested_placement": self.requested_placement,
            "resolved_placement": self.resolved_placement,
            "same_node_feasible": self.same_node_feasible,
            "placement_feasible": self.placement_feasible,
            "diagnostic_override": self.diagnostic_override,
            "prefill_cluster_devices": self.prefill_devices,
            "decode_cluster_devices": self.decode_devices,
            "total_pd_devices": self.total_pd_devices,
            "devices_per_node": self.devices_per_node,
            "infeasibility_reason": self.infeasibility_reason,
        }


@dataclass(frozen=True)
class KVCacheTransferLUTCurve:
    """Validated KV transfer lookup table for one link_type."""

    lut_path: str
    link_type: str
    points_bytes: tuple[int, ...]
    points_time_ms: tuple[float, ...]
    fitted_bandwidth_gbps: float
    fitted_alpha_latency_ms: float

    def predict_time_ms(self, effective_bytes: float, pricing_mode: str = "piecewise_linear") -> float:
        if not math.isfinite(effective_bytes) or effective_bytes < 0:
            raise ValueError(f"effective_bytes must be finite and >= 0, got {effective_bytes!r}")
        slope = 1.0 / (self.fitted_bandwidth_gbps * GBPS_TO_BYTES_PER_MS)
        if pricing_mode == "linear_fit":
            return self.fitted_alpha_latency_ms + effective_bytes * slope
        if pricing_mode != "piecewise_linear":
            raise ValueError(f"lut_pricing_mode must be one of {list(LUT_PRICING_MODES)}, got {pricing_mode!r}")
        xs, ys = self.points_bytes, self.points_time_ms
        if effective_bytes < xs[0]:
            y0 = min(self.fitted_alpha_latency_ms, ys[0])
            return y0 + (effective_bytes / float(xs[0])) * (ys[0] - y0)
        if effective_bytes > xs[-1]:
            return ys[-1] + (effective_bytes - float(xs[-1])) * slope
        idx = bisect_left(xs, effective_bytes)
        if idx < len(xs) and xs[idx] == effective_bytes:
            return ys[idx]
        x_lo, x_hi = float(xs[idx - 1]), float(xs[idx])
        return ys[idx - 1] + ((effective_bytes - x_lo) / (x_hi - x_lo)) * (ys[idx] - ys[idx - 1])


@dataclass(frozen=True)
class ResolvedKVCacheTransferProfile:
    """Resolved KV transfer pricing and placement contract."""

    pricing_source: str
    link_type: str
    transfer_protocol: str
    effective_bandwidth_gbps: float
    base_latency_ms: float
    extra_latency_ms: float = 0.0
    placement: Optional[PDTopologyPlacement] = None
    lut_path: Optional[str] = None
    lut_pricing_mode: Optional[str] = None
    lut_curve: Optional[KVCacheTransferLUTCurve] = None

    def predict_time_ms(self, effective_bytes: float) -> float:
        if self.lut_curve is not None:
            return self.lut_curve.predict_time_ms(
                effective_bytes, self.lut_pricing_mode or "piecewise_linear"
            ) + self.extra_latency_ms
        bw_bytes_per_ms = self.effective_bandwidth_gbps * GBPS_TO_BYTES_PER_MS
        return self.base_latency_ms + self.extra_latency_ms + (effective_bytes / bw_bytes_per_ms)

    def format_log_summary(self) -> str:
        p = self.placement
        prefix = (
            f"placement={p.resolved_placement} (req={p.requested_placement}, "
            f"P={p.prefill_devices}+D={p.decode_devices}={p.total_pd_devices}, "
            f"node_cap={p.devices_per_node}, feasible={p.placement_feasible}), "
            if p is not None
            else ""
        )
        return (
            f"[KV_TRANSFER_LINK] {prefix}link_type={self.link_type}, "
            f"protocol={self.transfer_protocol}, bw={self.effective_bandwidth_gbps:.2f} Gbps, "
            f"base_latency={self.base_latency_ms:.4f} ms, extra_latency={self.extra_latency_ms:.4f} ms, "
            f"source={self.pricing_source}"
        )

    def to_metrics_dict(self) -> dict[str, Any]:
        p = self.placement
        return {
            "link_type": self.link_type,
            "transfer_protocol": self.transfer_protocol,
            "pricing_source": self.pricing_source,
            "effective_bandwidth_gbps": float(self.effective_bandwidth_gbps),
            "base_latency_ms": float(self.base_latency_ms),
            "extra_latency_ms": float(self.extra_latency_ms),
            "lut_path": self.lut_path,
            "lut_pricing_mode": self.lut_pricing_mode,
            "pd_placement": p.resolved_placement if p else None,
            "placement_feasible": p.placement_feasible if p else None,
            "diagnostic_override": p.diagnostic_override if p else False,
            "topology_placement": p.to_metrics_dict() if p else None,
        }


def _fit_ols(xs: tuple[int, ...], ys: tuple[float, ...], path: str, link_type: str) -> tuple[float, float]:
    if len(xs) == 1:
        if ys[0] <= 0:
            raise ValueError(f"Non-positive LUT time for {link_type!r} in {path}: {ys[0]}")
        return (float(xs[0]) * 8.0) / (ys[0] * 1e6), 0.0
    n = float(len(xs))
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom > 0 else 0.0
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError(f"LUT {link_type!r} in {path} must have positive time-vs-bytes slope")
    return (8.0 / slope) / 1e6, max(0.0, my - slope * mx)


def load_kv_cache_transfer_lut(
    lut_path: str, link_type: str, *, expected_direction: Optional[str] = None
) -> KVCacheTransferLUTCurve:
    """Load and validate one link_type curve from kv_cache_transfer_lut.csv."""
    if not lut_path or not os.path.isfile(lut_path):
        raise ValueError(f"KV cache transfer LUT file not found: {lut_path!r}")
    target = link_type.strip()
    if not target:
        raise ValueError("link_type must be non-empty")

    with open(lut_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = [c for c in _REQUIRED_LUT_COLS if c not in fields]
        if missing:
            raise ValueError(f"KV transfer LUT {lut_path!r} missing columns: {missing}")
        available: set[str] = set()
        rows: list[tuple[int, dict[str, Any]]] = []
        for ln, row in enumerate(reader, start=2):
            lt = (row.get("link_type") or "").strip()
            if lt:
                available.add(lt)
            if lt == target:
                rows.append((ln, row))

    if not rows:
        raise ValueError(f"link_type={target!r} not found in {lut_path!r}; available: {sorted(available)}")

    dirs = {(r.get("direction") or "").strip() for _, r in rows if (r.get("direction") or "").strip()}
    if dirs:
        if expected_direction is not None:
            if expected_direction not in dirs:
                raise ValueError(
                    f"Direction mismatch for {target!r} in {lut_path!r}: "
                    f"expected {expected_direction!r}, found {sorted(dirs)}"
                )
            rows = [(ln, r) for ln, r in rows if (r.get("direction") or "").strip() == expected_direction]
        elif len(dirs) > 1:
            raise ValueError(f"Ambiguous directions {sorted(dirs)} for {target!r} in {lut_path!r}")

    by_bytes: dict[int, float] = {}
    for ln, row in rows:
        b_val = float((row.get("kv_cache_bytes") or "").strip())
        t_val = float((row.get("time_stats.kv_cache_transfer.median") or "").strip())
        if not (math.isfinite(b_val) and b_val > 0 and int(b_val) == b_val):
            raise ValueError(f"Invalid kv_cache_bytes on line {ln} of {lut_path!r}")
        if not (math.isfinite(t_val) and t_val >= 0):
            raise ValueError(f"Invalid median time on line {ln} of {lut_path!r}")
        b_int = int(b_val)
        if b_int in by_bytes and not math.isclose(by_bytes[b_int], t_val, abs_tol=1e-9):
            raise ValueError(f"Conflicting median times for {b_int}B ({target!r}) in {lut_path!r}")
        by_bytes[b_int] = t_val

    items = sorted(by_bytes.items())
    xs, ys = tuple(k for k, _ in items), tuple(v for _, v in items)
    for i in range(1, len(ys)):
        if ys[i] + 1e-9 < ys[i - 1]:
            raise ValueError(f"Non-monotonic LUT for {target!r} in {lut_path!r}: {xs[i-1]}->{xs[i]}B")
    bw, alpha = _fit_ols(xs, ys, lut_path, target)
    return KVCacheTransferLUTCurve(lut_path, target, xs, ys, bw, alpha)


def evaluate_pd_node_placement(
    cluster_config: Any,
    sys_arch: str,
    requested_placement: str,
    *,
    allow_infeasible_override: bool = False,
) -> PDTopologyPlacement:
    """Evaluate physical P/D node placement feasibility and resolve placement mode."""
    if requested_placement not in (*PD_PLACEMENTS, "legacy_explicit"):
        raise ValueError(f"pd_placement must be one of {list(PD_PLACEMENTS)}, got {requested_placement!r}")

    p_rep = getattr(cluster_config, "prefill_cluster_num_replicas", None)
    p_cfg = getattr(cluster_config, "prefill_replica_config", None)
    if sys_arch == "pd-af-disaggregation":
        d_name = ClusterType.DECODE_ATTN.name
        d_rep = getattr(cluster_config, "decode_attn_cluster_num_replicas", None)
        d_cfg = getattr(cluster_config, "decode_attn_replica_config", None)
    elif sys_arch == "pd-disaggregation":
        d_name = ClusterType.DECODE.name
        d_rep = getattr(cluster_config, "decode_cluster_num_replicas", None)
        d_cfg = getattr(cluster_config, "decode_replica_config", None)
    else:
        raise ValueError(f"PD placement requires disaggregated sys_arch, got {sys_arch!r}")
    if not p_rep or p_cfg is None or not d_rep or d_cfg is None:
        raise ValueError(f"Missing PREFILL or {d_name} cluster configuration for PD placement")

    p_devs = int(p_rep) * int(p_cfg.world_size)
    d_devs = int(d_rep) * int(d_cfg.world_size)
    p_cap = int(p_cfg.node_config.num_devices_per_node)
    d_cap = int(d_cfg.node_config.num_devices_per_node)
    node_cap = p_cap if p_cap == d_cap else None

    p_dev_key = _sku_key(p_cfg, "device_config", "device")
    d_dev_key = _sku_key(d_cfg, "device_config", "device")
    p_net_key = _sku_key(p_cfg, "node_config", "network_device")
    d_net_key = _sku_key(d_cfg, "node_config", "network_device")
    shared_node_sku = p_net_key if p_net_key == d_net_key else None

    reason: Optional[str] = None
    if p_dev_key != d_dev_key or p_net_key != d_net_key:
        reason = f"heterogeneous P/D SKU ({p_cfg.device}/{p_cfg.network_device} vs {d_cfg.device}/{d_cfg.network_device})"
    elif node_cap is None:
        reason = f"mismatched P/D node capacities ({p_cap} vs {d_cap})"
    elif p_devs + d_devs > node_cap:
        reason = f"PREFILL ({p_devs}) + {d_name} ({d_devs}) = {p_devs + d_devs} devices exceeds single-node capacity ({node_cap}/node)"
    same_feasible = reason is None

    feasible, override = True, False
    if requested_placement in ("auto", "legacy_explicit"):
        resolved = "same_node" if same_feasible else "cross_node"
    elif requested_placement == "cross_node":
        resolved = "cross_node"
    elif same_feasible:
        resolved = "same_node"
    elif allow_infeasible_override:
        resolved, feasible, override = "same_node", False, True
    else:
        raise ValueError(
            f"Infeasible pd_placement='same_node': {reason}. Use 'auto'/'cross_node' "
            "or pass --analytical_kv_cache_transfer_config_allow_infeasible_placement_override for sensitivity."
        )

    return PDTopologyPlacement(
        requested_placement, resolved, same_feasible, feasible, override, p_devs, d_devs, node_cap, reason, shared_node_sku
    )


def _validate_lut_manual_conflicts(kv_config: Any) -> None:
    if not (
        math.isclose(float(kv_config.network_bandwidth_gbps), DEFAULT_NETWORK_BANDWIDTH_GBPS, abs_tol=1e-12)
        and math.isclose(float(kv_config.network_latency_ms), DEFAULT_NETWORK_LATENCY_MS, abs_tol=1e-12)
    ):
        raise ValueError(
            "Cannot combine kv_cache_transfer lut_path with non-default network_bandwidth_gbps/network_latency_ms; "
            "use extra_latency_ms for additive latency sensitivity."
        )


def _resolve_requested_placement_with_link(req_pl: Optional[str], exp_link: Optional[str]) -> str:
    eff_req = req_pl or "auto"
    if exp_link is not None:
        loc = _link_locality(exp_link)
        if req_pl in ("same_node", "cross_node") and loc is not None and loc != req_pl:
            raise ValueError(f"Conflicting KV transfer settings: pd_placement={req_pl!r} with link_type={exp_link!r}")
        if req_pl in (None, "auto") and loc is not None:
            eff_req = loc
    return eff_req


def _validate_explicit_protocol_conflict(resolved_placement: Optional[str], proto: str) -> None:
    if resolved_placement == "cross_node" and proto.lower() == "nvlink":
        raise ValueError(
            "Conflicting KV transfer settings: resolved pd_placement='cross_node' cannot use transfer_protocol='nvlink'"
        )


def build_standalone_kv_transfer_profile(kv_config: Any) -> ResolvedKVCacheTransferProfile:
    """Build a profile when AnalyticalKVCacheTransferPredictor is instantiated without SimulationConfig."""
    lut_path = (getattr(kv_config, "lut_path", None) or "").strip() or None
    req_pl = getattr(kv_config, "pd_placement", None)
    exp_link = (getattr(kv_config, "link_type", None) or "").strip() or None
    extra = float(getattr(kv_config, "extra_latency_ms", 0.0) or 0.0)
    if lut_path is not None:
        _validate_lut_manual_conflicts(kv_config)
        _resolve_requested_placement_with_link(req_pl, exp_link)
        link = exp_link
        if link is None:
            if req_pl == "same_node":
                link = DEFAULT_INTRA_NODE_LINK_TYPE
            elif req_pl == "cross_node":
                link = DEFAULT_INTER_NODE_LINK_TYPE
            else:
                raise ValueError("Standalone predictor with lut_path requires link_type or concrete pd_placement")
        curve = load_kv_cache_transfer_lut(lut_path, link)
        return ResolvedKVCacheTransferProfile(
            "lut", link, infer_protocol_from_link_type(link, kv_config.transfer_protocol),
            curve.fitted_bandwidth_gbps, curve.fitted_alpha_latency_ms, extra,
            lut_path=lut_path, lut_pricing_mode=kv_config.lut_pricing_mode, lut_curve=curve,
        )
    proto = str(getattr(kv_config, "transfer_protocol", "rdma")).strip() or "rdma"
    _validate_explicit_protocol_conflict(req_pl, proto)
    return ResolvedKVCacheTransferProfile(
        "analytical_explicit", f"explicit_{proto.lower()}", proto,
        float(kv_config.network_bandwidth_gbps), float(kv_config.network_latency_ms), extra,
    )


def resolve_kv_cache_transfer_profile(
    kv_config: Any, cluster_config: Any, sys_arch: str
) -> ResolvedKVCacheTransferProfile:
    """Resolve the authoritative KV cache transfer profile for a disaggregated run."""
    lut_path = (getattr(kv_config, "lut_path", None) or "").strip() or None
    req_pl = getattr(kv_config, "pd_placement", None)
    exp_link = (getattr(kv_config, "link_type", None) or "").strip() or None
    extra = float(getattr(kv_config, "extra_latency_ms", 0.0) or 0.0)
    allow_ov = bool(getattr(kv_config, "allow_infeasible_placement_override", False))

    if lut_path is None and req_pl is None:
        placement = evaluate_pd_node_placement(cluster_config, sys_arch, "legacy_explicit")
        proto = str(getattr(kv_config, "transfer_protocol", "rdma")).strip() or "rdma"
        if proto.lower() == "nvlink" and not placement.same_node_feasible:
            logger.warning(
                "[KV_TRANSFER_PLACEMENT_WARNING] transfer_protocol='nvlink' on infeasible single-node topology (%s).",
                placement.infeasibility_reason,
            )
        return ResolvedKVCacheTransferProfile(
            "analytical_explicit", f"explicit_{proto.lower()}", proto,
            float(kv_config.network_bandwidth_gbps), float(kv_config.network_latency_ms), extra, placement,
        )

    if lut_path is not None:
        _validate_lut_manual_conflicts(kv_config)
        eff_req = _resolve_requested_placement_with_link(req_pl, exp_link)
        placement = evaluate_pd_node_placement(
            cluster_config, sys_arch, eff_req, allow_infeasible_override=allow_ov
        )
        if exp_link is not None:
            link = exp_link
        elif placement.resolved_placement == "same_node":
            if not _supports_nvlink_intra_node(placement.shared_node_sku):
                raise ValueError(
                    f"No default intra-node KV link_type for non-NVLink node SKU "
                    f"{placement.shared_node_sku!r}; pass explicit lut_path and link_type."
                )
            link = DEFAULT_INTRA_NODE_LINK_TYPE
        else:
            link = DEFAULT_INTER_NODE_LINK_TYPE
        expected_dir = "prefill_to_decode_attn" if sys_arch == "pd-af-disaggregation" else "prefill_to_decode"
        curve = load_kv_cache_transfer_lut(lut_path, link, expected_direction=expected_dir)
        if placement.diagnostic_override:
            logger.warning(
                "[KV_PLACEMENT_DIAGNOSTIC_OVERRIDE] Running infeasible same_node (%s) with link_type=%s.",
                placement.infeasibility_reason, link,
            )
        return ResolvedKVCacheTransferProfile(
            "lut", link, infer_protocol_from_link_type(link, str(kv_config.transfer_protocol)),
            curve.fitted_bandwidth_gbps, curve.fitted_alpha_latency_ms, extra, placement,
            lut_path, kv_config.lut_pricing_mode, curve,
        )

    placement = evaluate_pd_node_placement(
        cluster_config, sys_arch, req_pl, allow_infeasible_override=allow_ov
    )
    proto = str(getattr(kv_config, "transfer_protocol", "rdma")).strip() or "rdma"
    _validate_explicit_protocol_conflict(placement.resolved_placement, proto)
    if placement.diagnostic_override:
        logger.warning(
            "[KV_PLACEMENT_DIAGNOSTIC_OVERRIDE] Running infeasible same_node (%s) in explicit analytical mode.",
            placement.infeasibility_reason,
        )
    return ResolvedKVCacheTransferProfile(
        "analytical_explicit", f"explicit_{proto.lower()}", proto,
        float(kv_config.network_bandwidth_gbps), float(kv_config.network_latency_ms), extra, placement,
    )
