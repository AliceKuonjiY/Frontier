"""Device SRAM/memory budget accounting and profiling CSV contract loading.

Units follow Frontier's binary memory convention: nominal capacity in GiB
(1024^3 bytes), usable capacity and planning overhead in MiB (1024^2 bytes).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from typing import Any, Optional

from frontier.config.device_sku_config import BINARY_MIB_BYTES
from frontier.execution_time_predictor.measurement_input_paths import (
    resolve_measurement_input_paths,
)
from frontier.types import ClusterType, MeasurementType

_FLOAT_COLS = ("rank_fp8_weight_mb", "sram_headroom_mb")
_BOOL_COLS = ("sram_fit_170mb", "sram_fit_with_10mb_overhead", "weight_streaming_required")
SRAM_PROFILE_COLUMNS = _FLOAT_COLS + _BOOL_COLS


def _pct(numerator: int, denominator: int) -> float:
    return (float(numerator) / float(denominator)) * 100.0 if denominator > 0 else 0.0


def _mib(num_bytes: int) -> float:
    return float(num_bytes) / float(BINARY_MIB_BYTES)


def _parse_bool(raw: Optional[str], col: str, path: str) -> Optional[bool]:
    text = (raw or "").strip().lower()
    if not text:
        return None
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean {raw!r} in {col!r} of {path}")


def _parse_float(raw: Optional[str], col: str, path: str) -> Optional[float]:
    text = (raw or "").strip()
    if not text:
        return None
    val = float(text)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite number {raw!r} in {col!r} of {path}")
    return val


@dataclass(frozen=True)
class SRAMProfileContract:
    """Optional SRAM metadata parsed from linear_op profiling CSV."""

    rank_fp8_weight_mb: Optional[float] = None
    sram_headroom_mb: Optional[float] = None
    sram_fit_170mb: Optional[bool] = None
    sram_fit_with_10mb_overhead: Optional[bool] = None
    weight_streaming_required: Optional[bool] = None

    @property
    def rejects_all_resident(self) -> bool:
        return (
            self.sram_fit_with_10mb_overhead is False
            or self.weight_streaming_required is True
        )

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "rank_fp8_weight_mb": self.rank_fp8_weight_mb,
            "sram_headroom_mb": self.sram_headroom_mb,
            "sram_fit_170mb": self.sram_fit_170mb,
            "sram_fit_with_10mb_overhead": self.sram_fit_with_10mb_overhead,
            "weight_streaming_required": self.weight_streaming_required,
        }


def load_sram_profile_contract(
    file_path: str, tensor_parallel_size: int
) -> Optional[SRAMProfileContract]:
    """Load optional SRAM columns for one TP slice; fail fast on conflicting rows."""
    if not file_path or not os.path.isfile(file_path):
        return None

    target_tp = int(tensor_parallel_size)
    with open(file_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "num_tensor_parallel_workers" not in fields or not any(
            col in fields for col in SRAM_PROFILE_COLUMNS
        ):
            return None

        observed: dict[str, set[Any]] = {col: set() for col in SRAM_PROFILE_COLUMNS}
        matched = 0
        for row in reader:
            raw_tp = (row.get("num_tensor_parallel_workers") or "").strip()
            if not raw_tp or int(float(raw_tp)) != target_tp:
                continue
            matched += 1
            for col in _FLOAT_COLS:
                if col in fields:
                    observed[col].add(_parse_float(row.get(col), col, file_path))
            for col in _BOOL_COLS:
                if col in fields:
                    observed[col].add(_parse_bool(row.get(col), col, file_path))

    if matched == 0:
        return None

    resolved: dict[str, Any] = {}
    for col in SRAM_PROFILE_COLUMNS:
        vals = {v for v in observed[col] if v is not None}
        if len(vals) > 1:
            raise ValueError(
                f"Conflicting values for {col!r} at TP={target_tp} in {file_path}: {sorted(vals)}"
            )
        resolved[col] = next(iter(vals)) if vals else None
    return SRAMProfileContract(**resolved)


def resolve_sram_profile_contract_for_cluster(
    predictor_config: Any,
    replica_config: Any,
    cluster_type: Optional[ClusterType],
) -> Optional[SRAMProfileContract]:
    if (
        cluster_type != ClusterType.DECODE_FFN
        or predictor_config is None
        or replica_config is None
        or getattr(predictor_config, "enable_dummy_mode", False)
    ):
        return None
    model_cfg = getattr(replica_config, "model_config", None)
    model_name = (
        model_cfg.get_name()
        if model_cfg is not None and hasattr(model_cfg, "get_name")
        else str(getattr(replica_config, "model_name", ""))
    )
    paths = resolve_measurement_input_paths(
        predictor_config,
        MeasurementType.KERNEL_ONLY,
        device=str(getattr(replica_config, "device", "")),
        model=model_name,
        network_device=str(getattr(replica_config, "network_device", "")),
    )
    return load_sram_profile_contract(
        paths.compute, int(getattr(replica_config, "moe_tensor_parallel_size", 1) or 1)
    )


@dataclass(frozen=True)
class DeviceMemoryBudget:
    """Exact per-device memory budget shared by planner, scheduler logs, and metrics."""

    nominal_memory_bytes: int
    usable_memory_bytes: int
    effective_utilization: float
    requested_memory_bytes: int
    resident_weight_bytes: int
    staging_buffer_bytes: int
    planning_overhead_bytes: int
    non_kv_cache_overhead_bytes: int
    streamed_ffn_layers: int = 0

    @property
    def resident_plus_buffer_bytes(self) -> int:
        return self.resident_weight_bytes + self.staging_buffer_bytes

    @property
    def total_overhead_bytes(self) -> int:
        return self.planning_overhead_bytes + self.non_kv_cache_overhead_bytes

    @property
    def reserved_with_overhead_bytes(self) -> int:
        return self.resident_plus_buffer_bytes + self.total_overhead_bytes

    @property
    def available_memory_bytes(self) -> int:
        return self.requested_memory_bytes - self.reserved_with_overhead_bytes

    @property
    def fits_usable_budget(self) -> bool:
        return self.available_memory_bytes > 0

    @property
    def resident_plus_buffer_of_nominal_percent(self) -> float:
        return _pct(self.resident_plus_buffer_bytes, self.nominal_memory_bytes)

    @property
    def resident_plus_buffer_of_usable_percent(self) -> float:
        return _pct(self.resident_plus_buffer_bytes, self.usable_memory_bytes)


def format_decode_ffn_budget_log(
    budget: DeviceMemoryBudget,
    sram_contract: Optional[SRAMProfileContract] = None,
) -> str:
    cap = max(0, budget.available_memory_bytes)
    msg = (
        f"[ACTIVATION_MEMORY_CAPACITY] "
        f"nominal={budget.nominal_memory_bytes}B ({_mib(budget.nominal_memory_bytes):.2f} MiB), "
        f"usable={budget.usable_memory_bytes}B ({_mib(budget.usable_memory_bytes):.2f} MiB), "
        f"resident={budget.resident_weight_bytes}B ({_mib(budget.resident_weight_bytes):.2f} MiB), "
        f"staging_buffer={budget.staging_buffer_bytes}B ({_mib(budget.staging_buffer_bytes):.2f} MiB), "
        f"resident_plus_buffer={budget.resident_plus_buffer_bytes}B ({_mib(budget.resident_plus_buffer_bytes):.2f} MiB), "
        f"planning_overhead={budget.planning_overhead_bytes}B ({_mib(budget.planning_overhead_bytes):.2f} MiB), "
        f"reserved_with_overhead={budget.reserved_with_overhead_bytes}B ({_mib(budget.reserved_with_overhead_bytes):.2f} MiB), "
        f"activation_capacity={cap}B ({_mib(cap):.2f} MiB), "
        f"streamed_ffn_layers={budget.streamed_ffn_layers}"
    )
    if sram_contract is not None:
        msg += (
            f", csv_sram_fit_with_10mb_overhead={sram_contract.sram_fit_with_10mb_overhead}"
            f", csv_weight_streaming_required={sram_contract.weight_streaming_required}"
        )
    return msg


def build_memory_budget_metrics(budget: DeviceMemoryBudget) -> dict[str, Any]:
    return {
        "nominal_memory_bytes": int(budget.nominal_memory_bytes),
        "nominal_memory_mib": _mib(budget.nominal_memory_bytes),
        "usable_memory_bytes": int(budget.usable_memory_bytes),
        "usable_memory_mib": _mib(budget.usable_memory_bytes),
        "planning_overhead_bytes": int(budget.planning_overhead_bytes),
        "planning_overhead_mib": _mib(budget.planning_overhead_bytes),
        "resident_weight_bytes": int(budget.resident_weight_bytes),
        "staging_buffer_bytes": int(budget.staging_buffer_bytes),
        "resident_plus_buffer_bytes": int(budget.resident_plus_buffer_bytes),
        "reserved_with_overhead_bytes": int(budget.reserved_with_overhead_bytes),
        "usable_headroom_bytes": int(budget.available_memory_bytes),
        "fits_usable_budget": bool(budget.fits_usable_budget),
        "utilization_percent": {
            "resident_of_nominal": _pct(budget.resident_weight_bytes, budget.nominal_memory_bytes),
            "resident_plus_buffer_of_nominal": budget.resident_plus_buffer_of_nominal_percent,
            "resident_of_usable": _pct(budget.resident_weight_bytes, budget.usable_memory_bytes),
            "resident_plus_buffer_of_usable": budget.resident_plus_buffer_of_usable_percent,
            "reserved_with_overhead_of_nominal": _pct(
                budget.reserved_with_overhead_bytes, budget.nominal_memory_bytes
            ),
            "reserved_with_overhead_of_usable": _pct(
                budget.reserved_with_overhead_bytes, budget.usable_memory_bytes
            ),
        },
    }
