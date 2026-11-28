"""Device SRAM/memory budget accounting and profiling CSV contract loading.

Units follow Frontier's binary memory convention: nominal capacity in GiB
(1024^3 bytes), usable capacity and planning overhead in MiB (1024^2 bytes).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
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
_PRECISION_MIN_BYTES = {
    "FP4": 0.5,
    "INT4": 0.5,
    "FP8": 1.0,
    "INT8": 1.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "FP32": 4.0,
}


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
    profiling_precision: Optional[str] = None

    @property
    def rejects_all_resident(self) -> bool:
        return (
            self.sram_fit_with_10mb_overhead is False
            or self.weight_streaming_required is True
        )

    def rejects_all_resident_for_weight_bytes(
        self, mlp_weight_bytes_per_element: float
    ) -> bool:
        if not self.rejects_all_resident:
            return False
        precision_key = (self.profiling_precision or "FP8").strip().upper()
        min_bytes = _PRECISION_MIN_BYTES.get(precision_key, 1.0)
        return float(mlp_weight_bytes_per_element) >= min_bytes

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "rank_fp8_weight_mb": self.rank_fp8_weight_mb,
            "sram_headroom_mb": self.sram_headroom_mb,
            "sram_fit_170mb": self.sram_fit_170mb,
            "sram_fit_with_10mb_overhead": self.sram_fit_with_10mb_overhead,
            "weight_streaming_required": self.weight_streaming_required,
        }


def _effective_mlp_precision_label(model_cfg: Any) -> Optional[str]:
    if model_cfg is None:
        return None
    quant = getattr(model_cfg, "quantization_config", None)
    quant_method = getattr(quant, "quant_method", None) if quant is not None else None
    if quant_method:
        raw_quantized_ops = getattr(quant, "quantized_operations", None)
        quantized_ops = (
            {
                "attn_pre_proj",
                "attn_post_proj",
                "mlp_up_proj",
                "mlp_down_proj",
                "moe_grouped_gemm",
            }
            if raw_quantized_ops is None
            else set(raw_quantized_ops)
        )
        if "mlp_up_proj" in quantized_ops:
            return str(quant_method).strip().upper()
    default_prec = (
        model_cfg.get_default_precision()
        if hasattr(model_cfg, "get_default_precision")
        else None
    )
    if default_prec is not None:
        return str(getattr(default_prec, "name", default_prec)).strip().upper()
    return None


def _row_mlp_precision(
    row: dict[str, Any], has_op_precision_col: bool, has_precision_col: bool
) -> str:
    if has_op_precision_col:
        raw_by_op = (row.get("profiling_precision_by_op") or "").strip()
        if raw_by_op:
            try:
                parsed = json.loads(raw_by_op)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                op_prec = parsed.get("mlp_up_proj") or parsed.get("mlp_down_proj")
                if isinstance(op_prec, str) and op_prec.strip():
                    return op_prec.strip().upper()
    if has_precision_col:
        return (row.get("profiling_precision") or "").strip().upper()
    return ""


def load_sram_profile_contract(
    file_path: str,
    tensor_parallel_size: int,
    profiling_precision: Optional[str] = None,
) -> Optional[SRAMProfileContract]:
    """Load optional SRAM columns for one TP slice; fail fast on conflicting rows."""
    if not file_path or not os.path.isfile(file_path):
        return None

    target_tp = int(tensor_parallel_size)
    target_precision = (
        profiling_precision.strip().upper() if profiling_precision else None
    )
    with open(file_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "num_tensor_parallel_workers" not in fields or not any(
            col in fields for col in SRAM_PROFILE_COLUMNS
        ):
            return None

        has_precision_col = "profiling_precision" in fields
        has_op_precision_col = "profiling_precision_by_op" in fields
        tp_rows: list[tuple[str, dict[str, Any]]] = []
        for row in reader:
            raw_tp = (row.get("num_tensor_parallel_workers") or "").strip()
            if not raw_tp or int(float(raw_tp)) != target_tp:
                continue
            tp_rows.append(
                (_row_mlp_precision(row, has_op_precision_col, has_precision_col), row)
            )

    if not tp_rows:
        return None

    if target_precision:
        exact_rows = [
            item for item in tp_rows if not item[0] or item[0] == target_precision
        ]
        selected_rows = exact_rows if exact_rows else tp_rows
    else:
        selected_rows = tp_rows

    observed: dict[str, set[Any]] = {col: set() for col in SRAM_PROFILE_COLUMNS}
    observed_precisions: set[str] = set()
    for row_precision, row in selected_rows:
        if row_precision:
            observed_precisions.add(row_precision)
        for col in _FLOAT_COLS:
            if col in fields:
                observed[col].add(_parse_float(row.get(col), col, file_path))
        for col in _BOOL_COLS:
            if col in fields:
                observed[col].add(_parse_bool(row.get(col), col, file_path))

    resolved: dict[str, Any] = {}
    for col in SRAM_PROFILE_COLUMNS:
        vals = {v for v in observed[col] if v is not None}
        if len(vals) > 1:
            raise ValueError(
                f"Conflicting values for {col!r} at TP={target_tp} in {file_path}: {sorted(vals)}"
            )
        resolved[col] = next(iter(vals)) if vals else None
    resolved["profiling_precision"] = (
        next(iter(observed_precisions)) if len(observed_precisions) == 1 else None
    )
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
        paths.compute,
        int(getattr(replica_config, "moe_tensor_parallel_size", 1) or 1),
        profiling_precision=_effective_mlp_precision_label(model_cfg),
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
    def usable_headroom_bytes(self) -> int:
        return self.usable_memory_bytes - self.reserved_with_overhead_bytes

    @property
    def available_memory_bytes(self) -> int:
        return self.requested_memory_bytes - self.reserved_with_overhead_bytes

    @property
    def fits_usable_budget(self) -> bool:
        return self.usable_headroom_bytes > 0

    @property
    def fits_requested_budget(self) -> bool:
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
        f"requested={budget.requested_memory_bytes}B ({_mib(budget.requested_memory_bytes):.2f} MiB), "
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
        "effective_utilization": float(budget.effective_utilization),
        "requested_memory_bytes": int(budget.requested_memory_bytes),
        "requested_memory_mib": _mib(budget.requested_memory_bytes),
        "planning_overhead_bytes": int(budget.planning_overhead_bytes),
        "planning_overhead_mib": _mib(budget.planning_overhead_bytes),
        "non_kv_cache_overhead_bytes": int(budget.non_kv_cache_overhead_bytes),
        "resident_weight_bytes": int(budget.resident_weight_bytes),
        "staging_buffer_bytes": int(budget.staging_buffer_bytes),
        "resident_plus_buffer_bytes": int(budget.resident_plus_buffer_bytes),
        "reserved_with_overhead_bytes": int(budget.reserved_with_overhead_bytes),
        "usable_headroom_bytes": int(budget.usable_headroom_bytes),
        "requested_headroom_bytes": int(budget.available_memory_bytes),
        "fits_usable_budget": bool(budget.fits_usable_budget),
        "fits_requested_budget": bool(budget.fits_requested_budget),
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
