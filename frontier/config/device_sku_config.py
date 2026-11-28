from dataclasses import dataclass, field
import math
from typing import Optional

from frontier.config.base_fixed_config import BaseFixedConfig
from frontier.logger import init_logger
from frontier.types import DeviceSKUType

logger = init_logger(__name__)

BINARY_MIB_BYTES = 1024**2
BINARY_GIB_BYTES = 1024**3


def validate_memory_mb_value(value: object, field_name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number in MiB, got={value!r}")
    numeric = float(value)
    if numeric < 0.0 if allow_zero else numeric <= 0.0:
        op = ">=" if allow_zero else ">"
        raise ValueError(f"{field_name} must be {op} 0, got={value!r}")
    return numeric


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: float
    gpu_platform: str = "cuda"
    usable_memory_mb: Optional[float] = None
    planning_overhead_mb: float = 0.0

    def __post_init__(self) -> None:
        if self.usable_memory_mb is not None:
            self.usable_memory_mb = validate_memory_mb_value(
                self.usable_memory_mb, "usable_memory_mb", allow_zero=False
            )
            if self.get_usable_memory_bytes() > self.get_nominal_memory_bytes():
                raise ValueError("usable_memory_mb cannot exceed nominal device memory")
        self.planning_overhead_mb = validate_memory_mb_value(
            self.planning_overhead_mb, "planning_overhead_mb", allow_zero=True
        )
        if self.get_planning_overhead_bytes() >= self.get_usable_memory_bytes():
            raise ValueError("planning_overhead_mb must be strictly smaller than usable memory")

    def get_nominal_memory_bytes(self) -> int:
        return int(round(float(self.total_memory_gb) * BINARY_GIB_BYTES))

    def get_usable_memory_bytes(self) -> int:
        if self.usable_memory_mb is None:
            return self.get_nominal_memory_bytes()
        return int(round(float(self.usable_memory_mb) * BINARY_MIB_BYTES))

    def get_planning_overhead_bytes(self) -> int:
        return int(round(float(self.planning_overhead_mb) * BINARY_MIB_BYTES))


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A100

@dataclass
class A800DeviceSKUConfig(BaseDeviceSKUConfig):
    # A800的性能与A100相近，但有一些差异
    fp16_tflops: int = 320  # 可以根据实际情况调整
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A800
    
@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H100


@dataclass
class H800DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 989
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H800


@dataclass
class H200DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1979
    total_memory_gb: int = 141

    @staticmethod
    def get_type():
        return DeviceSKUType.H200


@dataclass
class H20DeviceSKUConfig(BaseDeviceSKUConfig):
    # H20 values are grounded in local H20 worker inventory and the bundled
    # MegaScale-Infer hardware table used by this repository's evaluation notes.
    fp16_tflops: int = 148
    total_memory_gb: int = 96

    @staticmethod
    def get_type():
        return DeviceSKUType.H20


@dataclass
class RtxPro6000DeviceSKUConfig(BaseDeviceSKUConfig):
    # NVIDIA RTX PRO 6000 Blackwell Server Edition: 96 GB GDDR7 and
    # 1 PFLOP FP16/BF16 Tensor Core throughput per NVIDIA's official specs.
    fp16_tflops: int = 1000
    total_memory_gb: int = 96

    @staticmethod
    def get_type():
        return DeviceSKUType.RTX_PRO_6000


@dataclass
class MI355XDeviceSKUConfig(BaseDeviceSKUConfig):
    """AMD Instinct MI355X device metadata."""

    fp16_tflops: int = 2516
    total_memory_gb: int = 288
    gpu_platform: str = "rocm"

    @staticmethod
    def get_type():
        return DeviceSKUType.MI355X


@dataclass
class HPP2DeviceSKUConfig(BaseDeviceSKUConfig):
    """HPP2.0 die/rank: 32 APC, ~380 TFLOPS FP16, 192 MiB physical APC DM.

    ``total_memory_gb`` is the nominal per-die/rank APC DM (0.1875 GiB =
    192 MiB = 201,326,592 B), not per card. Under the SCALE runtime, each of
    the 32 APCs reserves 4 x 176 KiB in Bank0 (22 MiB total), leaving 170 MiB
    (178,257,920 B) allocatable: 20 full 256 KiB banks per APC (160 MiB) plus
    4 x 80 KiB Bank0 remainders (10 MiB). ``usable_memory_mb`` and
    ``planning_overhead_mb`` use binary MiB (1024^2 bytes), matching
    ``total_memory_gb`` (binary GiB).

    The HPP2 decode-FFN role is priced from KERNEL_ONLY tables, so the
    CUDA-style event family is only a simulator bookkeeping default here.
    """

    fp16_tflops: int = 380
    total_memory_gb: float = 0.1875
    usable_memory_mb: Optional[float] = 170.0
    planning_overhead_mb: float = 10.0
    gpu_platform: str = "cuda"

    @staticmethod
    def get_type():
        return DeviceSKUType.HPP2
