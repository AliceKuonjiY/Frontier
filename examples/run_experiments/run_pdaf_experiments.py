#!/usr/bin/env python3
"""Qwen3-4B PD-AF 一键实验脚本。

设计目标
========
1. 对当前工作树里的 ``Frontier`` 直接跑，不依赖已安装包，方便改完模拟器立即复跑。
2. 每次运行写到独立的 ``outputs/runs/<时间戳>_<commit>/``，不会覆盖旧结果。
3. 记录 git commit / dirty 状态、Python 环境、完整配置和失败原因，便于审计。
4. 支持按 profile / stage / 单个实验名筛选，支持断点续跑。
5. 运行结束自动生成 ``summary.md``，可直接作为写报告的数据底稿。

常用命令
========
    # 默认 core：配比 + 关键在线 + M2N + 权重流式
    ./run_experiments.sh

    # 全量（包含 AF2/AF4 对照，不含 400 请求长跑）
    ./run_experiments.sh --profile full --workers 3

    # 只重跑配比
    ./run_experiments.sh --stage gpu_scale_burst --workers 4

    # 只跑一个配置
    ./run_experiments.sh --name g16_p8_d8_tp2

    # 多 attention replica 并行（当前可跑）
    ./run_experiments.sh --stage attn_stage_parallel

    # 单 replica 内 PP>1（需要模拟器先放开 PD-AF 限制）
    ./run_experiments.sh --stage attn_stage_parallel --include-experimental

    # 只打印将执行的命令，不跑仿真
    ./run_experiments.sh --dry-run

    # 400 请求多种子长跑（显式指定 stage）
    ./run_experiments.sh --stage online_long --workers 2

注意
====
- AF2/AF4 目前仍受 issue #6 影响：attention 按微批串行执行，overlap
  收益远小于拆分成本。core 里只保留少量 AF 对照点，默认推荐 AF1。
- M2N 默认使用 HostBounce 分段口径：``p2p_protocol=host_bounce``，
  366 Gbps 只计 IB 段，PCIe 段由 HPP2 CPU-overhead 表按层计。
- 跨节点 KV 仍按 NVLink 3172.5 Gbps 计（issue #10），报告里需注明。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as package_version

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
# examples/run_experiments -> examples -> Frontier
REPO_ROOT = HERE.parents[1]
RUNS_ROOT = REPO_ROOT / "outputs" / "runs"
DEFAULT_WORKERS = 3
DEFAULT_TRAINING_THREADS = 1
DEFAULT_TIMEOUT_S = 3600
NO_GIT_SESSION_ID = uuid.uuid4().hex

# main() 会根据 --output-dir 重新绑定这些全局变量；模块级默认值只为
# --dry-run / --list 时 build_command() 能正常构造命令。
OUT_ROOT = RUNS_ROOT / "_unset" / "outputs"
METRICS_ROOT = OUT_ROOT / "qwen3_4b" / "offline_batch"
ATTEMPTS_ROOT = RUNS_ROOT / "_unset" / "attempts"
RESULTS_CSV = RUNS_ROOT / "_unset" / "sweep_results.csv"

# Used by the Ctrl-C path: running children get terminated, queued futures
# get cancelled, and the checkpoint keeps only results written before the
# interrupt.
CANCEL_EVENT = threading.Event()
_RUNNING_LOCK = threading.Lock()
_RUNNING_PROCS: set[subprocess.Popen] = set()

RUNTIME_MODULES = (
    "ddsketch",
    "fasteners",
    "numpy",
    "pandas",
    "plotly",
    "yaml",
    "sklearn",
    "scipy",
    "tqdm",
)

REQUIRED_DATA = [
    "data/config/models/Qwen3-4B.json",
    "data/profiling/compute/hpp2/Qwen3-4B/linear_op_kernel_only.csv",
    "data/profiling/compute/h200/Qwen3-4B/linear_op.csv",
    "data/profiling/compute/h200/Qwen3-4B/linear_op_kernel_only.csv",
    "data/profiling/compute/h200/Qwen3-4B/attention.csv",
    "data/profiling/compute/h200/Qwen3-4B/attention_kernel_only.csv",
    "data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only.csv",
    "data/profiling/network/a2f_h200_to_hpp2/m2n_transfer_lut.csv",
    "data/profiling/network/hpp2_16rank/all_reduce.csv",
    "data/profiling/network/hpp2_16rank/weight_stream_lut.csv",
    "data/profiling/network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv",
]


# ---------------------------------------------------------------------------
# 实验定义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    name: str
    stage: str
    p_rep: int
    p_tp: int
    d_rep: int
    d_tp: int
    attn_pp: int = 1
    f_rep: int = 1
    f_tp: int = 16
    af_mb: int = 1
    d_batch: int = 8
    max_tokens_in_batch: int = 16384
    num_req: int = 32
    prefill_tokens: int = 512
    decode_tokens: int = 64
    arrival_pattern: str = "burst"
    qps: float = 0.0
    seed: int = 42
    weight_stream_layers: int = 0
    m2n_bw_gbps: float = 366.0
    m2n_latency_ms: float = 0.0
    m2n_protocol: str = "host_bounce"
    skip_cpu_overhead: bool = False
    comm_label: str = "HostBounce_IB366G"
    tier: str = "core"
    note: str = ""

    @property
    def h200_prefill_cards(self) -> int:
        return self.p_rep * self.p_tp

    @property
    def h200_decode_cards(self) -> int:
        return self.d_rep * self.d_tp * self.attn_pp

    @property
    def h200_total_cards(self) -> int:
        return self.h200_prefill_cards + self.h200_decode_cards

    @property
    def hpp2_total_cards(self) -> int:
        return self.f_rep * (self.f_tp // 2)

    @property
    def p_card_ratio(self) -> float:
        return self.h200_prefill_cards / self.h200_total_cards if self.h200_total_cards else 0.0


def _add(
    cfgs: list[Experiment],
    name: str,
    stage: str,
    prefill: tuple[int, int],
    decode: tuple[int, int],
    **kwargs: Any,
) -> None:
    cfgs.append(
        Experiment(
            name=name,
            stage=stage,
            p_rep=prefill[0],
            p_tp=prefill[1],
            d_rep=decode[0],
            d_tp=decode[1],
            **kwargs,
        )
    )


BURST_SPECS: list[tuple[str, tuple[int, int], tuple[int, int]]] = [
    # 2 H200
    ("g2_p1_d1_tp1", (1, 1), (1, 1)),
    # 4 H200
    ("g4_p2_d2_tp2", (1, 2), (1, 2)),
    ("g4_p2_d2_tp1", (2, 1), (2, 1)),
    ("g4_p1_d3_tp1", (1, 1), (3, 1)),
    ("g4_p3_d1_tp1", (3, 1), (1, 1)),
    # 6 H200
    ("g6_p2_d4_tp2", (1, 2), (2, 2)),
    ("g6_p4_d2_tp2", (2, 2), (1, 2)),
    ("g6_p2_d4_tp1", (2, 1), (4, 1)),
    ("g6_p3_d3_tp1", (3, 1), (3, 1)),
    # 8 H200
    ("g8_p4_d4_tp4", (1, 4), (1, 4)),
    ("g8_p4_d4_tp2", (2, 2), (2, 2)),
    ("g8_p2_d6_tp2", (1, 2), (3, 2)),
    ("g8_p6_d2_tp2", (3, 2), (1, 2)),
    ("g8_p4_d4_tp1", (4, 1), (4, 1)),
    # 12 H200
    ("g12_p4_d8_tp2", (2, 2), (4, 2)),
    ("g12_p6_d6_tp2", (3, 2), (3, 2)),
    ("g12_p8_d4_tp2", (4, 2), (2, 2)),
    ("g12_p4_d8_tp4", (1, 4), (2, 4)),
    ("g12_p8_d4_tp4", (2, 4), (1, 4)),
    # 16 H200
    ("g16_p4_d12_tp2", (2, 2), (6, 2)),
    ("g16_p6_d10_tp2", (3, 2), (5, 2)),
    ("g16_p8_d8_tp2", (4, 2), (4, 2)),
    ("g16_p10_d6_tp2", (5, 2), (3, 2)),
    ("g16_p8_d8_tp4", (2, 4), (2, 4)),
    ("g16_p8_d8_tp1", (8, 1), (8, 1)),
]

AF_PROBE_BASES: list[tuple[str, tuple[int, int], tuple[int, int], str]] = [
    ("g8_p4_d4_tp2", (2, 2), (2, 2), "full"),
    ("g12_p4_d8_tp2", (2, 2), (4, 2), "core"),
    ("g16_p6_d10_tp2", (3, 2), (5, 2), "core"),
    ("g16_p8_d8_tp2", (4, 2), (4, 2), "core"),
]

POISSON_TOPOS: list[tuple[str, tuple[int, int], tuple[int, int], str]] = [
    ("g2_tp1", (1, 1), (1, 1), "core"),
    ("g4_tp1", (2, 1), (2, 1), "core"),
    ("g8_tp2", (2, 2), (2, 2), "core"),
    ("g12_p6d6_tp2", (3, 2), (3, 2), "core"),
    ("g16_p8d8_tp2", (4, 2), (4, 2), "core"),
    ("g12_p4d8_tp2", (2, 2), (4, 2), "core"),
    ("g8_p4d4_tp1", (4, 1), (4, 1), "core"),
    ("g16_p6d10_tp2", (3, 2), (5, 2), "full"),
]

ONLINE_SPECS: list[tuple[str, tuple[int, int], tuple[int, int], int, int]] = [
    # name, P, D, QPS, requests per seed
    ("g16_p8d8_tp2", (4, 2), (4, 2), 250, 400),
    ("g16_p6d10_tp2", (3, 2), (5, 2), 250, 400),
    ("g12_p4d8_tp2", (2, 2), (4, 2), 150, 400),
    ("g8_p4d4_tp1", (4, 1), (4, 1), 100, 400),
]


def build_all_configs() -> list[Experiment]:
    cfgs: list[Experiment] = []

    # 0) 基线校验：8 请求，512:64，对应 RUN_QWEN3_4B_PD_AF.md
    for stream in (0, 3):
        _add(
            cfgs,
            f"verify_baseline_stream{stream}",
            "verify",
            (1, 8),
            (1, 8),
            num_req=8,
            max_tokens_in_batch=8192,
            weight_stream_layers=stream,
            tier="core",
        )

    # 1) 配比主扫描：32 请求 burst，固定 1 个 FFN replica（8 张 HPP2）
    for name, prefill, decode in BURST_SPECS:
        _add(
            cfgs,
            name,
            "gpu_scale_burst",
            prefill,
            decode,
            num_req=32,
            tier="core",
        )

    # 1b) AF2/AF4 对照：只保留少量点，避免默认 profile 被 AF 长跑拖慢
    for base_name, prefill, decode, tier in AF_PROBE_BASES:
        for af in (2, 4):
            _add(
                cfgs,
                f"{base_name}_af{af}",
                "gpu_scale_burst",
                prefill,
                decode,
                af_mb=af,
                num_req=32,
                tier=tier,
                note=f"AF{af} 对照；当前受 issue #6 影响，仅用于量化回归",
            )

    # 1c) 多个 attention stage/replica 并行。
    #     - 先给当前可跑的“多 D replica”配置：保持 H200 总数不变，
    #       增加 decode-attn 副本数、降低每副本 TP，让更多 attention
    #       cohort 在不同 replica 上并行。
    #     - 再给单 replica 内 PP>1 的实验配置（tier=experimental）。
    #       当前 PD-AF 校验要求 decode_attn num_pipeline_stages=1；
    #       模拟器放开后可用 --include-experimental 跑。
    for name, prefill, decode, note in (
        ("ap_g16_p4d8_tp1_af1", (4, 2), (8, 1), "8:8，8 个 D replica 并行（TP1）"),
        ("ap_g16_p2d12_tp1_af1", (2, 2), (12, 1), "4:12，12 个 D replica 并行（TP1）"),
        ("ap_g16_p6d10_tp1_af1", (6, 1), (10, 1), "6:10，10 个 D replica 并行（TP1）"),
        ("ap_g24_p4d8_tp2_af1", (4, 2), (8, 2), "8:16，8 个 D replica 并行（TP2）"),
        ("ap_g24_p8d4_tp2_af1", (8, 2), (4, 2), "16:8，4 个 D replica（TP2）"),
        ("ap_g32_p4d12_tp2_af1", (4, 2), (12, 2), "8:24，12 个 D replica 并行（TP2）"),
        ("ap_g32_p8d8_tp2_af1", (8, 2), (8, 2), "16:16，8 个 D replica 并行（TP2）"),
    ):
        _add(
            cfgs,
            name,
            "attn_stage_parallel",
            prefill,
            decode,
            num_req=32,
            tier="core",
            note=note,
        )

    # 1d) 多 D replica + AF2：验证“更多 attention 并行 + 每副本 2 个 stage”
    for name, prefill, decode, note in (
        ("ap_g16_p4d8_tp1_af2", (4, 2), (8, 1), "8:8，8 个 D replica + AF2"),
        ("ap_g16_p2d12_tp1_af2", (2, 2), (12, 1), "4:12，12 个 D replica + AF2"),
    ):
        _add(
            cfgs,
            name,
            "attn_stage_parallel",
            prefill,
            decode,
            af_mb=2,
            num_req=32,
            tier="full",
            note=note,
        )

    for name, prefill, decode, af, note in (
        ("ap_g16_p4d8_tp1_af4", (4, 2), (8, 1), 4, "8:8，8 个 D replica + AF4"),
        ("ap_g16_p2d12_tp1_af4", (2, 2), (12, 1), 4, "4:12，12 个 D replica + AF4"),
    ):
        _add(
            cfgs,
            name,
            "attn_stage_parallel",
            prefill,
            decode,
            af_mb=af,
            num_req=32,
            tier="full",
            note=note,
        )

    for name, prefill, decode, pp, af, total, note in (
        ("ap_pp_g16_p4d2_tp2_pp2_af1", (4, 2), (2, 2), 2, 1, 16, "8:8，D 侧 PP2，2 个 D replica"),
        ("ap_pp_g16_p4d1_tp2_pp4_af1", (4, 2), (1, 2), 4, 1, 16, "8:8，D 侧 PP4，1 个 D replica"),
        ("ap_pp_g16_p4d2_tp2_pp2_af2", (4, 2), (2, 2), 2, 2, 16, "8:8，D 侧 PP2 + AF2"),
        ("ap_pp_g16_p4d2_tp2_pp2_af4", (4, 2), (2, 2), 2, 4, 16, "8:8，D 侧 PP2 + AF4"),
        ("ap_pp_g24_p4d4_tp2_pp2_af1", (4, 2), (4, 2), 2, 1, 24, "8:16，D 侧 PP2，4 个 D replica"),
        ("ap_pp_g24_p4d2_tp2_pp4_af1", (4, 2), (2, 2), 4, 1, 24, "8:16，D 侧 PP4，2 个 D replica"),
        ("ap_pp_g32_p4d6_tp2_pp2_af1", (4, 2), (6, 2), 2, 1, 32, "8:24，D 侧 PP2，6 个 D replica"),
    ):
        _add(
            cfgs,
            name,
            "attn_stage_parallel",
            prefill,
            decode,
            attn_pp=pp,
            af_mb=af,
            num_req=32,
            tier="experimental",
            note=note + f"；H200={total}；需模拟器支持 PD-AF DECODE_ATTN PP>1",
        )

    # 2) Poisson 在线扫描：64 请求，QPS 20/50/100/150/250
    for topo_name, prefill, decode, tier in POISSON_TOPOS:
        qps_values = (20, 50, 100, 150, 250)
        if topo_name in ("g12_p4d8_tp2", "g8_p4d4_tp1"):
            qps_values = (50, 100, 150, 250)
        for qps in qps_values:
            _add(
                cfgs,
                f"pois_{topo_name}_qps{qps}",
                "gpu_scale_poisson",
                prefill,
                decode,
                num_req=64,
                arrival_pattern="poisson",
                qps=float(qps),
                tier=tier,
            )

    # 2b) AF2/AF4 在线对照：只跑 16 卡 8:8，QPS 100/150/250
    for af in (2, 4):
        for qps in (100, 150, 250):
            _add(
                cfgs,
                f"pois_g16_p8d8_tp2_af{af}_qps{qps}",
                "gpu_scale_poisson",
                (4, 2),
                (4, 2),
                af_mb=af,
                num_req=64,
                arrival_pattern="poisson",
                qps=float(qps),
                tier="full",
                note=f"AF{af} 在线对照；仅 full profile",
            )

    # 2c) 官方 10-case 校准集里的 400 请求 AF4 点
    _add(
        cfgs,
        "pois_g16_tp2_af4_qps250_n400",
        "gpu_scale_poisson",
        (4, 2),
        (4, 2),
        af_mb=4,
        num_req=400,
        arrival_pattern="poisson",
        qps=250.0,
        tier="calibration",
        note="官方 10-case 校准集的 400 请求在线点；当前 AF4 仅用于回归对照",
    )

    # 3) M2N 敏感性：AF1 为默认，Direct vs HostBounce，启动延迟 0/5/10/20/50 us
    m2n_latencies = (0, 20)  # core
    full_m2n_latencies = (0, 5, 10, 20, 50)

    for lat_us in full_m2n_latencies:
        tier = "core" if lat_us in m2n_latencies else "full"
        for protocol, bw, skip, label in (
            ("rdma", 366.0, True, "Direct_IB366G"),
            ("host_bounce", 366.0, False, "HostBounce_IB366G"),
        ):
            _add(
                cfgs,
                f"m2n_g16_p8d8_af1_{label.lower()}_lat{lat_us}us",
                "m2n_sensitivity",
                (4, 2),
                (4, 2),
                num_req=32,
                m2n_bw_gbps=bw,
                m2n_latency_ms=lat_us / 1000.0,
                m2n_protocol=protocol,
                skip_cpu_overhead=skip,
                comm_label=label,
                tier=tier,
            )

    for lat_us in (0, 20):
        for bw, label in ((100.0, "HostBounce_100G"), (50.0, "HostBounce_50G")):
            _add(
                cfgs,
                f"m2n_g16_p8d8_af1_hb{int(bw)}_lat{lat_us}us",
                "m2n_sensitivity",
                (4, 2),
                (4, 2),
                num_req=32,
                m2n_bw_gbps=bw,
                m2n_latency_ms=lat_us / 1000.0,
                m2n_protocol="host_bounce",
                skip_cpu_overhead=False,
                comm_label=label,
                tier="full",
            )

    for lat_us in (0, 20):
        _add(
            cfgs,
            f"m2n_g8_p4d4_af1_hb366_lat{lat_us}us",
            "m2n_sensitivity",
            (2, 2),
            (2, 2),
            num_req=32,
            m2n_bw_gbps=366.0,
            m2n_latency_ms=lat_us / 1000.0,
            m2n_protocol="host_bounce",
            skip_cpu_overhead=False,
            comm_label="HostBounce_IB366G",
            tier="full",
        )

    for af in (2, 4):
        _add(
            cfgs,
            f"m2n_g16_p8d8_af{af}_hb366_lat20us",
            "m2n_sensitivity",
            (4, 2),
            (4, 2),
            af_mb=af,
            num_req=32,
            m2n_bw_gbps=366.0,
            m2n_latency_ms=0.020,
            m2n_protocol="host_bounce",
            skip_cpu_overhead=False,
            comm_label="HostBounce_IB366G",
            tier="full",
            note=f"AF{af} M2N 对照；当前受 issue #6 影响",
        )

    # 4) 权重流式：0/1/2/3 层，AF1 为主；AF2/AF4 只做 full 对照
    for ws in (0, 1, 2, 3):
        _add(
            cfgs,
            f"ws{ws}_g16_p8d8_af1_f1",
            "weight_stream",
            (4, 2),
            (4, 2),
            num_req=32,
            weight_stream_layers=ws,
            tier="core",
        )
        for af in (2, 4):
            _add(
                cfgs,
                f"ws{ws}_g16_p8d8_af{af}_f1",
                "weight_stream",
                (4, 2),
                (4, 2),
                af_mb=af,
                num_req=32,
                weight_stream_layers=ws,
                tier="full",
                note=f"AF{af} 流式对照；当前受 issue #6 影响",
            )

    for ws in (1, 2, 3):
        _add(
            cfgs,
            f"ws{ws}_g16_p8d8_af1_f2",
            "weight_stream",
            (4, 2),
            (4, 2),
            f_rep=2,
            num_req=32,
            weight_stream_layers=ws,
            tier="core",
        )
        for af in (2, 4):
            _add(
                cfgs,
                f"ws{ws}_g16_p8d8_af{af}_f2",
                "weight_stream",
                (4, 2),
                (4, 2),
                f_rep=2,
                af_mb=af,
                num_req=32,
                weight_stream_layers=ws,
                tier="full",
                note=f"AF{af} + 双 FFN 流式对照；当前受 issue #6 影响",
            )

    for label, prefill, decode in (
        ("p4d12", (2, 2), (6, 2)),
        ("p6d10", (3, 2), (5, 2)),
        ("p10d6", (5, 2), (3, 2)),
    ):
        _add(
            cfgs,
            f"ws3_g16_{label}_af1_f1",
            "weight_stream",
            prefill,
            decode,
            num_req=32,
            weight_stream_layers=3,
            tier="core",
        )
        _add(
            cfgs,
            f"ws3_g16_{label}_af4_f1",
            "weight_stream",
            prefill,
            decode,
            af_mb=4,
            num_req=32,
            weight_stream_layers=3,
            tier="full",
            note="AF4 流式配比对照；当前受 issue #6 影响",
        )

    _add(
        cfgs,
        "ws3_g8_p4d4_af1_f1",
        "weight_stream",
        (2, 2),
        (2, 2),
        num_req=32,
        weight_stream_layers=3,
        tier="full",
    )

    # 5) 400 请求多种子长跑：显式 --stage online_long 才跑
    for base_name, prefill, decode, qps, num_req in ONLINE_SPECS:
        for seed in (1, 2, 3, 4, 5):
            _add(
                cfgs,
                f"online_{base_name}_qps{qps}_req{num_req}_seed{seed}",
                "online_long",
                prefill,
                decode,
                num_req=num_req,
                arrival_pattern="poisson",
                qps=float(qps),
                seed=seed,
                tier="long",
                note="400 请求 × 5 种子；在线容量/置信区间",
            )

    return cfgs


PROFILE_STAGES = {
    "smoke": {"verify", "gpu_scale_burst"},
    "core": {"verify", "gpu_scale_burst", "attn_stage_parallel", "gpu_scale_poisson", "m2n_sensitivity", "weight_stream"},
    "full": {"verify", "gpu_scale_burst", "attn_stage_parallel", "gpu_scale_poisson", "m2n_sensitivity", "weight_stream"},
}

SMOKE_NAMES = {
    "verify_baseline_stream0",
    "g16_p8_d8_tp2",
    "g16_p6_d10_tp2",
    "g12_p4_d8_tp2",
    "g16_p8_d8_tp2_af4",
}

# Frontier 仓库自带的 10-case 校准集（tests/performance/qwen3_4b_pdaf）。
# 运行后可直接用 compare.py 与 calibrated_cases.csv 对比。
CALIBRATION10_NAMES = {
    "verify_baseline_stream0",
    "verify_baseline_stream3",
    "g16_p8_d8_tp2",
    "g16_p8_d8_tp2_af4",
    "g12_p4_d8_tp2",
    "g8_p4_d4_tp1",
    "g8_p4_d4_tp2",
    "ws3_g16_p8d8_af4_f1",
    "ws3_g16_p8d8_af4_f2",
    "pois_g16_tp2_af4_qps250_n400",
}


# ---------------------------------------------------------------------------
# 命令与指标
# ---------------------------------------------------------------------------


def build_command(
    exp: Experiment,
    metrics_output_dir: Path | None = None,
    training_threads: int = DEFAULT_TRAINING_THREADS,
    cache_dir: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "frontier.main",
        "--simulation_mode",
        "offline",
        "--sys_arch",
        "pd-af-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_config_prefill_cluster_num_replicas",
        str(exp.p_rep),
        "--cluster_config_decode_attn_cluster_num_replicas",
        str(exp.d_rep),
        "--cluster_config_decode_ffn_cluster_num_replicas",
        str(exp.f_rep),
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch",
        str(exp.af_mb),
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch",
        str(exp.af_mb),
        "--cluster_config_decode_attn_micro_batch_size",
        str(exp.d_batch),
        # PREFILL
        "--cluster_config_prefill_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size",
        str(exp.p_tp),
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size",
        str(exp.p_tp),
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_total_expert_num",
        "1",
        "--cluster_config_prefill_replica_config_router_topk",
        "1",
        "--cluster_config_prefill_replica_config_device",
        "h200",
        "--cluster_config_prefill_replica_config_network_device",
        "h200_dgx",
        "--cluster_config_prefill_replica_config_memory_margin_fraction",
        "0.2",
        # DECODE_ATTN
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size",
        str(exp.d_tp),
        "--cluster_config_decode_attn_replica_config_num_pipeline_stages",
        str(exp.attn_pp),
        "--cluster_config_decode_attn_replica_config_device",
        "h200",
        "--cluster_config_decode_attn_replica_config_network_device",
        "h200_dgx",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction",
        "0.2",
        # DECODE_FFN
        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size",
        str(exp.f_tp),
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size",
        "1",
        "--cluster_config_decode_ffn_replica_config_total_expert_num",
        "1",
        "--cluster_config_decode_ffn_replica_config_router_topk",
        "1",
        "--cluster_config_decode_ffn_replica_config_device",
        "hpp2",
        "--cluster_config_decode_ffn_replica_config_streamed_ffn_layers",
        str(exp.weight_stream_layers),
        "--cluster_config_decode_ffn_replica_config_network_device",
        "hpp2_16rank",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction",
        "0.0",
        # Schedulers
        "--cluster_config_prefill_replica_scheduler_config_type",
        "vllm_v1",
        "--cluster_config_prefill_replica_scheduler_config_max_tokens_in_batch",
        str(exp.max_tokens_in_batch),
        "--cluster_config_decode_attn_replica_scheduler_config_type",
        "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_max_tokens_in_batch",
        str(exp.max_tokens_in_batch),
        "--cluster_config_decode_ffn_replica_scheduler_config_type",
        "orca",
        # Communication
        "--cc_backend_config_type",
        "analytical",
        "--m2n_transfer_config_type",
        "analytical",
        "--kv_cache_transfer_config_type",
        "analytical",
        "--analytical_m2_n_transfer_config_memory_bandwidth_gbps",
        str(exp.m2n_bw_gbps),
        "--analytical_m2_n_transfer_config_network_latency_ms",
        str(exp.m2n_latency_ms),
        "--analytical_m2_n_transfer_config_p2p_protocol",
        exp.m2n_protocol,
        "--no-analytical_m2_n_transfer_config_enable_p2p_optimization",
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
        "3172.5",
        "--analytical_kv_cache_transfer_config_network_latency_ms",
        "0.0208",
        "--analytical_kv_cache_transfer_config_transfer_protocol",
        "nvlink",
        "--analytical_cc_backend_config_intra_node_bandwidth_gbps",
        "2984.0",
        "--analytical_cc_backend_config_network_bandwidth_gbps",
        "292.8",
        "--analytical_cc_backend_config_network_latency_us",
        "0.0",
        # Model / routing
        "--replica_config_model_name",
        "Qwen3-4B",
        "--replica_config_moe_routing_distribution_type",
        "balanced",
        "--replica_config_moe_routing_seed",
        "42",
        # Scheduler batch
        "--vllm_v1_scheduler_config_max_tokens_in_batch",
        str(exp.max_tokens_in_batch),
        "--vllm_v1_scheduler_config_long_prefill_token_threshold",
        "64",
        "--vllm_v1_scheduler_config_block_size",
        "16",
        "--vllm_v1_scheduler_config_num_blocks",
        "4096",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        # Workload
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        str(exp.num_req),
        "--synthetic_request_generator_config_seed",
        str(exp.seed),
        "--length_generator_config_type",
        "fixed",
        "--fixed_request_length_generator_config_prefill_tokens",
        str(exp.prefill_tokens),
        "--fixed_request_length_generator_config_decode_tokens",
        str(exp.decode_tokens),
        "--fixed_request_length_generator_config_seed",
        str(exp.seed),
        "--random_forrest_execution_time_predictor_config_k_fold_cv_splits",
        "3",
        "--random_forrest_execution_time_predictor_config_num_training_job_threads",
        str(training_threads),
        # Metrics
        "--metrics_config_output_dir",
        str(metrics_output_dir or OUT_ROOT),
        "--metrics_config_cache_dir",
        str(cache_dir or (OUT_ROOT.parent / "predictor_cache")),
        "--metrics_config_run_id",
        exp.name,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics",
        "--metrics_config_store_token_completion_metrics",
        "--metrics_config_store_utilization_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        "--log_level",
        "warning",
    ]

    if exp.arrival_pattern == "burst":
        cmd.extend(
            [
                "--interval_generator_config_type",
                "static",
                "--static_request_interval_generator_config_seed",
                str(exp.seed),
            ]
        )
    elif exp.arrival_pattern == "poisson":
        cmd.extend(
            [
                "--interval_generator_config_type",
                "poisson",
                "--poisson_request_interval_generator_config_qps",
                str(exp.qps),
                "--poisson_request_interval_generator_config_seed",
                str(exp.seed),
                "--offline_use_generated_request_arrivals",
            ]
        )
    else:
        raise ValueError(f"unknown arrival_pattern={exp.arrival_pattern!r}")

    if exp.skip_cpu_overhead:
        cmd.append("--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling")
    else:
        cmd.append("--no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling")
        if exp.weight_stream_layers in (1, 2, 3):
            cmd.extend(
                [
                    "--random_forrest_execution_time_predictor_config_cpu_overhead_kernel_only_input_file",
                    f"./data/profiling/cpu_overhead/{{NETWORK_DEVICE}}/{{MODEL}}/cpu_overheads_kernel_only_stream{exp.weight_stream_layers}.csv",
                ]
            )

    return cmd


def _deep_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


class IncompleteRunError(RuntimeError):
    """The simulator exited successfully but did not finish every request."""


def extract_metrics(
    exp: Experiment,
    metrics_root: Path | None = None,
) -> dict[str, Any] | None:
    base = (metrics_root or METRICS_ROOT) / exp.name
    sys_path = base / "system_metrics.json"
    req_path = base / "request_metrics.csv"
    if not sys_path.exists() or not req_path.exists():
        return None

    with sys_path.open(encoding="utf-8") as f:
        system = json.load(f)
    with req_path.open(encoding="utf-8", newline="") as f:
        requests = list(csv.DictReader(f))

    def mean_col(key: str) -> float:
        vals: list[float] = []
        for request in requests:
            text = request.get(key)
            if text in (None, ""):
                continue
            try:
                value = float(text)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                vals.append(value)
        return sum(vals) / len(vals) if vals else float("nan")

    def numeric(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IncompleteRunError(f"{label} is not numeric: {value!r}")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise IncompleteRunError(f"{label} is not finite: {value!r}")
        return parsed

    def stat_section(name: str, required_keys: tuple[str, ...], required: bool) -> dict[str, Any]:
        section = system.get(name)
        if not isinstance(section, dict):
            if required:
                raise IncompleteRunError(f"{name} missing")
            return {}
        if "error" in section:
            if required:
                raise IncompleteRunError(f"{name} error: {section.get('error')}")
            return {}
        for key in required_keys:
            numeric(section.get(key), f"{name}.{key}")
        return section

    def stat(section: dict[str, Any], key: str) -> float:
        try:
            return numeric(section.get(key), key)
        except IncompleteRunError:
            return float("nan")

    sim = system.get("simulation_metadata", {})
    preempt = system.get("preemption_statistics", {})

    total_raw = sim.get("total_requests")
    completed_raw = sim.get("completed_requests")
    if total_raw is None or completed_raw is None:
        raise IncompleteRunError(
            "simulation_metadata missing total_requests/completed_requests"
        )
    total_requests = int(total_raw)
    completed_requests = int(completed_raw)
    if total_requests != exp.num_req:
        raise IncompleteRunError(
            f"simulated {total_requests} requests, expected {exp.num_req}"
        )
    if completed_requests != exp.num_req:
        raise IncompleteRunError(
            f"completed {completed_requests}/{exp.num_req} requests"
        )

    ttft = stat_section("ttft_statistics", ("mean",), required=True)
    tpot_required = exp.decode_tokens > 1
    tpot = stat_section("tpot_statistics", ("mean",), required=tpot_required)
    e2e = stat_section("request_e2e_time_statistics", ("mean",), required=True)
    tp = stat_section(
        "throughput_metrics",
        (
            "total_duration_ms",
            "requests_per_second",
            "tokens_per_second",
            "decode_tokens_per_second",
        ),
        required=True,
    )
    m2n = system.get("m2n_transfer_statistics", {})
    kv = system.get("kv_cache_transfer_statistics", {})
    af = system.get("af_pipeline_overlap_statistics", {})

    # 当前 PD-AF 输出没有 cluster_utilization 字段，缺测写 NaN，不写假 0。
    util = system.get("cluster_utilization", system.get("utilization_metrics", {}))
    prefill_util = float("nan")
    decode_attn_util = float("nan")
    decode_ffn_util = float("nan")
    if isinstance(util, dict):
        for key, value in util.items():
            key_l = key.lower()
            val = value.get("mean", value) if isinstance(value, dict) else value
            if not isinstance(val, (int, float)):
                continue
            if "prefill" in key_l:
                prefill_util = float(val)
            elif "attn" in key_l:
                decode_attn_util = float(val)
            elif "ffn" in key_l:
                decode_ffn_util = float(val)

    config_path = base / "config.json"
    effective_prefill_max = float("nan")
    effective_decode_attn_max = float("nan")
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            run_config = json.load(f)
        effective_prefill_max = _deep_get(
            run_config, "cluster_config.prefill_replica_scheduler_config_max_tokens_in_batch", float("nan")
        )
        effective_decode_attn_max = _deep_get(
            run_config,
            "cluster_config.decode_attn_replica_scheduler_config_max_tokens_in_batch",
            float("nan"),
        )

    makespan_ms = numeric(tp.get("total_duration_ms"), "throughput_metrics.total_duration_ms")
    out_tps = numeric(
        tp.get("decode_tokens_per_second"),
        "throughput_metrics.decode_tokens_per_second",
    )
    row = asdict(exp)
    row.update(
        {
            "h200_prefill_cards": exp.h200_prefill_cards,
            "h200_decode_cards": exp.h200_decode_cards,
            "h200_total_cards": exp.h200_total_cards,
            "hpp2_total_cards": exp.hpp2_total_cards,
            "p_card_ratio": round(exp.p_card_ratio, 4),
            "effective_prefill_max_tokens": effective_prefill_max,
            "effective_decode_attn_max_tokens": effective_decode_attn_max,
            "ttft_mean_ms": stat(ttft, "mean"),
            "ttft_p50_ms": stat(ttft, "p50"),
            "ttft_p90_ms": stat(ttft, "p90"),
            "ttft_p99_ms": stat(ttft, "p99"),
            "tpot_mean_ms": stat(tpot, "mean"),
            "tpot_p50_ms": stat(tpot, "p50"),
            "tpot_p90_ms": stat(tpot, "p90"),
            "tpot_p99_ms": stat(tpot, "p99"),
            "e2e_mean_ms": stat(e2e, "mean"),
            "e2e_p90_ms": stat(e2e, "p90"),
            "e2e_max_ms": stat(e2e, "max"),
            "requests_per_second": stat(tp, "requests_per_second"),
            "tokens_per_second": stat(tp, "tokens_per_second"),
            "decode_tokens_per_second": out_tps,
            "decode_tps_per_gpu": out_tps / exp.h200_total_cards if exp.h200_total_cards else float("nan"),
            "total_duration_ms": makespan_ms,
            "decode_attn_e2e_mean_ms": mean_col("cluster_decode_attn_e2e_time"),
            "decode_ffn_e2e_mean_ms": mean_col("cluster_decode_ffn_e2e_time"),
            "prefill_e2e_mean_ms": mean_col("prefill_e2e_time"),
            "ttft_kv_transfer_mean_ms": mean_col("ttft_kv_transfer"),
            "tpot_transfer_mean_ms": mean_col("tpot_transfer"),
            "m2n_total_ms": float(m2n.get("total_transfer_time_ms", 0.0)),
            "m2n_bytes": int(m2n.get("total_data_transferred_bytes", 0)),
            "m2n_transfers": int(m2n.get("total_transfers", 0)),
            "kv_total_ms": float(kv.get("total_transfer_time_ms", 0.0)),
            "kv_transfers": int(kv.get("total_transfers", 0)),
            "prefill_util": prefill_util,
            "decode_attn_util": decode_attn_util,
            "decode_ffn_util": decode_ffn_util,
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "preemption_events": int(preempt.get("total_preemption_events", 0)),
            "af_attn_active_ms": float(af.get("decode_attn_active_time_ms", 0.0)),
            "af_ffn_active_ms": float(af.get("decode_ffn_active_time_ms", 0.0)),
            "af_m2n_active_ms": float(af.get("m2n_active_time_ms", 0.0)),
            "af_overlap_savings_ms": float(af.get("pipeline_overlap_savings_ms", 0.0)),
            "af_ffn_hidden_ratio": float(af.get("ffn_hidden_ratio", 0.0)),
        }
    )
    return row


def _terminate_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        return
    try:
        proc.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def _terminate_all() -> None:
    with _RUNNING_LOCK:
        procs = list(_RUNNING_PROCS)
    for proc in procs:
        _terminate_proc(proc)


def write_publish_marker(row: dict[str, Any]) -> None:
    dst = Path(row["published_dir"])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "publish_metadata.json").write_text(
        json.dumps(
            {
                "attempt_dir": row.get("metrics_dir"),
                "run_fingerprint": row.get("run_fingerprint"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def published_matches(row: dict[str, Any]) -> bool:
    if not row:
        return False
    raw_dst = row.get("published_dir")
    if not isinstance(raw_dst, str) or not raw_dst.strip():
        return False
    dst = Path(raw_dst)
    meta_path = dst / "publish_metadata.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        meta.get("attempt_dir") == row.get("metrics_dir")
        and meta.get("run_fingerprint") == row.get("run_fingerprint")
    )


def ensure_published(row: dict[str, Any]) -> bool:
    if not row:
        return False
    if published_matches(row):
        return True
    raw_src = row.get("metrics_dir")
    raw_dst = row.get("published_dir")
    if (
        not isinstance(raw_src, str)
        or not raw_src.strip()
        or not isinstance(raw_dst, str)
        or not raw_dst.strip()
    ):
        return False
    src = Path(raw_src)
    dst = Path(raw_dst)
    if not src.is_dir():
        return False
    try:
        publish_metrics(src, dst)
        write_publish_marker(row)
    except Exception:
        return False
    return True


def withdraw_published(exp: Experiment, output_dir: Path) -> None:
    """Remove the stable published copy while a rerun is pending.

    Historical artifacts stay under attempts/; readers of the fixed
    outputs/.../<name> path must not see the previous successful run while the
    replacement is in progress.
    """
    published = METRICS_ROOT / exp.name
    if not published.exists():
        return
    withdrawn = output_dir / "withdrawn" / f"{exp.name}.{time.time_ns()}"
    withdrawn.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(published, withdrawn)
    except FileNotFoundError:
        pass


def publish_metrics(src: Path, dst: Path) -> None:
    """Copy one validated attempt to the stable published path.

    The attempt directory stays immutable so an older CSV row keeps pointing
    at the exact metrics it was produced from.  The published copy is built in
    a staging directory and renamed into place; a crash cannot expose a
    partially copied result.
    """
    if not src.is_dir():
        raise FileNotFoundError(f"attempt metrics directory missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = dst.with_name(f".{dst.name}.new.{time.time_ns()}")
    backup: Path | None = None
    try:
        shutil.copytree(src, staging)
        if dst.exists():
            backup = dst.with_name(f".{dst.name}.old.{time.time_ns()}")
            os.replace(dst, backup)
        os.replace(staging, dst)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and not dst.exists():
            os.replace(backup, dst)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def run_one(
    exp: Experiment,
    timeout_s: int,
    source: dict[str, Any],
    run_fp: str,
    output_dir: Path,
    training_threads: int = DEFAULT_TRAINING_THREADS,
    cache_dir: Path | None = None,
) -> tuple[str, dict[str, Any] | None, str | None, float]:
    if CANCEL_EVENT.is_set():
        return exp.name, None, "cancelled", 0.0

    def source_matches() -> bool:
        return current_source(exclude_paths=[output_dir]) == source

    if not source_matches():
        return (
            exp.name,
            None,
            "source changed before start; result rejected",
            0.0,
        )

    ATTEMPTS_ROOT.mkdir(parents=True, exist_ok=True)
    attempt_root = ATTEMPTS_ROOT / (
        f"{exp.name}_{time.time_ns()}_{os.getpid()}_{threading.get_ident()}"
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    log_path = attempt_root / "run.log"
    attempt_metrics_root = attempt_root / "qwen3_4b" / "offline_batch"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["WANDB_DISABLED"] = "true"
    env["VIDUR_DISABLE_WANDB"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")

    cmd = build_command(exp, attempt_root, training_threads, cache_dir)
    (attempt_root / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    t0 = time.time()
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )

    with log_path.open("w", encoding="utf-8") as logf:
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=REPO_ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
        except OSError as exc:
            return exp.name, None, f"spawn failed: {exc}", 0.0
        with _RUNNING_LOCK:
            _RUNNING_PROCS.add(proc)
            # Close the race with the main thread's KeyboardInterrupt handler:
            # if cancellation arrived while Popen was starting, the process is
            # now registered and must be terminated before we enter wait().
            cancelled_after_spawn = CANCEL_EVENT.is_set()
        if cancelled_after_spawn:
            _terminate_proc(proc)
            with _RUNNING_LOCK:
                _RUNNING_PROCS.discard(proc)
            return exp.name, None, "cancelled", round(time.time() - t0, 1)
        try:
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _terminate_proc(proc)
                rc = -1
                logf.write(f"\n[TIMEOUT] exceeded {timeout_s}s\n")
        finally:
            with _RUNNING_LOCK:
                _RUNNING_PROCS.discard(proc)

    elapsed = round(time.time() - t0, 1)
    if CANCEL_EVENT.is_set():
        return exp.name, None, "cancelled", elapsed
    if not source_matches():
        return exp.name, None, "source changed during run; result rejected", elapsed
    if rc != 0:
        tail = ""
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = " | ".join(lines[-4:])
        except OSError:
            pass
        return exp.name, None, f"exit={rc}; {tail}", elapsed
    try:
        row = extract_metrics(exp, attempt_metrics_root)
    except IncompleteRunError as exc:
        return exp.name, None, str(exc), elapsed
    except Exception as exc:  # noqa: BLE001 - keep the worker alive
        return exp.name, None, f"metrics error: {exc}", elapsed
    if row is None:
        return exp.name, None, "missing metrics files", elapsed
    attempt_result_dir = attempt_metrics_root / exp.name
    canonical_dir = METRICS_ROOT / exp.name
    row.update(source)
    row["config_fingerprint"] = config_fingerprint(exp)
    row["run_fingerprint"] = run_fp
    row["attempt_dir"] = str(attempt_root)
    row["metrics_dir"] = str(attempt_result_dir)
    row["published_dir"] = str(canonical_dir)
    row["predictor_cache_dir"] = str(cache_dir) if cache_dir else ""
    row["log_path"] = str(log_path)
    row["wall_seconds"] = elapsed
    return exp.name, row, None, elapsed


# ---------------------------------------------------------------------------
# 输出 / 汇总
# ---------------------------------------------------------------------------


def git_info(exclude_paths: Iterable[Path] = ()) -> dict[str, Any]:
    excluded = tuple(p.resolve() for p in exclude_paths)

    def _is_excluded(rel: str) -> bool:
        if not excluded:
            return False
        full = (REPO_ROOT / rel).resolve()
        return any(full == root or root in full.parents for root in excluded)

    def _run(*args: str, text: bool = True):
        try:
            kwargs = {"stderr": subprocess.DEVNULL}
            if text:
                kwargs.update(text=True, errors="replace")
            output = subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), *args],
                **kwargs,
            )
            return output.strip() if text else output
        except (OSError, subprocess.CalledProcessError):
            return "" if text else b""

    pathspec: list[str] = []
    for root in excluded:
        try:
            rel = root.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if rel == Path("."):
            continue
        pathspec.append(f":(exclude,top){rel.as_posix()}")
    source_scope = ["--", ".", *pathspec] if pathspec else []
    status = _run("status", "--porcelain", *source_scope)
    diff = _run("diff", "--binary", "HEAD", *source_scope, text=False)
    untracked_digest = hashlib.sha256()
    for rel in sorted(filter(None, _run("ls-files", "--others", "--exclude-standard", "-z").split("\0"))):
        if _is_excluded(rel):
            continue
        try:
            st = (REPO_ROOT / rel).stat()
        except OSError:
            continue
        untracked_digest.update(rel.encode("utf-8"))
        untracked_digest.update(str(st.st_size).encode("ascii"))
        untracked_digest.update(str(st.st_mtime_ns).encode("ascii"))
    return {
        "head": _run("rev-parse", "HEAD"),
        "short_head": _run("rev-parse", "--short", "HEAD"),
        "branch": _run("branch", "--show-current"),
        "last_commit": _run("log", "-1", "--oneline"),
        "dirty": bool(status),
        "status_porcelain": status,
        "diff_hash": _sha256_bytes(diff)[:16],
        "untracked_hash": untracked_digest.hexdigest()[:16],
        "available": bool(_run("rev-parse", "HEAD")),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "missing"


def environment_identity() -> dict[str, str]:
    dependencies = {
        "scikit-learn": _package_version("scikit-learn"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scipy": _package_version("scipy"),
    }
    payload = {
        "python_version": platform.python_version(),
        "python_executable": os.path.realpath(sys.executable),
        "dependencies": dependencies,
    }
    return {
        "source_python_version": payload["python_version"],
        "source_python_executable": payload["python_executable"],
        "source_sklearn_version": dependencies["scikit-learn"],
        "source_numpy_version": dependencies["numpy"],
        "source_pandas_version": dependencies["pandas"],
        "source_scipy_version": dependencies["scipy"],
        "source_environment_fingerprint": _sha256_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=False)
        )[:16],
    }


def current_source(
    git: dict[str, Any] | None = None,
    exclude_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Identity of the working tree that produced a result row."""
    git = git or git_info(exclude_paths=exclude_paths)
    status = git.get("status_porcelain", "")
    git_available = bool(git.get("head"))
    source = {
        "source_commit": git.get("head") or f"nogit-{NO_GIT_SESSION_ID}",
        "source_short_commit": git.get("short_head") or "nogit",
        "source_git_available": git_available,
        "source_nogit_session": "" if git_available else NO_GIT_SESSION_ID,
        "source_branch": git.get("branch") or "",
        "source_dirty": bool(git.get("dirty")),
        "source_status_hash": _sha256_text(status)[:16],
        "source_diff_hash": git.get("diff_hash", ""),
        "source_untracked_hash": git.get("untracked_hash", ""),
    }
    source.update(environment_identity())
    return source


def config_fingerprint(exp: Experiment) -> str:
    payload = json.dumps(
        asdict(exp),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def run_fingerprint(exp: Experiment, source: dict[str, Any]) -> str:
    payload = json.dumps(
        {"source": source, "config": asdict(exp)},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def source_fingerprint(source: dict[str, Any]) -> str:
    payload = json.dumps(
        source,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def pending_marker_active(path: Path, current_fingerprint: str | None) -> bool:
    """Whether a pending marker belongs to the current source snapshot.

    Corrupt or old-format markers are treated as active conservatively.  A
    valid marker from another source snapshot is stale and must not block a
    restored historical result.
    """
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    recorded = data.get("run_fingerprint") if isinstance(data, dict) else None
    if not isinstance(recorded, str) or not recorded:
        return True
    return current_fingerprint is not None and recorded == current_fingerprint


def load_existing_rows(results_csv: Path) -> list[dict[str, Any]]:
    if not results_csv.exists():
        return []
    with results_csv.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[row["name"]] = row
    return latest


def load_existing(results_csv: Path) -> dict[str, dict[str, Any]]:
    return latest_by_name(load_existing_rows(results_csv))


def _csv_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return value


def _existing_csv_header(results_csv: Path) -> list[str]:
    if not results_csv.exists():
        return ["name"]
    try:
        with results_csv.open(encoding="utf-8", newline="") as f:
            return next(csv.reader(f))
    except (OSError, StopIteration, csv.Error):
        return ["name"]


def save_csv(rows: list[dict[str, Any]], results_csv: Path) -> None:
    """Atomically publish the checkpoint.

    A crash during a truncate-and-write could otherwise leave a zero-byte or
    half-written CSV.  The temporary file lives on the same filesystem and is
    atomically renamed into place.  An empty result set still publishes a
    header-only CSV, which is used when no current-source result remains.
    """
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = _existing_csv_header(results_csv)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{results_csv.name}.",
        suffix=".tmp",
        dir=results_csv.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows([{k: _csv_safe(v) for k, v in row.items()} for row in rows])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, results_csv)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_int(value: Any, default: int = 1) -> int:
    parsed = _to_float(value, default=float("nan"))
    if math.isnan(parsed):
        return default
    return int(parsed)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_summary(rows: list[dict[str, Any]], path: Path, manifest: dict[str, Any], failures: list[tuple[str, str]]) -> None:
    all_rows = rows
    current_fps = manifest.get("current_fingerprints") or {
        e.get("name"): e.get("run_fingerprint")
        for e in manifest.get("experiments", [])
        if isinstance(e, dict) and e.get("name")
    }
    pending_set = set(manifest.get("pending", []))
    # Rankings and calibration views must use only rows produced by the same
    # source tree and experiment config.  Historical rows stay in the CSV and
    # in the provenance table, but they do not participate in the tables below.
    rows = [
        r for r in all_rows
        if r.get("run_fingerprint")
        and r["run_fingerprint"] == current_fps.get(r["name"])
        and r["name"] not in pending_set
    ]
    by_name = {r["name"]: r for r in rows}
    out: list[str] = []
    out.append("# Qwen3-4B PD-AF 实验汇总")
    out.append("")
    out.append(f"- 开始时间：{manifest.get('started_at', '—')}")
    out.append(f"- 结束时间：{manifest.get('finished_at', '—')}")
    out.append(f"- Frontier：`{manifest.get('git', {}).get('short_head', '—')}` "
               f"({manifest.get('git', {}).get('branch', '—')})"
               f"{'，工作树有未提交修改' if manifest.get('git', {}).get('dirty') else ''}")
    if manifest.get("selection_mode", "profile") == "profile":
        out.append(
            f"- profile：`{manifest.get('profile_requested', manifest.get('profile', '—'))}`；"
            f"workers={manifest.get('workers', '—')}"
        )
    else:
        out.append(
            f"- selection：`{manifest.get('selection_label', '—')}`；"
            f"workers={manifest.get('workers', '—')}"
        )
    cancelled = int(manifest.get("num_cancelled", 0) or 0)
    row_names = {r.get("name") for r in all_rows}
    pending_in_table = pending_set & row_names
    historical = max(0, len(all_rows) - len(rows) - len(pending_in_table))
    out.append(
        f"- 结果 {len(all_rows)} 条（历史尝试 {manifest.get('num_attempts', '—')}；"
        f"当前源码 {len(rows)}，历史 {historical}，待重跑 {len(pending_set)}；"
        f"本次运行 {manifest.get('num_run', '—')}，"
        f"缓存 {manifest.get('num_cached', '—')}）；失败 {len(failures)}"
        + (f"；中断取消 {cancelled}" if cancelled else "")
    )
    out.append("")

    verify = [r for r in rows if r.get("stage") == "verify"]
    if verify:
        out.append("## 基线校验")
        out.append("")
        out.append(_table(
            ["配置", "TTFT ms", "TPOT ms", "E2E ms", "输出 tok/s", "wall s"],
            [
                [
                    r["name"],
                    _fmt(_to_float(r.get("ttft_mean_ms"))),
                    _fmt(_to_float(r.get("tpot_mean_ms")), 3),
                    _fmt(_to_float(r.get("e2e_mean_ms"))),
                    _fmt(_to_float(r.get("decode_tokens_per_second")), 0),
                    _fmt(_to_float(r.get("wall_seconds"), 0.0), 1),
                ]
                for r in sorted(verify, key=lambda x: x["name"])
            ],
        ))
        out.append("")

    burst = [r for r in rows if r.get("stage") == "gpu_scale_burst"]
    if burst:
        out.append("## Burst 配比（32 请求，512:64）")
        out.append("")
        burst_sorted = sorted(
            burst,
            key=lambda x: (_to_int(x.get("h200_total_cards"), 0), -_to_float(x.get("decode_tokens_per_second"), 0.0)),
        )
        out.append(_table(
            ["配置", "H200", "P", "D", "AF", "TTFT", "TPOT", "E2E", "输出 tok/s", "tok/s/卡"],
            [
                [
                    r.get("name", "—"),
                    r.get("h200_total_cards", "—"),
                    f"{r.get('p_rep', '—')}×TP{r.get('p_tp', '—')}",
                    f"{r.get('d_rep', '—')}×TP{r.get('d_tp', '—')}",
                    r.get("af_mb", "—"),
                    _fmt(_to_float(r.get("ttft_mean_ms"))),
                    _fmt(_to_float(r.get("tpot_mean_ms")), 3),
                    _fmt(_to_float(r.get("e2e_mean_ms"))),
                    _fmt(_to_float(r.get("decode_tokens_per_second")), 0),
                    _fmt(_to_float(r.get("decode_tps_per_gpu")), 0),
                ]
                for r in burst_sorted
            ],
        ))
        out.append("")

    attn_parallel = [r for r in rows if r.get("stage") == "attn_stage_parallel"]
    if attn_parallel:
        out.append("## 多 attention stage/replica 并行")
        out.append("")
        out.append(_table(
            ["配置", "H200", "P", "D（副本×TP×PP）", "AF", "TTFT", "TPOT", "E2E", "输出 tok/s", "说明"],
            [
                [
                    r.get("name", "—"),
                    r.get("h200_total_cards", "—"),
                    f"{r.get('p_rep', '—')}×TP{r.get('p_tp', '—')}",
                    f"{r.get('d_rep', '—')}×TP{r.get('d_tp', '—')}×PP{_to_int(r.get('attn_pp'), 1)}",
                    r.get("af_mb", "—"),
                    _fmt(_to_float(r.get("ttft_mean_ms"))),
                    _fmt(_to_float(r.get("tpot_mean_ms")), 3),
                    _fmt(_to_float(r.get("e2e_mean_ms"))),
                    _fmt(_to_float(r.get("decode_tokens_per_second")), 0),
                    r.get("note", ""),
                ]
                for r in sorted(
                    attn_parallel,
                    key=lambda x: (_to_int(x.get("h200_total_cards"), 0), -_to_float(x.get("decode_tokens_per_second"), 0.0)),
                )
            ],
        ))
        out.append("")

    poisson = [r for r in rows if r.get("stage") == "gpu_scale_poisson"]
    if poisson:
        out.append("## Poisson 在线")
        out.append("")
        poi_sorted = sorted(
            poisson,
            key=lambda x: (
                _to_int(x.get("num_req"), 0),
                x["name"].split("_qps")[0],
                _to_float(x.get("qps"), 0.0),
            ),
        )
        out.append(_table(
            ["配置", "请求数", "QPS", "TTFT", "TPOT", "E2E", "输出 tok/s", "完成请求"],
            [
                [
                    r["name"],
                    r.get("num_req", "—"),
                    _fmt(_to_float(r.get("qps")), 0),
                    _fmt(_to_float(r.get("ttft_mean_ms"))),
                    _fmt(_to_float(r.get("tpot_mean_ms")), 3),
                    _fmt(_to_float(r.get("e2e_mean_ms"))),
                    _fmt(_to_float(r.get("decode_tokens_per_second")), 0),
                    r.get("completed_requests", "—"),
                ]
                for r in poi_sorted
            ],
        ))
        out.append("")

    m2n = [r for r in rows if r.get("stage") == "m2n_sensitivity"]
    if m2n:
        out.append("## M2N 敏感性")
        out.append("")
        out.append(_table(
            ["配置", "协议", "带宽 Gbps", "延迟 us", "M2N IB 累计 ms", "TPOT", "E2E", "输出 tok/s"],
            [
                [
                    r.get("name", "—"),
                    r.get("comm_label", "—"),
                    _fmt(_to_float(r.get("m2n_bw_gbps")), 0),
                    _fmt(_to_float(r.get("m2n_latency_ms")) * 1000.0, 0),
                    _fmt(_to_float(r.get("m2n_total_ms"))),
                    _fmt(_to_float(r.get("tpot_mean_ms")), 3),
                    _fmt(_to_float(r.get("e2e_mean_ms"))),
                    _fmt(_to_float(r.get("decode_tokens_per_second")), 0),
                ]
                for r in sorted(m2n, key=lambda x: x["name"])
            ],
        ))
        out.append("")
        out.append(
            "> HostBounce 的 PCIe 拷贝在 FFN CPU-overhead 中计费；"
            "本列只统计 M2N IB 传输事件。"
        )
        out.append("")

    stream = [r for r in rows if r.get("stage") == "weight_stream"]
    if stream:
        out.append("## 权重流式")
        out.append("")
        out.append(_table(
            ["配置", "流式层数", "FFN 副本", "AF", "TTFT", "TPOT", "E2E", "输出 tok/s"],
            [
                [
                    r.get("name", "—"),
                    r.get("weight_stream_layers", "—"),
                    r.get("f_rep", "—"),
                    r.get("af_mb", "—"),
                    _fmt(_to_float(r.get("ttft_mean_ms"))),
                    _fmt(_to_float(r.get("tpot_mean_ms")), 3),
                    _fmt(_to_float(r.get("e2e_mean_ms"))),
                    _fmt(_to_float(r.get("decode_tokens_per_second")), 0),
                ]
                for r in sorted(stream, key=lambda x: x["name"])
            ],
        ))
        out.append("")

    online = [r for r in rows if r.get("stage") == "online_long"]
    if online:
        out.append("## 400 请求多种子")
        out.append("")
        out.append(_table(
            ["配置", "QPS", "完成请求", "TTFT", "TPOT", "E2E", "输出 tok/s"],
            [
                [
                    r["name"],
                    _fmt(_to_float(r.get("qps")), 0),
                    r.get("completed_requests", "—"),
                    _fmt(_to_float(r.get("ttft_mean_ms"))),
                    _fmt(_to_float(r.get("tpot_mean_ms")), 3),
                    _fmt(_to_float(r.get("e2e_mean_ms"))),
                    _fmt(_to_float(r.get("decode_tokens_per_second")), 0),
                ]
                for r in sorted(online, key=lambda x: x["name"])
            ],
        ))
        out.append("")

    key_pairs = [
        ("16 卡 8:8", "g16_p8_d8_tp2"),
        ("16 卡 6:10", "g16_p6_d10_tp2"),
        ("16 卡 4:12", "g16_p4_d12_tp2"),
        ("12 卡 4:8", "g12_p4_d8_tp2"),
        ("8 卡 4:4 TP1", "g8_p4_d4_tp1"),
        ("16 卡 8:8 AF2", "g16_p8_d8_tp2_af2"),
        ("16 卡 8:8 AF4", "g16_p8_d8_tp2_af4"),
        ("16 卡 8:8 D8×TP1", "ap_g16_p4d8_tp1_af1"),
        ("16 卡 4:12 D12×TP1", "ap_g16_p2d12_tp1_af1"),
        ("24 卡 8:16 D8×TP2", "ap_g24_p4d8_tp2_af1"),
        ("32 卡 16:16 D8×TP2", "ap_g32_p8d8_tp2_af1"),
        ("32 卡 8:24 D12×TP2", "ap_g32_p4d12_tp2_af1"),
        ("16 卡 8:8 D×PP2（实验）", "ap_pp_g16_p4d2_tp2_pp2_af1"),
        ("16 卡 8:8 D×PP4（实验）", "ap_pp_g16_p4d1_tp2_pp4_af1"),
    ]
    key_rows = []
    for label, name in key_pairs:
        r = by_name.get(name)
        if r is None:
            continue
        key_rows.append([
            label,
            _fmt(_to_float(r.get("ttft_mean_ms"))),
            _fmt(_to_float(r.get("tpot_mean_ms")), 3),
            _fmt(_to_float(r.get("e2e_mean_ms"))),
            _fmt(_to_float(r.get("decode_tokens_per_second")), 0),
        ])
    if key_rows:
        out.append("## 关键对照")
        out.append("")
        out.append(_table(["配置", "TTFT ms", "TPOT ms", "E2E ms", "输出 tok/s"], key_rows))
        out.append("")

    if all_rows:
        out.append("## 数据来源")
        out.append("")
        provenance_rows = []
        for r in all_rows:
            name = r.get("name", "—")
            fp = r.get("run_fingerprint")
            if name in pending_set:
                state = "待重跑"
            elif not fp:
                state = "无指纹"
            elif fp == current_fps.get(name):
                state = "当前"
            else:
                state = "旧/不匹配"
            provenance_rows.append([
                name,
                r.get("source_short_commit", "—"),
                r.get("source_dirty", "—"),
                (r.get("source_environment_fingerprint") or "—")[:8],
                (fp or "—")[:12],
                state,
            ])
        out.append(_table(
            ["配置", "commit", "dirty", "env", "run_fingerprint", "当前源码"],
            provenance_rows,
        ))
        stale_count = sum(
            1 for row in provenance_rows if row[5] in ("旧/不匹配", "无指纹", "待重跑")
        )
        if stale_count:
            out.append("")
            out.append(
                f"> 注意：本表包含 {stale_count} 条旧源码或无指纹结果，"
                "不要把它们当作当前模拟器产物。"
            )
        out.append("")

    pending_only = sorted(pending_set - row_names)
    if pending_only:
        out.append("## 待重跑（尚无成功结果）")
        out.append("")
        for name in pending_only:
            out.append(f"- `{name}`")
        out.append("")

    if failures:
        out.append("## 失败")
        out.append("")
        for name, err in failures:
            out.append(f"- `{name}`：{err}")
        out.append("")

    path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------------
# 预检 / 主流程
# ---------------------------------------------------------------------------


class OutputDirLock:
    """Exclusive inter-process lock for one output directory.

    The lock covers the whole read-checkpoint -> run -> publish cycle.  Without
    it, two runners on the same directory would each replace the CSV from their
    own stale snapshot and one process would silently lose the other's rows.
    """

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / ".sweep.lock"
        self._fh = None
        self._locked = False

    def __enter__(self) -> "OutputDirLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def acquire(self) -> "OutputDirLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+", encoding="utf-8")
        if fcntl is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"等待输出目录锁：{self.path}", flush=True)
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            self._locked = True
        elif msvcrt is not None:  # pragma: no cover - Windows
            self._fh.seek(0)
            self._fh.write(" ")
            self._fh.flush()
            while True:
                try:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    self._locked = True
                    break
                except OSError:
                    time.sleep(0.2)
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if self._locked and fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            elif self._locked and msvcrt is not None:  # pragma: no cover - Windows
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._fh.close()
            self._fh = None
            self._locked = False


def _create_unique_run_dir(commit: str) -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"_{attempt + 1}"
        candidate = RUNS_ROOT / f"{stamp}_{commit}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"无法为 {stamp}_{commit} 创建唯一输出目录")


def resolve_output_dir(path: Path | None, *, create: bool) -> Path:
    """Resolve --output-dir: relative paths live under outputs/runs.

    Default runs use an exclusively-created timestamp/commit directory so two
    processes started in the same second cannot collide.  Explicit directories
    are created if needed and serialized by OutputDirLock in main().
    """
    if path is None:
        commit = git_info()["short_head"] or "nogit"
        if create:
            return _create_unique_run_dir(commit)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return RUNS_ROOT / f"{stamp}_{commit}"
    resolved = path.resolve() if path.is_absolute() else (RUNS_ROOT / path).resolve()
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def preflight(selection_label: str, stages: set[str], selected: list[Experiment]) -> None:
    errors: list[str] = []
    if not (REPO_ROOT / ".git").exists():
        errors.append(f"Frontier 不是 git 仓库：{REPO_ROOT}")
    if not (REPO_ROOT / "frontier" / "main.py").exists():
        errors.append(f"找不到 frontier/main.py：{REPO_ROOT}")
    git_probe = git_info()
    if not git_probe.get("head"):
        errors.append(
            "无法读取 git HEAD；请确认这是完整仓库且 git 可用 "
            "（无 git 环境不能安全复用实验缓存）"
        )
    for rel in REQUIRED_DATA:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"缺少 profiling 文件：{rel}")

    # The streamed-weight CPU-overhead tables are scenario-specific.  Missing
    # tables must fail before any run: the predictor would otherwise price the
    # HostBounce PCIe segment as zero for that stream depth.
    missing_stream_tables: dict[str, list[str]] = {}
    for exp in selected:
        if exp.skip_cpu_overhead or exp.weight_stream_layers not in (1, 2, 3):
            continue
        rel = (
            "data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/"
            f"cpu_overheads_kernel_only_stream{exp.weight_stream_layers}.csv"
        )
        if not (REPO_ROOT / rel).exists():
            missing_stream_tables.setdefault(rel, []).append(exp.name)
    for rel, names in sorted(missing_stream_tables.items()):
        errors.append(
            f"缺少流式 CPU-overhead 表：{rel}（配置：{', '.join(names[:4])}"
            f"{' 等' if len(names) > 4 else ''}）"
        )
    missing_modules: list[str] = []
    for module_name in RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - 环境检查
            missing_modules.append(f"{module_name} ({exc.__class__.__name__})")
    if missing_modules:
        errors.append("当前 Python 缺少运行时依赖：" + ", ".join(missing_modules))

    if errors:
        print("预检失败：", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "\n可用以下命令创建环境：\n"
            "  cd Frontier\n"
            "  uv venv .venv\n"
            "  uv pip install --python .venv/bin/python -e .",
            file=sys.stderr,
        )
        raise SystemExit(2)

    git = git_info()
    print("预检通过")
    print(f"  Frontier : {REPO_ROOT}")
    print(f"  commit   : {git['short_head']} ({git['branch']})")
    print(f"  dirty    : {git['dirty']}")
    print(f"  selection: {selection_label}")
    print(f"  stages   : {', '.join(sorted(stages))}")
    print(f"  Python   : {sys.executable} ({platform.python_version()})")
    if git["branch"] != "qwen3-4b-pd-af-simulator":
        print(f"  WARNING  : 当前分支不是 qwen3-4b-pd-af-simulator，而是 {git['branch']}")


def select_configs(args: argparse.Namespace) -> list[Experiment]:
    all_cfgs = build_all_configs()
    by_name = {c.name: c for c in all_cfgs}
    if len(by_name) != len(all_cfgs):
        raise RuntimeError("实验名重复，请检查 build_all_configs()")

    if args.suite == "calibration10":
        selected = [c for c in all_cfgs if c.name in CALIBRATION10_NAMES]
    elif args.stage:
        stage_set = set(args.stage)
        unknown = stage_set - {c.stage for c in all_cfgs}
        if unknown:
            raise SystemExit(f"未知 stage：{sorted(unknown)}；可选：{sorted({c.stage for c in all_cfgs})}")
        selected = [c for c in all_cfgs if c.stage in stage_set]
        # calibration tier 只在 --suite calibration10 或显式 --name 时跑，
        # 避免 --stage gpu_scale_poisson 意外带上 400 请求长跑。
        # experimental tier 同理，需要 --include-experimental 或 --name。
        if not args.name:
            selected = [c for c in selected if c.tier != "calibration"]
        if not args.name and not args.include_experimental:
            selected = [c for c in selected if c.tier != "experimental"]
    elif args.name:
        # 显式点名时忽略 profile/tier，任何 tier 的配置都能单独跑。
        selected = list(all_cfgs)
    else:
        selected = [c for c in all_cfgs if c.stage in PROFILE_STAGES[args.profile]]
        if args.profile == "smoke":
            selected = [c for c in selected if c.name in SMOKE_NAMES]
        elif args.profile == "core":
            allowed_tiers = {"core"}
            if args.include_experimental:
                allowed_tiers.add("experimental")
            selected = [c for c in selected if c.tier in allowed_tiers]
        elif args.profile == "full":
            allowed_tiers = {"core", "full"}
            if args.include_experimental:
                allowed_tiers.add("experimental")
            selected = [c for c in selected if c.tier in allowed_tiers]

    if args.name:
        name_set = set(args.name)
        unknown = name_set - {c.name for c in all_cfgs}
        if unknown:
            raise SystemExit(f"未知实验名：{sorted(unknown)}")
        selected = [c for c in selected if c.name in name_set]

    return selected


def main() -> int:
    global OUT_ROOT, METRICS_ROOT, ATTEMPTS_ROOT, RESULTS_CSV
    parser = argparse.ArgumentParser(
        description="Qwen3-4B PD-AF 一键实验脚本（Frontier 工作树，支持断点续跑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_STAGES), default="core",
                        help="实验集：smoke=5 个冒烟点；core=配比+在线+关键敏感性；full=再加 AF2/AF4 全对照（默认 core）")
    parser.add_argument("--stage", action="append", default=[],
                        help="只跑指定 stage，可重复；指定后忽略 profile 的 stage 过滤，但仍受 --name 过滤")
    parser.add_argument("--name", action="append", default=[],
                        help="只跑指定实验名，可重复")
    parser.add_argument("--suite", choices=["calibration10"], default=None,
                        help="跑官方 10-case 校准集；结果可与 tests/performance/qwen3_4b_pdaf/calibrated_cases.csv 对比")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录；默认 outputs/runs/<时间戳>_<commit>，相对路径按 outputs/runs 解析")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"并行 worker 数（默认 {DEFAULT_WORKERS}；内存 <8GB 建议不超过 3）")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                        help=f"单次仿真超时秒数（默认 {DEFAULT_TIMEOUT_S}）")
    parser.add_argument("--training-threads", type=int, default=DEFAULT_TRAINING_THREADS,
                        help=f"每个仿真 sklearn GridSearchCV 的 n_jobs（默认 {DEFAULT_TRAINING_THREADS}）")
    parser.add_argument("--include-experimental", action="store_true",
                        help="包含 tier=experimental 的配置（例如 DECODE_ATTN PP>1，需要模拟器先支持）")
    parser.add_argument("--force", action="store_true",
                        help="强制重跑选中配置；保留同一目录内其他 stage 的已有结果")
    parser.add_argument("--list", action="store_true",
                        help="只列出配置数量和名称，不运行")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将执行的命令，不创建输出目录、不跑仿真")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers 必须 >= 1")
    if args.training_threads < 1:
        parser.error("--training-threads 必须 >= 1")

    selected = select_configs(args)
    if not selected:
        print("没有匹配的实验。用 --list 查看配置。")
        return 0

    if args.suite:
        selection_mode = "suite"
        selection_label = f"suite={args.suite}"
    elif args.stage:
        selection_mode = "stage"
        selection_label = "stages=" + ",".join(args.stage)
    elif args.name:
        selection_mode = "name"
        selection_label = f"names={len(args.name)}"
    else:
        selection_mode = "profile"
        selection_label = f"profile={args.profile}"
    if args.name and selection_mode != "name":
        selection_label += f"; names={len(args.name)}"

    experimental = [c for c in selected if c.tier == "experimental" or c.attn_pp > 1]
    if experimental:
        print(
            f"WARNING: 选中 {len(experimental)} 个实验性配置（DECODE_ATTN PP>1）。"
            "当前 PD-AF 校验可能仍要求 num_pipeline_stages=1；"
            "模拟器未放开时会直接失败。"
        )

    if args.list:
        counts: dict[str, int] = {}
        for c in selected:
            counts[c.stage] = counts.get(c.stage, 0) + 1
        print(f"selection={selection_label}  selected={len(selected)}")
        for stage, count in sorted(counts.items()):
            print(f"  {stage:22s}: {count}")
        for c in selected:
            print(f"  {c.name:48s} stage={c.stage:20s} tier={c.tier:12s} "
                  f"H200={c.h200_total_cards:2d} AF={c.af_mb} PP={c.attn_pp} ws={c.weight_stream_layers}")
        return 0

    stages = {c.stage for c in selected}
    preflight(selection_label, stages, selected)

    if args.dry_run:
        preview_dir = resolve_output_dir(args.output_dir, create=False)
        OUT_ROOT = preview_dir / "outputs"
        METRICS_ROOT = OUT_ROOT / "qwen3_4b" / "offline_batch"
        RESULTS_CSV = preview_dir / "sweep_results.csv"
        preview_cache = preview_dir / "predictor_cache" / source_fingerprint(
            current_source()
        )
        print(f"\n将执行 {len(selected)} 个配置（dry-run，不创建目录）：")
        for c in selected:
            print(f"  {c.name}")
        print("\n第一条命令：")
        print(
            shlex.join(
                build_command(
                    selected[0],
                    training_threads=args.training_threads,
                    cache_dir=preview_cache,
                )
            )
        )
        return 0

    output_dir = resolve_output_dir(args.output_dir, create=True)

    OUT_ROOT = output_dir / "outputs"
    METRICS_ROOT = OUT_ROOT / "qwen3_4b" / "offline_batch"
    ATTEMPTS_ROOT = output_dir / "attempts"
    RESULTS_CSV = output_dir / "sweep_results.csv"
    current_csv = output_dir / "current_results.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    old_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame) -> None:
        raise KeyboardInterrupt

    # SIGTERM must use the same cancellation path as Ctrl-C; children run in
    # separate sessions, so default termination would leave them alive.
    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        lock = OutputDirLock(output_dir).acquire()
    except KeyboardInterrupt:
        signal.signal(signal.SIGTERM, old_sigterm)
        print("\n等待目录锁时收到终止信号，退出。", file=sys.stderr)
        return 130
    except BaseException:
        signal.signal(signal.SIGTERM, old_sigterm)
        raise
    try:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        run_manifest_path = output_dir / "manifests" / f"{run_id}.json"
        run_summary_path = output_dir / "summaries" / f"{run_id}.md"
        run_commands_path = output_dir / "commands" / f"{run_id}.txt"
        pending_dir = output_dir / "pending"
        for parent in (
            run_manifest_path.parent,
            run_summary_path.parent,
            run_commands_path.parent,
            pending_dir,
        ):
            parent.mkdir(parents=True, exist_ok=True)

        def pending_path(exp: Experiment) -> Path:
            return pending_dir / f"{exp.name}.json"

        # Results are reusable only when both the experiment config and the
        # working-tree identity match.  A new commit or an uncommitted simulator
        # edit must not silently inherit an older row.
        CANCEL_EVENT.clear()
        git = git_info(exclude_paths=[output_dir])
        source = current_source(git, exclude_paths=[output_dir])
        source_fp = source_fingerprint(source)
        predictor_cache_dir = output_dir / "predictor_cache" / source_fp
        predictor_cache_dir.mkdir(parents=True, exist_ok=True)
        all_cfgs = build_all_configs()
        all_run_fps = {c.name: run_fingerprint(c, source) for c in all_cfgs}
        config_fps = {c.name: config_fingerprint(c) for c in selected}
        run_fps = {c.name: all_run_fps[c.name] for c in selected}

        history_rows = load_existing_rows(RESULTS_CSV)
        rows_by_name: dict[str, dict[str, Any]] = latest_by_name(history_rows)
        current_by_name: dict[str, dict[str, Any]] = {}
        for existing_row in history_rows:
            if existing_row.get("run_fingerprint") == all_run_fps.get(existing_row["name"]):
                current_by_name[existing_row["name"]] = existing_row

        # The fixed published path may belong to another source snapshot after
        # switching A -> B -> A.  Restore it from the matching immutable attempt.
        cfg_by_name = {c.name: c for c in all_cfgs}
        pending_now: set[str] = set()
        for marker in pending_dir.glob("*.json"):
            if pending_marker_active(marker, all_run_fps.get(marker.stem)):
                pending_now.add(marker.stem)
            else:
                marker.unlink(missing_ok=True)
        # A pending case must not expose a fixed published copy, even when the
        # current selection does not include it.
        for name in pending_now:
            exp = cfg_by_name.get(name)
            if exp is not None:
                withdraw_published(exp, output_dir)
        for name in list(current_by_name):
            if name in pending_now:
                continue
            if not ensure_published(current_by_name[name]):
                del current_by_name[name]

        # Withdraw published copies whose source does not match the current
        # snapshot; they remain recoverable from attempts/ if the source returns.
        for name in rows_by_name:
            if name in current_by_name or name in pending_now:
                continue
            exp = cfg_by_name.get(name)
            if exp is not None:
                withdraw_published(exp, output_dir)

        def cache_state(exp: Experiment) -> str:
            if pending_marker_active(pending_path(exp), run_fps[exp.name]):
                return "pending"
            row = current_by_name.get(exp.name)
            if row is not None:
                if ensure_published(row):
                    return "current"
                del current_by_name[exp.name]
            return "stale" if exp.name in rows_by_name else "missing"

        stale = [c for c in selected if cache_state(c) == "stale"]
        to_run = [c for c in selected if args.force or cache_state(c) != "current"]

        manifest: dict[str, Any] = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "run_id": run_id,
            "run_manifest": str(run_manifest_path),
            "run_summary": str(run_summary_path),
            "run_commands": str(run_commands_path),
            "profile": None if selection_mode != "profile" else args.profile,
            "profile_requested": args.profile,
            "selection_mode": selection_mode,
            "selection_label": selection_label,
            "suite": args.suite,
            "stage_filters": list(args.stage),
            "name_filters": list(args.name),
            "include_experimental": bool(args.include_experimental),
            "force": bool(args.force),
            "stages": sorted(stages),
            "workers": args.workers,
            "training_threads": args.training_threads,
            "timeout_s": args.timeout,
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "repo_root": str(REPO_ROOT),
            "output_dir": str(output_dir),
            "git": git,
            "source": source,
            "source_fingerprint": source_fp,
            "predictor_cache_dir": str(predictor_cache_dir),
            "current_fingerprints": all_run_fps,
            "num_selected": len(selected),
            "num_run": len(to_run),
            "num_cached": len(selected) - len(to_run),
            "num_stale": 0 if args.force else len(stale),
            "pending": [],
            "num_pending": 0,
            "num_ok": 0,
            "num_failed": 0,
            "num_cancelled": 0,
            "interrupted": False,
            "failures": [],
            "experiments": [
                {
                    **asdict(c),
                    "config_fingerprint": config_fps[c.name],
                    "run_fingerprint": run_fps[c.name],
                }
                for c in selected
            ],
        }
        manifest_path = output_dir / "manifest.json"
        manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
        manifest_path.write_text(manifest_text, encoding="utf-8")
        run_manifest_path.write_text(manifest_text, encoding="utf-8")
        commands_text = "\n\n".join(
            shlex.join(
                build_command(
                    c,
                    training_threads=args.training_threads,
                    cache_dir=predictor_cache_dir,
                )
            )
            for c in selected
        )
        (output_dir / "commands.txt").write_text(commands_text, encoding="utf-8")
        run_commands_path.write_text(commands_text, encoding="utf-8")

        def preferred_row(name: str) -> dict[str, Any] | None:
            return current_by_name.get(name) or rows_by_name.get(name)

        def ordered_rows() -> list[dict[str, Any]]:
            ordered = [
                preferred_row(c.name)
                for c in selected
                if preferred_row(c.name) is not None
            ]
            seen = {row["name"] for row in ordered}
            for name in rows_by_name:
                if name not in seen:
                    row = preferred_row(name)
                    if row is not None:
                        ordered.append(row)
            return ordered

        def current_rows_now() -> list[dict[str, Any]]:
            pending = {
                p.stem
                for p in pending_dir.glob("*.json")
                if pending_marker_active(p, all_run_fps.get(p.stem))
            }
            return [
                r for r in current_by_name.values()
                if r["name"] not in pending
            ]

        print()
        print(f"输出目录 : {output_dir}")
        print(f"配置数   : {len(selected)}（缓存 {len(selected) - len(to_run)}，待跑 {len(to_run)}）")
        if stale and not args.force:
            print(f"旧结果   : {len(stale)} 个源码/配置指纹不匹配，将重跑")
        pending_before = [
            c
            for c in selected
            if pending_marker_active(pending_path(c), run_fps[c.name])
        ]
        if pending_before:
            print(f"待重跑   : {len(pending_before)} 个上次中断/失败未完成")
        print(f"workers  : {args.workers}")
        print()

        failures: list[tuple[str, str]] = []
        ok_names: set[str] = set()
        done = 0
        interrupted = False
        # Mark every selected attempt before launching it.  If this process is
        # interrupted or dies, the old row is not treated as current on the next
        # resume; the run is retried instead.
        for exp in to_run:
            pending_path(exp).write_text(
                json.dumps(
                    {"run_fingerprint": run_fps[exp.name], "run_id": run_id},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            withdraw_published(exp, output_dir)

        # Withdraw stale current-source rows immediately; report scripts must not
        # keep reading the previous current_results.csv while reruns are pending.
        save_csv(current_rows_now(), current_csv)

        t_start = time.time()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        futures: dict[concurrent.futures.Future, Experiment] = {}
        try:
            # Keep submission inside the interrupt handler and register every
            # future incrementally.  Ctrl-C during submission must still cancel
            # queued work and terminate already-started children.
            for c in to_run:
                future = pool.submit(
                    run_one,
                    c,
                    args.timeout,
                    source,
                    run_fps[c.name],
                    output_dir,
                    args.training_threads,
                    predictor_cache_dir,
                )
                futures[future] = c
            for future in concurrent.futures.as_completed(futures):
                done += 1
                exp = futures[future]
                try:
                    name, row, err, elapsed = future.result()
                except Exception as exc:  # noqa: BLE001 - report worker crash
                    name, row, err, elapsed = exp.name, None, f"worker crashed: {exc}", 0.0
                if not err:
                    # Publishing on the main thread closes the Ctrl-C window where
                    # a worker had already written the fixed metrics path but the
                    # pending marker and CSV were never updated.
                    if current_source(exclude_paths=[output_dir]) != source:
                        err = "source changed before publish; result rejected"
                    else:
                        try:
                            publish_metrics(
                                Path(row["metrics_dir"]),
                                Path(row["published_dir"]),
                            )
                            write_publish_marker(row)
                        except Exception as exc:  # noqa: BLE001 - report as failure
                            err = f"publish failed: {exc}"
                    if not err and current_source(exclude_paths=[output_dir]) != source:
                        withdraw_published(exp, output_dir)
                        err = "source changed during publish; result rejected"
                if err:
                    failures.append((name, err))
                    withdraw_published(exp, output_dir)
                    print(f"[{done}/{len(to_run)}] FAIL {name}: {err} ({elapsed}s)", flush=True)
                else:
                    rows_by_name[name] = row
                    current_by_name[name] = row
                    history_rows.append(row)
                    ok_names.add(name)
                    save_csv(history_rows, RESULTS_CSV)
                    pending_path(exp).unlink(missing_ok=True)
                    print(
                        f"[{done}/{len(to_run)}] OK   {name:46s} ({elapsed:6.1f}s) | "
                        f"TTFT={row['ttft_mean_ms']:7.2f}ms TPOT={row['tpot_mean_ms']:6.3f}ms "
                        f"E2E={row['e2e_mean_ms']:7.1f}ms out_tps={row['decode_tokens_per_second']:8.0f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            interrupted = True
            print(
                "\n收到 Ctrl-C / SIGTERM：取消排队任务、终止在跑仿真，并保存已完成结果。",
                file=sys.stderr,
            )
            CANCEL_EVENT.set()
            for future in futures:
                future.cancel()
            _terminate_all()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        # A pending case must never leave a published fixed path behind, including
        # results that finished in a worker while the main loop was interrupted.
        for exp in selected:
            if pending_marker_active(pending_path(exp), run_fps[exp.name]):
                withdraw_published(exp, output_dir)

        ordered = ordered_rows()
        save_csv(history_rows, RESULTS_CSV)
        pending_names = sorted(
            p.stem
            for p in pending_dir.glob("*.json")
            if pending_marker_active(p, all_run_fps.get(p.stem))
        )
        pending_set = set(pending_names)
        current_rows = current_rows_now()
        save_csv(current_rows, current_csv)

        num_cancelled = max(0, len(to_run) - len(ok_names) - len(failures))
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["num_ok"] = len(ok_names)
        manifest["num_failed"] = len(failures)
        manifest["num_cancelled"] = num_cancelled if interrupted else 0
        manifest["interrupted"] = interrupted
        manifest["num_results"] = len(ordered)
        manifest["num_attempts"] = len(history_rows)
        manifest["num_current_results"] = len(current_rows)
        manifest["pending"] = pending_names
        manifest["num_pending"] = len(pending_names)
        manifest["current_results_csv"] = str(current_csv)
        manifest["failures"] = [{"name": n, "error": e} for n, e in failures]
        manifest["elapsed_s"] = round(time.time() - t_start, 1)
        manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
        manifest_path.write_text(manifest_text, encoding="utf-8")
        run_manifest_path.write_text(manifest_text, encoding="utf-8")

        summary_path = output_dir / "summary.md"
        write_summary(ordered, summary_path, manifest, failures)
        write_summary(ordered, run_summary_path, manifest, failures)

        print()
        print(
            f"完成：本次运行 {len(ok_names)} 成功，{len(failures)} 失败，"
            f"{manifest['num_cancelled']} 取消；"
            f"目录共 {len(ordered)} 条结果（缓存 {len(selected) - len(to_run)}）"
        )
        print(f"全部结果 : {RESULTS_CSV}")
        print(f"当前源码 : {current_csv}")
        print(f"汇总报告 : {summary_path}")
        print(f"运行清单 : {manifest_path}")
        if interrupted:
            return 130
        if failures:
            print("\n失败列表：")
            for name, err in failures:
                print(f"  - {name}: {err}")
            return 1
        return 0


    finally:
        lock.release()
        signal.signal(signal.SIGTERM, old_sigterm)

if __name__ == "__main__":
    raise SystemExit(main())
