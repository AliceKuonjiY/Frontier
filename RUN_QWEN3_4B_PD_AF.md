# Qwen3-4B PD-AF 仿真运行手册

## 已安装内容

Profiling 数据（来自 `scale/qwen3-4b-pd-af.zip`）已安装到：

```text
data/profiling/compute/hpp2/Qwen3-4B/linear_op_kernel_only.csv
data/profiling/compute/h200/Qwen3-4B/{linear_op.csv,linear_op_kernel_only.csv,
                                      attention.csv,attention_kernel_only.csv}
data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only.csv
data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only_stream{1,2,3}.csv
data/profiling/network/a2f_h200_to_hpp2/m2n_transfer_lut.csv
data/profiling/network/hpp2_16rank/{all_reduce.csv,weight_stream_lut.csv}
data/profiling/network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv
```

模型配置：

```text
data/config/models/Qwen3-4B.json
```

运行脚本：

```text
examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh
```

## 运行

在 Frontier 仓库根目录创建一次 uv 环境：

```bash
cd /home/yhu/Desktop/scale/Frontier
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e ".[test]"
```

然后运行：

```bash
cd /home/yhu/Desktop/scale/Frontier
export PYTHONPATH=$PWD
PYTHON_BIN=.venv/bin/python \
  bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh
```

也可以先激活 venv，再直接用 `bash`：

```bash
source .venv/bin/activate
bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh
```

常用覆盖参数：

```bash
PYTHON_BIN=.venv/bin/python \
NUM_REQUESTS=8 PREFILL_TOKENS=512 DECODE_TOKENS=64 ARRIVAL_PATTERN=burst \
RUN_ID=my_run bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh

# 让请求按生成的时间到达，名义到达率为每秒 1 个请求
PYTHON_BIN=.venv/bin/python ARRIVAL_PATTERN=poisson QPS=1 RUN_ID=poisson1 \
bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh

# 默认 WEIGHT_STREAM_LAYERS=2（严格二进制 170 MiB 可用 + 10 MiB overhead 下最小可行档位）；
# 2/3 可行，0/1 默认 fail fast
PYTHON_BIN=.venv/bin/python WEIGHT_STREAM_LAYERS=3 RUN_ID=stream3 \
bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh

# 仅诊断对照：显式放宽预算为名义 192 MiB / 0 overhead 运行全常驻（会输出 WARNING）
PYTHON_BIN=.venv/bin/python WEIGHT_STREAM_LAYERS=0 \
HPP2_USABLE_MEMORY_MB=192 HPP2_PLANNING_OVERHEAD_MB=0 RUN_ID=diag_stream0 \
bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh
```

Metrics 输出到：

```text
outputs/qwen3-4b-pd-af/qwen3_4b/offline_batch/<RUN_ID>/
├── request_metrics.csv
├── system_metrics.json
└── op_precision_metadata.csv
```

## Baseline 结果（8 个请求同时到达，每个输入 512、输出 64 个 token）

> **SRAM 预算口径说明（Issue #7）**：HPP2 单 Die 物理 APC DM 为 `192 MiB`（`201.33 MB`），SCALE 运行时固定保留 `22 MiB` 后可用上限为 **`170 MiB`**（`178.26 MB`），预留 **`10 MiB`**（4×80 KiB Bank0 碎片/规划余量）后有效权重+暂存+激活预算为 **`160 MiB`**（`167.77 MB`）。因此 `WEIGHT_STREAM_LAYERS=0/1` 均不可行（默认 fail fast），最小可行配置为 **`WEIGHT_STREAM_LAYERS=2`**（脚本当前默认值），`WEIGHT_STREAM_LAYERS=3` 亦可行。下方保留的 `stream-0` 数据仅为 Issue #7 修正前的历史理论对照，新版 `stream-2` baseline 与配比结果待后续统一重跑更新。

默认 `ARRIVAL_PATTERN=burst` 让所有请求在时刻 0 到达，该模式下 `QPS` 不参与调度。下面两组权重策略沿用 PR #1 的 P→A NVLink 参考参数（3172.5 Gbit/s，启动延迟 0.0208 ms）。比较 GPU 与 GPU+LPU 时，两端应使用相同的到达序列和请求长度。

`system_metrics.json` 的 `throughput_metrics.decode_tokens_per_second` 是整轮生成的输出 token 数除以从首个请求到达到最后一个请求完成的时间。`tokens_per_second` 使用同一分母，分子同时计入输入和输出。本轮两种口径的分子分别是 512 和 4,608；画解码输出吞吐图时取前者。

```text
TTFT                 21.300 ms
TPOT                  1.661 ms
request E2E         125.926 ms
输出 tokens/s       4,065.87
总 tokens/s        36,592.79  （含输入和输出）
requests/s            63.53  （仿真聚合值）
DECODE_ATTN E2E      90.89 ms / 63 个后续 decode step
DECODE_FFN E2E        9.46 ms / 63 个后续 decode step
KV transfer           1.69 ms 总计（8 次传输）
M2N transfer          4.06 ms 总计（4536 次传输，IB 段；PCIe 段在 FFN residence 的 CPU overhead 中）
```

`DECODE_FFN E2E 9.46 ms` 是 HostBounce residence，包含每层 FFN 输入/输出的 PCIe CPU overhead，不是 FFN 计算时间。  
每步 FFN 计算相比 H200 attention 很小；对这个 4B dense 模型，AF 分离是
**attention 主导**。HPP2 FFN 集群仍有价值，但更适合作为容量/通信研究，
而不是延迟瓶颈。

全常驻 smoke（`1 request，prefill 64，decode 4`）：

```text
TTFT 8.736 ms, TPOT 1.395 ms, DECODE_ATTN E2E 4.053 ms, FFN E2E 0.064 ms
```

流式 3 层 smoke（同一 workload，33 层常驻 + 3 层流式）：

```text
TTFT 8.736 ms, TPOT 2.004 ms, DECODE_ATTN E2E 4.053 ms, FFN E2E 1.891 ms
```

流式 3 层跑 8-request baseline：

```text
TTFT              21.300 ms
TPOT               2.270 ms
request E2E      164.297 ms
输出 tokens/s   3,116.31
总 tokens/s    28,046.76  （含输入和输出）
DECODE_ATTN E2E   90.895 ms / 63 个后续 decode step
DECODE_FFN E2E    47.830 ms / 63 个后续 decode step
```

流式 3 层是一阶上界：权重流式时间被分摊到 36 层的 CPU overhead 表里，
没有模拟逐层流式加载与 FFN 计算的重叠。

## 已做的 Frontier 改动

1. **HPP2 device / node SKU 与 SRAM 预算口径**
   - `DeviceSKUType.HPP2`、`HPP2DeviceSKUConfig`（每 die 名义 `192 MiB` APC DM、SCALE 可用 `usable_memory_mb=170.0` MiB、`planning_overhead_mb=10.0` MiB，380 TFLOPS FP16）。
   - `NodeSKUType.HPP2_16RANK`，8 卡 = 16 rank。
   - `ReplicaConfig` 与 `ClusterConfig`（`decode_ffn`）支持显式覆盖 `usable_memory_mb` / `planning_overhead_mb`（均为二进制 MiB = $1024^2$ 字节，与 `total_memory_gb` 二进制 GiB 保持一致）。
   - `MemoryPlanner` 与 `qwen3_4b_pd_af.sh` 默认按 `170 MiB` 可用 + `10 MiB` overhead 执行 fail-fast 检查，并读取 `linear_op_kernel_only.csv` 中的 `sram_fit_with_10mb_overhead` / `weight_streaming_required`；TP16 下 `stream-0/1`（`160.31 MiB + 10 MiB = 170.31 MiB > 170 MiB`）被拒绝，`stream-2`（`155.86 + 10 = 165.86 MiB`，余 `4.14 MiB`）与 `stream-3`（`151.41 + 10 = 161.41 MiB`，余 `8.59 MiB`）可行。仅当用户显式放宽 `usable_memory_mb` / `planning_overhead_mb` 时允许以诊断模式运行并打印 `WARNING`。

2. **混合精度 profiling 元数据**
   - 新增可选 CSV 列 `profiling_precision_by_op`。
   - `ProfilingMetadata`、CSV 解析器和
     `QuantizationManager.register_profiling_metadata()` 现在支持按算子精度。
   - 元数据注册只覆盖文件中实际存在的算子，因此 FFN 表不会再把
     `emb`/`attn_rope`/norms 标成 FP8。
   - `op_precision_metadata.csv` 已验证：`mlp_up_proj`/`mlp_down_proj`/
     `attn_pre_proj`/`attn_post_proj` = FP8，`mlp_act`/norms/`attn_rope`/`emb`
     = BF16，全部 `data_source=profiling`（没有 2 倍 approximation）。

3. **DECODE_FFN 的 CPU overhead 与跨集群隔离**
   - CPU 表过滤与去重签名在 FFN 角色下改用 `moe_tensor_parallel_size`，
     因此 TP16 HPP2 表能被正确加载。
   - 共享模型管理器按 `(family, network_device|model|tp|csv)` 隔离 CPU
     overhead 模型，避免 `DECODE_FFN`（HPP2）的 PCIe host-bounce 与流式权重
     CPU overhead 泄露给同属 `kernel_only` family 的 `DECODE_ATTN`（H200）。
   - 运行脚本在 `WEIGHT_STREAM_LAYERS=1|2|3` 时保留 `{NETWORK_DEVICE}/{MODEL}`
     路径占位符，确保只有 `hpp2_16rank` 加载流式表。

4. **FP8 常驻权重与名义/可用双口径内存核算**
   - `ParamCounter` 和 `system_metrics.model_weight_memory` 按配置的 MLP
     权重宽度计算（FP8 为 1 byte），不再固定按 2 bytes。
   - 在 TP16 配置下，每个 HPP2 rank 的 36 层 FFN 权重共 168,099,840 字节（`160.31 MiB` / `168.10 MB`）。流式 2 层时，34 层常驻（`151.41 MiB`）+ 1 层暂存（`4.45 MiB`）合计 `155.86 MiB`（`163.43 MB`），加 `10 MiB` planning overhead 为 `165.86 MiB`（余 `4.14 MiB`）；流式 3 层时，33 层常驻（`146.95 MiB`）+ 1 层暂存（`4.45 MiB`）合计 `151.41 MiB`（`158.76 MB`），加 `10 MiB` overhead 为 `161.41 MiB`（余 `8.59 MiB`）。
   - `system_metrics.json` 的 `model_weight_memory.DECODE_FFN` 同时输出 nominal（`192 MiB`）与 usable（`170 MiB`）口径下的利用率（含 staging buffer 与 +10 MiB overhead），`memory_utilization_percent` 口径为 `resident_plus_buffer / usable`，并通过 `memory_budget`、`weight_residency` 与 `profiling_sram_contract` 完整记录核算分项。`DECODE_FFN` 初始化日志 `[ACTIVATION_MEMORY_CAPACITY]` 同步打印上述全部分项。

5. **HPP2 M2N / CPU overhead 口径**
   - M2N 统一按 **HostBounce 分段** 解释：`p2p_protocol=host_bounce`，`memory_bandwidth_gbps=366` 只计 IB 段；每层 FFN 输入/输出的 PCIe 拷贝由 HPP2 CPU-overhead 表的 `prepare_inputs_e2e` / `process_model_outputs` 按层计。
   - 两段串行，不要再用 122.44 Gbps 的 HostBounce 单段有效带宽，否则会重复计 PCIe。
   - 如果实测路径是 Direct RDMA，则 CPU 表应清零、M2N 用 366 Gbps；当前脚本默认按 HostBounce。

## 从 LUT 拟合的 analytical 参数

| 路径 | 设置 |
|---|---|
| M2N HostBounce（分段） | IB 366 Gbps，0 ms latency，关闭 p2p speedup；每层 HPP2 PCIe 23 GB/s 在 CPU overhead 表 |
| KV prefill→decode_attn（NVLink） | 3172.5 Gbps，0.0208 ms latency |
| HPP2 ring all-reduce | 2984 Gbps 节点内，0 latency |
| H200 clusters | 使用同一套 analytical CC 设置；未安装 H200 all-reduce 表 |

> HPP2 M2N 采用 **HostBounce 分段口径**：`analytical_m2_n_transfer_config_memory_bandwidth_gbps=366` 只计 IB 段；每层 FFN 的输入/输出 PCIe 拷贝（23 GB/s）由 HPP2 CPU-overhead 表的 `prepare_inputs_e2e` / `process_model_outputs` 按层计。两段串行，不要再叠加 122.44 Gbps 的 HostBounce 单段有效带宽，否则会重复计 PCIe。

## 注意事项

- HPP2 FFN 表是 **SOL 理论值**，不是实测；H200 attention/linear 表是
  2026-11-26 实测。
- 模拟器按每个 HPP2 die 严格二进制 `170 MiB` 可用 + `10 MiB` planning overhead（有效 `160 MiB` = 20 个完整 256 KiB bank，相对名义 `192 MiB` 扣除了 SCALE 运行时保留的 `22 MiB`）执行容量规划。注意上游 `linear_op_kernel_only.csv` 中的 `rank_fp8_weight_mb=168.100` 为十进制 MB（$10^6$ B），而 `sram_headroom_mb=1.900` 混用了十进制权重与 `170`，实际二进制权重为 `160.31 MiB`；在严格二进制 `170 MiB / 10 MiB` 下 `stream-0/1` 不可行、`stream-2/3` 可行。最终能否部署仍需核对激活和工作区的实际峰值。
- H200 CPU overhead 表没有安装；H200 host 项按 0 计并打 warning。
  HPP2 CPU overhead 已使用。
- `moe_grouped_gemm`/`moe_*` 的 warning 来自共享默认 MoE 路径；
  Qwen3-4B 是 dense，不会执行这些算子。
- Attention/MTP 常驻内存仍按 2 bytes 近似；只有 MLP 项跟随 FP8。
  H200 容量不受这个影响。
- `uv pip install -e ".[test]"` 会安装仿真器依赖：
  `numpy`、`pandas`、`scikit-learn`、`scipy`、`plotly`、`fasteners>=0.19`、
  `ddsketch`、`pyyaml` 和 `pytest`。
