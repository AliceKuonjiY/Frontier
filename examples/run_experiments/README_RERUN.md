# Qwen3-4B PD-AF 一键复跑说明

这个目录里有一键实验脚本，改完 `Frontier` 模拟器后不需要手工拼 100 多条命令。

## 0. 参数说明

| 参数                        | 默认值                 | 作用                          | 注意                                             |
| --------------------------- | ---------------------- | ----------------------------- | ------------------------------------------------ |
| --profile {smoke,core,full} | core                   | 选择实验集                    | full 不含 experimental、calibration、online_long |
| --stage STAGE               | 无                     | 只跑指定 stage，可重复        | 指定后忽略 profile 的 stage 过滤                 |
| --name NAME                 | 无                     | 只跑指定实验名，可重复        | 显式点名时忽略 profile/tier，任何 tier 都能跑    |
| --suite {calibration10}     | 无                     | 跑官方 10-case 校准集         | 优先于 --stage 和 --profile                      |
| --include-experimental      | 关闭                   | 加入 tier=experimental 的配置 | 主要是 DECODE_ATTN PP>1，当前模拟器可能仍拒绝    |
| --output-dir PATH           | runs/<时间戳>_<commit> | 指定输出目录                  | 复用同一目录即断点续跑                           |
| --workers N                 | 3                      | 并行仿真进程数                | 8 GB 内存建议不超过 3                            |
| --timeout N                 | 3600                   | 单次仿真超时秒数              | 400 请求长跑可设 7200                            |
| --force                     | 关闭                   | 忽略已有 CSV，全部重跑        | 会重写 CSV，见下面“注意”                         |
| --list                      | -                      | 只列出选中的配置              | 不跑仿真                                         |
| --dry-run                   | -                      | 预检并打印配置和第一条命令    | 不创建输出目录、不跑仿真                         |
| -h/--help                   | -                      | 帮助                          |                                                  |

## 1. 最快用法

```bash
# 从工作区根目录：
cd Frontier/examples/run_experiments
# 或者，如果已经在 Frontier 仓库根目录：
# cd examples/run_experiments
./run_experiments.sh
```

默认跑 `core` 实验集（87 个配置）：

- 2 个基线校验（8 请求，stream 0/3）
- 31 个 Burst 配比（2/4/6/8/12/16 张 H200，含 6 个 AF2/AF4 对照）
- 7 个多 attention replica 并行点（8:8 / 4:12 / 6:10 / 8:16 / 16:8 / 8:24 / 16:16）
- 33 个 Poisson 在线点（64 请求，QPS 20/50/100/150/250）
- 4 个 M2N 敏感性点（AF1，Direct vs HostBounce，0/20 µs）
- 10 个权重流式点（0/1/2/3 层，AF1，单/双 FFN 副本，流式下的 P:D 配比）

每个配置写到独立目录：

```text
Frontier/outputs/runs/<YYYYmmdd_HHMMSS>_<commit>/
├── manifest.json          # git commit / dirty / Python / 实验清单
├── commands.txt           # 实际执行的完整命令
├── sweep_results.csv      # 增量写入；中断后可用同一目录续跑
├── summary.md             # 自动生成的汇总表
├── logs/<name>.log        # 单次仿真日志
└── outputs/qwen3_4b/offline_batch/<name>/
    ├── system_metrics.json
    ├── request_metrics.csv
    └── config.json
```

## 2. 常用命令

```bash
# 改完模拟器后，建议先跑 5 个冒烟点，约 2–5 分钟
./run_experiments.sh --profile smoke --workers 3

# 全量实验（含 AF2/AF4 全对照，约 1.5–3 小时，取决于机器）
./run_experiments.sh --profile full --workers 3

# 只重跑配比主表
./run_experiments.sh --stage gpu_scale_burst --workers 4

# 只跑一个配置
./run_experiments.sh --name g16_p8_d8_tp2

# 只跑 Poisson 在线
./run_experiments.sh --stage gpu_scale_poisson --workers 3

# 只跑多 attention replica 并行（当前可跑）
./run_experiments.sh --stage attn_stage_parallel --workers 3

# 加上单 replica 内 PP>1 的实验配置（需要模拟器先放开 PD-AF 限制）
./run_experiments.sh --stage attn_stage_parallel --include-experimental --workers 2

# 只跑 400 请求 × 5 种子长跑（很慢，显式指定）
./run_experiments.sh --stage online_long --workers 2

# 跑官方 10-case 校准集，结果可直接喂给 compare.py
./run_experiments.sh --suite calibration10 --workers 3

# 只看配置，不跑
./run_experiments.sh --list

# 只打印将执行的命令，不创建目录
./run_experiments.sh --dry-run
```

## 3. profile / stage / suite 的区别

| 名称                     | 内容                                                 | 大约配置数 |
| ------------------------ | ---------------------------------------------------- | ---------: |
| `--profile smoke`        | 基线 + 4 个关键配比（含 AF4 回归）                   |          5 |
| `--profile core`（默认） | 配比 + 多 attention replica + 在线 + 关键 M2N + 流式 |         87 |
| `--profile full`         | core + AF2/AF4 全对照 + 全部 M2N/流式                |        136 |
| `--suite calibration10`  | 官方 `tests/performance/qwen3_4b_pdaf` 10-case       |         10 |
| `--stage online_long`    | 400 请求 × 5 种子（不在任何 profile 里默认跑）       |         20 |

`--stage` 会覆盖 profile 的 stage 过滤；`--name` 再在结果里做交集。  
例如 `--stage gpu_scale_burst --name g16_p8_d8_tp2` 只跑一个配比点。

### 多 attention stage / replica 并行

`attn_stage_parallel` 分两类：

1. **多 replica 并行（当前可直接跑，tier=core）**
   保持 H200 总数不变，增加 decode-attn 副本数、降低每副本 TP：

   | 配置                   | H200 | P     | D      |
   | ---------------------- | ---: | ----- | ------ |
   | `ap_g16_p4d8_tp1_af1`  |   16 | 4×TP2 | 8×TP1  |
   | `ap_g16_p2d12_tp1_af1` |   16 | 2×TP2 | 12×TP1 |
   | `ap_g16_p6d10_tp1_af1` |   16 | 6×TP1 | 10×TP1 |
   | `ap_g24_p4d8_tp2_af1`  |   24 | 4×TP2 | 8×TP2  |
   | `ap_g24_p8d4_tp2_af1`  |   24 | 8×TP2 | 4×TP2  |
   | `ap_g32_p4d12_tp2_af1` |   32 | 4×TP2 | 12×TP2 |
   | `ap_g32_p8d8_tp2_af1`  |   32 | 8×TP2 | 8×TP2  |

   `--stage attn_stage_parallel` 默认会跑这 7 个 AF1 点，外加 2 个
   AF2 变体（`ap_g16_p4d8_tp1_af2`、`ap_g16_p2d12_tp1_af2`）和 2 个
   AF4 变体（`ap_g16_p4d8_tp1_af4`、`ap_g16_p2d12_tp1_af4`）；
   `--profile core` 只包含 7 个 AF1 点。

2. **单 replica 内 PP>1（tier=experimental）**
   例如 `ap_pp_g16_p4d2_tp2_pp2_af1` 表示 D 侧 2 个 replica、
   每个 replica `TP2 × PP2`。当前 PD-AF 校验要求
   `decode_attn_replica_config_num_pipeline_stages=1`，所以这些配置
   只有模拟器放开限制后才能跑；默认不选，需加 `--include-experimental`。

运行后 CSV 会多一列 `attn_pp`；`H200` 列按
`P_replica×P_TP + D_replica×D_TP×attn_pp` 计算。

## 4. 修改模拟器后怎么复跑

脚本直接使用当前工作树：

```text
cwd      = Frontier
PYTHONPATH = Frontier
python   = Frontier/.venv/bin/python
```

因此只要改的是 `Frontier/frontier/...`，不需要 `pip install`，也不需要提交。
每次运行的 `manifest.json` 会记录：

```json
{
  "git": {
    "head": "...",
    "short_head": "...",
    "branch": "qwen3-4b-pd-af-simulator",
    "dirty": true,
    "status_porcelain": " M frontier/..."
  }
}
```

如果工作树是 dirty，报告里可以直接写“基于 `<commit>` + 未提交修改”。

## 5. 断点续跑

默认每次运行都会新建 `outputs/runs/<时间戳>_<commit>/`，不会复用旧结果。  
`--output-dir` 的相对路径按 `Frontier/outputs/runs/` 解析。  
如果中途 Ctrl-C 或某台机器重启，用同一个目录续跑：

```bash
./run_experiments.sh --output-dir 20261128_101500_9e4fa41
```

已出现在 `sweep_results.csv` 里的配置会跳过；要强制重跑选中的配置
（同一目录内其他 stage 的结果会保留）：

```bash
./run_experiments.sh --output-dir 20261128_101500_9e4fa41 --force
```

## 6. 和官方 10-case 校准集对比

`Frontier/tests/performance/qwen3_4b_pdaf/` 里有一份固定的 10-case 校准集。
先跑：

```bash
./run_experiments.sh --suite calibration10 --output-dir calib_after
```

然后：

```bash
cd /home/yhu/Desktop/scale/Frontier
.venv/bin/python -m tests.performance.qwen3_4b_pdaf.compare \
  tests/performance/qwen3_4b_pdaf/calibrated_cases.csv \
  outputs/runs/calib_after/sweep_results.csv \
  /tmp/pdaf_compare
```

`compare.py` 会检查 10 个 case 是否齐全、请求数是否完整、拓扑/AF/流式/QPS/seed
是否匹配，并在全部通过后原子发布对比表。注意它要求：

- `total_requests` 和 `completed_requests` 来自 `system_metrics.json`；
- `prefill_util` 用 `NA` 或 `[0,1]` 小数；
- 400 请求 Poisson 用 `pois_g16_tp2_af4_qps250_n400`，不要拿 64 请求点代替。

本脚本的 CSV 已包含这些列；缺测的利用率会写 `NA`。

## 7. 环境准备

脚本会按以下顺序找 Python：

1. `PYTHON_BIN`
2. `scale/.venv/bin/python`
3. `Frontier/.venv/bin/python`
4. `/tmp/opencode/frontier-venv/bin/python`
5. 系统 `python3`

推荐在 `Frontier` 下建 uv 环境：

```bash
cd /home/yhu/Desktop/scale/Frontier
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

如果依赖缺失，脚本会在预检阶段直接报错，不会跑一半才失败。

## 8. 口径与已知限制

- **AF2/AF4**：当前仍受 issue #6 影响。`core` 只保留少量 AF 对照点；
  默认配比排名请用 AF1。AF4 的 M2N 总字节已与 AF1 对齐，但 attention
  按微批串行执行，净收益为负。
- **M2N**：默认 `p2p_protocol=host_bounce`，366 Gbps 只计 IB 段；
  每层 PCIe 段由 HPP2 CPU-overhead 表计。不要再叠加 122.44 Gbps
  的单段有效带宽。
- **KV**：当前仍按 NVLink 3172.5 Gbps 计，12/16 卡跨节点时偏乐观
  （issue #10）。
- **HPP2 SRAM**：仍按 192 MiB 名义容量；170 MB 可用口径尚未落地
  （issue #7），全常驻结论需保留余量警告。
- **HPP2 FFN 表**：是 SOL 估算，不是实测；绝对 FFN 时间不可信
  （issue #9）。
- **利用率**：当前 `system_metrics.json` 没有 cluster utilization 字段，
  CSV 写 `NA`，不要当 0 用。
- **AF4 400 请求**：官方 10-case 里的 `pois_g16_tp2_af4_qps250_n400`
  是回归对照，不代表推荐配置；它的 E2E 会随时间增长，不能声称稳态容量。

## 9. 新增实验

实验矩阵在 `run_pdaf_experiments.py` 的 `build_all_configs()` 里。  
新增一个点只要加一条 `_add(...)`，并给它一个唯一的 `name`、`stage` 和
`tier`：

```python
_add(
    cfgs,
    "g16_p8_d8_tp2_af1_qps300",
    "gpu_scale_poisson",
    (4, 2),
    (4, 2),
    num_req=128,
    arrival_pattern="poisson",
    qps=300.0,
    tier="core",
)
```

- `tier="core"`：默认 `core` 会跑；
- `tier="full"`：只有 `--profile full` 或显式 `--name/--stage` 会跑；
- `tier="calibration"`：只有 `--suite calibration10` 或显式 `--name` 会跑；
- `tier="experimental"`：只有 `--include-experimental` 或显式 `--name` 会跑；
- `tier="long"`：只有 `--stage online_long` 会跑。

改完先跑：

```bash
./run_experiments.sh --list
./run_experiments.sh --dry-run
```

确认配置数和命令无误后再正式跑。

## 10. 结果怎么写进报告

`summary.md` 已经按 stage 分好表，包含：

- 基线 TTFT/TPOT/E2E/输出 tok/s；
- Burst 配比的 H200 卡数、P/D 拓扑、AF、TTFT、TPOT、E2E、tok/s/卡；
- Poisson 每个 QPS 点的 TTFT/TPOT/E2E/完成请求数；
- M2N 协议/带宽/延迟对比；
- 权重流式的层数、FFN 副本、AF 对比；
- 关键对照：16 卡 8:8 / 6:10 / 4:12、12 卡 4:8、8 卡 TP1、AF1/AF2/AF4。

写新报告时建议：

1. 先引用 `manifest.json` 的 commit/dirty 状态；
2. 用 `summary.md` 的数字，不要手工抄旧报告；
3. 明确 AF4 是“对照/回归”，不是推荐；
4. 12 卡 4:8 同时给 `vs 16 卡 8:8` 和 `vs 16 卡 4:12` 两个比例；
5. 在线容量只引用 `online_long` 或官方 400 请求点，不引用 64 请求 QPS 250；
6. 保留 issue #7/#8/#9/#10 的限制说明。

## 11. 目录中的历史结果

- `outputs/runs/`：新脚本的正式输出目录，每次运行独立，带 `manifest.json`。
- 旧版 105 点报告包在仓库外的 `report/pdaf_sweep/`
  （`run_extended_sweep.py`、`sweep_extended_results.csv`、`figures/`）。
  它是 Windows 路径和旧 AF4 artifact，仅保留用于对照，不要直接复跑。
