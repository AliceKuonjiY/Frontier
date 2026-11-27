# Qwen3-4B PD-AF ten-case calibration

These are **simulator** measurements, not a hardware or a steady-state benchmark.
The configuration matches the segmented HostBounce recipe (IB 366 Gbit/s, zero
startup latency; HPP2 PCIe transfers in the CPU-overhead model). Each request
has 512 input and 64 output tokens. Burst means simultaneous arrivals. The
Poisson case has 400 requests, offered QPS 250 and seed 42. `AF` denotes the
configured number of microbatches, not a measured speedup.

## Publishing a comparison

From the repository root, with Python 3.10+:

```sh
python -m tests.performance.qwen3_4b_pdaf.compare before.csv candidate.csv results/
```

Both inputs need `name`, `ttft_mean_ms`, `tpot_mean_ms`, `decode_tokens_per_second`.
Supply `total_requests` and `completed_requests` from each run’s `system_metrics.json` on both sides, not the configured count.
The candidate also needs `prefill_e2e_mean_ms` and `prefill_util` (`NA` if absent).
The historical 105-case sweep alone is not a complete baseline: append the
separate **400**-request Poisson row, not the 64-request run. A larger input is
accepted; all ten named cases must exist exactly once on each side. Missing,
duplicate and invalid pairs fail before either published CSV is replaced.
Use a **new output directory** for each publication; previous batches are immutable.
The two CSVs become visible together in a single same-filesystem directory rename.

Output throughput is total generated decode tokens divided by total simulated
wall time (including drain); TTFT and TPOT are means over completed requests.
The older pre-overlap sweep and its 19%/22% streaming results are **not**
measurements of the current scheduler. Do not combine them into one ranking.

## Recorded sample, not a universal ranking

[Ten calibrated cases](calibrated_cases.csv) were run against the simulator based on
`f4540bd` with the role-TP selection in `1d91121`, including the merged overlap
scheduler (#4), norm/collective accounting (#5), and segmented HostBounce.
All 640 requests completed. The subsequent typed-ownership guard selects the
same Qwen3-4B compute rows: every measured H200 norm at TP 1/2/4/8 declares
`post_attention_layernorm`, replicated at operator TP1. It matters for sparse
or partial profiles, not for this sample.

- Under 32-request burst, 16 H200 (8:8, TP2, AF1) delivered 13,699 tok/s;
  12 H200 (4:8, TP2, AF1) delivered 11,823. At 16 H200, AF4 delivered **4,587**
  tok/s despite measurable stage overlap. These are finite simulated workloads.
- With three streamed layers at AF4, doubling HPP2 from 8 to 16 cards improved
  throughput from 3,002 to 4,576 tok/s (+52.45%). Not an equal-resource comparison.
- Under Poisson QPS 250 with AF4, the 400-request sample delivered 4,732 tok/s.
  Mean E2E increased from 526 ms (first 50) through 2,099 ms (middle 50) to
  3,649 ms (last 50). Do not claim this configuration sustains the offered load.
