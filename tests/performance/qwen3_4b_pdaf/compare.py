"""Publish the two calibrated sweep tables only after all ten pairs validate.

CLI inputs are CSVs from the baseline and candidate runs, not output files.
The baseline may be a larger sweep, but must include the 400-request case.
"""
import argparse
import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory


# Prefill replicas, decode-attention replicas, H200 TP, FFN replicas, AF depth, streamed layers.
CASES = {
    "verify_baseline_stream0": (1, 1, 8, 1, 1, 0), "verify_baseline_stream3": (1, 1, 8, 1, 1, 3),
    "g16_p8_d8_tp2": (4, 4, 2, 1, 1, 0), "g16_p8_d8_tp2_af4": (4, 4, 2, 1, 4, 0),
    "g12_p4_d8_tp2": (2, 4, 2, 1, 1, 0), "g8_p4_d4_tp1": (4, 4, 1, 1, 1, 0),
    "g8_p4_d4_tp2": (2, 2, 2, 1, 1, 0), "ws3_g16_p8d8_af4_f1": (4, 4, 2, 1, 4, 3),
    "ws3_g16_p8d8_af4_f2": (4, 4, 2, 2, 4, 3), "pois_g16_tp2_af4_qps250_n400": (4, 4, 2, 1, 4, 0),
}


def expected_requests(name):
    if name.startswith("verify_baseline_"):
        return 8
    return 400 if name == "pois_g16_tp2_af4_qps250_n400" else 32


def expected_parameters(name):
    p, d, tp, f, af, stream = CASES[name]
    poisson = name.startswith("pois_")
    return dict(p_rep=p, d_rep=d, p_tp=tp, d_tp=tp, f_tp=16, f_rep=f, af_mb=af,
                weight_stream_layers=stream, arrival_pattern="poisson" if poisson else "burst",
                qps=250 if poisson else 0, seed=42, prefill_tokens=512, decode_tokens=64, d_batch=8)


def parameters_from_config(cfg):
    c, g = cfg["cluster_config"], cfg["request_generator_config"]
    interval, lengths = g["interval_generator_config"], g["length_generator_config"]
    return dict(p_rep=c["prefill_cluster_num_replicas"], d_rep=c["decode_attn_cluster_num_replicas"],
                p_tp=c["prefill_replica_config"]["attn_tensor_parallel_size"],
                d_tp=c["decode_attn_replica_config"]["attn_tensor_parallel_size"],
                f_rep=c["decode_ffn_cluster_num_replicas"], f_tp=c["decode_ffn_replica_config"]["moe_tensor_parallel_size"],
                af_mb=c["decode_attn_af_pipeline_num_micro_batch"], d_batch=c["decode_attn_micro_batch_size"],
                weight_stream_layers=c["decode_ffn_replica_config"]["streamed_ffn_layers"],
                prefill_tokens=lengths["prefill_tokens"], decode_tokens=lengths["decode_tokens"],
                arrival_pattern="burst" if interval["name"]=="static" else interval["name"],
                qps=interval.get("qps", 0), seed=interval["seed"])


def _index(rows, label):
    by_name = {}
    for row in rows:
        name = row["name"]
        if name not in CASES:
            continue
        if name in by_name:
            raise ValueError(f"Duplicate {label} case: {name}")
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Malformed {label} row for {name}")
        for field in ("total_requests", "completed_requests"):
            if str(row.get(field)) != str(expected_requests(name)):
                raise ValueError(f"Incomplete {label} case {name}: {field}={row.get(field)!r}, "
                                 f"expected {expected_requests(name)}")
        for field, expected in expected_parameters(name).items():
            value = row.get(field)
            if value is None or (value != expected if isinstance(expected, str) else float(value) != expected):
                raise ValueError(f"Mismatched {label} case {name}: {field}={value!r}, expected {expected!r}")
        by_name[name] = row
    missing = [name for name in CASES if name not in by_name]
    if missing:
        raise ValueError(f"Missing {label} cases: {', '.join(missing)}")
    return by_name


def publish(before_rows, after_rows, output_dir):
    before = _index(before_rows, "before")
    after = _index(after_rows, "after")
    pairs = []
    for name in CASES:
        pair = {"name": name}
        for source, label, digits in (("ttft_mean_ms", "ttft", 4), ("tpot_mean_ms", "tpot", 4),
                                      ("decode_tokens_per_second", "out_tps", 2)):
            b, a = float(before[name][source]), float(after[name][source])
            if not all(math.isfinite(v) and v > 0 for v in (b, a)):
                raise ValueError(f"Invalid {source} for {name}: before={b}, after={a}")
            suffix = "" if label == "out_tps" else "_ms"
            pair[f"before_{label}{suffix}"] = round(b, digits)
            pair[f"after_{label}{suffix}"] = round(a, digits)
            pair[f"{label}_pct"] = round((a - b) / b * 100, 3)
        prefill = float(after[name]["prefill_e2e_mean_ms"])
        if not math.isfinite(prefill) or prefill < 0:
            raise ValueError(f"Invalid prefill E2E for {name}: {prefill}")
        pair["after_prefill_e2e_ms"] = round(prefill, 4)
        util = after[name]["prefill_util"]
        if util != "NA" and not 0 <= float(util) <= 1:
            raise ValueError(f"Invalid prefill_util for {name}: expected NA or a fraction in [0, 1]")
        pair["after_prefill_util"] = util
        pairs.append(pair)

    out = Path(output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Choose a new output directory; preserving {out}")
    tables = {"after_fix_results.csv": [after[name] for name in CASES],
              "before_after_comparison.csv": pairs}
    # Publish one immutable directory, not two independent file replacements.
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "batch"
        stage.mkdir()
        for filename, rows in tables.items():
            with (stage / filename).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        stage.rename(out)


def _read_csv(stream):
    reader = csv.DictReader(stream, strict=True)
    header = reader.fieldnames
    if not header or not all(header) or len(set(header)) != len(header):
        raise ValueError(f"Missing or duplicate column name in {stream.name}")
    rows = []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Wrong column count in {stream.name}, line {reader.line_num}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        with args.before.open(encoding="utf-8-sig", newline="") as b, args.after.open(encoding="utf-8-sig", newline="") as a:
            before, after = _read_csv(b), _read_csv(a)
        publish(before, after, args.output)
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        parser.exit(1, f"Failed to publish comparison: {exc}\n")


if __name__ == "__main__":
    main()
