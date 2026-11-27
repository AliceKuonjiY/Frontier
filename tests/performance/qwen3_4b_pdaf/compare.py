"""Publish the two calibrated sweep tables only after all ten pairs validate.

CLI inputs are CSVs from the baseline and candidate runs, not output files.
The baseline may be a larger sweep, but must include the 400-request case.
"""
import argparse
import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory


CASES = (
    "verify_baseline_stream0", "verify_baseline_stream3", "g16_p8_d8_tp2",
    "g16_p8_d8_tp2_af4", "g12_p4_d8_tp2", "g8_p4_d4_tp1", "g8_p4_d4_tp2",
    "ws3_g16_p8d8_af4_f1", "ws3_g16_p8d8_af4_f2", "pois_g16_tp2_af4_qps250_n400",
)


def expected_requests(name):
    if name.startswith("verify_baseline_"):
        return 8
    return 400 if name == "pois_g16_tp2_af4_qps250_n400" else 32


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
        if not str(util).strip():
            raise ValueError(f"Missing prefill_util for {name}; use NA if unavailable")
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
