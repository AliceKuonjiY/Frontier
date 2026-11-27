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


def _index(rows, label):
    by_name = {}
    for row in rows:
        name = row["name"]
        if name not in CASES:
            continue
        if name in by_name:
            raise ValueError(f"Duplicate {label} case: {name}")
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
        pair["after_prefill_util"] = after[name]["prefill_util"]
        pairs.append(pair)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"after_fix_results.csv": [after[name] for name in CASES],
              "before_after_comparison.csv": pairs}
    # Stage both in the destination filesystem. Validation/serialization failures
    # cannot truncate an existing table; replacement is atomic per file.
    with TemporaryDirectory(dir=out) as tmp:
        for filename, rows in tables.items():
            with (Path(tmp) / filename).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        for filename in tables:
            (Path(tmp) / filename).replace(out / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        with args.before.open(encoding="utf-8-sig", newline="") as b, args.after.open(encoding="utf-8-sig", newline="") as a:
            before, after = list(csv.DictReader(b)), list(csv.DictReader(a))
        publish(before, after, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Failed to publish comparison: {exc}\n")


if __name__ == "__main__":
    main()
