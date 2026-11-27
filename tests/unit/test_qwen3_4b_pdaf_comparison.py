import csv
import subprocess
import sys

import pytest

from tests.performance.qwen3_4b_pdaf.compare import CASES, expected_parameters, expected_requests, publish


def _rows():
    return [dict(name=name, total_requests=expected_requests(name), completed_requests=expected_requests(name), ttft_mean_ms=3, tpot_mean_ms=2, decode_tokens_per_second=5,
                 prefill_e2e_mean_ms=3, prefill_util="NA", **expected_parameters(name)) for name in CASES]


@pytest.mark.parametrize("bad_side", ["before", "after", "duplicate", "invalid"])
def test_failed_comparison_preserves_both_previous_tables(tmp_path, bad_side):
    before, after = _rows(), _rows()
    files = [tmp_path / x for x in ("before_after_comparison.csv", "after_fix_results.csv")]
    for file in files:
        file.write_bytes(b"last complete run")
    if bad_side == "duplicate":
        after.append(after[-1])
    elif bad_side == "invalid":
        after[-1]["tpot_mean_ms"] = float("nan")
    else:
        (before if bad_side == "before" else after).pop()
    with pytest.raises(ValueError, match=list(CASES)[-1]):
        publish(before, after, tmp_path)
    assert all(file.read_bytes() == b"last complete run" for file in files)


def test_publish_orders_all_pairs_and_computes_changes(tmp_path):
    after = _rows()[::-1]
    after[0]["decode_tokens_per_second"] = 6
    tmp_path = tmp_path / "published"
    publish(_rows(), after, tmp_path)
    for filename in ("after_fix_results.csv", "before_after_comparison.csv"):
        with (tmp_path / filename).open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["name"] for r in rows] == list(CASES)
    assert float(rows[-1]["out_tps_pct"]) == 20


@pytest.mark.parametrize("malformation", ["short", "duplicate_header", "long"])
def test_cli_rejects_malformed_csv_without_creating_either_output(tmp_path, malformation):
    rows = _rows()
    fields = list(rows[0])
    inputs = [tmp_path / name for name in ("before.csv", "candidate.csv")]
    for path in inputs:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields + ([fields[0]] if path == inputs[1] and malformation == "duplicate_header" else []))
            for i, row in enumerate(rows):
                values = list(row.values())
                if path == inputs[1] and i == len(rows) - 1:
                    values = values[:-1] if malformation == "short" else values + [1]
                writer.writerow(values)
    out = tmp_path / "new"
    result = subprocess.run([sys.executable, "-m", "tests.performance.qwen3_4b_pdaf.compare",
                             *inputs, out], capture_output=True, text=True)
    assert result.returncode == 1 and "Failed to publish" in result.stderr
    assert not out.exists()


def test_completed_batch_cannot_be_overwritten(tmp_path):
    out = tmp_path / "batch1"
    publish(_rows(), _rows(), out)
    snapshot = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        publish(_rows(), _rows()[::-1], out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == snapshot


@pytest.mark.parametrize("side", ["before", "after"])
def test_rejects_time_limited_poisson_run_even_with_valid_timings(tmp_path, side):
    before, after = _rows(), _rows()
    (before if side == "before" else after)[-1]["completed_requests"] = 200
    out = tmp_path / "batch"
    with pytest.raises(ValueError, match="completed_requests=200"):
        publish(before, after, out)
    assert not out.exists()


@pytest.mark.parametrize(("field", "value"), [("af_mb", 2), ("seed", 0), ("arrival_pattern", "burst"),
                                               ("prefill_util", "nan"), ("prefill_util", "inf"), ("prefill_util", "text")])
def test_rejects_invalid_case_parameters_and_utilization(tmp_path, field, value):
    after = _rows()
    after[-1][field] = value
    with pytest.raises(ValueError):
        publish(_rows(), after, tmp_path / "batch")
    assert not (tmp_path / "batch").exists()
