import csv

import pytest

from tests.performance.qwen3_4b_pdaf.compare import CASES, publish


def _rows():
    return [dict(name=name, ttft_mean_ms=3, tpot_mean_ms=2, decode_tokens_per_second=5,
                 prefill_e2e_mean_ms=3, prefill_util="NA") for name in CASES]


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
    with pytest.raises(ValueError, match=CASES[-1]):
        publish(before, after, tmp_path)
    assert all(file.read_bytes() == b"last complete run" for file in files)


def test_publish_orders_all_pairs_and_computes_changes(tmp_path):
    after = _rows()[::-1]
    after[0]["decode_tokens_per_second"] = 6
    publish(_rows(), after, tmp_path)
    for filename in ("after_fix_results.csv", "before_after_comparison.csv"):
        with (tmp_path / filename).open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["name"] for r in rows] == list(CASES)
    assert float(rows[-1]["out_tps_pct"]) == 20
