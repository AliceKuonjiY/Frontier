"""Regression tests for examples/run_experiments/run_pdaf_experiments.py."""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "run_experiments"
    / "run_pdaf_experiments.py"
)
_SPEC = importlib.util.spec_from_file_location("pdaf_run_experiments", RUNNER_PATH)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _experiment(name: str = "g16_p8_d8_tp2") -> runner.Experiment:
    return next(c for c in runner.build_all_configs() if c.name == name)


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_short_commit": "a" * 7,
        "source_branch": "main",
        "source_dirty": False,
        "source_status_hash": "0" * 16,
        "source_diff_hash": "1" * 16,
        "source_untracked_hash": "2" * 16,
    }


def test_save_csv_is_atomic_and_publishes_header_only_on_empty(tmp_path: Path) -> None:
    path = tmp_path / "sweep.csv"
    runner.save_csv([{"name": "x", "value": 1.0}], path)
    assert next(csv.reader(path.open())) == ["name", "value"]

    runner.save_csv([], path)
    assert next(csv.reader(path.open())) == ["name", "value"]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_fingerprints_change_with_config_and_source() -> None:
    exp = _experiment()
    source = _source()
    config_fp = runner.config_fingerprint(exp)
    run_fp = runner.run_fingerprint(exp, source)

    assert config_fp == runner.config_fingerprint(exp)
    assert run_fp == runner.run_fingerprint(exp, source)
    assert config_fp != runner.config_fingerprint(dataclasses.replace(exp, af_mb=2))
    assert run_fp != runner.run_fingerprint(exp, {**source, "source_diff_hash": "3" * 16})


def test_extract_metrics_rejects_incomplete_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp = _experiment("verify_baseline_stream0")
    base = tmp_path / exp.name
    base.mkdir()
    (base / "system_metrics.json").write_text(
        json.dumps(
            {
                "simulation_metadata": {
                    "total_requests": 8,
                    "completed_requests": 7,
                }
            }
        )
    )
    (base / "request_metrics.csv").write_text("Request Id\n")
    monkeypatch.setattr(runner, "METRICS_ROOT", tmp_path)

    with pytest.raises(runner.IncompleteRunError, match="completed 7/8"):
        runner.extract_metrics(exp)


def test_preflight_rejects_missing_streamed_cpu_overhead_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "frontier").mkdir()
    (tmp_path / "frontier" / "main.py").write_text("")
    for rel in runner.REQUIRED_DATA:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    exp = _experiment("ws3_g16_p8d8_af1_f1")
    with pytest.raises(SystemExit):
        runner.preflight("core", {exp.stage}, [exp])


def test_run_one_honours_cancellation_before_spawn() -> None:
    runner.CANCEL_EVENT.set()
    try:
        name, row, err, elapsed = runner.run_one(
            _experiment(), 1, _source(), "fingerprint"
        )
    finally:
        runner.CANCEL_EVENT.clear()
    assert (name, row, err, elapsed) == ("g16_p8_d8_tp2", None, "cancelled", 0.0)
