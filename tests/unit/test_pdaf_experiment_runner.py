"""Regression tests for examples/run_experiments/run_pdaf_experiments.py."""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import subprocess
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
    assert run_fp != runner.run_fingerprint(
        exp, {**source, "source_environment_fingerprint": "4" * 16}
    )


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


def test_git_info_excludes_output_dir_from_status_and_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "--allow-empty", "-qm", "base"],
        cwd=repo,
        check=True,
    )
    tracked = repo / "src" / "tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("x = 1\n")
    subprocess.run(["git", "add", "src/tracked.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "source"],
        cwd=repo,
        check=True,
    )

    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    baseline = runner.git_info()
    assert baseline["status_porcelain"] == ""

    out = repo / "src" / "out"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    filtered = runner.git_info(exclude_paths=[out])
    assert filtered["status_porcelain"] == ""
    assert filtered["diff_hash"] == baseline["diff_hash"]
    assert filtered["untracked_hash"] == baseline["untracked_hash"]


def test_summary_handles_na_and_empty_numeric_fields(tmp_path: Path) -> None:
    rows = [
        {
            "name": "case",
            "stage": "verify",
            "run_fingerprint": "fp",
            "ttft_mean_ms": "NA",
            "tpot_mean_ms": "",
            "e2e_mean_ms": "NA",
            "decode_tokens_per_second": "NA",
            "wall_seconds": "",
        }
    ]
    manifest = {
        "current_fingerprints": {"case": "fp"},
        "pending": [],
        "experiments": [],
        "selection_mode": "profile",
        "profile_requested": "core",
        "git": {},
        "workers": 1,
        "num_run": 0,
        "num_cached": 1,
        "num_cancelled": 0,
    }
    out = tmp_path / "summary.md"
    runner.write_summary(rows, out, manifest, [])
    text = out.read_text()
    assert "NaN" in text
    assert "case" in text


def test_synthetic_generator_seed_changes_arrivals() -> None:
    from frontier.config.request_generator_config import (
        FixedRequestLengthGeneratorConfig,
        PoissonRequestIntervalGeneratorConfig,
        SyntheticRequestGeneratorConfig,
    )
    from frontier.request_generator.synthetic_request_generator import (
        SyntheticRequestGenerator,
    )

    def arrivals(seed: int) -> list[float]:
        cfg = SyntheticRequestGeneratorConfig(
            seed=seed,
            num_requests=8,
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(
                qps=250, seed=seed
            ),
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=512, decode_tokens=64, seed=seed
            ),
        )
        requests = SyntheticRequestGenerator(cfg).generate_requests()
        return [round(request.arrived_at, 9) for request in requests]

    assert arrivals(1) == arrivals(1)
    assert arrivals(1) != arrivals(2)


def test_build_command_passes_synthetic_seed() -> None:
    exp = _experiment("online_g16_p8d8_tp2_qps250_req400_seed3")
    cmd = runner.build_command(exp)
    idx = cmd.index("--synthetic_request_generator_config_seed")
    assert cmd[idx + 1] == "3"
    idx = cmd.index("--poisson_request_interval_generator_config_seed")
    assert cmd[idx + 1] == "3"
    idx = cmd.index(
        "--random_forrest_execution_time_predictor_config_num_training_job_threads"
    )
    assert cmd[idx + 1] == "1"
    cmd = runner.build_command(exp, training_threads=2)
    idx = cmd.index(
        "--random_forrest_execution_time_predictor_config_num_training_job_threads"
    )
    assert cmd[idx + 1] == "2"


def test_default_output_dirs_are_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path)
    first = runner.resolve_output_dir(None, create=True)
    second = runner.resolve_output_dir(None, create=True)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_run_one_terminates_if_cancel_arrives_during_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ATTEMPTS_ROOT", tmp_path / "attempts")
    monkeypatch.setattr(runner, "METRICS_ROOT", tmp_path / "metrics")
    terminated: list[object] = []

    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.pid = 12345
            runner.CANCEL_EVENT.set()

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise AssertionError("run_one must not wait after cancellation")

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner, "_terminate_proc", lambda proc: terminated.append(proc))
    monkeypatch.setattr(runner, "current_source", lambda exclude_paths=(): _source())

    try:
        name, row, err, _ = runner.run_one(
            _experiment(), 1, _source(), "fingerprint", tmp_path / "out"
        )
    finally:
        runner.CANCEL_EVENT.clear()
        runner._RUNNING_PROCS.clear()

    assert (name, row, err) == ("g16_p8_d8_tp2", None, "cancelled")
    assert len(terminated) == 1


def test_withdraw_published_moves_fixed_metrics_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics_root = tmp_path / "metrics"
    monkeypatch.setattr(runner, "METRICS_ROOT", metrics_root)
    exp = _experiment()
    published = metrics_root / exp.name
    published.mkdir(parents=True)
    (published / "system_metrics.json").write_text("{}")

    out = tmp_path / "out"
    runner.withdraw_published(exp, out)

    assert not published.exists()
    assert list((out / "withdrawn").rglob("system_metrics.json"))


def test_run_one_rejects_result_if_source_changes_during_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ATTEMPTS_ROOT", tmp_path / "attempts")
    monkeypatch.setattr(runner, "METRICS_ROOT", tmp_path / "metrics")
    source = _source()
    changed = {**source, "source_diff_hash": "9" * 16}
    calls = iter((source, changed))
    monkeypatch.setattr(
        runner, "current_source", lambda exclude_paths=(): next(calls)
    )

    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.pid = 12345

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    name, row, err, _ = runner.run_one(
        _experiment(), 1, source, "fingerprint", tmp_path / "out"
    )
    assert (name, row) == ("g16_p8_d8_tp2", None)
    assert "source changed during run" in (err or "")


def test_run_one_honours_cancellation_before_spawn(tmp_path: Path) -> None:
    runner.CANCEL_EVENT.set()
    try:
        name, row, err, elapsed = runner.run_one(
            _experiment(), 1, _source(), "fingerprint", tmp_path / "out"
        )
    finally:
        runner.CANCEL_EVENT.clear()
    assert (name, row, err, elapsed) == ("g16_p8_d8_tp2", None, "cancelled", 0.0)
