from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "pr_labeler.py"
CONFIG_PATH = REPO_ROOT / ".github" / "pr-labeler.json"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pr-labeler.yml"


def _load_labeler():
    spec = importlib.util.spec_from_file_location("pr_labeler", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pr_labeler_workflow_and_config_are_valid() -> None:
    assert WORKFLOW_PATH.is_file()
    assert CONFIG_PATH.is_file()
    assert SCRIPT_PATH.is_file()

    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert set(triggers["pull_request"]["types"]) >= {
        "opened",
        "synchronize",
        "reopened",
        "edited",
    }
    assert "pr_number" in triggers["workflow_dispatch"]["inputs"]

    label_job = workflow["jobs"]["label"]
    assert label_job["permissions"] == {
        "contents": "read",
        "issues": "write",
        "pull-requests": "write",
    }
    assert label_job["timeout-minutes"] <= 10

    # Ensure workflow_dispatch input is strictly validated as positive integer
    # and never interpolated directly into shell `run:` commands.
    steps_by_name = {step["name"]: step for step in label_job["steps"]}
    resolve_step = steps_by_name["Resolve PR number"]
    assert "^[1-9][0-9]*$" in resolve_step["run"]
    apply_step = steps_by_name["Apply PR labels"]
    assert "${{" not in apply_step["run"]
    assert apply_step["env"]["PR_NUMBER"] == "${{ steps.pr.outputs.number }}"
    assert apply_step["env"]["GITHUB_REPO"] == "${{ github.repository }}"

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    label_meta = config["labels"]
    hex_re = re.compile(r"^[0-9a-fA-F]{6}$")
    for name, meta in label_meta.items():
        assert hex_re.match(meta["color"]), f"invalid color for {name}"
        assert len(meta["description"]) <= 100, f"description too long for {name}"

    # Every label referenced by any rule must have metadata defined.
    referenced = set(config["type_prefixes"].values())
    referenced.update(config["path_rules"].keys())
    referenced.update(config["area_rules"].keys())
    referenced.update(entry["label"] for entry in config["size_rules"]["thresholds"])
    missing = referenced - set(label_meta.keys())
    assert not missing, f"labels missing metadata: {missing}"


def test_glob_matching_and_paginated_json_parsing() -> None:
    mod = _load_labeler()

    assert mod.matches_any_glob("README.md", ["*.md"])
    assert mod.matches_any_glob("RUN_QWEN3_4B_PD_AF.md", ["*.md"])
    assert not mod.matches_any_glob("tests/performance/qwen3_4b_pdaf/README.md", ["*.md"])
    assert mod.matches_any_glob("docs/cli/README.md", ["docs/**"])
    assert mod.matches_any_glob("foo.csv", ["**/*.csv"])
    assert mod.matches_any_glob("data/profiling/h200/a.csv", ["**/*.csv"])
    assert mod.matches_any_glob("frontier/moe_ep_workload.py", ["frontier/moe_*"])
    assert not mod.matches_any_glob("frontier/sub/moe_ep.py", ["frontier/moe_*"])
    assert mod.matches_any_glob("tests/e2e/pd_af_parity/runner.py", ["**/*pd_af*/**"])
    assert mod.matches_any_glob("tests/performance/qwen3_4b_pdaf/compare.py", ["**/*pdaf*/**"])

    parsed = mod.parse_paginated_json_array('[{"a": 1}]\n[{"a": 2}, {"a": 3}]')
    assert parsed == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_compute_pr_labels_title_precedence_areas_and_size() -> None:
    mod = _load_labeler()
    config = mod.load_config(CONFIG_PATH)

    # Title prefix `ci:` must take precedence over head branch `feat/...`,
    # and hyphenated branch names in the title must not falsely trigger `area/pd-af`.
    ci_res = mod.compute_pr_labels(
        title="ci: add Codex PR review workflow for qwen3-4b-pd-af-simulator",
        head_branch="feat/codex-pr-review",
        files=[
            {"filename": ".github/workflows/codex-review.yml", "additions": 310, "deletions": 0},
            {"filename": ".github/scripts/gemini_review_helper.py", "additions": 88, "deletions": 5},
            {"filename": "tests/unit/test_gemini_review_helper.py", "additions": 140, "deletions": 0},
        ],
        config=config,
    )
    assert "ci" in ci_res["labels"]
    assert "enhancement" not in ci_res["labels"]
    assert "area/pd-af" not in ci_res["labels"]
    assert "tests" not in ci_res["labels"]
    assert ci_res["size_label"] == "size/M"

    # PD-AF feature PR with CSV noise ignored from size calculation.
    pdaf_res = mod.compute_pr_labels(
        title="feat(pd-af): complete 2-microbatch A/F overlap simulation and metrics",
        head_branch="feat/pdaf-af-microbatch-overlap",
        files=[
            {"filename": "frontier/metrics/af_overlap_tracker.py", "additions": 204, "deletions": 0},
            {"filename": "frontier/scheduler/utils/pdaf_transfer.py", "additions": 19, "deletions": 4},
            {"filename": "data/profiling/huge.csv", "additions": 5000, "deletions": 0},
            {"filename": "tests/unit/test_pdaf_af_microbatch_overlap.py", "additions": 100, "deletions": 0},
        ],
        config=config,
    )
    assert set(pdaf_res["labels"]) == {
        "enhancement",
        "area/pd-af",
        "area/metrics",
        "area/scheduler",
        "area/profiling",
        "size/M",
    }
    assert pdaf_res["net_changes"] == 327
    assert pdaf_res["ignored_size_files"] == 1

    # Files nested inside pd_af / pdaf subdirectories and RUN_QWEN3_4B_PD_AF.md
    # must match `area/pd-af` even without title/branch keywords.
    for path in (
        "tests/e2e/pd_af_parity/runner.py",
        "tests/performance/qwen3_4b_pdaf/compare.py",
        "RUN_QWEN3_4B_PD_AF.md",
    ):
        sub_res = mod.compute_pr_labels(
            title="Update runner script",
            head_branch="update-runner",
            files=[{"filename": path, "additions": 10, "deletions": 2}],
            config=config,
        )
        assert "area/pd-af" in sub_res["labels"], f"{path} should match area/pd-af"

    # Test-only PR gets `tests` label even without `test:` title prefix.
    test_only = mod.compute_pr_labels(
        title="Add unit tests for overlap metrics",
        head_branch="add-overlap-tests",
        files=[
            {"filename": "tests/unit/test_foo.py", "additions": 30, "deletions": 5},
        ],
        config=config,
    )
    assert set(test_only["labels"]) == {"tests", "size/XS"}

    # Mixed changes on a `test/` branch without a `test:` title prefix must NOT get `tests`.
    mixed_test_branch = mod.compute_pr_labels(
        title="Adjust simulator scheduling logic",
        head_branch="test/scheduler-experiment",
        files=[
            {"filename": "frontier/scheduler/foo.py", "additions": 20, "deletions": 5},
            {"filename": "tests/unit/test_foo.py", "additions": 30, "deletions": 0},
        ],
        config=config,
    )
    assert "tests" not in mixed_test_branch["labels"]
    assert "area/scheduler" in mixed_test_branch["labels"]

    # Fallback to branch prefix when title has no conventional prefix.
    branch_fallback = mod.compute_pr_labels(
        title="修复 MoE 路由不均衡",
        head_branch="fix/moe-routing",
        files=[
            {"filename": "frontier/moe_routing_runtime.py", "additions": 40, "deletions": 15},
        ],
        config=config,
    )
    assert set(branch_fallback["labels"]) == {"bug", "area/moe", "size/S"}


def test_plan_label_updates_swaps_stale_size_and_preserves_manual_labels() -> None:
    mod = _load_labeler()
    to_add, to_remove = mod.plan_label_updates(
        existing_pr_labels=["help wanted", "enhancement", "size/S"],
        desired_labels=["bug", "area/pd-af", "size/L"],
        exclusive_size_labels=["size/XS", "size/S", "size/M", "size/L", "size/XL"],
    )
    assert to_add == ["area/pd-af", "bug", "size/L"]
    assert to_remove == ["size/S"]


def test_ensure_repo_labels_exist_handles_422_already_exists_in_stdout() -> None:
    mod = _load_labeler()

    def fake_gh_api(endpoint, *, method="GET", paginate=False, fields=None, json_payload=None):
        if method == "GET" and endpoint.startswith("repos/AliceKuonjiY/Frontier/labels"):
            return 0, json.dumps([{"name": "bug"}]), ""
        if method == "POST" and endpoint == "repos/AliceKuonjiY/Frontier/labels":
            name = dict(fields or {}).get("name")
            if name == "area/pd-af":
                # Simulate real `gh api` 422 behavior: JSON details on stdout, summary on stderr.
                stdout = json.dumps(
                    {
                        "message": "Validation Failed",
                        "errors": [
                            {"resource": "Label", "code": "already_exists", "field": "name"}
                        ],
                    }
                )
                return 1, stdout, "gh: Validation Failed (HTTP 422)"
            return 0, "{}", ""
        raise AssertionError(f"Unexpected gh api call: {method} {endpoint}")

    with mock.patch.object(mod, "run_gh_api", side_effect=fake_gh_api):
        created = mod.ensure_repo_labels_exist(
            repo="AliceKuonjiY/Frontier",
            labels_to_add=["bug", "area/pd-af", "size/M"],
            label_metadata={
                "area/pd-af": {"color": "5319e7", "description": "PD-AF"},
                "size/M": {"color": "f0ad4e", "description": "Medium PR"},
            },
        )
    assert created == ["size/M"]


def test_apply_pr_labels_e2e_and_fork_skip_and_dry_run() -> None:
    mod = _load_labeler()
    calls: list[tuple[str, str, list[tuple[str, str]] | None, dict | None]] = []

    def fake_same_repo_gh_api(
        endpoint, *, method="GET", paginate=False, fields=None, json_payload=None
    ):
        calls.append((method, endpoint, fields, json_payload))
        if method == "GET" and endpoint == "repos/AliceKuonjiY/Frontier/pulls/99":
            return (
                0,
                json.dumps(
                    {
                        "title": "feat(pd-af): test overlap",
                        "head": {
                            "ref": "feat/test",
                            "repo": {"full_name": "AliceKuonjiY/Frontier"},
                        },
                        "labels": [{"name": "help wanted"}, {"name": "size/S"}],
                    }
                ),
                "",
            )
        if method == "GET" and endpoint.startswith("repos/AliceKuonjiY/Frontier/pulls/99/files"):
            page1 = json.dumps([{"filename": "frontier/metrics/a.py", "additions": 250, "deletions": 10}])
            page2 = json.dumps([{"filename": "data/profiling/x.csv", "additions": 9000, "deletions": 0}])
            return 0, f"{page1}\n{page2}", ""
        if method == "GET" and endpoint.startswith("repos/AliceKuonjiY/Frontier/labels"):
            return 0, json.dumps([{"name": "enhancement"}, {"name": "help wanted"}]), ""
        if method == "POST" and endpoint == "repos/AliceKuonjiY/Frontier/labels":
            name = dict(fields or {}).get("name")
            if name == "area/metrics":
                return (
                    1,
                    '{"message":"Validation Failed","errors":[{"code":"already_exists"}]}',
                    "gh: Validation Failed (HTTP 422)",
                )
            return 0, "{}", ""
        if method == "POST" and endpoint == "repos/AliceKuonjiY/Frontier/issues/99/labels":
            return 0, "[]", ""
        if method == "DELETE" and endpoint == "repos/AliceKuonjiY/Frontier/issues/99/labels/size%2FS":
            # Also verify 404 on already-deleted label is tolerated gracefully.
            return 1, '{"message":"Not Found"}', "gh: Not Found (HTTP 404)"
        raise AssertionError(f"Unexpected call: {method} {endpoint}")

    # 1. Full apply flow (including 422 race on stdout & URL-encoded DELETE 404 tolerance).
    with mock.patch.object(mod, "run_gh_api", side_effect=fake_same_repo_gh_api):
        summary = mod.apply_pr_labels(
            repo="AliceKuonjiY/Frontier",
            pr_number=99,
            config_path=CONFIG_PATH,
            dry_run=False,
        )
    assert summary["skipped_fork"] is False
    assert set(summary["added_labels"]) == {
        "area/metrics",
        "area/pd-af",
        "area/profiling",
        "enhancement",
        "size/M",
    }
    assert summary["removed_labels"] == ["size/S"]
    delete_endpoints = [ep for m, ep, _, _ in calls if m == "DELETE"]
    assert delete_endpoints == ["repos/AliceKuonjiY/Frontier/issues/99/labels/size%2FS"]
    md = mod.format_step_summary(summary)
    assert "### PR #99 Auto Labeler Summary" in md
    assert "`size/M`" in md
    assert "`size/S`" in md

    # 2. dry_run=True performs zero mutating calls.
    calls.clear()
    with mock.patch.object(mod, "run_gh_api", side_effect=fake_same_repo_gh_api):
        dry_summary = mod.apply_pr_labels(
            repo="AliceKuonjiY/Frontier",
            pr_number=99,
            config_path=CONFIG_PATH,
            dry_run=True,
        )
    assert dry_summary["dry_run"] is True
    assert all(m == "GET" for m, _, _, _ in calls)

    # 3. Fork PR (via workflow_dispatch) must be skipped before any write API calls.
    fork_calls: list[tuple[str, str]] = []

    def fake_fork_gh_api(endpoint, *, method="GET", paginate=False, fields=None, json_payload=None):
        fork_calls.append((method, endpoint))
        if method == "GET" and endpoint == "repos/AliceKuonjiY/Frontier/pulls/77":
            return (
                0,
                json.dumps(
                    {
                        "title": "fix(pd-af): external fork PR",
                        "head": {
                            "ref": "fix/external",
                            "repo": {"full_name": "SomeoneElse/Frontier"},
                        },
                        "labels": [{"name": "size/S"}],
                    }
                ),
                "",
            )
        raise AssertionError(f"Fork PR must not trigger further API calls: {method} {endpoint}")

    with mock.patch.object(mod, "run_gh_api", side_effect=fake_fork_gh_api):
        fork_summary = mod.apply_pr_labels(
            repo="AliceKuonjiY/Frontier",
            pr_number=77,
            config_path=CONFIG_PATH,
            dry_run=False,
        )
    assert fork_summary["skipped_fork"] is True
    assert fork_calls == [("GET", "repos/AliceKuonjiY/Frontier/pulls/77")]
    assert "Skipped" in mod.format_step_summary(fork_summary)
