from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

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

    parsed = mod.parse_paginated_json_array('[{"a": 1}]\n[{"a": 2}, {"a": 3}]')
    assert parsed == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_compute_pr_labels_title_precedence_areas_and_size() -> None:
    mod = _load_labeler()
    config = mod.load_config(CONFIG_PATH)

    # Title prefix `ci:` must take precedence over head branch `feat/...`.
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
