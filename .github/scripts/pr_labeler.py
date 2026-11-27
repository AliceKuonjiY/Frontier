#!/usr/bin/env python3
"""Deterministic PR auto-labeler for Frontier GitHub Actions.

Evaluates PR title, head branch, and changed files against `.github/pr-labeler.json`
to apply Type, Frontier Area, and mutually-exclusive Size labels. Missing labels are
created automatically on the repository, and manually-added non-size labels are
strictly preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

TITLE_PREFIX_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)*(?P<prefix>[a-zA-Z]+)(?:\([^)]*\))?!?\s*:"
)
BRANCH_PREFIX_RE = re.compile(r"^\s*(?P<prefix>[a-zA-Z]+)[/_-]")


def normalize_repo_path(path: str) -> str:
    """Normalize a repository-relative path without stripping dot-directories."""
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a POSIX-like path glob supporting `*`, `?`, and `**` into a regex."""
    cleaned = normalize_repo_path(pattern)
    out: list[str] = ["^"]
    i = 0
    n = len(cleaned)
    while i < n:
        if cleaned[i : i + 3] == "**/":
            out.append("(?:.*/)?")
            i += 3
        elif cleaned[i : i + 2] == "**":
            out.append(".*")
            i += 2
        elif cleaned[i] == "*":
            out.append("[^/]*")
            i += 1
        elif cleaned[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(cleaned[i]))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def matches_any_glob(path: str, patterns: list[str]) -> bool:
    normalized = normalize_repo_path(path)
    return any(glob_to_regex(pat).match(normalized) is not None for pat in patterns)


def parse_paginated_json_array(raw: str) -> list[Any]:
    """Parse one or more concatenated JSON arrays emitted by `gh api --paginate`."""
    decoder = json.JSONDecoder()
    items: list[Any] = []
    idx = 0
    length = len(raw)
    while idx < length:
        while idx < length and raw[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, end = decoder.raw_decode(raw, idx)
        if isinstance(obj, list):
            items.extend(obj)
        else:
            items.append(obj)
        idx = end
    return items


def run_gh_api(
    endpoint: str,
    *,
    method: str = "GET",
    paginate: bool = False,
    fields: list[tuple[str, str]] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> tuple[int, str, str]:
    cmd = ["gh", "api"]
    if method != "GET":
        cmd.extend(["--method", method])
    if paginate:
        cmd.append("--paginate")
    cmd.append(endpoint)
    for key, value in fields or []:
        cmd.extend(["-f", f"{key}={value}"])
    input_text: str | None = None
    if json_payload is not None:
        cmd.extend(["--input", "-"])
        input_text = json.dumps(json_payload)
    proc = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def compute_pr_labels(
    *,
    title: str,
    head_branch: str,
    files: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute desired labels and net line-change count for a PR."""
    desired: set[str] = set()
    reasons: dict[str, list[str]] = {}

    def add_label(label: str, reason: str) -> None:
        desired.add(label)
        reasons.setdefault(label, []).append(reason)

    filenames = [
        normalize_repo_path(str(item.get("filename", "")))
        for item in files
        if item.get("filename")
    ]

    # 1. Conventional-commit type label (PR title takes precedence over branch prefix).
    type_prefixes: dict[str, str] = {
        k.lower(): v for k, v in config.get("type_prefixes", {}).items()
    }
    title_match = TITLE_PREFIX_RE.match(title or "")
    matched_type_from_title = False
    if title_match:
        prefix = title_match.group("prefix").lower()
        if prefix in type_prefixes:
            add_label(type_prefixes[prefix], f"title prefix `{prefix}:`")
            matched_type_from_title = True

    if not matched_type_from_title:
        branch_match = BRANCH_PREFIX_RE.match(head_branch or "")
        if branch_match:
            prefix = branch_match.group("prefix").lower()
            if prefix in type_prefixes:
                add_label(type_prefixes[prefix], f"branch prefix `{prefix}/`")

    # 2. Path-based type rules.
    for label, rule in config.get("path_rules", {}).items():
        any_paths = rule.get("any_paths", [])
        all_paths = rule.get("all_paths", [])
        if any_paths and any(matches_any_glob(f, any_paths) for f in filenames):
            add_label(label, "changed paths")
        elif all_paths and filenames and all(
            matches_any_glob(f, all_paths) for f in filenames
        ):
            add_label(label, "all changed files match rule")

    # 3. Frontier area/module rules (paths + title + branch).
    for label, rule in config.get("area_rules", {}).items():
        any_paths = rule.get("any_paths", [])
        title_patterns = rule.get("title_patterns", [])
        branch_patterns = rule.get("branch_patterns", [])
        if any_paths and any(matches_any_glob(f, any_paths) for f in filenames):
            add_label(label, "changed paths")
            continue
        if title_patterns and any(
            re.search(pat, title or "", flags=re.IGNORECASE) for pat in title_patterns
        ):
            add_label(label, "PR title")
            continue
        if branch_patterns and any(
            re.search(pat, head_branch or "", flags=re.IGNORECASE)
            for pat in branch_patterns
        ):
            add_label(label, "head branch")

    # 4. Mutually-exclusive size label (excluding profiling/generated/binary noise).
    size_cfg = config.get("size_rules", {})
    ignore_paths = size_cfg.get("ignore_paths", [])
    net_changes = 0
    ignored_files = 0
    for item in files:
        fname = normalize_repo_path(str(item.get("filename", "")))
        if not fname:
            continue
        if matches_any_glob(fname, ignore_paths):
            ignored_files += 1
            continue
        additions = int(item.get("additions", 0) or 0)
        deletions = int(item.get("deletions", 0) or 0)
        net_changes += additions + deletions

    size_label: str | None = None
    thresholds = size_cfg.get("thresholds", [])
    for entry in thresholds:
        max_changes = entry.get("max_changes")
        if max_changes is None or net_changes <= int(max_changes):
            size_label = str(entry["label"])
            break
    if size_label:
        add_label(size_label, f"{net_changes} net changed lines")

    size_label_set = {str(entry["label"]) for entry in thresholds}
    return {
        "labels": sorted(desired),
        "reasons": {k: reasons[k] for k in sorted(reasons)},
        "size_label": size_label,
        "size_label_set": sorted(size_label_set),
        "net_changes": net_changes,
        "total_files": len(filenames),
        "ignored_size_files": ignored_files,
    }


def plan_label_updates(
    *,
    existing_pr_labels: list[str],
    desired_labels: list[str],
    exclusive_size_labels: list[str],
) -> tuple[list[str], list[str]]:
    """Return `(labels_to_add, labels_to_remove)`.

    Only stale mutually-exclusive size labels are removed; all manually applied
    labels and non-size labels remain untouched.
    """
    existing_lower = {lbl.lower(): lbl for lbl in existing_pr_labels}
    desired_lower = {lbl.lower(): lbl for lbl in desired_labels}
    size_lower = {lbl.lower() for lbl in exclusive_size_labels}

    to_add = [
        lbl for key, lbl in desired_lower.items() if key not in existing_lower
    ]
    to_remove = [
        orig
        for key, orig in existing_lower.items()
        if key in size_lower and key not in desired_lower
    ]
    return sorted(to_add), sorted(to_remove)


def ensure_repo_labels_exist(
    repo: str,
    labels_to_add: list[str],
    label_metadata: dict[str, dict[str, str]],
) -> list[str]:
    """Ensure every label in `labels_to_add` exists on the repository."""
    if not labels_to_add:
        return []

    rc, stdout, stderr = run_gh_api(
        f"repos/{repo}/labels?per_page=100", paginate=True
    )
    if rc != 0:
        raise RuntimeError(f"Failed to list repository labels: {stderr.strip()}")

    existing_items = parse_paginated_json_array(stdout)
    existing_lower = {
        str(item.get("name", "")).lower()
        for item in existing_items
        if isinstance(item, dict) and item.get("name")
    }

    created: list[str] = []
    for label in labels_to_add:
        if label.lower() in existing_lower:
            continue
        meta = label_metadata.get(label, {})
        color = str(meta.get("color", "ededed")).lstrip("#")
        description = str(meta.get("description", ""))[:100]
        rc, _, err = run_gh_api(
            f"repos/{repo}/labels",
            method="POST",
            fields=[
                ("name", label),
                ("color", color),
                ("description", description),
            ],
        )
        if rc == 0:
            created.append(label)
            existing_lower.add(label.lower())
        elif "already_exists" not in err.lower():
            raise RuntimeError(
                f"Failed to create repository label '{label}': {err.strip()}"
            )
    return created


def apply_pr_labels(
    *,
    repo: str,
    pr_number: int,
    config_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)

    rc, pr_stdout, pr_stderr = run_gh_api(f"repos/{repo}/pulls/{pr_number}")
    if rc != 0:
        raise RuntimeError(f"Failed to fetch PR #{pr_number}: {pr_stderr.strip()}")
    pr_data = json.loads(pr_stdout)

    rc, files_stdout, files_stderr = run_gh_api(
        f"repos/{repo}/pulls/{pr_number}/files?per_page=100",
        paginate=True,
    )
    if rc != 0:
        raise RuntimeError(
            f"Failed to fetch files for PR #{pr_number}: {files_stderr.strip()}"
        )
    files = parse_paginated_json_array(files_stdout)

    title = str(pr_data.get("title") or "")
    head_branch = str((pr_data.get("head") or {}).get("ref") or "")
    existing_pr_labels = [
        str(lbl["name"])
        for lbl in pr_data.get("labels", [])
        if isinstance(lbl, dict) and lbl.get("name")
    ]

    result = compute_pr_labels(
        title=title,
        head_branch=head_branch,
        files=files,
        config=config,
    )
    to_add, to_remove = plan_label_updates(
        existing_pr_labels=existing_pr_labels,
        desired_labels=result["labels"],
        exclusive_size_labels=result["size_label_set"],
    )

    created_repo_labels: list[str] = []
    if not dry_run:
        created_repo_labels = ensure_repo_labels_exist(
            repo=repo,
            labels_to_add=to_add,
            label_metadata=config.get("labels", {}),
        )
        if to_add:
            rc, _, err = run_gh_api(
                f"repos/{repo}/issues/{pr_number}/labels",
                method="POST",
                json_payload={"labels": to_add},
            )
            if rc != 0:
                raise RuntimeError(
                    f"Failed to add labels {to_add} to PR #{pr_number}: {err.strip()}"
                )
        for stale in to_remove:
            encoded = urllib.parse.quote(stale, safe="")
            rc, _, err = run_gh_api(
                f"repos/{repo}/issues/{pr_number}/labels/{encoded}",
                method="DELETE",
            )
            if rc != 0 and "404" not in err:
                raise RuntimeError(
                    f"Failed to remove stale label '{stale}' from PR #{pr_number}: {err.strip()}"
                )

    summary_data = {
        "pr_number": pr_number,
        "title": title,
        "head_branch": head_branch,
        "desired_labels": result["labels"],
        "reasons": result["reasons"],
        "existing_labels": sorted(existing_pr_labels),
        "added_labels": to_add,
        "removed_labels": to_remove,
        "created_repo_labels": created_repo_labels,
        "net_changes": result["net_changes"],
        "total_files": result["total_files"],
        "ignored_size_files": result["ignored_size_files"],
        "dry_run": dry_run,
    }
    return summary_data


def format_step_summary(data: dict[str, Any]) -> str:
    lines = [
        f"### PR #{data['pr_number']} Auto Labeler Summary",
        "",
        f"- **Matched labels**: {', '.join(f'`{l}`' for l in data['desired_labels']) or '(none)'}",
        f"- **Added**: {', '.join(f'`{l}`' for l in data['added_labels']) or '(none)'}",
        f"- **Removed (stale size)**: {', '.join(f'`{l}`' for l in data['removed_labels']) or '(none)'}",
        f"- **Created repo labels**: {', '.join(f'`{l}`' for l in data['created_repo_labels']) or '(none)'}",
        f"- **Net diff lines**: `{data['net_changes']}` across `{data['total_files']}` file(s) (`{data['ignored_size_files']}` ignored for size)",
    ]
    if data.get("reasons"):
        lines.append("")
        lines.append("| Label | Matched By |")
        lines.append("| --- | --- |")
        for label, matched_by in data["reasons"].items():
            lines.append(f"| `{label}` | {', '.join(matched_by)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-label a GitHub Pull Request.")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".github/pr-labeler.json"),
        help="Path to pr-labeler JSON config",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary_data = apply_pr_labels(
        repo=args.repo,
        pr_number=args.pr_number,
        config_path=args.config,
        dry_run=args.dry_run,
    )
    md = format_step_summary(summary_data)
    print(md.strip())

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary and not args.dry_run:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
