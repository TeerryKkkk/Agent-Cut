from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.family_runtime import build_reference_runner
from utils.family_registry import list_families, load_validator_module
from utils.io_utils import markdown_table, read_json, read_yaml, write_json, write_text
from utils.pathing import detect_project_paths
from utils.predictive_splits import load_family_split_summary


def replay_skill_on_task(paths, family_id: str, skill_payload: dict[str, Any], task_name: str) -> dict[str, Any]:
    validator_module = load_validator_module(paths, family_id)
    runner = build_reference_runner(
        paths,
        family_id,
        task_name,
        run_namespace="support_replay",
        run_label=f"{skill_payload['split_id']}__{skill_payload['skill_id']}",
    )
    runner.run_segment(skill_payload["role_start"], skill_payload["role_end"])
    run_dir = runner.run_dir
    artifact_path = validator_module.artifact_paths_for_run(run_dir)[skill_payload["role_end"]]
    validation = validator_module.validate_role(
        skill_payload["role_end"],
        artifact_path,
        workspace_root=paths.workspace_root,
        run_dir=run_dir,
    )
    return {
        "family_id": family_id,
        "split_id": skill_payload["split_id"],
        "skill_id": skill_payload["skill_id"],
        "task_name": task_name,
        "role_end": skill_payload["role_end"],
        "passed": validation.passed,
        "error_code": validation.error_code,
        "message": validation.message,
        "run_dir": paths.relative_to_workspace(run_dir),
    }


def _validate_skill_on_runner(paths, family_id: str, skill_payload: dict[str, Any], task_name: str, runner) -> dict[str, Any]:
    validator_module = load_validator_module(paths, family_id)
    runner.run_segment(skill_payload["role_start"], skill_payload["role_end"])
    run_dir = runner.run_dir
    artifact_path = validator_module.artifact_paths_for_run(run_dir)[skill_payload["role_end"]]
    validation = validator_module.validate_role(
        skill_payload["role_end"],
        artifact_path,
        workspace_root=paths.workspace_root,
        run_dir=run_dir,
    )
    return {
        "family_id": family_id,
        "split_id": skill_payload["split_id"],
        "skill_id": skill_payload["skill_id"],
        "task_name": task_name,
        "role_end": skill_payload["role_end"],
        "passed": validation.passed,
        "error_code": validation.error_code,
        "message": validation.message,
        "run_dir": paths.relative_to_workspace(run_dir),
    }


def replay_library(paths, family_id: str, split_id: str, *, summary_root: Path | None = None) -> dict[str, Any]:
    compiled_root = paths.compiled_skills_dir / family_id / split_id
    library_manifest = read_json(compiled_root / "library_manifest.json")

    rows: list[dict[str, Any]] = []
    skill_payloads = []
    for skill_entry in library_manifest["skills"]:
        skill_dir = paths.workspace_root / skill_entry["skill_dir"]
        skill_payloads.append(read_yaml(skill_dir / "skill.yaml"))

    support_tasks = list(library_manifest["support_tasks"])
    for task_name in support_tasks:
        runner = build_reference_runner(
            paths,
            family_id,
            task_name,
            run_namespace="support_replay",
            run_label=f"{split_id}__task_replay",
        )
        for skill_payload in skill_payloads:
            rows.append(_validate_skill_on_runner(paths, family_id, skill_payload, task_name, runner))

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    pass_rate = passed / total if total else 0.0
    failing_rows = [row for row in rows if not row["passed"]]
    summary = {
        "family_id": family_id,
        "split_id": split_id,
        "total_replays": total,
        "passed_replays": passed,
        "pass_rate": pass_rate,
        "target_pass_rate": 0.95,
        "passes_target": pass_rate >= 0.95,
        "results": rows,
        "failing_results": failing_rows,
    }

    resolved_summary_root = summary_root or (paths.results_dir / "support_replay" / family_id / split_id)
    resolved_summary_root.mkdir(parents=True, exist_ok=True)
    write_json(resolved_summary_root / "summary.json", summary)

    report_lines = [
        "# Support Replay",
        "",
        f"- family_id: `{family_id}`",
        f"- split_id: `{split_id}`",
        f"- total replays: `{total}`",
        f"- passed replays: `{passed}`",
        f"- pass rate: `{pass_rate:.3f}`",
        "",
        markdown_table(
            [
                {
                    "skill_id": row["skill_id"],
                    "task_name": row["task_name"],
                    "role_end": row["role_end"],
                    "status": "pass" if row["passed"] else "fail",
                    "error_code": row["error_code"] or "none",
                }
                for row in rows
            ],
            ["skill_id", "task_name", "role_end", "status", "error_code"],
        ),
        "",
    ]
    if failing_rows:
        report_lines.extend(
            [
                "Support replay did not meet the target pass rate.",
                "",
                markdown_table(
                    [
                        {
                            "skill_id": row["skill_id"],
                            "task_name": row["task_name"],
                            "error_code": row["error_code"] or "none",
                            "message": row["message"],
                        }
                        for row in failing_rows
                    ],
                    ["skill_id", "task_name", "error_code", "message"],
                ),
            ]
        )
    else:
        report_lines.append("Support replay met the target pass rate.")
    write_text(resolved_summary_root / "summary.md", "\n".join(report_lines) + "\n")
    return summary


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    for family_id in list_families(paths, family_type="predictive"):
        split_summary = load_family_split_summary(paths, family_id)
        for split_path in split_summary["split_paths"]:
            split_manifest = read_json(paths.workspace_root / split_path)
            summary = replay_library(paths, family_id, split_manifest["split_id"])
            print(
                f"{family_id}/{split_manifest['split_id']}: "
                f"support_replay_pass_rate={summary['pass_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
