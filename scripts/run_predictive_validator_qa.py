from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import family_dataset_names
from pipelines.predictive_runtime import list_predictive_families, run_reference_task
from utils.io_utils import markdown_table, read_json, write_json, write_text
from utils.pathing import detect_project_paths
from validators.predictive_roles import VALIDATOR_ROLES, artifact_paths_for_run, validate_role


@dataclass
class CorruptionCase:
    family_id: str
    task_name: str
    role: str
    artifact_path: Path
    run_dir: Path
    workspace_root: Path
    mutation_name: str


def _copy_tree(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _ensure_reference_run(paths, family_id: str, task_name: str) -> Path:
    run_dir = paths.runs_dir / "reference" / family_id / task_name
    if not (run_dir / "artifacts" / "metrics.json").exists():
        runner = run_reference_task(paths, family_id, task_name, run_namespace="reference")
        return runner.run_dir
    return run_dir


def _prepare_corruption_cases(results_dir: Path, family_id: str, task_name: str, run_dir: Path, workspace_root: Path) -> list[CorruptionCase]:
    cases: list[CorruptionCase] = []
    artifacts = artifact_paths_for_run(run_dir)
    task_case_root = results_dir / "cases" / family_id / task_name

    profile_path = _copy_file(artifacts["profile_json"], task_case_root / "profile_missing_target" / "profile.json")
    profile_payload = read_json(profile_path)
    profile_payload.pop("target_name", None)
    write_json(profile_path, profile_payload)
    cases.append(CorruptionCase(family_id, task_name, "profile_json", profile_path, run_dir, workspace_root, "profile_missing_target"))

    split_overlap_path = _copy_file(artifacts["split_spec_json"], task_case_root / "split_overlap" / "split_spec.json")
    split_overlap_payload = read_json(split_overlap_path)
    split_overlap_payload["train_row_ids"] = list(split_overlap_payload["train_row_ids"]) + [split_overlap_payload["test_row_ids"][0]]
    split_overlap_payload["train_size"] = len(split_overlap_payload["train_row_ids"])
    write_json(split_overlap_path, split_overlap_payload)
    cases.append(CorruptionCase(family_id, task_name, "split_spec_json", split_overlap_path, run_dir, workspace_root, "split_overlap"))

    split_target_path = _copy_file(artifacts["split_spec_json"], task_case_root / "split_wrong_target" / "split_spec.json")
    split_target_payload = read_json(split_target_path)
    split_target_payload["target_name"] = "__wrong_target__"
    write_json(split_target_path, split_target_payload)
    cases.append(CorruptionCase(family_id, task_name, "split_spec_json", split_target_path, run_dir, workspace_root, "split_wrong_target"))

    preprocess_dir = _copy_tree(artifacts["preprocess_bundle"], task_case_root / "preprocess_bad_manifest" / "preprocess_bundle")
    preprocess_manifest = read_json(preprocess_dir / "manifest.json")
    preprocess_manifest["transformed_feature_count"] = 0
    write_json(preprocess_dir / "manifest.json", preprocess_manifest)
    cases.append(CorruptionCase(family_id, task_name, "preprocess_bundle", preprocess_dir, run_dir, workspace_root, "preprocess_bad_manifest"))

    model_dir = _copy_tree(artifacts["model_bundle"], task_case_root / "model_broken_file" / "model_bundle")
    (model_dir / "model.joblib").write_text("not a joblib model", encoding="utf-8")
    cases.append(CorruptionCase(family_id, task_name, "model_bundle", model_dir, run_dir, workspace_root, "model_broken_file"))

    metrics_path = _copy_file(artifacts["metrics_json"], task_case_root / "metrics_inconsistent" / "metrics.json")
    metrics_payload = read_json(metrics_path)
    primary_metric = metrics_payload["primary_metric"]
    metrics_payload[primary_metric] = float(metrics_payload[primary_metric]) + 0.1
    write_json(metrics_path, metrics_payload)
    cases.append(CorruptionCase(family_id, task_name, "metrics_json", metrics_path, run_dir, workspace_root, "metrics_inconsistent"))

    report_path = _copy_file(artifacts["report_md"], task_case_root / "report_empty" / "report.md")
    report_path.write_text("", encoding="utf-8")
    cases.append(CorruptionCase(family_id, task_name, "report_md", report_path, run_dir, workspace_root, "report_empty"))

    return cases


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    qa_root = paths.results_dir / "predictive_substrate" / "validator_qa"
    if qa_root.exists():
        shutil.rmtree(qa_root)
    qa_root.mkdir(parents=True, exist_ok=True)

    valid_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, Any]] = {}

    for family_id in list_predictive_families(paths):
        family_valid_rows: list[dict[str, Any]] = []
        family_bad_rows: list[dict[str, Any]] = []
        corruption_cases: list[CorruptionCase] = []

        for task_name in family_dataset_names(paths, family_id):
            run_dir = _ensure_reference_run(paths, family_id, task_name)
            artifacts = artifact_paths_for_run(run_dir)
            for role in VALIDATOR_ROLES:
                result = validate_role(role, artifacts[role], workspace_root=paths.workspace_root, run_dir=run_dir)
                row = {
                    "family_id": family_id,
                    "task_name": task_name,
                    "role": role,
                    "status": "pass" if result.passed else "fail",
                    "error_code": result.error_code or "none",
                }
                valid_rows.append(row)
                family_valid_rows.append(row)

            corruption_cases.extend(_prepare_corruption_cases(qa_root, family_id, task_name, run_dir, paths.workspace_root))

        for case in corruption_cases:
            result = validate_role(case.role, case.artifact_path, workspace_root=case.workspace_root, run_dir=case.run_dir)
            row = {
                "family_id": case.family_id,
                "task_name": case.task_name,
                "role": case.role,
                "mutation": case.mutation_name,
                "status": "detected" if not result.passed else "missed",
                "error_code": result.error_code or "none",
            }
            bad_rows.append(row)
            family_bad_rows.append(row)

        family_false_positives = sum(1 for row in family_valid_rows if row["status"] != "pass")
        family_detected = sum(1 for row in family_bad_rows if row["status"] == "detected")
        family_summary = {
            "family_id": family_id,
            "total_valid_artifacts": len(family_valid_rows),
            "false_positives": family_false_positives,
            "false_positive_rate": _rate(family_false_positives, len(family_valid_rows)),
            "total_bad_artifacts": len(family_bad_rows),
            "detected_bad_artifacts": family_detected,
            "bad_detection_rate": _rate(family_detected, len(family_bad_rows)),
            "passes_target": _rate(family_detected, len(family_bad_rows)) >= 0.9
            and _rate(family_false_positives, len(family_valid_rows)) <= 0.05,
            "valid_results": family_valid_rows,
            "bad_results": family_bad_rows,
        }
        by_family[family_id] = family_summary
        family_root = qa_root / family_id
        family_root.mkdir(parents=True, exist_ok=True)
        write_json(family_root / "summary.json", family_summary)

    false_positives = sum(1 for row in valid_rows if row["status"] != "pass")
    detected = sum(1 for row in bad_rows if row["status"] == "detected")
    summary = {
        "total_valid_artifacts": len(valid_rows),
        "false_positives": false_positives,
        "false_positive_rate": _rate(false_positives, len(valid_rows)),
        "total_bad_artifacts": len(bad_rows),
        "detected_bad_artifacts": detected,
        "bad_detection_rate": _rate(detected, len(bad_rows)),
        "targets": {
            "min_bad_detection_rate": 0.9,
            "max_false_positive_rate": 0.05,
        },
        "passes_target": _rate(detected, len(bad_rows)) >= 0.9 and _rate(false_positives, len(valid_rows)) <= 0.05,
        "valid_results": valid_rows,
        "bad_results": bad_rows,
        "by_family": by_family,
    }
    write_json(qa_root / "summary.json", summary)

    report_lines = [
        "# Predictive Validator QA",
        "",
        f"- valid artifacts checked: `{len(valid_rows)}`",
        f"- false positives: `{false_positives}`",
        f"- false positive rate: `{summary['false_positive_rate']:.3f}`",
        f"- corrupted artifacts checked: `{len(bad_rows)}`",
        f"- detected corruptions: `{detected}`",
        f"- bad artifact detection rate: `{summary['bad_detection_rate']:.3f}`",
        "",
        "## Family Summary",
        "",
        markdown_table(
            [
                {
                    "family_id": family_id,
                    "bad_detection_rate": family_summary["bad_detection_rate"],
                    "false_positive_rate": family_summary["false_positive_rate"],
                    "passes_target": family_summary["passes_target"],
                }
                for family_id, family_summary in by_family.items()
            ],
            ["family_id", "bad_detection_rate", "false_positive_rate", "passes_target"],
        ),
        "",
    ]
    write_text(qa_root / "summary.md", "\n".join(report_lines) + "\n")

    for family_id, family_summary in by_family.items():
        print(
            f"{family_id}: detection_rate={family_summary['bad_detection_rate']:.3f} "
            f"false_positive_rate={family_summary['false_positive_rate']:.3f}"
        )
    print(f"Predictive validator QA summary: {paths.relative_to_workspace(qa_root / 'summary.json')}")


if __name__ == "__main__":
    main()
