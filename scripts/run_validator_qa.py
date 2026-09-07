from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.ref_openml_binary import run_reference_family
from utils.io_utils import markdown_table, read_json, write_json, write_text
from utils.pathing import detect_project_paths
from validators.predictive_roles import VALIDATOR_ROLES, artifact_paths_for_run, validate_role


@dataclass
class CorruptionCase:
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


def _prepare_corruption_cases(results_dir: Path, reference_runs: dict[str, Path], workspace_root: Path) -> list[CorruptionCase]:
    cases: list[CorruptionCase] = []
    for task_name, run_dir in reference_runs.items():
        artifacts = artifact_paths_for_run(run_dir)
        task_case_root = results_dir / "cases" / task_name

        profile_path = _copy_file(artifacts["profile_json"], task_case_root / "profile_missing_target" / "profile.json")
        profile_payload = read_json(profile_path)
        profile_payload.pop("target_name", None)
        write_json(profile_path, profile_payload)
        cases.append(CorruptionCase(task_name, "profile_json", profile_path, run_dir, workspace_root, "profile_missing_target"))

        split_path = _copy_file(artifacts["split_spec_json"], task_case_root / "split_overlap" / "split_spec.json")
        split_payload = read_json(split_path)
        split_payload["train_row_ids"] = list(split_payload["train_row_ids"]) + [split_payload["test_row_ids"][0]]
        split_payload["train_size"] = len(split_payload["train_row_ids"])
        write_json(split_path, split_payload)
        cases.append(CorruptionCase(task_name, "split_spec_json", split_path, run_dir, workspace_root, "split_overlap"))

        preprocess_dir = _copy_tree(artifacts["preprocess_bundle"], task_case_root / "preprocess_bad_manifest" / "preprocess_bundle")
        preprocess_manifest = read_json(preprocess_dir / "manifest.json")
        preprocess_manifest["transformed_feature_count"] = 0
        write_json(preprocess_dir / "manifest.json", preprocess_manifest)
        cases.append(CorruptionCase(task_name, "preprocess_bundle", preprocess_dir, run_dir, workspace_root, "preprocess_bad_manifest"))

        model_dir = _copy_tree(artifacts["model_bundle"], task_case_root / "model_broken_file" / "model_bundle")
        (model_dir / "model.joblib").write_text("not a joblib model", encoding="utf-8")
        cases.append(CorruptionCase(task_name, "model_bundle", model_dir, run_dir, workspace_root, "model_broken_file"))

        metrics_path = _copy_file(artifacts["metrics_json"], task_case_root / "metrics_inconsistent" / "metrics.json")
        metrics_payload = read_json(metrics_path)
        metrics_payload["balanced_accuracy"] = float(metrics_payload["balanced_accuracy"]) + 0.1
        write_json(metrics_path, metrics_payload)
        cases.append(CorruptionCase(task_name, "metrics_json", metrics_path, run_dir, workspace_root, "metrics_inconsistent"))

        report_path = _copy_file(artifacts["report_md"], task_case_root / "report_empty" / "report.md")
        report_path.write_text("", encoding="utf-8")
        cases.append(CorruptionCase(task_name, "report_md", report_path, run_dir, workspace_root, "report_empty"))

    return cases


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    qa_root = paths.results_dir / "validator_qa"
    qa_root.mkdir(parents=True, exist_ok=True)

    runs = run_reference_family(paths=paths)
    reference_runs = {runner.task_name: runner.run_dir for runner in runs}

    valid_rows: list[dict[str, Any]] = []
    false_positives = 0
    total_valid = 0
    for task_name, run_dir in reference_runs.items():
        artifacts = artifact_paths_for_run(run_dir)
        for role in VALIDATOR_ROLES:
            total_valid += 1
            result = validate_role(role, artifacts[role], workspace_root=paths.workspace_root, run_dir=run_dir)
            false_positives += 0 if result.passed else 1
            valid_rows.append(
                {
                    "task_name": task_name,
                    "role": role,
                    "status": "pass" if result.passed else "fail",
                    "error_code": result.error_code or "none",
                }
            )

    corruption_cases = _prepare_corruption_cases(qa_root, reference_runs, paths.workspace_root)
    bad_rows: list[dict[str, Any]] = []
    detected = 0
    for case in corruption_cases:
        result = validate_role(case.role, case.artifact_path, workspace_root=case.workspace_root, run_dir=case.run_dir)
        detected += 0 if result.passed else 1
        bad_rows.append(
            {
                "task_name": case.task_name,
                "role": case.role,
                "mutation": case.mutation_name,
                "status": "detected" if not result.passed else "missed",
                "error_code": result.error_code or "none",
            }
        )

    bad_detection_rate = detected / len(corruption_cases) if corruption_cases else 0.0
    false_positive_rate = false_positives / total_valid if total_valid else 0.0

    summary = {
        "total_valid_artifacts": total_valid,
        "false_positives": false_positives,
        "false_positive_rate": false_positive_rate,
        "total_bad_artifacts": len(corruption_cases),
        "detected_bad_artifacts": detected,
        "bad_detection_rate": bad_detection_rate,
        "targets": {
            "min_bad_detection_rate": 0.9,
            "max_false_positive_rate": 0.05,
        },
        "passes_target": bad_detection_rate >= 0.9 and false_positive_rate <= 0.05,
        "valid_results": valid_rows,
        "bad_results": bad_rows,
    }
    write_json(qa_root / "summary.json", summary)

    report_lines = [
        "# Validator QA",
        "",
        f"- valid artifacts checked: `{total_valid}`",
        f"- false positives: `{false_positives}`",
        f"- false positive rate: `{false_positive_rate:.3f}`",
        f"- corrupted artifacts checked: `{len(corruption_cases)}`",
        f"- detected corruptions: `{detected}`",
        f"- bad artifact detection rate: `{bad_detection_rate:.3f}`",
        "",
        "## Valid Artifact Results",
        "",
        markdown_table(valid_rows, ["task_name", "role", "status", "error_code"]),
        "",
        "## Corruption Results",
        "",
        markdown_table(bad_rows, ["task_name", "role", "mutation", "status", "error_code"]),
        "",
    ]
    if summary["passes_target"]:
        report_lines.append("Validator QA met the acceptance thresholds.")
    else:
        report_lines.extend(
            [
                "Validator QA did not meet the acceptance thresholds.",
                "",
                f"- required bad detection rate: `>= 0.900`, observed `{bad_detection_rate:.3f}`",
                f"- required false positive rate: `<= 0.050`, observed `{false_positive_rate:.3f}`",
            ]
        )
    write_text(qa_root / "summary.md", "\n".join(report_lines) + "\n")

    print(f"Validator QA detection rate: {bad_detection_rate:.3f}")
    print(f"Validator QA false positive rate: {false_positive_rate:.3f}")
    print(f"Validator QA summary: {paths.relative_to_workspace(qa_root / 'summary.json')}")


if __name__ == "__main__":
    main()
