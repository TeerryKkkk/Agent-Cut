from __future__ import annotations

import shutil
import traceback
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.family_registry import ensure_family_manifests, list_families, list_family_tasks, load_validator_module
from utils.io_utils import markdown_table, read_json, write_json, write_text
from utils.pathing import detect_project_paths


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


def _canonical_reference_run_dir(paths, family_id: str, task_name: str) -> Path:
    base_dir = paths.runs_dir / "reference" / family_id / task_name
    if (base_dir / "summary.json").exists():
        return base_dir
    candidates = sorted(path.parent for path in base_dir.glob("**/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No reference run summary found for {family_id}/{task_name}")
    return candidates[0]


def _prepare_f5_cases(results_dir: Path, family_id: str, task_name: str, run_dir: Path, workspace_root: Path) -> list[CorruptionCase]:
    cases = []
    raw_dir = _copy_tree(run_dir / "artifacts" / "reference_query_raw", results_dir / family_id / task_name / "raw_missing_celltype" / "reference_query_raw")
    query = ad.read_h5ad(raw_dir / "query_raw.h5ad")
    del query.obs["celltype"]
    query.write_h5ad(raw_dir / "query_raw.h5ad")
    cases.append(CorruptionCase(family_id, task_name, "reference_query_raw", raw_dir, run_dir, workspace_root, "raw_missing_celltype"))

    prepared_dir = _copy_tree(run_dir / "artifacts" / "prepared_query", results_dir / family_id / task_name / "prepared_gene_misalignment" / "prepared_query")
    query_prepared = ad.read_h5ad(prepared_dir / "query_prepared.h5ad")
    renamed = list(query_prepared.var_names)
    renamed[0] = f"{renamed[0]}__bad"
    query_prepared.var_names = renamed
    query_prepared.write_h5ad(prepared_dir / "query_prepared.h5ad")
    cases.append(CorruptionCase(family_id, task_name, "prepared_query", prepared_dir, run_dir, workspace_root, "prepared_gene_misalignment"))

    latent_dir = _copy_tree(run_dir / "artifacts" / "latent_or_graph", results_dir / family_id / task_name / "latent_missing_neighbors" / "latent_or_graph")
    reference = ad.read_h5ad(latent_dir / "reference_graph.h5ad")
    if "neighbors" in reference.uns:
        del reference.uns["neighbors"]
    reference.write_h5ad(latent_dir / "reference_graph.h5ad")
    cases.append(CorruptionCase(family_id, task_name, "latent_or_graph", latent_dir, run_dir, workspace_root, "latent_missing_neighbors"))

    prediction_dir = _copy_tree(run_dir / "artifacts" / "predicted_labels", results_dir / family_id / task_name / "predictions_empty" / "predicted_labels")
    pd.DataFrame(columns=["cell_id", "true_label", "predicted_label"]).to_csv(prediction_dir / "predictions.csv", index=False)
    cases.append(CorruptionCase(family_id, task_name, "predicted_labels", prediction_dir, run_dir, workspace_root, "predictions_empty"))

    metrics_path = _copy_file(run_dir / "artifacts" / "mapping_metrics" / "metrics.json", results_dir / family_id / task_name / "metrics_malformed" / "metrics.json")
    payload = read_json(metrics_path)
    payload.pop("acc_all", None)
    write_json(metrics_path, payload)
    cases.append(CorruptionCase(family_id, task_name, "mapping_metrics", metrics_path, run_dir, workspace_root, "metrics_malformed"))

    report_path = _copy_file(run_dir / "artifacts" / "report.md", results_dir / family_id / task_name / "report_empty" / "report.md")
    report_path.write_text("", encoding="utf-8")
    cases.append(CorruptionCase(family_id, task_name, "report_md", report_path, run_dir, workspace_root, "report_empty"))
    return cases


def _prepare_f6_cases(results_dir: Path, family_id: str, task_name: str, run_dir: Path, workspace_root: Path) -> list[CorruptionCase]:
    cases = []
    raw_dir = _copy_tree(run_dir / "artifacts" / "reference_query_raw", results_dir / family_id / task_name / "raw_missing_counts" / "reference_query_raw")
    reference = ad.read_h5ad(raw_dir / "reference_raw.h5ad")
    del reference.layers["counts"]
    reference.write_h5ad(raw_dir / "reference_raw.h5ad")
    cases.append(CorruptionCase(family_id, task_name, "reference_query_raw", raw_dir, run_dir, workspace_root, "raw_missing_counts"))

    prepared_dir = _copy_tree(run_dir / "artifacts" / "prepared_query", results_dir / family_id / task_name / "prepared_wrong_labels_key" / "prepared_query")
    combined = ad.read_h5ad(prepared_dir / "combined_prepared.h5ad")
    del combined.obs["celltype_scanvi"]
    combined.write_h5ad(prepared_dir / "combined_prepared.h5ad")
    cases.append(CorruptionCase(family_id, task_name, "prepared_query", prepared_dir, run_dir, workspace_root, "prepared_wrong_labels_key"))

    prepared_misaligned = _copy_tree(run_dir / "artifacts" / "prepared_query", results_dir / family_id / task_name / "prepared_gene_misalignment" / "prepared_query")
    query_prepared = ad.read_h5ad(prepared_misaligned / "query_prepared.h5ad")
    renamed = list(query_prepared.var_names)
    renamed[0] = f"{renamed[0]}__bad"
    query_prepared.var_names = renamed
    query_prepared.write_h5ad(prepared_misaligned / "query_prepared.h5ad")
    cases.append(CorruptionCase(family_id, task_name, "prepared_query", prepared_misaligned, run_dir, workspace_root, "prepared_gene_misalignment"))

    prediction_dir = _copy_tree(run_dir / "artifacts" / "predicted_labels", results_dir / family_id / task_name / "predictions_empty" / "predicted_labels")
    pd.DataFrame(columns=["cell_id", "true_label", "predicted_label"]).to_csv(prediction_dir / "predictions.csv", index=False)
    cases.append(CorruptionCase(family_id, task_name, "predicted_labels", prediction_dir, run_dir, workspace_root, "predictions_empty"))

    metrics_path = _copy_file(run_dir / "artifacts" / "mapping_metrics" / "metrics.json", results_dir / family_id / task_name / "metrics_malformed" / "metrics.json")
    payload = read_json(metrics_path)
    payload.pop("accuracy", None)
    write_json(metrics_path, payload)
    cases.append(CorruptionCase(family_id, task_name, "mapping_metrics", metrics_path, run_dir, workspace_root, "metrics_malformed"))

    report_path = _copy_file(run_dir / "artifacts" / "report.md", results_dir / family_id / task_name / "report_empty" / "report.md")
    report_path.write_text("", encoding="utf-8")
    cases.append(CorruptionCase(family_id, task_name, "report_md", report_path, run_dir, workspace_root, "report_empty"))
    return cases


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    qa_root = paths.results_dir / "singlecell_substrate" / "validator_qa"
    if qa_root.exists():
        shutil.rmtree(qa_root)
    qa_root.mkdir(parents=True, exist_ok=True)

    valid_rows = []
    bad_rows = []
    by_family = {}

    for family_id in list_families(paths, family_type="singlecell_mapping"):
        ensure_family_manifests(paths, family_id)
        validator_module = load_validator_module(paths, family_id)
        family_valid_rows = []
        family_bad_rows = []
        corruption_cases = []

        for task_name in list_family_tasks(paths, family_id):
            run_dir = _canonical_reference_run_dir(paths, family_id, task_name)
            artifact_paths = validator_module.artifact_paths_for_run(run_dir)
            for role in validator_module.VALIDATOR_ROLES:
                result = validator_module.validate_role(role, artifact_paths[role], workspace_root=paths.workspace_root, run_dir=run_dir)
                row = {
                    "family_id": family_id,
                    "task_name": task_name,
                    "role": role,
                    "status": "pass" if result.passed else "fail",
                    "error_code": result.error_code or "none",
                }
                valid_rows.append(row)
                family_valid_rows.append(row)

            if family_id == "scanpy_pancreas_ingest":
                corruption_cases.extend(_prepare_f5_cases(qa_root / "cases", family_id, task_name, run_dir, paths.workspace_root))
            else:
                corruption_cases.extend(_prepare_f6_cases(qa_root / "cases", family_id, task_name, run_dir, paths.workspace_root))

        for case in corruption_cases:
            try:
                result = validator_module.validate_role(case.role, case.artifact_path, workspace_root=case.workspace_root, run_dir=case.run_dir)
                row = {
                    "family_id": case.family_id,
                    "task_name": case.task_name,
                    "role": case.role,
                    "mutation": case.mutation_name,
                    "status": "detected" if not result.passed else "missed",
                    "error_code": result.error_code or "none",
                }
            except Exception as exc:
                row = {
                    "family_id": case.family_id,
                    "task_name": case.task_name,
                    "role": case.role,
                    "mutation": case.mutation_name,
                    "status": "detected",
                    "error_code": "validator_exception",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
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
            "passes_target": _rate(family_detected, len(family_bad_rows)) >= 0.9 and _rate(family_false_positives, len(family_valid_rows)) <= 0.05,
            "valid_results": family_valid_rows,
            "bad_results": family_bad_rows,
        }
        by_family[family_id] = family_summary
        family_dir = qa_root / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        write_json(family_dir / "summary.json", family_summary)
        print(f"{family_id}: detection_rate={family_summary['bad_detection_rate']:.3f} false_positive_rate={family_summary['false_positive_rate']:.3f}")

    false_positives = sum(1 for row in valid_rows if row["status"] != "pass")
    detected = sum(1 for row in bad_rows if row["status"] == "detected")
    summary = {
        "total_valid_artifacts": len(valid_rows),
        "false_positives": false_positives,
        "false_positive_rate": _rate(false_positives, len(valid_rows)),
        "total_bad_artifacts": len(bad_rows),
        "detected_bad_artifacts": detected,
        "bad_detection_rate": _rate(detected, len(bad_rows)),
        "targets": {"min_bad_detection_rate": 0.9, "max_false_positive_rate": 0.05},
        "passes_target": _rate(detected, len(bad_rows)) >= 0.9 and _rate(false_positives, len(valid_rows)) <= 0.05,
        "valid_results": valid_rows,
        "bad_results": bad_rows,
        "by_family": by_family,
    }
    write_json(qa_root / "summary.json", summary)
    report_lines = [
        "# Single-Cell Validator QA",
        "",
        f"- valid artifacts checked: `{len(valid_rows)}`",
        f"- false positives: `{false_positives}`",
        f"- false positive rate: `{summary['false_positive_rate']:.3f}`",
        f"- corrupted artifacts checked: `{len(bad_rows)}`",
        f"- detected corruptions: `{detected}`",
        f"- bad artifact detection rate: `{summary['bad_detection_rate']:.3f}`",
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
    print(f"Single-cell validator QA summary: {paths.relative_to_workspace(qa_root / 'summary.json')}")


if __name__ == "__main__":
    main()
