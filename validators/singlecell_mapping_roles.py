from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from utils.family_registry import load_family_metric_policy
from utils.io_utils import read_json
from utils.pathing import detect_project_paths

VALIDATOR_ROLES = [
    "reference_query_raw",
    "prepared_query",
    "latent_or_graph",
    "predicted_labels",
    "mapping_metrics",
    "report_md",
]
ROLE_TO_FILE = {
    "reference_query_raw": "reference_query_raw",
    "prepared_query": "prepared_query",
    "latent_or_graph": "latent_or_graph",
    "predicted_labels": "predicted_labels",
    "mapping_metrics": "mapping_metrics",
    "report_md": "report.md",
}


@dataclass
class ValidationResult:
    role: str
    passed: bool
    error_code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pass(role: str, message: str, details: dict[str, Any] | None = None) -> ValidationResult:
    return ValidationResult(role=role, passed=True, error_code="", message=message, details=details or {})


def _fail(role: str, error_code: str, message: str, details: dict[str, Any] | None = None) -> ValidationResult:
    return ValidationResult(role=role, passed=False, error_code=error_code, message=message, details=details or {})


def infer_workspace_root(path: Path) -> Path | None:
    try:
        return detect_project_paths(path).workspace_root
    except RuntimeError:
        return None


def artifact_paths_for_run(run_dir: Path) -> dict[str, Path]:
    artifacts_dir = run_dir / "artifacts"
    return {
        "reference_query_raw": artifacts_dir / "reference_query_raw",
        "prepared_query": artifacts_dir / "prepared_query",
        "latent_or_graph": artifacts_dir / "latent_or_graph",
        "predicted_labels": artifacts_dir / "predicted_labels",
        "mapping_metrics": artifacts_dir / "mapping_metrics" / "metrics.json",
        "report_md": artifacts_dir / "report.md",
    }


def _load_family_id(run_dir: Path) -> str:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)["family_id"]
    task_manifest_path = run_dir / "artifacts" / "reference_query_raw" / "task_manifest.json"
    return read_json(task_manifest_path)["family_id"]


def _required_metrics(workspace_root: Path | None, family_id: str) -> list[str]:
    if workspace_root is None:
        return []
    return list(load_family_metric_policy(type("Paths", (), {"configs_dir": workspace_root / "configs"})(), family_id)["required_metrics"])


def validate_reference_query_raw(artifact_dir: Path, family_id: str) -> ValidationResult:
    role = "reference_query_raw"
    try:
        reference = ad.read_h5ad(artifact_dir / "reference_raw.h5ad")
        query = ad.read_h5ad(artifact_dir / "query_raw.h5ad")
        task_manifest = read_json(artifact_dir / "task_manifest.json")
    except Exception as exc:
        return _fail(role, "raw_unreadable", f"Could not load raw reference/query artifacts: {exc}")

    if family_id == "scanpy_pancreas_ingest":
        for key in ["batch", "celltype"]:
            if key not in reference.obs.columns or key not in query.obs.columns:
                return _fail(role, "raw_missing_obs_key", f"Missing required obs key `{key}` in reference or query.", {"obs_key": key})
        return _pass(role, "Pancreas raw artifacts passed validation.", {"reference_rows": int(reference.n_obs), "query_rows": int(query.n_obs)})

    if family_id == "tabula_muris_label_transfer":
        if "counts" not in reference.layers or "counts" not in query.layers:
            return _fail(role, "raw_counts_layer_missing", "Raw Tabula Muris artifacts must preserve layers['counts'].")
        if "cell_ontology_class" not in reference.obs.columns:
            return _fail(role, "raw_reference_labels_missing", "Reference artifact is missing cell_ontology_class labels.")
        return _pass(role, "Tabula Muris raw artifacts passed validation.", {"reference_rows": int(reference.n_obs), "query_rows": int(query.n_obs), "tissue": task_manifest["tissue"]})

    return _fail(role, "family_unsupported", f"Unsupported single-cell family: {family_id}")


def validate_prepared_query(artifact_dir: Path, family_id: str) -> ValidationResult:
    role = "prepared_query"
    try:
        reference = ad.read_h5ad(artifact_dir / "reference_prepared.h5ad")
        query = ad.read_h5ad(artifact_dir / "query_prepared.h5ad")
        summary = read_json(artifact_dir / "summary.json")
    except Exception as exc:
        return _fail(role, "prepared_unreadable", f"Could not load prepared artifacts: {exc}")

    if list(reference.var_names) != list(query.var_names):
        return _fail(role, "prepared_gene_misalignment", "Reference and query prepared artifacts do not share the same var_names ordering.")
    if query.n_obs <= 0 or reference.n_obs <= 0 or query.n_vars <= 0:
        return _fail(role, "prepared_empty", "Prepared reference or query artifact is empty.")

    if family_id == "tabula_muris_label_transfer":
        combined_path = artifact_dir / "combined_prepared.h5ad"
        if not combined_path.exists():
            return _fail(role, "prepared_combined_missing", "Tabula Muris prepared artifacts are missing combined_prepared.h5ad.")
        combined = ad.read_h5ad(combined_path)
        if "counts" not in combined.layers:
            return _fail(role, "prepared_counts_layer_missing", "Combined prepared artifact must preserve layers['counts'].")
        if "celltype_scanvi" not in combined.obs.columns:
            return _fail(role, "prepared_scanvi_label_key_missing", "Combined prepared artifact is missing celltype_scanvi.")

    return _pass(role, "Prepared artifacts passed validation.", {"gene_count": summary["gene_count"], "query_rows": int(query.n_obs)})


def validate_latent_or_graph(artifact_dir: Path, family_id: str) -> ValidationResult:
    role = "latent_or_graph"
    try:
        summary = read_json(artifact_dir / "summary.json")
    except Exception as exc:
        return _fail(role, "latent_summary_unreadable", f"Could not read latent summary: {exc}")

    if family_id == "scanpy_pancreas_ingest":
        try:
            reference = ad.read_h5ad(artifact_dir / "reference_graph.h5ad")
        except Exception as exc:
            return _fail(role, "latent_reference_unreadable", f"Could not load pancreas reference graph artifact: {exc}")
        missing = []
        if "X_pca" not in reference.obsm:
            missing.append("X_pca")
        if "X_umap" not in reference.obsm:
            missing.append("X_umap")
        if "neighbors" not in reference.uns:
            missing.append("neighbors")
        if missing:
            return _fail(role, "latent_ingest_prerequisite_missing", "Reference graph is missing ingest prerequisites.", {"missing": missing})
        return _pass(role, "Pancreas latent/graph artifact passed validation.", summary)

    if family_id == "tabula_muris_label_transfer":
        try:
            adata = ad.read_h5ad(artifact_dir / "combined_latent.h5ad")
        except Exception as exc:
            return _fail(role, "latent_combined_unreadable", f"Could not load combined latent artifact: {exc}")
        valid_keys = [key for key in ["X_scVI", "X_scANVI"] if key in adata.obsm]
        if not valid_keys:
            return _fail(role, "latent_missing_embedding", "Combined latent artifact is missing X_scVI/X_scANVI embeddings.")
        for key in valid_keys:
            if adata.obsm[key].shape[0] != adata.n_obs:
                return _fail(role, "latent_shape_mismatch", f"Latent key {key} has invalid first dimension.", {"shape": list(adata.obsm[key].shape), "n_obs": int(adata.n_obs)})
        return _pass(role, "Tabula Muris latent artifact passed validation.", {"latent_keys": valid_keys, "n_obs": int(adata.n_obs)})

    return _fail(role, "family_unsupported", f"Unsupported single-cell family: {family_id}")


def validate_predicted_labels(artifact_dir: Path, family_id: str, run_dir: Path | None = None) -> ValidationResult:
    role = "predicted_labels"
    try:
        frame = pd.read_csv(artifact_dir / "predictions.csv")
    except Exception as exc:
        return _fail(role, "predictions_unreadable", f"Could not read predictions CSV: {exc}")

    if frame.empty:
        return _fail(role, "predictions_empty", "Predictions CSV is empty.")
    required_columns = {"cell_id", "true_label", "predicted_label"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        return _fail(role, "predictions_missing_columns", "Predictions CSV is missing required columns.", {"missing_columns": missing})

    if run_dir is not None and family_id == "scanpy_pancreas_ingest":
        query = ad.read_h5ad(run_dir / "artifacts" / "latent_or_graph" / "query_ingested.h5ad")
        reference = ad.read_h5ad(run_dir / "artifacts" / "latent_or_graph" / "reference_graph.h5ad")
        reference_vocab = set(reference.obs["celltype"].astype(str).tolist())
        expected_cell_ids = list(map(str, query.obs_names))
        observed_cell_ids = frame["cell_id"].astype(str).tolist()
        if len(frame) != int(query.n_obs):
            return _fail(role, "predictions_row_count_mismatch", "Predictions row count does not match query cells.", {"rows": len(frame), "query_rows": int(query.n_obs)})
        if observed_cell_ids != expected_cell_ids:
            return _fail(
                role,
                "predictions_cell_order_mismatch",
                "Prediction cell_id ordering does not match the latent/query cell order.",
                {"first_expected": expected_cell_ids[:3], "first_observed": observed_cell_ids[:3]},
            )
        if not set(frame["predicted_label"].astype(str)).issubset(reference_vocab):
            return _fail(role, "predictions_out_of_vocab", "Predicted labels are not a subset of the reference vocabulary.")

    if run_dir is not None and family_id == "tabula_muris_label_transfer":
        latent = ad.read_h5ad(run_dir / "artifacts" / "latent_or_graph" / "combined_latent.h5ad")
        query_rows = int((latent.obs["tech"].astype(str) == "10x").sum())
        reference_labels = set(latent.obs.loc[latent.obs["tech"].astype(str) == "SS2", "cell_ontology_class"].astype(str))
        expected_cell_ids = list(map(str, latent.obs.loc[latent.obs["tech"].astype(str) == "10x"].index))
        observed_cell_ids = frame["cell_id"].astype(str).tolist()
        if len(frame) != query_rows:
            return _fail(role, "predictions_row_count_mismatch", "Predictions row count does not match query cells.", {"rows": len(frame), "query_rows": query_rows})
        if observed_cell_ids != expected_cell_ids:
            return _fail(
                role,
                "predictions_cell_order_mismatch",
                "Prediction cell_id ordering does not match the latent/query cell order.",
                {"first_expected": expected_cell_ids[:3], "first_observed": observed_cell_ids[:3]},
            )
        if not set(frame["predicted_label"].astype(str)).issubset(reference_labels | {"Unknown"}):
            return _fail(role, "predictions_out_of_vocab", "Predicted labels are inconsistent with the reference training labels.")

    return _pass(role, "Predicted-label artifact passed validation.", {"row_count": len(frame)})


def validate_mapping_metrics(metrics_path: Path, family_id: str, workspace_root: Path | None = None) -> ValidationResult:
    role = "mapping_metrics"
    workspace_root = workspace_root or infer_workspace_root(metrics_path)
    try:
        payload = read_json(metrics_path)
    except Exception as exc:
        return _fail(role, "metrics_unreadable", f"Could not read mapping metrics JSON: {exc}")

    required_metrics = _required_metrics(workspace_root, family_id)
    missing_metrics = [metric for metric in required_metrics if metric not in payload]
    if missing_metrics:
        return _fail(role, "metrics_missing_fields", "Mapping metrics JSON is missing required fields.", {"missing_metrics": missing_metrics})
    confusion_path = payload.get("confusion_matrix_path")
    if confusion_path is None:
        return _fail(role, "metrics_confusion_missing", "Mapping metrics JSON is missing confusion_matrix_path.")
    if workspace_root is None or not (workspace_root / confusion_path).exists():
        return _fail(role, "metrics_confusion_missing_file", "confusion_matrix_path does not resolve to an existing file.", {"confusion_matrix_path": confusion_path})

    run_dir = metrics_path.parents[2]
    predictions_path = run_dir / "artifacts" / "predicted_labels" / "predictions.csv"
    if predictions_path.exists():
        prediction_frame = pd.read_csv(predictions_path)
        if {"true_label", "predicted_label"}.issubset(prediction_frame.columns) and not prediction_frame.empty:
            primary_metric = "acc_all" if family_id == "scanpy_pancreas_ingest" else "accuracy"
            recomputed = float((prediction_frame["true_label"].astype(str) == prediction_frame["predicted_label"].astype(str)).mean())
            stored_value = payload.get(primary_metric)
            if stored_value is not None and abs(float(stored_value) - recomputed) > 1.0e-9:
                return _fail(
                    role,
                    "metrics_prediction_mismatch",
                    "Stored mapping metrics are inconsistent with the saved predicted_labels artifact.",
                    {"metric_name": primary_metric, "stored": float(stored_value), "recomputed": recomputed},
                )

    return _pass(role, "Mapping metrics JSON passed validation.", {"primary_metric": payload.get("primary_metric"), "task_name": payload.get("task_name")})


def validate_report_md(report_path: Path, family_id: str) -> ValidationResult:
    role = "report_md"
    try:
        content = report_path.read_text(encoding="utf-8")
    except Exception as exc:
        return _fail(role, "report_unreadable", f"Could not read report markdown: {exc}")

    if not content.strip():
        return _fail(role, "report_empty", "Report markdown is empty.")

    if family_id == "scanpy_pancreas_ingest":
        required_strings = ["# Mapping Report:", "## Task", "## Metrics", "reference_batch", "query_batch", "seed", "deterministic_note"]
    else:
        required_strings = ["# Mapping Report:", "## Task", "## Metrics", "tissue", "seed", "warning_note", "## Reference Query Summary"]
    missing = [item for item in required_strings if item not in content]
    if missing:
        return _fail(role, "report_missing_sections", "Report markdown is missing required content.", {"missing": missing})
    return _pass(role, "Report markdown passed validation.")


def validate_role(
    role: str,
    artifact_path: Path,
    *,
    workspace_root: Path | None = None,
    run_dir: Path | None = None,
) -> ValidationResult:
    resolved_family_id = _load_family_id(run_dir) if run_dir is not None else None
    if resolved_family_id is None:
        raise ValueError("single-cell role validation requires run_dir to infer the family id.")

    if role == "reference_query_raw":
        return validate_reference_query_raw(artifact_path, resolved_family_id)
    if role == "prepared_query":
        return validate_prepared_query(artifact_path, resolved_family_id)
    if role == "latent_or_graph":
        return validate_latent_or_graph(artifact_path, resolved_family_id)
    if role == "predicted_labels":
        return validate_predicted_labels(artifact_path, resolved_family_id, run_dir=run_dir)
    if role == "mapping_metrics":
        return validate_mapping_metrics(artifact_path, resolved_family_id, workspace_root=workspace_root)
    if role == "report_md":
        return validate_report_md(artifact_path, resolved_family_id)
    raise ValueError(f"Unsupported single-cell validator role: {role}")


def validate_run_directory(run_dir: Path, workspace_root: Path | None = None) -> dict[str, ValidationResult]:
    artifact_paths = artifact_paths_for_run(run_dir)
    return {
        role: validate_role(role, artifact_paths[role], workspace_root=workspace_root, run_dir=run_dir)
        for role in VALIDATOR_ROLES
    }
