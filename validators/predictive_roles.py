from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import compute_binary_metrics, compute_regression_metrics, normalize_metric_name
from utils.io_utils import read_json

VALIDATOR_ROLES = [
    "profile_json",
    "split_spec_json",
    "preprocess_bundle",
    "model_bundle",
    "metrics_json",
    "report_md",
]

CLASSIFICATION_METRICS = ["accuracy", "balanced_accuracy", "macro_f1"]
CLASSIFICATION_PROBABILITY_METRICS = ["roc_auc", "average_precision"]
REGRESSION_METRICS = ["mae", "rmse", "r2", "spearman"]


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
    resolved = path.resolve()
    for candidate in [resolved.parent, *resolved.parents]:
        if (candidate / "configs").exists() and (candidate / "apikey.txt").exists():
            return candidate
    return None


def artifact_paths_for_run(run_dir: Path) -> dict[str, Path]:
    artifact_dir = run_dir / "artifacts"
    return {
        "raw_data": artifact_dir / "raw_data" / "raw_data.parquet",
        "profile_json": artifact_dir / "profile.json",
        "split_spec_json": artifact_dir / "split_spec.json",
        "preprocess_bundle": artifact_dir / "preprocess_bundle",
        "model_bundle": artifact_dir / "model_bundle",
        "metrics_json": artifact_dir / "metrics.json",
        "report_md": artifact_dir / "report.md",
    }


def _task_mode(payload: dict[str, Any]) -> str:
    task_type = str(payload.get("task_type", "")).lower()
    if "regression" in task_type:
        return "regression"
    if "classification" in task_type:
        return "classification"

    primary_metric = payload.get("primary_metric")
    if primary_metric is not None:
        normalized = normalize_metric_name(str(primary_metric))
        if normalized in REGRESSION_METRICS:
            return "regression"
    return "classification"


def _metric_names_for_payload(payload: dict[str, Any], probability_frame: pd.DataFrame | None = None) -> list[str]:
    task_mode = _task_mode(payload)
    if task_mode == "regression":
        return list(REGRESSION_METRICS)

    metric_names = list(CLASSIFICATION_METRICS)
    if probability_frame is not None and not probability_frame.empty:
        metric_names.extend(CLASSIFICATION_PROBABILITY_METRICS)
    return metric_names


def _load_optional_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def validate_profile_json(profile_path: Path) -> ValidationResult:
    role = "profile_json"
    try:
        payload = read_json(profile_path)
    except Exception as exc:
        return _fail(role, "profile_unreadable", f"Could not read profile JSON: {exc}")

    required = {
        "family_id",
        "task_name",
        "task_id",
        "dataset_id",
        "target_name",
        "row_count",
        "column_count",
        "feature_count",
        "feature_names",
        "dtypes",
        "missing_counts",
        "target_present",
        "split_source",
    }
    missing = sorted(required - set(payload))
    if missing:
        return _fail(role, "profile_missing_fields", "Profile is missing required fields.", {"missing_fields": missing})

    if not payload["target_present"]:
        return _fail(role, "profile_target_missing", "Profile states that the target column is absent.")
    if int(payload["row_count"]) <= 0 or int(payload["feature_count"]) <= 0:
        return _fail(
            role,
            "profile_invalid_shape",
            "Profile row or feature counts are not positive.",
            {"row_count": payload["row_count"], "feature_count": payload["feature_count"]},
        )

    feature_names = list(payload["feature_names"])
    summarized_features = (
        set(payload.get("categorical_columns", []))
        | set(payload.get("numerical_columns", []))
        | set(payload.get("molecule_columns", []))
    )
    if summarized_features != set(feature_names):
        return _fail(
            role,
            "profile_feature_partition_mismatch",
            "Feature summaries do not cover the declared feature set exactly.",
            {"feature_names": feature_names, "summarized_features": sorted(summarized_features)},
        )

    if _task_mode(payload) == "classification":
        class_distribution = payload.get("class_distribution")
        if not isinstance(class_distribution, dict) or not class_distribution:
            return _fail(role, "profile_class_distribution_missing", "Classification profile must include class distribution.")
        class_total = sum(int(value) for value in class_distribution.values())
        if class_total != int(payload["row_count"]):
            return _fail(
                role,
                "profile_class_count_mismatch",
                "Class distribution does not sum to the row count.",
                {"class_total": class_total, "row_count": payload["row_count"]},
            )
    else:
        target_summary = payload.get("target_summary", {})
        required_summary_fields = {"mean", "std", "min", "max"}
        missing_summary_fields = sorted(required_summary_fields - set(target_summary))
        if missing_summary_fields:
            return _fail(
                role,
                "profile_target_summary_missing",
                "Regression profile must include target summary statistics.",
                {"missing_fields": missing_summary_fields},
            )
        if float(target_summary["min"]) > float(target_summary["max"]):
            return _fail(role, "profile_target_summary_invalid", "Target summary min exceeds max.")

    return _pass(
        role,
        "Profile JSON passed validation.",
        {"row_count": payload["row_count"], "feature_count": payload["feature_count"]},
    )


def validate_split_spec_json(split_spec_path: Path, raw_data_path: Path | None = None) -> ValidationResult:
    role = "split_spec_json"
    try:
        payload = read_json(split_spec_path)
    except Exception as exc:
        return _fail(role, "split_unreadable", f"Could not read split spec JSON: {exc}")

    required = {
        "family_id",
        "task_name",
        "task_id",
        "dataset_id",
        "target_name",
        "row_id_field",
        "train_row_ids",
        "test_row_ids",
        "train_size",
        "test_size",
        "split_source",
    }
    missing = sorted(required - set(payload))
    if missing:
        return _fail(role, "split_missing_fields", "Split spec is missing required fields.", {"missing_fields": missing})

    split_sets = {
        "train": list(payload["train_row_ids"]),
        "test": list(payload["test_row_ids"]),
    }
    if "valid_row_ids" in payload:
        split_sets["valid"] = list(payload["valid_row_ids"])
    if "train_val_row_ids" in payload:
        split_sets["train_val"] = list(payload["train_val_row_ids"])

    size_fields = {
        "train": payload["train_size"],
        "test": payload["test_size"],
        "valid": payload.get("valid_size"),
        "train_val": payload.get("train_val_size"),
    }
    for split_name, row_ids in split_sets.items():
        expected_size = size_fields.get(split_name)
        if expected_size is not None and len(row_ids) != int(expected_size):
            return _fail(
                role,
                "split_size_mismatch",
                "Split sizes do not match the row id lists.",
                {"split_name": split_name, "observed_size": len(row_ids), "expected_size": expected_size},
            )

    explicit_splits = ["train", "valid", "test"] if "valid" in split_sets else ["train", "test"]
    for left_index, left_name in enumerate(explicit_splits):
        for right_name in explicit_splits[left_index + 1 :]:
            overlap = sorted(set(split_sets[left_name]) & set(split_sets[right_name]))
            if overlap:
                return _fail(
                    role,
                    "split_overlap",
                    "Explicit split row ids overlap.",
                    {
                        "left_split": left_name,
                        "right_split": right_name,
                        "overlap_count": len(overlap),
                        "overlap_preview": overlap[:5],
                    },
                )

    if "train_val" in split_sets:
        train_val_ids = set(split_sets["train_val"])
        train_valid_union = set(split_sets["train"])
        if "valid" in split_sets:
            train_valid_union |= set(split_sets["valid"])
        if train_valid_union != train_val_ids:
            return _fail(
                role,
                "split_train_val_mismatch",
                "Train/valid ids do not reconstruct the declared train_val split.",
                {
                    "train_valid_union_size": len(train_valid_union),
                    "train_val_size": len(train_val_ids),
                },
            )
        overlap = sorted(train_val_ids & set(split_sets["test"]))
        if overlap:
            return _fail(
                role,
                "split_train_val_test_overlap",
                "train_val ids overlap with test ids.",
                {"overlap_count": len(overlap), "overlap_preview": overlap[:5]},
            )

    if raw_data_path is not None:
        try:
            raw_df = pd.read_parquet(raw_data_path)
        except Exception as exc:
            return _fail(role, "split_raw_unreadable", f"Could not read raw data for split validation: {exc}")

        if payload["row_id_field"] not in raw_df.columns:
            return _fail(
                role,
                "split_row_id_missing",
                "Raw data does not contain the declared row id field.",
                {"row_id_field": payload["row_id_field"]},
            )
        known_ids = set(raw_df[payload["row_id_field"]].astype(int).tolist())
        referenced_ids = set()
        for row_ids in split_sets.values():
            referenced_ids |= set(int(item) for item in row_ids)
        unknown = sorted(referenced_ids - known_ids)
        if unknown:
            return _fail(
                role,
                "split_unknown_row_ids",
                "Split references row ids not present in raw data.",
                {"unknown_preview": unknown[:5]},
            )
        if payload["target_name"] not in raw_df.columns:
            return _fail(
                role,
                "split_target_mismatch",
                "Split target metadata does not match the raw data columns.",
                {"target_name": payload["target_name"]},
            )

    return _pass(
        role,
        "Split spec JSON passed validation.",
        {f"{split_name}_size": len(row_ids) for split_name, row_ids in split_sets.items()},
    )


def validate_preprocess_bundle(bundle_dir: Path) -> ValidationResult:
    role = "preprocess_bundle"
    try:
        manifest = read_json(bundle_dir / "manifest.json")
        preprocessor = joblib.load(bundle_dir / "preprocessor.joblib")
        train_matrix = np.load(bundle_dir / "train_matrix.npy", allow_pickle=False)
        test_matrix = np.load(bundle_dir / "test_matrix.npy", allow_pickle=False)
        train_labels = np.load(bundle_dir / "train_labels.npy", allow_pickle=False)
        test_labels = np.load(bundle_dir / "test_labels.npy", allow_pickle=False)
        valid_matrix = _load_optional_array(bundle_dir / "valid_matrix.npy")
        valid_labels = _load_optional_array(bundle_dir / "valid_labels.npy")
    except Exception as exc:
        return _fail(role, "preprocess_unreadable", f"Could not load preprocess bundle: {exc}")

    if manifest.get("fit_scope") != "train_only":
        return _fail(
            role,
            "preprocess_fit_scope_invalid",
            "Preprocess bundle must document train-only fitting.",
            {"fit_scope": manifest.get("fit_scope")},
        )
    if int(manifest.get("transformed_feature_count", 0)) <= 0:
        return _fail(
            role,
            "preprocess_empty_feature_space",
            "Transformed feature dimension must be positive.",
            {"transformed_feature_count": manifest.get("transformed_feature_count")},
        )
    if train_matrix.shape[0] != len(train_labels) or test_matrix.shape[0] != len(test_labels):
        return _fail(
            role,
            "preprocess_row_count_mismatch",
            "Feature matrices and labels have inconsistent row counts.",
            {
                "train_matrix_rows": int(train_matrix.shape[0]),
                "train_labels": int(len(train_labels)),
                "test_matrix_rows": int(test_matrix.shape[0]),
                "test_labels": int(len(test_labels)),
            },
        )

    transformed_feature_count = int(manifest["transformed_feature_count"])
    if train_matrix.shape[1] != transformed_feature_count or test_matrix.shape[1] != transformed_feature_count:
        return _fail(
            role,
            "preprocess_feature_count_mismatch",
            "Matrix feature counts do not match the manifest.",
            {
                "train_features": int(train_matrix.shape[1]),
                "test_features": int(test_matrix.shape[1]),
                "manifest_features": transformed_feature_count,
            },
        )

    has_validation_split = bool(manifest.get("has_validation_split", False))
    if has_validation_split and (valid_matrix is None or valid_labels is None):
        return _fail(role, "preprocess_validation_missing", "Validation split is declared but validation arrays are missing.")
    if valid_matrix is not None and valid_labels is not None:
        if valid_matrix.shape[0] != len(valid_labels):
            return _fail(
                role,
                "preprocess_validation_row_count_mismatch",
                "Validation matrix and labels have inconsistent row counts.",
                {
                    "valid_matrix_rows": int(valid_matrix.shape[0]),
                    "valid_labels": int(len(valid_labels)),
                },
            )
        if valid_matrix.shape[1] != transformed_feature_count:
            return _fail(
                role,
                "preprocess_validation_feature_count_mismatch",
                "Validation matrix feature count does not match the manifest.",
                {
                    "valid_features": int(valid_matrix.shape[1]),
                    "manifest_features": transformed_feature_count,
                },
            )

    if not hasattr(preprocessor, "transform"):
        return _fail(role, "preprocess_missing_pipeline", "Loaded preprocess object does not expose transform().")

    if manifest.get("feature_modality") == "molecule_fingerprint":
        required_manifest_fields = {"featurizer_name", "radius", "n_bits", "canonicalize_smiles"}
        missing_fields = sorted(required_manifest_fields - set(manifest))
        if missing_fields:
            return _fail(
                role,
                "preprocess_molecule_manifest_missing",
                "Molecule preprocess bundle is missing featurizer manifest fields.",
                {"missing_fields": missing_fields},
            )

    return _pass(
        role,
        "Preprocess bundle passed validation.",
        {
            "transformed_feature_count": transformed_feature_count,
            "has_validation_split": has_validation_split,
        },
    )


def validate_model_bundle(bundle_dir: Path) -> ValidationResult:
    role = "model_bundle"
    try:
        manifest = read_json(bundle_dir / "manifest.json")
        model = joblib.load(bundle_dir / "model.joblib")
        predictions = pd.read_csv(bundle_dir / "predictions.csv")
        smoke_input = np.load(bundle_dir / "smoke_input.npy", allow_pickle=False)
    except Exception as exc:
        return _fail(role, "model_unreadable", f"Could not load model bundle: {exc}")

    if "actual" not in predictions.columns or "predicted" not in predictions.columns:
        return _fail(role, "model_prediction_columns_missing", "Predictions must contain actual and predicted columns.")
    if not hasattr(model, "predict"):
        return _fail(role, "model_missing_predict", "Loaded model does not expose predict().")
    if len(predictions) != int(manifest.get("prediction_row_count", -1)):
        return _fail(
            role,
            "model_prediction_count_mismatch",
            "Prediction row count does not match the manifest.",
            {"predictions_rows": int(len(predictions)), "manifest_rows": manifest.get("prediction_row_count")},
        )

    try:
        smoke_predictions = model.predict(smoke_input)
    except Exception as exc:
        return _fail(role, "model_smoke_inference_failed", f"Smoke inference failed: {exc}")

    if len(smoke_predictions) != int(manifest.get("smoke_row_count", -1)):
        return _fail(
            role,
            "model_smoke_row_count_mismatch",
            "Smoke inference output length does not match the manifest.",
            {"smoke_predictions": int(len(smoke_predictions)), "manifest_smoke_rows": manifest.get("smoke_row_count")},
        )

    probability_columns = list(manifest.get("probability_columns", []))
    missing_probability_columns = [column for column in probability_columns if column not in predictions.columns]
    if missing_probability_columns:
        return _fail(
            role,
            "model_probability_columns_missing",
            "Predictions are missing declared probability columns.",
            {"missing_probability_columns": missing_probability_columns},
        )

    return _pass(role, "Model bundle passed validation.", {"prediction_row_count": manifest["prediction_row_count"]})


def validate_metrics_json(metrics_path: Path, workspace_root: Path | None = None) -> ValidationResult:
    role = "metrics_json"
    try:
        payload = read_json(metrics_path)
    except Exception as exc:
        return _fail(role, "metrics_unreadable", f"Could not read metrics JSON: {exc}")

    required = {
        "family_id",
        "task_name",
        "task_id",
        "dataset_id",
        "target_name",
        "primary_metric",
        "metric_tolerance",
        "prediction_artifact",
    }
    missing = sorted(required - set(payload))
    if missing:
        return _fail(role, "metrics_missing_fields", "Metrics JSON is missing required fields.", {"missing_fields": missing})

    resolved_workspace = workspace_root or infer_workspace_root(metrics_path)
    if resolved_workspace is None:
        return _fail(role, "metrics_workspace_unknown", "Could not infer workspace root for metric recomputation.")

    prediction_path = resolved_workspace / Path(payload["prediction_artifact"])
    if not prediction_path.exists():
        return _fail(
            role,
            "metrics_prediction_missing",
            "Prediction artifact referenced by metrics JSON does not exist.",
            {"prediction_artifact": payload["prediction_artifact"]},
        )

    prediction_frame = pd.read_csv(prediction_path)
    probability_columns = [column for column in prediction_frame.columns if column.startswith("proba_")]
    probability_frame = prediction_frame[probability_columns] if probability_columns else None

    tolerance = float(payload["metric_tolerance"])
    expected_metric_names = {
        normalize_metric_name(str(payload["primary_metric"])),
        *[normalize_metric_name(str(name)) for name in payload.get("secondary_metrics", [])],
    }
    expected_metric_names |= set(_metric_names_for_payload(payload, probability_frame))
    missing_metrics = sorted(metric_name for metric_name in expected_metric_names if metric_name not in payload)
    if missing_metrics:
        return _fail(
            role,
            "metrics_expected_values_missing",
            "Metrics JSON is missing expected metric values.",
            {"missing_metrics": missing_metrics},
        )

    if _task_mode(payload) == "regression":
        recomputed = compute_regression_metrics(
            prediction_frame["actual"].astype(float),
            prediction_frame["predicted"].astype(float),
        )
        inferred_positive_label = None
    else:
        recomputed, inferred_positive_label = compute_binary_metrics(
            prediction_frame["actual"].astype(str),
            prediction_frame["predicted"].astype(str),
            probability_frame,
        )

    mismatches = {}
    for metric_name in expected_metric_names:
        expected_value = recomputed.get(metric_name)
        observed_value = payload.get(metric_name)
        if expected_value is None:
            continue
        if observed_value is None or abs(float(observed_value) - float(expected_value)) > tolerance:
            mismatches[metric_name] = {"expected": expected_value, "observed": observed_value}

    if probability_frame is not None and payload.get("positive_label") != inferred_positive_label:
        return _fail(
            role,
            "metrics_positive_label_mismatch",
            "Stored positive label does not match the prediction artifact.",
            {"stored": payload.get("positive_label"), "inferred": inferred_positive_label},
        )
    if mismatches:
        return _fail(
            role,
            "metrics_inconsistent",
            "Metrics are inconsistent with recomputation from predictions.",
            {"mismatches": mismatches},
        )

    primary_metric = normalize_metric_name(str(payload["primary_metric"]))
    return _pass(
        role,
        "Metrics JSON passed validation.",
        {"primary_metric": primary_metric, "primary_value": payload[primary_metric]},
    )


def validate_report_md(
    report_path: Path,
    split_spec_path: Path | None = None,
    metrics_path: Path | None = None,
) -> ValidationResult:
    role = "report_md"
    try:
        content = report_path.read_text(encoding="utf-8")
    except Exception as exc:
        return _fail(role, "report_unreadable", f"Could not read report: {exc}")

    if not content.strip():
        return _fail(role, "report_empty", "Report is empty.")

    required_strings = ["# Reference Report:", "## Split", "## Metrics", "deterministic note", "## Warnings"]
    missing_sections = [item for item in required_strings if item not in content]
    if missing_sections:
        return _fail(
            role,
            "report_missing_sections",
            "Report is missing required sections.",
            {"missing_sections": missing_sections},
        )

    if split_spec_path is not None and split_spec_path.exists():
        split_spec = read_json(split_spec_path)
        expected_strings = [
            split_spec["task_name"],
            str(split_spec["train_size"]),
            str(split_spec["test_size"]),
        ]
        if "valid_size" in split_spec:
            expected_strings.append(str(split_spec["valid_size"]))
        missing_split_strings = [item for item in expected_strings if item not in content]
        if missing_split_strings:
            return _fail(
                role,
                "report_split_summary_missing",
                "Report does not contain the expected split summary.",
                {"expected": missing_split_strings},
            )

    if metrics_path is not None and metrics_path.exists():
        metrics = read_json(metrics_path)
        if metrics["task_name"] not in content:
            return _fail(
                role,
                "report_task_name_missing",
                "Report does not contain the task name.",
                {"task_name": metrics["task_name"]},
            )
        metric_names = [metrics["primary_metric"]]
        metric_names.extend(
            metric_name
            for metric_name in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
                "average_precision",
                "mae",
                "rmse",
                "r2",
                "spearman",
            ]
            if metric_name in metrics
        )
        missing_metric_mentions = [metric_name for metric_name in metric_names if str(metric_name) not in content]
        if missing_metric_mentions:
            return _fail(
                role,
                "report_metric_summary_missing",
                "Report does not mention the required metrics.",
                {"missing_metric_mentions": missing_metric_mentions},
            )

    return _pass(role, "Report markdown passed validation.")


def validate_role(
    role: str,
    artifact_path: Path,
    workspace_root: Path | None = None,
    run_dir: Path | None = None,
) -> ValidationResult:
    if role == "profile_json":
        return validate_profile_json(artifact_path)
    if role == "split_spec_json":
        raw_data_path = artifact_paths_for_run(run_dir)["raw_data"] if run_dir is not None else None
        return validate_split_spec_json(artifact_path, raw_data_path=raw_data_path)
    if role == "preprocess_bundle":
        return validate_preprocess_bundle(artifact_path)
    if role == "model_bundle":
        return validate_model_bundle(artifact_path)
    if role == "metrics_json":
        return validate_metrics_json(artifact_path, workspace_root=workspace_root)
    if role == "report_md":
        split_spec_path = artifact_paths_for_run(run_dir)["split_spec_json"] if run_dir is not None else None
        metrics_path = artifact_paths_for_run(run_dir)["metrics_json"] if run_dir is not None else None
        return validate_report_md(artifact_path, split_spec_path=split_spec_path, metrics_path=metrics_path)
    raise ValueError(f"Unsupported role validator: {role}")


def validate_run_directory(run_dir: Path, workspace_root: Path | None = None) -> dict[str, ValidationResult]:
    artifacts = artifact_paths_for_run(run_dir)
    results: dict[str, ValidationResult] = {}
    for role in VALIDATOR_ROLES:
        results[role] = validate_role(
            role,
            artifacts[role],
            workspace_root=workspace_root,
            run_dir=run_dir,
        )
    return results
