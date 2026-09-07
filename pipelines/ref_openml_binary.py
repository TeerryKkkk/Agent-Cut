from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from provenance.trace_utils import TraceRecorder
from tasks.openml_tasks import export_openml_task, load_exported_metadata
from utils.io_utils import read_json, read_yaml, write_json, write_text
from utils.pathing import ProjectPaths, detect_project_paths, ensure_workspace_dirs

FAMILY_ID = "openml_tabular_binary"
ROLE_TEMPLATE = [
    "raw_data",
    "profile_json",
    "split_spec_json",
    "preprocess_bundle",
    "model_bundle",
    "metrics_json",
    "report_md",
]
STEP_SPECS = [
    ("step_01_materialize_raw_data", "openml_task", "raw_data", "materialize_raw_data"),
    ("step_02_profile_dataset", "raw_data", "profile_json", "build_profile"),
    ("step_03_define_split", "profile_json", "split_spec_json", "build_split_spec"),
    ("step_04_fit_preprocess", "split_spec_json", "preprocess_bundle", "build_preprocess_bundle"),
    ("step_05_fit_model", "preprocess_bundle", "model_bundle", "build_model_bundle"),
    ("step_06_compute_metrics", "model_bundle", "metrics_json", "build_metrics"),
    ("step_07_write_report", "metrics_json", "report_md", "build_report"),
]


@dataclass
class RoleArtifact:
    role: str
    paths: list[Path]
    metadata: dict[str, Any]


def load_family_config(paths: ProjectPaths) -> dict[str, Any]:
    return read_yaml(paths.configs_dir / "families.yaml")["families"][FAMILY_ID]


def load_metrics_config(paths: ProjectPaths) -> dict[str, Any]:
    return read_yaml(paths.configs_dir / "metrics.yaml")["families"][FAMILY_ID]


def infer_positive_label(probability_columns: list[str]) -> str | None:
    if len(probability_columns) != 2:
        return None
    return probability_columns[-1].replace("proba_", "", 1)


def compute_binary_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    probability_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, float], str | None]:
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
    }

    positive_label = None
    if probability_frame is not None:
        probability_columns = [column for column in probability_frame.columns if column.startswith("proba_")]
        positive_label = infer_positive_label(probability_columns)
        if positive_label is not None:
            positive_scores = probability_frame[f"proba_{positive_label}"]
            actual_binary = (actual.astype(str) == positive_label).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(actual_binary, positive_scores))
            metrics["average_precision"] = float(average_precision_score(actual_binary, positive_scores))

    return metrics, positive_label


class OpenMLBinaryReferenceRunner:
    FAMILY_ID = FAMILY_ID
    ROLE_TEMPLATE = ROLE_TEMPLATE
    STEP_SPECS = STEP_SPECS

    def __init__(
        self,
        paths: ProjectPaths,
        task_name: str,
        *,
        run_namespace: str = "reference",
        run_label: str | None = None,
    ) -> None:
        ensure_workspace_dirs(paths)
        export_openml_task(paths, task_name)

        self.paths = paths
        self.task_name = task_name
        self.family_config = load_family_config(paths)
        self.metrics_config = load_metrics_config(paths)
        self.metadata = load_exported_metadata(paths, task_name)
        self.run_namespace = run_namespace
        self.run_label = run_label

        run_root = paths.runs_dir / run_namespace / FAMILY_ID / task_name
        self.run_dir = run_root if run_label is None else run_root / run_label
        self.artifacts_dir = self.run_dir / "artifacts"
        self.trace_path = self.run_dir / "trace.jsonl"
        self.prov_path = self.run_dir / "prov.json"
        self.summary_path = self.run_dir / "summary.json"
        self.trace = TraceRecorder(self.trace_path)
        self.role_artifacts: dict[str, RoleArtifact] = {}

        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _rel(self, path: Path) -> str:
        return self.paths.relative_to_workspace(path)

    def _role_dir(self, role: str) -> Path:
        if role in {"raw_data", "preprocess_bundle", "model_bundle"}:
            return self.artifacts_dir / role
        return self.artifacts_dir

    def _record_success(
        self,
        *,
        step_id: str,
        role_in: str,
        role_out: str,
        function_name: str,
        started_at: float,
        input_paths: list[Path],
        output_paths: list[Path],
        details: dict[str, Any] | None = None,
    ) -> None:
        self.trace.record(
            step_id=step_id,
            role_in=role_in,
            role_out=role_out,
            input_artifacts=[self._rel(path) for path in input_paths],
            output_artifacts=[self._rel(path) for path in output_paths],
            function=function_name,
            status="pass",
            wall_clock_s=time.perf_counter() - started_at,
            details=details or {},
        )

    def _slice_from_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw_artifact = self.role_artifacts["raw_data"]
        split_artifact = self.role_artifacts["split_spec_json"]
        raw_data_path = raw_artifact.paths[0]
        split_path = split_artifact.paths[0]

        raw_df = pd.read_parquet(raw_data_path)
        split_spec = read_json(split_path)
        train_ids = set(split_spec["train_row_ids"])
        test_ids = set(split_spec["test_row_ids"])
        train_df = raw_df[raw_df["row_id"].isin(train_ids)].copy().sort_values("row_id").reset_index(drop=True)
        test_df = raw_df[raw_df["row_id"].isin(test_ids)].copy().sort_values("row_id").reset_index(drop=True)
        return train_df, test_df

    def materialize_raw_data(self) -> RoleArtifact:
        started_at = time.perf_counter()
        export_dir = self.paths.exports_dir / self.task_name
        input_paths = [export_dir / "raw_data.parquet", export_dir / "metadata.json"]
        role_dir = self._role_dir("raw_data")
        role_dir.mkdir(parents=True, exist_ok=True)

        raw_data_path = role_dir / "raw_data.parquet"
        manifest_path = role_dir / "manifest.json"
        raw_df = pd.read_parquet(export_dir / "raw_data.parquet")
        raw_df.to_parquet(raw_data_path, index=False)
        manifest = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "task_id": self.metadata["task_id"],
            "dataset_id": self.metadata["dataset_id"],
            "target_name": self.metadata["target_name"],
            "row_count": int(len(raw_df)),
            "column_count": int(raw_df.shape[1]),
            "row_id_field": "row_id",
            "source_export": self._rel(export_dir / "raw_data.parquet"),
        }
        write_json(manifest_path, manifest)
        artifact = RoleArtifact("raw_data", [raw_data_path, manifest_path], manifest)
        self.role_artifacts["raw_data"] = artifact
        self._record_success(
            step_id="step_01_materialize_raw_data",
            role_in="openml_task",
            role_out="raw_data",
            function_name="materialize_raw_data",
            started_at=started_at,
            input_paths=input_paths,
            output_paths=artifact.paths,
            details={"row_count": manifest["row_count"], "column_count": manifest["column_count"]},
        )
        return artifact

    def build_profile(self) -> RoleArtifact:
        started_at = time.perf_counter()
        raw_df = pd.read_parquet(self.role_artifacts["raw_data"].paths[0])
        target_name = self.metadata["target_name"]
        profile_path = self._role_dir("profile_json") / "profile.json"

        feature_columns = [column for column in raw_df.columns if column not in {target_name, "row_id"}]
        profile = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "task_id": self.metadata["task_id"],
            "dataset_id": self.metadata["dataset_id"],
            "target_name": target_name,
            "row_count": int(len(raw_df)),
            "column_count": int(len(feature_columns) + 1),
            "feature_count": int(len(feature_columns)),
            "feature_names": feature_columns,
            "categorical_columns": self.metadata["categorical_columns"],
            "numerical_columns": self.metadata["numerical_columns"],
            "dtypes": {column: str(raw_df[column].dtype) for column in feature_columns + [target_name]},
            "missing_counts": {column: int(raw_df[column].isna().sum()) for column in feature_columns + [target_name]},
            "target_present": target_name in raw_df.columns,
            "class_distribution": {str(label): int(count) for label, count in raw_df[target_name].value_counts(dropna=False).items()},
            "split_source": self.metadata["split_source"],
        }
        write_json(profile_path, profile)
        artifact = RoleArtifact("profile_json", [profile_path], profile)
        self.role_artifacts["profile_json"] = artifact
        self._record_success(
            step_id="step_02_profile_dataset",
            role_in="raw_data",
            role_out="profile_json",
            function_name="build_profile",
            started_at=started_at,
            input_paths=self.role_artifacts["raw_data"].paths,
            output_paths=artifact.paths,
            details={"feature_count": profile["feature_count"]},
        )
        return artifact

    def build_split_spec(self) -> RoleArtifact:
        started_at = time.perf_counter()
        export_dir = self.paths.exports_dir / self.task_name
        raw_df = pd.read_parquet(self.role_artifacts["raw_data"].paths[0])
        split_indices = read_json(export_dir / "split_indices.json")
        split_path = self._role_dir("split_spec_json") / "split_spec.json"
        target_name = self.metadata["target_name"]

        train_ids = split_indices["train_row_ids"]
        test_ids = split_indices["test_row_ids"]
        train_df = raw_df[raw_df["row_id"].isin(train_ids)]
        test_df = raw_df[raw_df["row_id"].isin(test_ids)]
        split_spec = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "task_id": self.metadata["task_id"],
            "dataset_id": self.metadata["dataset_id"],
            "target_name": target_name,
            "row_id_field": "row_id",
            "train_row_ids": train_ids,
            "test_row_ids": test_ids,
            "train_size": int(len(train_ids)),
            "test_size": int(len(test_ids)),
            "split_source": self.metadata["split_source"],
            "repeat": self.metadata["split_rule"]["repeat"],
            "fold": self.metadata["split_rule"]["fold"],
            "sample": self.metadata["split_rule"]["sample"],
            "train_class_distribution": {str(label): int(count) for label, count in train_df[target_name].value_counts(dropna=False).items()},
            "test_class_distribution": {str(label): int(count) for label, count in test_df[target_name].value_counts(dropna=False).items()},
        }
        write_json(split_path, split_spec)
        artifact = RoleArtifact("split_spec_json", [split_path], split_spec)
        self.role_artifacts["split_spec_json"] = artifact
        self._record_success(
            step_id="step_03_define_split",
            role_in="profile_json",
            role_out="split_spec_json",
            function_name="build_split_spec",
            started_at=started_at,
            input_paths=self.role_artifacts["profile_json"].paths + self.role_artifacts["raw_data"].paths,
            output_paths=artifact.paths,
            details={"train_size": split_spec["train_size"], "test_size": split_spec["test_size"]},
        )
        return artifact

    def build_preprocess_bundle(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("preprocess_bundle")
        role_dir.mkdir(parents=True, exist_ok=True)

        train_df, test_df = self._slice_from_split()
        target_name = self.metadata["target_name"]
        feature_columns = [column for column in self.metadata["feature_names"] if column in train_df.columns]
        numeric_columns = [column for column in self.metadata["numerical_columns"] if column in feature_columns]
        categorical_columns = [column for column in self.metadata["categorical_columns"] if column in feature_columns]

        x_train = train_df[feature_columns]
        x_test = test_df[feature_columns]
        y_train = train_df[target_name]
        y_test = test_df[target_name]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

        train_matrix = preprocessor.fit_transform(x_train)
        test_matrix = preprocessor.transform(x_test)
        feature_names = list(preprocessor.get_feature_names_out())

        preprocessor_path = role_dir / "preprocessor.joblib"
        train_matrix_path = role_dir / "train_matrix.npy"
        test_matrix_path = role_dir / "test_matrix.npy"
        train_labels_path = role_dir / "train_labels.npy"
        test_labels_path = role_dir / "test_labels.npy"
        manifest_path = role_dir / "manifest.json"

        joblib.dump(preprocessor, preprocessor_path)
        np.save(train_matrix_path, train_matrix)
        np.save(test_matrix_path, test_matrix)
        np.save(train_labels_path, y_train.astype(str).to_numpy(dtype="U"))
        np.save(test_labels_path, y_test.astype(str).to_numpy(dtype="U"))

        manifest = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "target_name": target_name,
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "transformed_feature_count": int(len(feature_names)),
            "transformed_feature_names": feature_names,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "fit_scope": "train_only",
            "split_spec_path": self._rel(self.role_artifacts["split_spec_json"].paths[0]),
            "reference_recipe": self.family_config["reference_recipe"],
        }
        write_json(manifest_path, manifest)
        artifact = RoleArtifact(
            "preprocess_bundle",
            [
                preprocessor_path,
                train_matrix_path,
                test_matrix_path,
                train_labels_path,
                test_labels_path,
                manifest_path,
            ],
            manifest,
        )
        self.role_artifacts["preprocess_bundle"] = artifact
        self._record_success(
            step_id="step_04_fit_preprocess",
            role_in="split_spec_json",
            role_out="preprocess_bundle",
            function_name="build_preprocess_bundle",
            started_at=started_at,
            input_paths=self.role_artifacts["split_spec_json"].paths + self.role_artifacts["raw_data"].paths,
            output_paths=artifact.paths,
            details={"transformed_feature_count": manifest["transformed_feature_count"]},
        )
        return artifact

    def build_model_bundle(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("model_bundle")
        role_dir.mkdir(parents=True, exist_ok=True)

        preprocess_artifact = self.role_artifacts["preprocess_bundle"]
        train_matrix = np.load(preprocess_artifact.paths[1], allow_pickle=False)
        test_matrix = np.load(preprocess_artifact.paths[2], allow_pickle=False)
        y_train = np.load(preprocess_artifact.paths[3], allow_pickle=False)
        y_test = np.load(preprocess_artifact.paths[4], allow_pickle=False)

        recipe = self.family_config["reference_recipe"]["estimator"]
        model = LogisticRegression(
            solver=recipe["solver"],
            class_weight=recipe["class_weight"],
            max_iter=recipe["max_iter"],
            random_state=self.family_config["reference_recipe"]["random_seed"],
        )
        model.fit(train_matrix, y_train)

        predictions = model.predict(test_matrix)
        prediction_frame = pd.DataFrame({"actual": y_test, "predicted": predictions})
        probability_frame = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(test_matrix)
            probability_columns = []
            for index, class_name in enumerate(model.classes_):
                column_name = f"proba_{class_name}"
                prediction_frame[column_name] = probabilities[:, index]
                probability_columns.append(column_name)
            probability_frame = prediction_frame[probability_columns]

        model_path = role_dir / "model.joblib"
        predictions_path = role_dir / "predictions.csv"
        smoke_input_path = role_dir / "smoke_input.npy"
        manifest_path = role_dir / "manifest.json"

        joblib.dump(model, model_path)
        prediction_frame.to_csv(predictions_path, index=False)
        np.save(smoke_input_path, test_matrix[:5])
        manifest = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "target_name": self.metadata["target_name"],
            "estimator_kind": "LogisticRegression",
            "estimator_params": recipe,
            "class_labels": [str(label) for label in model.classes_],
            "prediction_row_count": int(len(prediction_frame)),
            "smoke_row_count": int(min(5, len(prediction_frame))),
            "preprocess_bundle_path": self._rel(preprocess_artifact.paths[-1]),
            "probability_columns": list(probability_frame.columns) if probability_frame is not None else [],
        }
        write_json(manifest_path, manifest)
        artifact = RoleArtifact(
            "model_bundle",
            [model_path, predictions_path, smoke_input_path, manifest_path],
            manifest,
        )
        self.role_artifacts["model_bundle"] = artifact
        self._record_success(
            step_id="step_05_fit_model",
            role_in="preprocess_bundle",
            role_out="model_bundle",
            function_name="build_model_bundle",
            started_at=started_at,
            input_paths=preprocess_artifact.paths,
            output_paths=artifact.paths,
            details={"prediction_row_count": manifest["prediction_row_count"]},
        )
        return artifact

    def build_metrics(self) -> RoleArtifact:
        started_at = time.perf_counter()
        metrics_path = self._role_dir("metrics_json") / "metrics.json"
        prediction_frame = pd.read_csv(self.role_artifacts["model_bundle"].paths[1])
        probability_columns = [column for column in prediction_frame.columns if column.startswith("proba_")]
        metrics, positive_label = compute_binary_metrics(
            prediction_frame["actual"].astype(str),
            prediction_frame["predicted"].astype(str),
            prediction_frame[probability_columns] if probability_columns else None,
        )
        metrics_payload = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "task_id": self.metadata["task_id"],
            "dataset_id": self.metadata["dataset_id"],
            "target_name": self.metadata["target_name"],
            "primary_metric": self.metrics_config["primary_metric"],
            "secondary_metrics": self.metrics_config["secondary_metrics"],
            "metric_tolerance": self.metrics_config["recomputation"]["metric_tolerance"],
            "prediction_artifact": self._rel(self.role_artifacts["model_bundle"].paths[1]),
            "positive_label": positive_label,
            **metrics,
        }
        write_json(metrics_path, metrics_payload)
        artifact = RoleArtifact("metrics_json", [metrics_path], metrics_payload)
        self.role_artifacts["metrics_json"] = artifact
        self._record_success(
            step_id="step_06_compute_metrics",
            role_in="model_bundle",
            role_out="metrics_json",
            function_name="build_metrics",
            started_at=started_at,
            input_paths=self.role_artifacts["model_bundle"].paths,
            output_paths=artifact.paths,
            details={"primary_metric_value": metrics_payload[self.metrics_config["primary_metric"]]},
        )
        return artifact

    def build_report(self) -> RoleArtifact:
        started_at = time.perf_counter()
        report_path = self._role_dir("report_md") / "report.md"
        split_spec = self.role_artifacts["split_spec_json"].metadata
        metrics_payload = self.role_artifacts["metrics_json"].metadata
        warnings: list[str] = []
        if metrics_payload["balanced_accuracy"] < 0.5:
            warnings.append("Balanced accuracy is below 0.5.")

        lines = [
            f"# Reference Report: {self.task_name}",
            "",
            "## Family",
            "",
            f"- family_id: `{FAMILY_ID}`",
            f"- family_type: `{self.family_config['family_type']}`",
            f"- role_template: `{', '.join(self.family_config['role_template'])}`",
            "",
            "## Split",
            "",
            f"- task_id: `{self.metadata['task_id']}`",
            f"- dataset_id: `{self.metadata['dataset_id']}`",
            f"- target_name: `{self.metadata['target_name']}`",
            f"- train_size: `{split_spec['train_size']}`",
            f"- test_size: `{split_spec['test_size']}`",
            f"- split_source: `{split_spec['split_source']}`",
            "",
            "## Reference Recipe",
            "",
            "- deterministic note: `train-only preprocessing fit and fixed LogisticRegression recipe`",
            f"- seed: `{self.family_config['reference_recipe']['random_seed']}`",
            "",
            "## Metrics",
            "",
            f"- accuracy: `{metrics_payload['accuracy']:.6f}`",
            f"- balanced_accuracy: `{metrics_payload['balanced_accuracy']:.6f}`",
            f"- macro_f1: `{metrics_payload['macro_f1']:.6f}`",
        ]
        if "roc_auc" in metrics_payload:
            lines.append(f"- roc_auc: `{metrics_payload['roc_auc']:.6f}`")
        if "average_precision" in metrics_payload:
            lines.append(f"- average_precision: `{metrics_payload['average_precision']:.6f}`")

        lines.extend(["", "## Warnings", ""])
        if warnings:
            lines.extend([f"- {warning}" for warning in warnings])
        else:
            lines.append("- none")

        write_text(report_path, "\n".join(lines) + "\n")
        artifact = RoleArtifact("report_md", [report_path], {"warnings": warnings})
        self.role_artifacts["report_md"] = artifact
        self._record_success(
            step_id="step_07_write_report",
            role_in="metrics_json",
            role_out="report_md",
            function_name="build_report",
            started_at=started_at,
            input_paths=self.role_artifacts["metrics_json"].paths + self.role_artifacts["split_spec_json"].paths,
            output_paths=artifact.paths,
            details={"warning_count": len(warnings)},
        )
        return artifact

    def run_until(self, end_role: str = "report_md") -> dict[str, RoleArtifact]:
        for step_id, role_in, role_out, function_name in STEP_SPECS:
            if role_out in self.role_artifacts:
                if role_out == end_role:
                    break
                continue
            getattr(self, function_name)()
            if role_out == end_role:
                break

        self._write_run_summary()
        return self.role_artifacts

    def run_segment(self, start_role: str, end_role: str) -> dict[str, RoleArtifact]:
        start_index = ROLE_TEMPLATE.index(start_role)
        end_index = ROLE_TEMPLATE.index(end_role)
        if start_index > end_index:
            raise ValueError(f"Invalid segment order: {start_role} -> {end_role}")
        if start_role == "raw_data" and "raw_data" not in self.role_artifacts:
            self.materialize_raw_data()
        if start_role != "raw_data":
            self.run_until(start_role)
        for _, _, role_out, function_name in STEP_SPECS:
            if ROLE_TEMPLATE.index(role_out) <= start_index:
                continue
            if role_out in self.role_artifacts:
                continue
            getattr(self, function_name)()
            if role_out == end_role:
                break

        self._write_run_summary()
        return self.role_artifacts

    def _write_run_summary(self) -> None:
        artifact_registry = {
            role: [self._rel(path) for path in artifact.paths]
            for role, artifact in self.role_artifacts.items()
        }
        summary = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "run_namespace": self.run_namespace,
            "run_label": self.run_label,
            "role_template": ROLE_TEMPLATE,
            "artifact_registry": artifact_registry,
            "trace_path": self._rel(self.trace_path),
            "provenance_path": self._rel(self.prov_path),
            "workspace_layout_mode": self.paths.layout_mode,
        }
        write_json(self.summary_path, summary)
        self.trace.write_summary(
            self.prov_path,
            {
                "family_id": FAMILY_ID,
                "task_name": self.task_name,
                "metadata": self.metadata,
                "artifact_registry": artifact_registry,
                "reference_recipe": self.family_config["reference_recipe"],
            },
        )


def run_reference_task(task_name: str, paths: ProjectPaths | None = None) -> OpenMLBinaryReferenceRunner:
    resolved_paths = paths or detect_project_paths()
    runner = OpenMLBinaryReferenceRunner(resolved_paths, task_name)
    runner.run_until("report_md")
    return runner


def run_reference_family(task_names: list[str] | None = None, paths: ProjectPaths | None = None) -> list[OpenMLBinaryReferenceRunner]:
    resolved_paths = paths or detect_project_paths()
    names = task_names or ["adult", "credit-g", "bank-marketing"]
    runs = []
    for task_name in names:
        runs.append(run_reference_task(task_name, paths=resolved_paths))
    return runs


def main() -> None:
    paths = detect_project_paths()
    runs = run_reference_family(paths=paths)
    print("Reference pipeline complete.")
    for run in runs:
        print(f"{run.task_name}: {run.paths.relative_to_workspace(run.run_dir)}")


if __name__ == "__main__":
    main()
