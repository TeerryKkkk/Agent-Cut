from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import compute_binary_metrics
from pipelines.tdc_reference_base import RoleArtifact, TDCReferenceRunnerBase
from utils.io_utils import read_json, write_json, write_text
from utils.pathing import detect_project_paths


class TDCBinaryReferenceRunner(TDCReferenceRunnerBase):
    def build_model_bundle(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("model_bundle")
        role_dir.mkdir(parents=True, exist_ok=True)

        preprocess_artifact = self.role_artifacts["preprocess_bundle"]
        train_matrix = np.load(preprocess_artifact.paths[1], allow_pickle=False)
        valid_matrix = np.load(preprocess_artifact.paths[2], allow_pickle=False)
        test_matrix = np.load(preprocess_artifact.paths[3], allow_pickle=False)
        y_train = np.load(preprocess_artifact.paths[4], allow_pickle=False).astype(str)
        y_valid = np.load(preprocess_artifact.paths[5], allow_pickle=False).astype(str)
        y_test = np.load(preprocess_artifact.paths[6], allow_pickle=False).astype(str)

        estimator_spec = self.family_config["reference_recipe"]["estimator"]
        model = LogisticRegression(
            solver=estimator_spec["solver"],
            class_weight=estimator_spec["class_weight"],
            max_iter=estimator_spec["max_iter"],
            random_state=self.family_config["reference_recipe"]["random_seed"],
        )
        model.fit(train_matrix, y_train)

        predictions = model.predict(test_matrix)
        prediction_frame = pd.DataFrame({"actual": y_test, "predicted": predictions})
        probability_columns: list[str] = []
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(test_matrix)
            for index, class_name in enumerate(model.classes_):
                column_name = f"proba_{class_name}"
                prediction_frame[column_name] = probabilities[:, index]
                probability_columns.append(column_name)

        model_path = role_dir / "model.joblib"
        predictions_path = role_dir / "predictions.csv"
        smoke_input_path = role_dir / "smoke_input.npy"
        manifest_path = role_dir / "manifest.json"

        joblib.dump(model, model_path)
        prediction_frame.to_csv(predictions_path, index=False)
        np.save(smoke_input_path, test_matrix[:5])
        manifest = {
            "family_id": self.family_id,
            "task_name": self.task_name,
            "task_type": self.metadata["expected_task_type"],
            "target_name": self.metadata["target_name"],
            "estimator_kind": estimator_spec["kind"],
            "estimator_params": estimator_spec,
            "class_labels": [str(label) for label in model.classes_],
            "prediction_row_count": int(len(prediction_frame)),
            "smoke_row_count": int(min(5, len(prediction_frame))),
            "probability_columns": probability_columns,
            "training_row_count": int(len(y_train)),
            "valid_row_count": int(len(y_valid)),
            "fit_data_source": "train_only",
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
        primary_metric = self.primary_metric_name()
        metrics_payload = {
            "family_id": self.family_id,
            "task_name": self.task_name,
            "task_id": self.metadata["task_id"],
            "dataset_id": self.metadata["dataset_id"],
            "target_name": self.metadata["target_name"],
            "task_type": self.metadata["expected_task_type"],
            "official_primary_metric": self.dataset_spec["official_metric"],
            "primary_metric": primary_metric,
            "secondary_metrics": self.metric_policy["secondary_metrics"],
            "metric_tolerance": self.metric_policy["recomputation"]["metric_tolerance"],
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
            details={"primary_metric_value": metrics_payload[primary_metric]},
        )
        return artifact

    def build_report(self) -> RoleArtifact:
        started_at = time.perf_counter()
        report_path = self._role_dir("report_md") / "report.md"
        split_spec = self.role_artifacts["split_spec_json"].metadata
        metrics_payload = self.role_artifacts["metrics_json"].metadata
        warnings: list[str] = []
        if metrics_payload["roc_auc"] < 0.5:
            warnings.append("AUROC is below 0.5.")

        lines = [
            f"# Reference Report: {self.task_name}",
            "",
            "## Family",
            "",
            f"- family_id: `{self.family_id}`",
            f"- benchmark_group: `{self.metadata['benchmark_group']}`",
            f"- benchmark_key: `{self.metadata['benchmark_key']}`",
            f"- role_template: `{', '.join(self.family_config['role_template'])}`",
            "",
            "## Split",
            "",
            f"- task_id: `{self.metadata['task_id']}`",
            f"- dataset_id: `{self.metadata['dataset_id']}`",
            f"- target_name: `{self.metadata['target_name']}`",
            f"- train_size: `{split_spec['train_size']}`",
            f"- valid_size: `{split_spec['valid_size']}`",
            f"- test_size: `{split_spec['test_size']}`",
            f"- split_source: `{split_spec['split_source']}`",
            f"- official_split_type: `{split_spec['official_split_type']}`",
            "",
            "## Reference Recipe",
            "",
            "- deterministic note: `SMILES char-ngram hashing with fixed LogisticRegression baseline`",
            f"- seed: `{self.family_config['reference_recipe']['random_seed']}`",
            "",
            "## Metrics",
            "",
            f"- accuracy: `{metrics_payload['accuracy']:.6f}`",
            f"- balanced_accuracy: `{metrics_payload['balanced_accuracy']:.6f}`",
            f"- macro_f1: `{metrics_payload['macro_f1']:.6f}`",
            f"- roc_auc: `{metrics_payload['roc_auc']:.6f}`",
            f"- average_precision: `{metrics_payload['average_precision']:.6f}`",
            "",
            "## Warnings",
            "",
        ]
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


def run_reference_task(task_name: str, family_id: str, paths=None) -> TDCBinaryReferenceRunner:
    resolved_paths = paths or detect_project_paths()
    runner = TDCBinaryReferenceRunner(resolved_paths, task_name, family_id)
    runner.run_until("report_md")
    return runner
