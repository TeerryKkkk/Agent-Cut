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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import (
    PREDICTIVE_ROLE_TEMPLATE,
    PREDICTIVE_STEP_SPECS,
    featurize_smiles_hashing,
    load_dataset_catalog,
    load_family_config,
    load_family_metric_policy,
    reference_preprocessor_identity,
    resolve_primary_metric,
)
from provenance.trace_utils import TraceRecorder
from tasks.tdc_tasks import export_tdc_dataset, load_exported_tdc_metadata
from utils.io_utils import read_json, write_json, write_text
from utils.pathing import ProjectPaths, ensure_workspace_dirs


@dataclass
class RoleArtifact:
    role: str
    paths: list[Path]
    metadata: dict[str, Any]


class TDCReferenceRunnerBase:
    ROLE_TEMPLATE = PREDICTIVE_ROLE_TEMPLATE
    STEP_SPECS = PREDICTIVE_STEP_SPECS

    def __init__(
        self,
        paths: ProjectPaths,
        task_name: str,
        family_id: str,
        *,
        run_namespace: str = "reference",
        run_label: str | None = None,
    ) -> None:
        ensure_workspace_dirs(paths)
        export_tdc_dataset(paths, task_name)

        self.paths = paths
        self.task_name = task_name
        self.family_id = family_id
        self.family_config = load_family_config(paths, family_id)
        self.metric_policy = load_family_metric_policy(paths, family_id)
        self.dataset_catalog = load_dataset_catalog(paths)
        self.dataset_spec = self.dataset_catalog[task_name]
        self.metadata = load_exported_tdc_metadata(paths, task_name)
        self.run_namespace = run_namespace
        self.run_label = run_label

        run_root = paths.runs_dir / run_namespace / family_id / task_name
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

    def _split_payload(self) -> dict[str, Any]:
        return read_json(self.paths.exports_dir / self.task_name / "split_indices.json")

    def _raw_frame(self) -> pd.DataFrame:
        return pd.read_parquet(self.role_artifacts["raw_data"].paths[0])

    def _frame_from_ids(self, row_ids: list[int]) -> pd.DataFrame:
        raw_df = self._raw_frame()
        return raw_df[raw_df["row_id"].isin(row_ids)].copy().sort_values("row_id").reset_index(drop=True)

    def _slice_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split_spec = self.role_artifacts["split_spec_json"].metadata
        train_df = self._frame_from_ids(split_spec["train_row_ids"])
        valid_df = self._frame_from_ids(split_spec["valid_row_ids"])
        test_df = self._frame_from_ids(split_spec["test_row_ids"])
        return train_df, valid_df, test_df

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
            "family_id": self.family_id,
            "task_name": self.task_name,
            "task_id": self.metadata["task_id"],
            "dataset_id": self.metadata["dataset_id"],
            "benchmark_group": self.metadata["benchmark_group"],
            "benchmark_key": self.metadata["benchmark_key"],
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
            role_in="source_dataset",
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
        raw_df = self._raw_frame()
        target_name = self.metadata["target_name"]
        molecule_column = self.metadata["molecule_column"]
        id_column = self.metadata["id_column"]
        profile_path = self._role_dir("profile_json") / "profile.json"

        normalized_smiles = raw_df[molecule_column].fillna("").astype(str).str.strip()
        valid_smiles = int(normalized_smiles.ne("").sum())

        profile = {
            "family_id": self.family_id,
            "task_name": self.task_name,
            "task_id": self.metadata["task_id"],
            "dataset_id": self.metadata["dataset_id"],
            "target_name": target_name,
            "task_type": self.metadata["expected_task_type"],
            "row_count": int(len(raw_df)),
            "column_count": int(raw_df.shape[1] - 1),
            "feature_count": 1,
            "feature_names": [molecule_column],
            "categorical_columns": [],
            "numerical_columns": [],
            "molecule_columns": [molecule_column],
            "auxiliary_columns": [id_column],
            "dtypes": {column: str(raw_df[column].dtype) for column in raw_df.columns if column != "row_id"},
            "missing_counts": {column: int(raw_df[column].isna().sum()) for column in raw_df.columns if column != "row_id"},
            "target_present": target_name in raw_df.columns,
            "class_distribution": (
                {str(label): int(count) for label, count in raw_df[target_name].value_counts(dropna=False).items()}
                if "classification" in self.metadata["expected_task_type"]
                else {}
            ),
            "target_summary": (
                {}
                if "classification" in self.metadata["expected_task_type"]
                else {
                    "mean": float(raw_df[target_name].mean()),
                    "std": float(raw_df[target_name].std(ddof=0)),
                    "min": float(raw_df[target_name].min()),
                    "max": float(raw_df[target_name].max()),
                }
            ),
            "valid_smiles_count": valid_smiles,
            "invalid_smiles_count": int(len(raw_df) - valid_smiles),
            "unique_drug_count": int(normalized_smiles.nunique(dropna=False)),
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
            details={"valid_smiles_count": profile["valid_smiles_count"]},
        )
        return artifact

    def build_split_spec(self) -> RoleArtifact:
        started_at = time.perf_counter()
        split_payload = self._split_payload()
        split_path = self._role_dir("split_spec_json") / "split_spec.json"

        split_spec = {
            "family_id": self.family_id,
            "task_name": self.task_name,
            "task_id": self.metadata["task_id"],
            "dataset_id": self.metadata["dataset_id"],
            "target_name": self.metadata["target_name"],
            "task_type": self.metadata["expected_task_type"],
            "row_id_field": "row_id",
            "train_row_ids": split_payload["train_row_ids"],
            "valid_row_ids": split_payload["valid_row_ids"],
            "test_row_ids": split_payload["test_row_ids"],
            "train_val_row_ids": split_payload["train_val_row_ids"],
            "train_size": int(len(split_payload["train_row_ids"])),
            "valid_size": int(len(split_payload["valid_row_ids"])),
            "test_size": int(len(split_payload["test_row_ids"])),
            "train_val_size": int(len(split_payload["train_val_row_ids"])),
            "benchmark_group": self.metadata["benchmark_group"],
            "benchmark_key": self.metadata["benchmark_key"],
            "benchmark_category": self.metadata["benchmark_category"],
            "split_source": self.metadata["split_source"],
            "official_split_type": self.metadata["official_split_type"],
            "validation_split_seed": self.family_config["reference_recipe"]["random_seed"],
            "repeat": None,
            "fold": None,
            "sample": None,
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
            details={"train_size": split_spec["train_size"], "valid_size": split_spec["valid_size"], "test_size": split_spec["test_size"]},
        )
        return artifact

    def build_preprocess_bundle(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("preprocess_bundle")
        role_dir.mkdir(parents=True, exist_ok=True)

        train_df, valid_df, test_df = self._slice_frames()
        molecule_column = self.metadata["molecule_column"]
        target_name = self.metadata["target_name"]
        recipe = self.family_config["reference_recipe"]["preprocessing"]["molecule"]

        train_matrix, train_smiles, train_valid_mask = featurize_smiles_hashing(
            train_df[molecule_column],
            n_features=recipe["n_features"],
            ngram_range=tuple(recipe["ngram_range"]),
        )
        valid_matrix, valid_smiles, valid_valid_mask = featurize_smiles_hashing(
            valid_df[molecule_column],
            n_features=recipe["n_features"],
            ngram_range=tuple(recipe["ngram_range"]),
        )
        test_matrix, test_smiles, test_valid_mask = featurize_smiles_hashing(
            test_df[molecule_column],
            n_features=recipe["n_features"],
            ngram_range=tuple(recipe["ngram_range"]),
        )

        preprocessor = reference_preprocessor_identity()
        preprocessor.fit(train_matrix)

        preprocessor_path = role_dir / "preprocessor.joblib"
        train_matrix_path = role_dir / "train_matrix.npy"
        valid_matrix_path = role_dir / "valid_matrix.npy"
        test_matrix_path = role_dir / "test_matrix.npy"
        train_labels_path = role_dir / "train_labels.npy"
        valid_labels_path = role_dir / "valid_labels.npy"
        test_labels_path = role_dir / "test_labels.npy"
        manifest_path = role_dir / "manifest.json"

        joblib.dump(preprocessor, preprocessor_path)
        np.save(train_matrix_path, train_matrix.astype(np.float32))
        np.save(valid_matrix_path, valid_matrix.astype(np.float32))
        np.save(test_matrix_path, test_matrix.astype(np.float32))
        np.save(train_labels_path, train_df[target_name].to_numpy())
        np.save(valid_labels_path, valid_df[target_name].to_numpy())
        np.save(test_labels_path, test_df[target_name].to_numpy())

        manifest = {
            "family_id": self.family_id,
            "task_name": self.task_name,
            "task_type": self.metadata["expected_task_type"],
            "target_name": target_name,
            "transformed_feature_count": int(train_matrix.shape[1]),
            "fit_scope": "train_only",
            "has_validation_split": True,
            "feature_modality": "molecule_string_hashing",
            "feature_columns": [molecule_column],
            "auxiliary_columns": [self.metadata["id_column"]],
            "featurizer_name": recipe["featurizer"],
            "n_features": recipe["n_features"],
            "ngram_range": recipe["ngram_range"],
            "canonicalize_smiles": bool(recipe["canonicalize_smiles"]),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
            "valid_smiles_count": {
                "train": int(sum(train_valid_mask)),
                "valid": int(sum(valid_valid_mask)),
                "test": int(sum(test_valid_mask)),
            },
            "canonical_smiles_preview": {
                "train": train_smiles[:3],
                "valid": valid_smiles[:3],
                "test": test_smiles[:3],
            },
            "split_spec_path": self._rel(self.role_artifacts["split_spec_json"].paths[0]),
            "reference_recipe": self.family_config["reference_recipe"],
        }
        write_json(manifest_path, manifest)
        artifact = RoleArtifact(
            "preprocess_bundle",
            [
                preprocessor_path,
                train_matrix_path,
                valid_matrix_path,
                test_matrix_path,
                train_labels_path,
                valid_labels_path,
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

    def _write_run_summary(self) -> None:
        artifact_registry = {
            role: [self._rel(path) for path in artifact.paths]
            for role, artifact in self.role_artifacts.items()
        }
        summary = {
            "family_id": self.family_id,
            "task_name": self.task_name,
            "run_namespace": self.run_namespace,
            "run_label": self.run_label,
            "role_template": PREDICTIVE_ROLE_TEMPLATE,
            "artifact_registry": artifact_registry,
            "trace_path": self._rel(self.trace_path),
            "provenance_path": self._rel(self.prov_path),
            "workspace_layout_mode": self.paths.layout_mode,
        }
        write_json(self.summary_path, summary)
        self.trace.write_summary(
            self.prov_path,
            {
                "family_id": self.family_id,
                "task_name": self.task_name,
                "metadata": self.metadata,
                "artifact_registry": artifact_registry,
                "reference_recipe": self.family_config["reference_recipe"],
            },
        )

    def run_until(self, end_role: str = "report_md") -> dict[str, RoleArtifact]:
        for _, _, role_out, function_name in PREDICTIVE_STEP_SPECS:
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
        start_index = PREDICTIVE_ROLE_TEMPLATE.index(start_role)
        end_index = PREDICTIVE_ROLE_TEMPLATE.index(end_role)
        if start_index > end_index:
            raise ValueError(f"Invalid segment order: {start_role} -> {end_role}")
        if start_role == "raw_data" and "raw_data" not in self.role_artifacts:
            self.materialize_raw_data()
        if start_role != "raw_data":
            self.run_until(start_role)
        for _, _, role_out, function_name in PREDICTIVE_STEP_SPECS:
            if PREDICTIVE_ROLE_TEMPLATE.index(role_out) <= start_index:
                continue
            if role_out in self.role_artifacts:
                continue
            getattr(self, function_name)()
            if role_out == end_role:
                break

        self._write_run_summary()
        return self.role_artifacts

    def primary_metric_name(self) -> str:
        return resolve_primary_metric(self.metric_policy, self.task_name)
