from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.io_utils import read_json, read_yaml, write_json
from utils.pathing import ProjectPaths

OPENML_RETRIES = 3


def load_family_config(paths: ProjectPaths, family_id: str = "openml_tabular_binary") -> dict[str, Any]:
    payload = read_yaml(paths.configs_dir / "families.yaml")
    return payload["families"][family_id]


def load_dataset_catalog(paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    payload = read_yaml(paths.configs_dir / "datasets.yaml")
    return payload["datasets"]


def list_family_tasks(paths: ProjectPaths, family_id: str = "openml_tabular_binary") -> list[str]:
    catalog = load_dataset_catalog(paths)
    return [name for name, spec in catalog.items() if spec["family_id"] == family_id]


def _ensure_openml(cache_dir: Path):
    os.environ["OPENML_CACHE_DIR"] = str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    import openml

    openml.config.cache_directory = str(cache_dir)
    return openml


def export_openml_task(paths: ProjectPaths, task_name: str, force: bool = False) -> dict[str, Any]:
    dataset_catalog = load_dataset_catalog(paths)
    family_config = load_family_config(paths)
    task_spec = dataset_catalog[task_name]
    export_dir = paths.exports_dir / task_name
    metadata_path = export_dir / "metadata.json"

    split_path = export_dir / "split_indices.json"
    raw_path = export_dir / "raw_data.parquet"
    if not force and metadata_path.exists() and split_path.exists() and raw_path.exists():
        return read_json(metadata_path)

    export_dir.mkdir(parents=True, exist_ok=True)
    split_rule = family_config["split_rule"]

    last_error: Exception | None = None
    for _ in range(OPENML_RETRIES):
        try:
            openml = _ensure_openml(paths.openml_cache_dir)
            task = openml.tasks.get_task(
                task_spec["task_id"],
                download_data=True,
                download_splits=True,
            )
            dataset = task.get_dataset()
            x_df, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe",
                target=task.target_name,
            )
            train_idx, test_idx = task.get_train_test_split_indices(
                repeat=split_rule["repeat"],
                fold=split_rule["fold"],
                sample=split_rule["sample"],
            )

            full_df = x_df.copy()
            full_df.insert(0, "row_id", np.arange(len(full_df)))
            full_df[task.target_name] = y
            raw_data_path = export_dir / "raw_data.parquet"
            train_path = export_dir / "train.csv"
            test_path = export_dir / "test.csv"
            split_indices_path = export_dir / "split_indices.json"

            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            test_df = full_df.iloc[test_idx].reset_index(drop=True)
            full_df.reset_index(drop=True).to_parquet(raw_data_path, index=False)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            write_json(
                split_indices_path,
                {
                    "row_id_field": "row_id",
                    "train_row_ids": [int(item) for item in full_df.iloc[train_idx]["row_id"].tolist()],
                    "test_row_ids": [int(item) for item in full_df.iloc[test_idx]["row_id"].tolist()],
                },
            )

            categorical_columns = [
                attribute_names[idx]
                for idx, is_categorical in enumerate(categorical_indicator)
                if is_categorical
            ]
            numerical_columns = [
                attribute_names[idx]
                for idx, is_categorical in enumerate(categorical_indicator)
                if not is_categorical
            ]

            metadata = {
                "task_name": task_name,
                "task_id": int(task.task_id),
                "dataset_id": int(dataset.dataset_id),
                "family_id": task_spec["family_id"],
                "target_name": task.target_name,
                "expected_task_type": task_spec["expected_task_type"],
                "feature_names": attribute_names,
                "categorical_columns": categorical_columns,
                "numerical_columns": numerical_columns,
                "row_count": int(len(full_df)),
                "train_row_count": int(len(train_df)),
                "test_row_count": int(len(test_df)),
                "split_source": "OpenML repeat=0 fold=0 sample=0 official split",
                "split_rule": split_rule,
                "development_role": task_spec["development_role"],
                "notes": task_spec["notes"],
                "export_source": "openml",
            }
            write_json(metadata_path, metadata)
            return metadata
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)

    raise RuntimeError(f"Failed to export OpenML task {task_name}: {last_error}")


def export_family_tasks(paths: ProjectPaths, family_id: str = "openml_tabular_binary", force: bool = False) -> list[dict[str, Any]]:
    exported = []
    for task_name in list_family_tasks(paths, family_id=family_id):
        exported.append(export_openml_task(paths, task_name, force=force))
    return exported


def load_exported_metadata(paths: ProjectPaths, task_name: str) -> dict[str, Any]:
    return read_json(paths.exports_dir / task_name / "metadata.json")


def load_exported_frame(paths: ProjectPaths, task_name: str) -> pd.DataFrame:
    export_dir = paths.exports_dir / task_name
    raw_path = export_dir / "raw_data.parquet"
    if raw_path.exists():
        return pd.read_parquet(raw_path)

    train_df = pd.read_csv(export_dir / "train.csv")
    test_df = pd.read_csv(export_dir / "test.csv")
    return pd.concat([train_df, test_df], axis=0, ignore_index=True)
