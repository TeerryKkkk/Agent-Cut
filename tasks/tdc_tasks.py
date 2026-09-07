from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import load_dataset_catalog, load_family_config
from utils.io_utils import read_json, write_json


def _tdc_metadata():
    from tdc import metadata

    return metadata


def verify_tdc_benchmark_group_imports() -> dict[str, Any]:
    from tdc.benchmark_group import admet_group, drugcombo_group, dti_dg_group

    return {
        "admet_group": admet_group.__name__,
        "drugcombo_group": drugcombo_group.__name__,
        "dti_dg_group": dti_dg_group.__name__,
    }


def _admet_group(paths):
    from tdc.benchmark_group import admet_group

    benchmark_root = paths.data_dir / "tdc_benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    return admet_group(path=str(benchmark_root))


def _instance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    copy["__match_key"] = (
        copy["Drug_ID"].astype(str)
        + "||"
        + copy["Drug"].astype(str)
        + "||"
        + copy["Y"].astype(str)
    )
    copy["__match_rank"] = copy.groupby("__match_key").cumcount()
    return copy


def _match_row_ids(source_frame: pd.DataFrame, subset_frame: pd.DataFrame) -> list[int]:
    indexed_source = _instance_frame(source_frame)
    indexed_subset = _instance_frame(subset_frame)
    merged = indexed_subset.merge(
        indexed_source[["row_id", "__match_key", "__match_rank"]],
        on=["__match_key", "__match_rank"],
        how="left",
    )
    if merged["row_id"].isna().any():
        missing = int(merged["row_id"].isna().sum())
        raise RuntimeError(f"Could not map {missing} TDC split rows back to source row ids.")
    return merged["row_id"].astype(int).tolist()


def _dataset_spec(paths, task_name: str) -> dict[str, Any]:
    return load_dataset_catalog(paths)[task_name]


def export_tdc_dataset(paths, task_name: str, force: bool = False) -> dict[str, Any]:
    dataset_spec = _dataset_spec(paths, task_name)
    family_spec = load_family_config(paths, dataset_spec["family_id"])
    export_dir = paths.exports_dir / task_name
    metadata_path = export_dir / "metadata.json"
    raw_data_path = export_dir / "raw_data.parquet"
    split_indices_path = export_dir / "split_indices.json"

    if not force and metadata_path.exists() and raw_data_path.exists() and split_indices_path.exists():
        return read_json(metadata_path)

    export_dir.mkdir(parents=True, exist_ok=True)
    group = _admet_group(paths)
    benchmark = group.get(dataset_spec["benchmark_name"])
    train_val = benchmark["train_val"].copy()
    test = benchmark["test"].copy()

    with pd.option_context("mode.copy_on_write", True):
        with __import__("contextlib").redirect_stderr(__import__("io").StringIO()):
            train_split, valid_split = group.get_train_valid_split(
                seed=family_spec["reference_recipe"]["random_seed"],
                benchmark=dataset_spec["benchmark_name"],
            )

    train_val["split_membership"] = "train_val"
    test["split_membership"] = "test"
    train_val = train_val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    full_df = pd.concat([train_val, test], axis=0, ignore_index=True)
    full_df.insert(0, "row_id", np.arange(len(full_df), dtype=int))
    full_df.to_parquet(raw_data_path, index=False)

    train_val_with_ids = full_df[full_df["split_membership"] == "train_val"].copy().reset_index(drop=True)
    test_with_ids = full_df[full_df["split_membership"] == "test"].copy().reset_index(drop=True)

    split_payload = {
        "row_id_field": "row_id",
        "official_split_type": dataset_spec["official_split_type"],
        "train_val_row_ids": train_val_with_ids["row_id"].astype(int).tolist(),
        "test_row_ids": test_with_ids["row_id"].astype(int).tolist(),
        "train_row_ids": _match_row_ids(train_val_with_ids, train_split),
        "valid_row_ids": _match_row_ids(train_val_with_ids, valid_split),
    }
    write_json(split_indices_path, split_payload)

    metadata_module = _tdc_metadata()
    metric_name = metadata_module.bm_metric_names[dataset_spec["benchmark_group"]][dataset_spec["benchmark_key"]]
    split_type = metadata_module.bm_split_names[dataset_spec["benchmark_group"]][dataset_spec["benchmark_key"]]
    metadata = {
        "task_name": task_name,
        "task_id": dataset_spec["benchmark_key"],
        "dataset_id": int(dataset_spec["dataset_id"]),
        "family_id": dataset_spec["family_id"],
        "benchmark_group": dataset_spec["benchmark_group"],
        "benchmark_key": dataset_spec["benchmark_key"],
        "benchmark_category": dataset_spec["benchmark_category"],
        "target_name": dataset_spec["target_name"],
        "id_column": dataset_spec["id_column"],
        "molecule_column": dataset_spec["molecule_column"],
        "expected_task_type": dataset_spec["expected_task_type"],
        "row_count": int(len(full_df)),
        "train_val_count": int(len(train_val)),
        "test_count": int(len(test)),
        "official_metric": metric_name,
        "official_split_type": split_type,
        "split_source": f"TDC {dataset_spec['benchmark_group']} benchmark split",
        "development_role": dataset_spec.get("development_role", "family_member"),
        "notes": dataset_spec["notes"],
        "export_source": "tdc_benchmark_group",
    }
    write_json(metadata_path, metadata)
    return metadata


def export_family_datasets(paths, family_id: str, force: bool = False) -> list[dict[str, Any]]:
    catalog = load_dataset_catalog(paths)
    exported = []
    for task_name, spec in catalog.items():
        if spec["family_id"] != family_id:
            continue
        if spec.get("source") != "tdc":
            continue
        exported.append(export_tdc_dataset(paths, task_name, force=force))
    return exported


def load_exported_tdc_metadata(paths, task_name: str) -> dict[str, Any]:
    return read_json(paths.exports_dir / task_name / "metadata.json")


def build_authoritative_tdc_registry(paths) -> dict[str, Any]:
    metadata_module = _tdc_metadata()
    dataset_catalog = load_dataset_catalog(paths)
    records: list[dict[str, Any]] = []

    for task_name, spec in dataset_catalog.items():
        if spec.get("source") != "tdc":
            continue
        records.append(
            {
                "task_name": task_name,
                "family_id": spec["family_id"],
                "benchmark_group": spec["benchmark_group"],
                "benchmark_category": spec["benchmark_category"],
                "benchmark_name": spec["benchmark_name"],
                "benchmark_key": spec["benchmark_key"],
                "dataset_id": int(spec["dataset_id"]),
                "official_metric": metadata_module.bm_metric_names[spec["benchmark_group"]][spec["benchmark_key"]],
                "official_split_type": metadata_module.bm_split_names[spec["benchmark_group"]][spec["benchmark_key"]],
            }
        )

    return {
        "verified_imports": verify_tdc_benchmark_group_imports(),
        "datasets": records,
    }
