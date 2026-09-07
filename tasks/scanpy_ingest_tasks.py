from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import scanpy as sc

from pipelines.singlecell_common import ensure_h5ad_download
from utils.family_registry import load_dataset_catalog, load_family_config
from utils.io_utils import read_json, write_json

FAMILY_ID = "scanpy_pancreas_ingest"
DATASET_KEY = "pancreas"


def _data_root(paths) -> Path:
    return paths.data_dir / "singlecell" / FAMILY_ID


def _dataset_path(paths) -> Path:
    return _data_root(paths) / "pancreas.h5ad"


def _dataset_metadata_path(paths) -> Path:
    return _data_root(paths) / "pancreas_dataset_metadata.json"


def _manifest_root(paths) -> Path:
    return paths.results_dir / "singlecell_substrate" / "manifests" / FAMILY_ID


def _task_dir(paths) -> Path:
    return _manifest_root(paths) / "tasks"


def _split_dir(paths) -> Path:
    return _manifest_root(paths) / "splits"


def _summary_path(paths) -> Path:
    return _manifest_root(paths) / "summary.json"


def _task_name(reference_batch: str, query_batch: str) -> str:
    return f"pancreas_ref_batch_{reference_batch}__query_batch_{query_batch}"


def _split_id(task_name: str) -> str:
    return f"{FAMILY_ID}__heldout__{task_name}"


def ensure_pancreas_dataset(paths) -> Path:
    dataset_spec = load_dataset_catalog(paths)[DATASET_KEY]
    return ensure_h5ad_download(dataset_spec["backup_url"], _dataset_path(paths))


def load_clean_pancreas(paths):
    dataset_path = ensure_pancreas_dataset(paths)
    adata = sc.read_h5ad(dataset_path)
    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].astype(str).astype("category")

    family_spec = load_family_config(paths, FAMILY_ID)
    counts = adata.obs["celltype"].value_counts()
    minority_count = int(family_spec["split_rule"]["pancreas_minority_celltypes_removed"])
    minority_celltypes = counts.index[-minority_count:].tolist()
    cleaned = adata[~adata.obs["celltype"].isin(minority_celltypes)].copy()
    cleaned.obs["batch"] = cleaned.obs["batch"].astype(str).astype("category")
    cleaned.obs["celltype"] = cleaned.obs["celltype"].astype(str).astype("category")
    return cleaned, minority_celltypes


def build_family_payload(paths) -> dict[str, Any]:
    family_spec = load_family_config(paths, FAMILY_ID)
    dataset_spec = load_dataset_catalog(paths)[DATASET_KEY]
    adata, minority_celltypes = load_clean_pancreas(paths)

    global_celltypes = sorted(adata.obs["celltype"].astype(str).unique().tolist())
    batch_counts = adata.obs.groupby("batch", observed=False).size().sort_index()
    batch_label_counts = adata.obs.groupby("batch", observed=False)["celltype"].nunique().sort_index()

    batch_rows = []
    valid_reference_batches = []
    for batch in sorted(adata.obs["batch"].astype(str).unique().tolist()):
        n_cells = int(batch_counts.loc[batch])
        label_count = int(batch_label_counts.loc[batch])
        coverage_ratio = label_count / max(1, len(global_celltypes))
        row = {
            "batch": batch,
            "n_cells": n_cells,
            "label_count": label_count,
            "coverage_ratio": round(coverage_ratio, 6),
            "is_valid_reference": (
                n_cells >= int(family_spec["split_rule"]["reference_batch_min_cells"])
                and coverage_ratio >= float(family_spec["split_rule"]["reference_batch_min_coverage_ratio"])
            ),
        }
        batch_rows.append(row)
        if row["is_valid_reference"]:
            valid_reference_batches.append(batch)

    tasks = []
    for reference_batch in valid_reference_batches:
        for query_batch in sorted(batch for batch in adata.obs["batch"].astype(str).unique().tolist() if batch != reference_batch):
            ref_subset = adata[adata.obs["batch"].astype(str) == reference_batch].copy()
            query_subset = adata[adata.obs["batch"].astype(str) == query_batch].copy()
            reference_labels = sorted(ref_subset.obs["celltype"].astype(str).unique().tolist())
            query_labels = sorted(query_subset.obs["celltype"].astype(str).unique().tolist())
            task_name = _task_name(reference_batch, query_batch)
            tasks.append(
                {
                    "family_id": FAMILY_ID,
                    "task_name": task_name,
                    "task_id": task_name,
                    "dataset_key": DATASET_KEY,
                    "reference_batch": reference_batch,
                    "query_batch": query_batch,
                    "reference_n_cells": int(ref_subset.n_obs),
                    "query_n_cells": int(query_subset.n_obs),
                    "reference_label_count": len(reference_labels),
                    "query_label_count": len(query_labels),
                    "global_label_count": len(global_celltypes),
                    "reference_coverage_ratio": round(len(reference_labels) / max(1, len(global_celltypes)), 6),
                    "query_label_overlap_with_reference": sorted(set(reference_labels) & set(query_labels)),
                    "reference_label_vocabulary": reference_labels,
                    "query_label_vocabulary": query_labels,
                    "split_rule": family_spec["split_rule"],
                    "source_dataset": {
                        "dataset_key": DATASET_KEY,
                        "backup_url": dataset_spec["backup_url"],
                    },
                }
            )

    tasks = sorted(tasks, key=lambda item: item["task_name"])
    splits = []
    for task in tasks:
        heldout_task = task["task_name"]
        support_tasks = [candidate["task_name"] for candidate in tasks if candidate["task_name"] != heldout_task]
        splits.append(
            {
                "family_id": FAMILY_ID,
                "split_id": _split_id(heldout_task),
                "split_kind": "leave_one_task_out",
                "partition_unit": family_spec["split_rule"]["partition_unit"],
                "heldout_task": heldout_task,
                "support_tasks": support_tasks,
                "support_task_count": len(support_tasks),
            }
        )

    default_smoke_split_id = splits[0]["split_id"] if splits else ""
    dataset_metadata = {
        "family_id": FAMILY_ID,
        "dataset_key": DATASET_KEY,
        "dataset_path": paths.relative_to_workspace(_dataset_path(paths)),
        "raw_shape": [int(adata.n_obs), int(adata.n_vars)],
        "global_celltype_count": len(global_celltypes),
        "minority_celltypes_removed": minority_celltypes,
        "batch_rows": batch_rows,
        "valid_reference_batches": valid_reference_batches,
    }
    family_manifest = {
        "family_id": FAMILY_ID,
        "family_type": family_spec["family_type"],
        "description": family_spec["description"],
        "role_template": family_spec["role_template"],
        "validator_roles": family_spec["validator_roles"],
        "split_rule": family_spec["split_rule"],
        "primary_metric": family_spec["primary_metric"],
        "reference_pipeline": family_spec["reference_pipeline"],
        "reference_recipe": family_spec["reference_recipe"],
        "dataset_metadata": dataset_metadata,
        "task_count": len(tasks),
        "split_count": len(splits),
        "default_smoke_split_id": default_smoke_split_id,
    }
    return {
        "family_manifest": family_manifest,
        "dataset_metadata": dataset_metadata,
        "tasks": tasks,
        "splits": splits,
        "default_smoke_split_id": default_smoke_split_id,
    }


def write_family_manifests(paths) -> dict[str, Any]:
    payload = build_family_payload(paths)
    manifest_root = _manifest_root(paths)
    manifest_root.mkdir(parents=True, exist_ok=True)
    _task_dir(paths).mkdir(parents=True, exist_ok=True)
    _split_dir(paths).mkdir(parents=True, exist_ok=True)

    family_manifest_path = manifest_root / "family_manifest.json"
    write_json(family_manifest_path, payload["family_manifest"])
    write_json(_dataset_metadata_path(paths), payload["dataset_metadata"])

    task_paths = []
    for task in payload["tasks"]:
        task_path = _task_dir(paths) / f"{task['task_name']}.json"
        write_json(task_path, task)
        task_paths.append(paths.relative_to_workspace(task_path))

    split_paths = []
    for split in payload["splits"]:
        split_path = _split_dir(paths) / f"{split['split_id']}.json"
        write_json(split_path, split)
        split_paths.append(paths.relative_to_workspace(split_path))

    summary = {
        "family_id": FAMILY_ID,
        "family_manifest_path": paths.relative_to_workspace(family_manifest_path),
        "dataset_metadata_path": paths.relative_to_workspace(_dataset_metadata_path(paths)),
        "task_count": len(payload["tasks"]),
        "split_count": len(payload["splits"]),
        "task_paths": task_paths,
        "split_paths": split_paths,
        "default_smoke_split_id": payload["default_smoke_split_id"],
    }
    write_json(_summary_path(paths), summary)
    return summary


def load_family_summary(paths) -> dict[str, Any]:
    if not _summary_path(paths).exists():
        return write_family_manifests(paths)
    return read_json(_summary_path(paths))


def list_task_names(paths) -> list[str]:
    summary = load_family_summary(paths)
    return [Path(task_path).stem for task_path in summary["task_paths"]]


def load_task_manifest(paths, task_name: str) -> dict[str, Any]:
    summary = load_family_summary(paths)
    task_path = _task_dir(paths) / f"{task_name}.json"
    if not task_path.exists():
        raise FileNotFoundError(f"Unknown {FAMILY_ID} task manifest: {task_name}. Known tasks: {summary['task_paths']}")
    return read_json(task_path)
