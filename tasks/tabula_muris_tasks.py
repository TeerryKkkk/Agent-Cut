from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import scanpy as sc

from pipelines.singlecell_common import ensure_h5ad_download, ensure_text_download
from utils.family_registry import load_dataset_catalog, load_family_config
from utils.io_utils import read_json, write_json

FAMILY_ID = "tabula_muris_label_transfer"
DROPLET_KEY = "TM_droplet"
FACS_KEY = "TM_facs"


def _data_root(paths) -> Path:
    return paths.data_dir / "singlecell" / FAMILY_ID


def _manifest_root(paths) -> Path:
    return paths.results_dir / "singlecell_substrate" / "manifests" / FAMILY_ID


def _task_dir(paths) -> Path:
    return _manifest_root(paths) / "tasks"


def _split_dir(paths) -> Path:
    return _manifest_root(paths) / "splits"


def _summary_path(paths) -> Path:
    return _manifest_root(paths) / "summary.json"


def _raw_path(paths, dataset_key: str) -> Path:
    return _data_root(paths) / f"{dataset_key}.h5ad"


def _gene_length_path(paths) -> Path:
    return _data_root(paths) / "gene_len.txt"


def _dataset_metadata_path(paths) -> Path:
    return _data_root(paths) / "tabula_muris_dataset_metadata.json"


def _task_name(tissue: str) -> str:
    slug = tissue.lower().replace(" ", "_").replace("/", "_")
    return f"tabula_muris_tissue_{slug}"


def _split_id(task_name: str) -> str:
    return f"{FAMILY_ID}__heldout__{task_name}"


def ensure_tabula_muris_files(paths) -> dict[str, Path]:
    catalog = load_dataset_catalog(paths)
    print("[tabula_muris] ensuring droplet raw h5ad", flush=True)
    droplet_path = ensure_h5ad_download(catalog[DROPLET_KEY]["backup_url"], _raw_path(paths, DROPLET_KEY))
    print("[tabula_muris] ensuring facs raw h5ad", flush=True)
    facs_path = ensure_h5ad_download(catalog[FACS_KEY]["backup_url"], _raw_path(paths, FACS_KEY))
    print("[tabula_muris] ensuring gene length table", flush=True)
    gene_length_path = ensure_text_download(catalog[FACS_KEY]["gene_length_url"], _gene_length_path(paths))
    return {
        "droplet_path": droplet_path,
        "facs_path": facs_path,
        "gene_length_path": gene_length_path,
    }


def _obs_frame(adata, tech: str) -> pd.DataFrame:
    frame = adata.obs[["tissue", "sex", "cell_ontology_class"]].copy()
    frame["tissue"] = frame["tissue"].astype(str)
    frame["sex"] = frame["sex"].astype(str)
    frame["cell_ontology_class"] = frame["cell_ontology_class"].astype(str)
    frame["tech"] = tech
    return frame


def build_family_payload(paths) -> dict[str, Any]:
    family_spec = load_family_config(paths, FAMILY_ID)
    files = ensure_tabula_muris_files(paths)
    print("[tabula_muris] reading backed obs metadata for droplet and facs", flush=True)

    droplet_backed = ad.read_h5ad(files["droplet_path"], backed="r")
    facs_backed = ad.read_h5ad(files["facs_path"], backed="r")
    droplet_obs = _obs_frame(droplet_backed, "10x")
    facs_obs = _obs_frame(facs_backed, "SS2")
    del droplet_backed
    del facs_backed

    common_tissues = sorted(set(droplet_obs["tissue"]) & set(facs_obs["tissue"]))
    rows = []
    for tissue in common_tissues:
        droplet_subset = droplet_obs[
            (droplet_obs["sex"] == "female")
            & (droplet_obs["tissue"] == tissue)
            & (droplet_obs["cell_ontology_class"] != "nan")
        ].copy()
        facs_subset = facs_obs[
            (facs_obs["sex"] == "female")
            & (facs_obs["tissue"] == tissue)
            & (facs_obs["cell_ontology_class"] != "nan")
        ].copy()
        row = {
            "tissue": tissue,
            "droplet_cells": int(len(droplet_subset)),
            "facs_cells": int(len(facs_subset)),
            "droplet_celltypes": int(droplet_subset["cell_ontology_class"].nunique()),
            "facs_celltypes": int(facs_subset["cell_ontology_class"].nunique()),
        }
        row["eligible"] = (
            row["droplet_cells"] >= int(family_spec["split_rule"]["min_cells_per_technology"])
            and row["facs_cells"] >= int(family_spec["split_rule"]["min_cells_per_technology"])
            and min(row["droplet_celltypes"], row["facs_celltypes"]) >= int(family_spec["split_rule"]["min_cell_ontology_classes"])
        )
        row["combined_cells"] = row["droplet_cells"] + row["facs_cells"]
        rows.append(row)

    eligible_rows = [row for row in rows if row["eligible"]]
    eligible_rows.sort(key=lambda row: (-row["combined_cells"], row["tissue"]))
    top_rows = eligible_rows[: int(family_spec["split_rule"]["top_tissues_limit"])]
    top_tissues = [row["tissue"] for row in top_rows]
    print(f"[tabula_muris] eligible_tissues={len(eligible_rows)} selected_tissues={top_tissues}", flush=True)

    tasks = []
    for row in top_rows:
        task_name = _task_name(row["tissue"])
        tasks.append(
            {
                "family_id": FAMILY_ID,
                "task_name": task_name,
                "task_id": task_name,
                "tissue": row["tissue"],
                "reference_dataset_key": FACS_KEY,
                "query_dataset_key": DROPLET_KEY,
                "reference_technology": "SS2",
                "query_technology": "10x",
                "sex_filter": "female",
                "reference_n_cells": row["facs_cells"],
                "query_n_cells": row["droplet_cells"],
                "reference_label_count": row["facs_celltypes"],
                "query_label_count": row["droplet_celltypes"],
                "split_rule": family_spec["split_rule"],
                "source_files": {
                    "droplet_path": paths.relative_to_workspace(files["droplet_path"]),
                    "facs_path": paths.relative_to_workspace(files["facs_path"]),
                    "gene_length_path": paths.relative_to_workspace(files["gene_length_path"]),
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
                "split_kind": "leave_one_tissue_out",
                "partition_unit": family_spec["split_rule"]["partition_unit"],
                "heldout_task": heldout_task,
                "support_tasks": support_tasks,
                "support_task_count": len(support_tasks),
            }
        )

    default_smoke_split_id = splits[0]["split_id"] if splits else ""
    dataset_metadata = {
        "family_id": FAMILY_ID,
        "droplet_path": paths.relative_to_workspace(files["droplet_path"]),
        "facs_path": paths.relative_to_workspace(files["facs_path"]),
        "gene_length_path": paths.relative_to_workspace(files["gene_length_path"]),
        "common_tissues": common_tissues,
        "tissue_rows": rows,
        "eligible_tissues": [row["tissue"] for row in eligible_rows],
        "selected_tissues": top_tissues,
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
