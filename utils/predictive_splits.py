from __future__ import annotations

from pathlib import Path
from typing import Any

from pipelines.predictive_common import family_dataset_names, load_family_config, load_family_metric_policy
from utils.io_utils import read_json, write_json


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def family_manifest_root(paths) -> Path:
    return paths.results_dir / "predictive_substrate" / "family_manifests"


def family_split_root(paths) -> Path:
    return paths.results_dir / "predictive_substrate" / "splits"


def build_family_manifest(paths, family_id: str) -> dict[str, Any]:
    family_spec = load_family_config(paths, family_id)
    metric_policy = load_family_metric_policy(paths, family_id)
    dataset_names = family_dataset_names(paths, family_id)
    return {
        "family_id": family_id,
        "family_type": family_spec["family_type"],
        "description": family_spec["description"],
        "source": family_spec.get("source"),
        "benchmark_group": family_spec.get("benchmark_group"),
        "benchmark_category": family_spec.get("benchmark_category"),
        "role_template": family_spec["role_template"],
        "split_rule": family_spec["split_rule"],
        "primary_metric": family_spec["primary_metric"],
        "metric_policy": metric_policy,
        "datasets": dataset_names,
        "dataset_count": len(dataset_names),
        "dev_split": family_spec["dev_split"],
        "reference_pipeline": family_spec["reference_pipeline"],
        "validator_roles": family_spec["validator_roles"],
    }


def build_leave_one_dataset_out_splits(paths, family_id: str) -> dict[str, Any]:
    family_manifest = build_family_manifest(paths, family_id)
    dataset_names = list(family_manifest["datasets"])
    family_spec = load_family_config(paths, family_id)
    dev_split = family_spec["dev_split"]

    splits: list[dict[str, Any]] = []
    for heldout_task in dataset_names:
        support_tasks = [task_name for task_name in dataset_names if task_name != heldout_task]
        split_id = f"{family_id}__heldout__{_slug(heldout_task)}"
        split_manifest = {
            "family_id": family_id,
            "split_id": split_id,
            "split_kind": "leave_one_dataset_out",
            "partition_unit": family_spec["split_rule"]["partition_unit"],
            "support_tasks": support_tasks,
            "heldout_task": heldout_task,
            "support_task_count": len(support_tasks),
            "selected_for_smoke": heldout_task == dev_split["heldout_task"],
            "selected_for_dev": set(support_tasks) == set(dev_split["support_tasks"]) and heldout_task == dev_split["heldout_task"],
        }
        splits.append(split_manifest)

    return {
        "family_manifest": family_manifest,
        "splits": splits,
        "dev_split": dev_split,
        "default_smoke_split_id": dev_split["split_id"],
    }


def write_family_split_manifests(paths, family_id: str) -> dict[str, Any]:
    payload = build_leave_one_dataset_out_splits(paths, family_id)

    manifest_root = family_manifest_root(paths)
    split_root = family_split_root(paths) / family_id
    manifest_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    family_manifest_path = manifest_root / f"{family_id}.json"
    write_json(family_manifest_path, payload["family_manifest"])

    split_paths: list[str] = []
    for split_manifest in payload["splits"]:
        split_path = split_root / f"{split_manifest['split_id']}.json"
        write_json(split_path, split_manifest)
        split_paths.append(paths.relative_to_workspace(split_path))

    summary = {
        "family_id": family_id,
        "family_manifest_path": paths.relative_to_workspace(family_manifest_path),
        "split_count": len(payload["splits"]),
        "split_paths": split_paths,
        "dev_split": payload["dev_split"],
    }
    write_json(split_root / "summary.json", summary)
    return summary


def load_family_manifest(paths, family_id: str) -> dict[str, Any]:
    return read_json(family_manifest_root(paths) / f"{family_id}.json")


def load_family_split_summary(paths, family_id: str) -> dict[str, Any]:
    return read_json(family_split_root(paths) / family_id / "summary.json")


def load_split_manifest(paths, family_id: str, split_id: str) -> dict[str, Any]:
    return read_json(family_split_root(paths) / family_id / f"{split_id}.json")
