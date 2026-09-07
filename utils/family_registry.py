from __future__ import annotations

import importlib
from typing import Any

from utils.io_utils import read_yaml


def load_family_catalog(paths) -> dict[str, dict[str, Any]]:
    return read_yaml(paths.configs_dir / "families.yaml")["families"]


def load_family_config(paths, family_id: str) -> dict[str, Any]:
    return load_family_catalog(paths)[family_id]


def load_dataset_catalog(paths) -> dict[str, dict[str, Any]]:
    return read_yaml(paths.configs_dir / "datasets.yaml")["datasets"]


def load_family_metric_policy(paths, family_id: str) -> dict[str, Any]:
    return read_yaml(paths.configs_dir / "metrics.yaml")["families"][family_id]


def list_families(paths, family_type: str | None = None) -> list[str]:
    families = load_family_catalog(paths)
    family_ids = [
        family_id
        for family_id, spec in families.items()
        if family_type is None or spec.get("family_type") == family_type
    ]
    return sorted(family_ids)


def family_result_namespace(paths, family_id: str) -> str:
    family_spec = load_family_config(paths, family_id)
    if "result_namespace" in family_spec:
        return str(family_spec["result_namespace"])
    if family_spec.get("family_type") == "predictive":
        return "predictive_substrate"
    return f"{family_spec['family_type']}_substrate"


def family_dataset_names(paths, family_id: str) -> list[str]:
    catalog = load_dataset_catalog(paths)
    names = [
        dataset_name
        for dataset_name, spec in catalog.items()
        if spec["family_id"] == family_id
    ]
    return sorted(names)


def _task_loader_spec(paths, family_id: str) -> dict[str, Any] | None:
    return load_family_config(paths, family_id).get("task_loader")


def has_dynamic_task_loader(paths, family_id: str) -> bool:
    return _task_loader_spec(paths, family_id) is not None


def _task_loader_module(paths, family_id: str):
    loader_spec = _task_loader_spec(paths, family_id)
    if loader_spec is None:
        raise ValueError(f"Family {family_id} does not define a dynamic task loader.")
    return importlib.import_module(loader_spec["module"])


def ensure_family_manifests(paths, family_id: str) -> dict[str, Any] | None:
    loader_spec = _task_loader_spec(paths, family_id)
    if loader_spec is None:
        return None

    module = _task_loader_module(paths, family_id)
    write_callable = getattr(module, loader_spec.get("write_callable", "write_family_manifests"))
    return write_callable(paths)


def list_family_tasks(paths, family_id: str) -> list[str]:
    loader_spec = _task_loader_spec(paths, family_id)
    if loader_spec is None:
        return family_dataset_names(paths, family_id)

    module = _task_loader_module(paths, family_id)
    list_callable = getattr(module, loader_spec.get("list_callable", "list_task_names"))
    return list_callable(paths)


def load_task_manifest(paths, family_id: str, task_name: str) -> dict[str, Any]:
    loader_spec = _task_loader_spec(paths, family_id)
    if loader_spec is None:
        return load_dataset_catalog(paths)[task_name]

    module = _task_loader_module(paths, family_id)
    load_callable = getattr(module, loader_spec.get("load_callable", "load_task_manifest"))
    return load_callable(paths, task_name)


def load_family_manifest_summary(paths, family_id: str) -> dict[str, Any]:
    loader_spec = _task_loader_spec(paths, family_id)
    if loader_spec is None:
        raise ValueError(f"Family {family_id} does not define a manifest summary loader.")

    module = _task_loader_module(paths, family_id)
    summary_callable = getattr(module, loader_spec.get("summary_callable", "load_family_summary"))
    return summary_callable(paths)


def load_reference_runner_class(paths, family_id: str):
    family_spec = load_family_config(paths, family_id)
    reference_spec = family_spec["reference_pipeline"]
    module = importlib.import_module(reference_spec["module"])
    return getattr(module, reference_spec["runner_class"])


def load_validator_module(paths, family_id: str):
    family_spec = load_family_config(paths, family_id)
    validator_module = family_spec.get("validator_module")
    if validator_module is None:
        raise ValueError(f"Family {family_id} does not define a validator_module.")
    return importlib.import_module(validator_module)


def load_runner_specs(paths, family_id: str) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    runner_class = load_reference_runner_class(paths, family_id)
    role_template = list(getattr(runner_class, "ROLE_TEMPLATE", load_family_config(paths, family_id)["role_template"]))
    step_specs = list(getattr(runner_class, "STEP_SPECS", []))
    if not step_specs:
        raise ValueError(f"Reference runner for {family_id} does not expose STEP_SPECS.")
    return role_template, step_specs
