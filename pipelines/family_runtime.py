from __future__ import annotations

import inspect
from typing import Any

from utils.family_registry import (
    list_families as list_registered_families,
    list_family_tasks,
    load_reference_runner_class,
)


def list_families(paths, family_type: str | None = None) -> list[str]:
    return list_registered_families(paths, family_type=family_type)


def build_reference_runner(
    paths,
    family_id: str,
    task_name: str,
    *,
    run_namespace: str = "reference",
    run_label: str | None = None,
    seed: int | None = None,
):
    runner_class = load_reference_runner_class(paths, family_id)
    signature = inspect.signature(runner_class)
    kwargs: dict[str, Any] = {
        "run_namespace": run_namespace,
        "run_label": run_label,
    }
    if "family_id" in signature.parameters:
        kwargs["family_id"] = family_id
    if seed is not None and "seed" in signature.parameters:
        kwargs["seed"] = seed
    return runner_class(paths, task_name, **kwargs)


def run_reference_task(
    paths,
    family_id: str,
    task_name: str,
    *,
    run_namespace: str = "reference",
    run_label: str | None = None,
    seed: int | None = None,
):
    runner = build_reference_runner(
        paths,
        family_id,
        task_name,
        run_namespace=run_namespace,
        run_label=run_label,
        seed=seed,
    )
    runner.run_until("report_md")
    return runner


def run_reference_family(
    paths,
    family_id: str,
    *,
    task_names: list[str] | None = None,
    run_namespace: str = "reference",
    seed: int | None = None,
) -> list[Any]:
    selected_task_names = task_names or list_family_tasks(paths, family_id)
    runs = []
    for task_name in selected_task_names:
        runs.append(
            run_reference_task(
                paths,
                family_id,
                task_name,
                run_namespace=run_namespace,
                seed=seed,
            )
        )
    return runs
