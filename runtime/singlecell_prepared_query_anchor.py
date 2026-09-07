from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipelines.family_runtime import build_reference_runner
from pipelines.singlecell_common import SINGLECELL_ROLE_TEMPLATE
from runtime.singlecell_workflow_evaluation import canonical_seed, role_index
from runtime.singlecell_repair_evaluation import (
    _first_failing_role,
    _link_prefix_artifacts,
    _metric_outcome,
    _reconstruct_run_state,
)
from utils.family_registry import load_validator_module
from utils.io_utils import append_jsonl

TARGET_FAULT_TYPE = "latent_or_graph_missing_required_key"
TARGET_SELECTED_BOUNDARY_ID = "partition_009"
TARGET_INTERNAL_ANCHOR = "prepared_query"
TARGET_FAILED_ROLE = "latent_or_graph"


def append_anchor_event(trace_path: Path, event_type: str, payload: dict[str, Any]) -> None:
    append_jsonl(trace_path, {"event_type": event_type, **payload})


def evaluate_prepared_query_anchor(
    paths,
    *,
    family_id: str,
    split_id: str,
    heldout_task: str,
    fault_type: str,
    selected_partition_id: str,
    fault_seed_run_dir: Path,
    trace_path: Path,
) -> dict[str, Any]:
    validator_module = load_validator_module(paths, family_id)
    validation_map = validator_module.validate_run_directory(fault_seed_run_dir, workspace_root=paths.workspace_root)
    actual_failed_role = _first_failing_role(validation_map)
    prepared_query_validation = validation_map["prepared_query"]

    trigger_parts = [
        f"fault_type={fault_type}",
        f"selected_partition_id={selected_partition_id}",
        f"actual_failed_role={actual_failed_role or ''}",
        f"prepared_query_valid={prepared_query_validation.passed}",
    ]
    route_allowed = (
        fault_type == TARGET_FAULT_TYPE
        and selected_partition_id == TARGET_SELECTED_BOUNDARY_ID
        and actual_failed_role == TARGET_FAILED_ROLE
        and prepared_query_validation.passed
    )
    note = ""
    if not route_allowed:
        if fault_type != TARGET_FAULT_TYPE:
            note = "Fault family is outside the focused Part-2 anchor path."
        elif selected_partition_id != TARGET_SELECTED_BOUNDARY_ID:
            note = "Selected boundary does not match the frozen reusable boundary."
        elif actual_failed_role != TARGET_FAILED_ROLE:
            note = "Observed failure role does not match the diagnosed latent_or_graph slice."
        else:
            note = f"prepared_query validation failed: {prepared_query_validation.error_code or 'unknown_error'}"
    else:
        note = "Prepared-query repair-only anchor selected."

    decision = {
        "family_id": family_id,
        "split_id": split_id,
        "heldout_task": heldout_task,
        "fault_type": fault_type,
        "selected_partition_id": selected_partition_id,
        "actual_failed_role": actual_failed_role or "",
        "prepared_query_validation": prepared_query_validation.as_dict(),
        "anchor_invoked": route_allowed,
        "anchor_name": TARGET_INTERNAL_ANCHOR if route_allowed else "",
        "rerun_start_role": TARGET_INTERNAL_ANCHOR if route_allowed else "",
        "route_trigger": "; ".join(trigger_parts),
        "note": note,
    }
    append_anchor_event(trace_path, "prepared_query_anchor_evaluated", decision)
    if route_allowed:
        append_anchor_event(
            trace_path,
            "prepared_query_anchor_selected",
            {
                "family_id": family_id,
                "split_id": split_id,
                "fault_type": fault_type,
                "selected_partition_id": selected_partition_id,
                "internal_anchor_name": TARGET_INTERNAL_ANCHOR,
                "rerun_start_role": TARGET_INTERNAL_ANCHOR,
                "route_trigger": decision["route_trigger"],
            },
        )
    return decision


def _executed_roles_from_runner(runner, anchor_role: str) -> list[str]:
    anchor_index = role_index(anchor_role)
    return [
        role
        for role in SINGLECELL_ROLE_TEMPLATE
        if role in runner.role_artifacts and role_index(role) > anchor_index
    ]


def _last_reached_role(runner, anchor_role: str) -> str:
    executed_roles = _executed_roles_from_runner(runner, anchor_role)
    if executed_roles:
        return executed_roles[-1]
    return anchor_role


def run_repair_only_anchor_rerun(
    paths,
    *,
    family_id: str,
    heldout_task: str,
    fault_seed_run_dir: Path,
    anchor_role: str,
    run_namespace: str,
    run_label: str,
    trace_path: Path,
) -> dict[str, Any]:
    source_state = _reconstruct_run_state(fault_seed_run_dir)
    seed = canonical_seed(paths, family_id, heldout_task)
    runner = build_reference_runner(
        paths,
        family_id,
        heldout_task,
        run_namespace=run_namespace,
        run_label=f"{run_label}__repair__{anchor_role}",
        seed=seed,
    )
    _link_prefix_artifacts(source_state, runner, anchor_role)
    append_anchor_event(
        trace_path,
        "prepared_query_anchor_rerun_started",
        {
            "family_id": family_id,
            "heldout_task": heldout_task,
            "anchor_role": anchor_role,
            "fault_seed_run_dir": fault_seed_run_dir.as_posix(),
            "repair_run_dir": runner.run_dir.as_posix(),
        },
    )

    started_at = time.perf_counter()
    error_message = ""
    try:
        runner.run_until("report_md")
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
    elapsed_seconds = round(time.perf_counter() - started_at, 6)

    metric_outcome = _metric_outcome(paths, family_id, heldout_task, runner.run_dir)
    executed_roles = _executed_roles_from_runner(runner, anchor_role)
    recovery_reached_artifact = _last_reached_role(runner, anchor_role)
    result = {
        "anchor_role": anchor_role,
        "run_dir": paths.relative_to_workspace(runner.run_dir),
        "run_trace_path": paths.relative_to_workspace(runner.trace_path),
        "executed_roles": executed_roles,
        "recovery_reached_artifact": recovery_reached_artifact,
        "elapsed_seconds": elapsed_seconds,
        "metric_outcome": metric_outcome,
        "error_message": error_message,
    }
    append_anchor_event(
        trace_path,
        "prepared_query_anchor_rerun_completed" if not error_message else "prepared_query_anchor_rerun_failed",
        {
            "anchor_role": anchor_role,
            "repair_run_dir": runner.run_dir.as_posix(),
            "run_trace_path": runner.trace_path.as_posix(),
            "executed_roles": executed_roles,
            "recovery_reached_artifact": recovery_reached_artifact,
            "elapsed_seconds": elapsed_seconds,
            "final_success": bool(metric_outcome["final_success"]),
            "error_message": error_message,
        },
    )
    return result
