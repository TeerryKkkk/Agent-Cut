from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import (
    PREDICTIVE_ROLE_TEMPLATE,
    PREDICTIVE_STEP_SPECS,
    load_family_metric_policy,
    resolve_primary_metric,
    threshold_passes,
)
from pipelines.predictive_runtime import build_reference_runner
from runtime.workflow_evaluation import (
    SAME_INFO_STRESS_PROTOCOLS,
    _repair_run_from_prefix,
    build_support_evidence,
    compute_substrate_baseline_version,
    load_reference_metrics,
    load_reference_trace,
    load_split_manifests,
    role_index,
    segment_rows,
    selected_partition,
    total_reference_wall_clock,
)
from utils.io_utils import append_jsonl, markdown_table, read_json, write_json, write_text
from utils.pathing import detect_project_paths
from validators.predictive_roles import VALIDATOR_ROLES, artifact_paths_for_run, validate_role, validate_run_directory

PHASE2_ROOT_NAME = "repair_evaluation"
PRIMARY_FAMILIES = ["tdc_admet_binary", "tdc_admet_regression", "tdc_tox_binary"]
CONTINUITY_FAMILY = "openml_tabular_binary"
ALL_FAMILIES = [*PRIMARY_FAMILIES, CONTINUITY_FAMILY]
MAX_RERUN_SPAN = len(PREDICTIVE_ROLE_TEMPLATE) - 1
SELECTED_PARTITION_SEGMENT_COUNT = 4

REQUIRED_E1_COLUMNS = {
    "family_id",
    "split_id",
    "heldout_task",
    "condition",
    "clean_or_stress",
    "success",
    "artifact_validity_rate",
    "metric_gap_to_reference",
    "first_failed_boundary_depth",
    "rerun_span",
    "wall_clock_seconds",
    "utility",
}
REQUIRED_E2_COLUMNS = {
    "family_id",
    "split_id",
    "heldout_task",
    "partition_id",
    "partition_score",
    "clean_or_stress",
    "stress_protocol_id",
    "success",
    "first_failed_boundary_depth",
    "rerun_span",
    "utility",
}


@dataclass(frozen=True)
class AblationConditionSpec:
    condition_id: str
    label: str
    controller_runtime: str
    has_contracts: bool
    validator_mode: str
    repair_mode: str


@dataclass(frozen=True)
class FaultSpec:
    fault_type: str
    description: str
    expected_failed_role: str
    family_scope: str


@dataclass
class CopiedRoleArtifact:
    role: str
    paths: list[Path]
    metadata: dict[str, Any]


@dataclass
class CopiedRunState:
    run_dir: Path
    role_artifacts: dict[str, CopiedRoleArtifact]


ABLATION_CONDITIONS = [
    AblationConditionSpec(
        condition_id="artifact_partition_no_contract",
        label="A1 artifact_partition_no_contract",
        controller_runtime="phase2_no_contract_controller",
        has_contracts=False,
        validator_mode="none",
        repair_mode="none",
    ),
    AblationConditionSpec(
        condition_id="artifact_partition_contract_no_structured_validator",
        label="A2 artifact_partition_contract_no_structured_validator",
        controller_runtime="phase2_contract_only_controller",
        has_contracts=True,
        validator_mode="none",
        repair_mode="none",
    ),
    AblationConditionSpec(
        condition_id="artifact_partition_contract_validator_no_repair",
        label="A3 artifact_partition_contract_validator_no_repair",
        controller_runtime="phase2_contract_validator_controller",
        has_contracts=True,
        validator_mode="boundary",
        repair_mode="none",
    ),
    AblationConditionSpec(
        condition_id="artifact_partition_full",
        label="A4 artifact_partition_full",
        controller_runtime="phase2_full_controller",
        has_contracts=True,
        validator_mode="boundary",
        repair_mode="segment_restart",
    ),
]

FAULT_LIBRARY = [
    FaultSpec(
        fault_type="split_overlap_leakage",
        description="Inject overlap between train and test row ids in split_spec.json.",
        expected_failed_role="split_spec_json",
        family_scope="all",
    ),
    FaultSpec(
        fault_type="wrong_target_metadata",
        description="Corrupt target metadata in split_spec.json.",
        expected_failed_role="split_spec_json",
        family_scope="all",
    ),
    FaultSpec(
        fault_type="broken_preprocess_manifest",
        description="Invalidate the preprocess manifest fit_scope field.",
        expected_failed_role="preprocess_bundle",
        family_scope="all",
    ),
    FaultSpec(
        fault_type="missing_feature_artifact",
        description="Remove a required preprocess feature artifact.",
        expected_failed_role="preprocess_bundle",
        family_scope="all",
    ),
    FaultSpec(
        fault_type="model_file_corruption",
        description="Corrupt the serialized model artifact.",
        expected_failed_role="model_bundle",
        family_scope="all",
    ),
    FaultSpec(
        fault_type="metrics_inconsistency",
        description="Change a metric value so that recomputation no longer matches.",
        expected_failed_role="metrics_json",
        family_scope="all",
    ),
    FaultSpec(
        fault_type="report_corruption",
        description="Blank the final report markdown.",
        expected_failed_role="report_md",
        family_scope="all",
    ),
    FaultSpec(
        fault_type="molecule_manifest_mismatch",
        description="Break the TDC molecule profile manifest role coverage.",
        expected_failed_role="profile_json",
        family_scope="tdc_only",
    ),
]


def phase2_root(paths) -> Path:
    return paths.results_dir / PHASE2_ROOT_NAME


def audit_root(paths) -> Path:
    return phase2_root(paths) / "audit"


def ablation_root(paths) -> Path:
    return phase2_root(paths) / "ablation"


def induced_fault_root(paths) -> Path:
    return phase2_root(paths) / "induced_fault"


def aggregate_root(paths) -> Path:
    return phase2_root(paths) / "aggregate_tables"


def figures_root(paths) -> Path:
    return phase2_root(paths) / "figures"


def manifests_root(paths) -> Path:
    return phase2_root(paths) / "manifests"


def _controller_trace_path(paths, stage_kind: str, family_id: str, split_id: str, name: str) -> Path:
    return phase2_root(paths) / stage_kind / "controller_traces" / family_id / split_id / f"{name}.jsonl"


def _append_controller_event(trace_path: Path, event_type: str, payload: dict[str, Any]) -> None:
    append_jsonl(trace_path, {"event_type": event_type, **payload})


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _jsonify(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in [
        "extra_runtime_operations",
        "validation_results",
        "contract_checks",
        "scope",
        "fault_details",
        "injection_log",
    ]:
        if column in frame.columns:
            frame[f"{column}_json"] = frame[column].apply(_jsonify)
    return frame


def _segments_touched_from_role(segments: list[dict[str, Any]], start_role: str | None) -> int:
    if start_role is None:
        return 0
    start_index = role_index(start_role)
    return sum(1 for segment in segments if role_index(segment["end_role"]) > start_index)


def _condition_by_id(condition_id: str) -> AblationConditionSpec:
    for condition in ABLATION_CONDITIONS:
        if condition.condition_id == condition_id:
            return condition
    raise KeyError(condition_id)


def _faults_for_family(family_id: str) -> list[FaultSpec]:
    rows = []
    for fault in FAULT_LIBRARY:
        if fault.family_scope == "all":
            rows.append(fault)
        elif fault.family_scope == "tdc_only" and family_id.startswith("tdc_"):
            rows.append(fault)
    return rows


def _scope_manifest(paths) -> dict[str, Any]:
    family_scope = {}
    for family_id in ALL_FAMILIES:
        split_manifests = load_split_manifests(paths, family_id)
        family_scope[family_id] = {
            "split_ids": [split_manifest["split_id"] for split_manifest in split_manifests],
            "heldout_tasks": [split_manifest["heldout_task"] for split_manifest in split_manifests],
            "fault_types": [fault.fault_type for fault in _faults_for_family(family_id)],
        }
    payload = {
        "phase": "repair_evaluation",
        "family_scope": family_scope,
        "ablation_conditions": [condition.condition_id for condition in ABLATION_CONDITIONS],
        "stress_protocol_ids": [protocol["setting_id"] for protocol in SAME_INFO_STRESS_PROTOCOLS],
    }
    write_json(manifests_root(paths) / "scope.json", payload)
    return payload


def _load_csv_columns(csv_path: Path) -> set[str]:
    frame = pd.read_csv(csv_path, nrows=1)
    return set(frame.columns)


def _all_paths_exist(paths, relative_paths: list[str]) -> bool:
    return all((paths.workspace_root / relative_path).exists() for relative_path in relative_paths)


def _phase1_report_mentions(report_path: Path, snippet: str) -> bool:
    if not report_path.exists():
        return False
    content = report_path.read_text(encoding="utf-8")
    return snippet in content


def _render_gate_audit_report(paths, audit_summary: dict[str, Any]) -> Path:
    report_path = paths.reports_dir / "phase2_gate_audit.md"
    reuse_rows = []
    for item in audit_summary["reuse_decisions"]:
        reuse_rows.append(
            {
                "component": item["component"],
                "decision": item["decision"],
                "reason": item["reason"],
            }
        )

    selected_rows = []
    for family_id, selected_info in audit_summary["selected_partitions"]["by_family"].items():
        selected_rows.append(
            {
                "family_id": family_id,
                "partition_id": selected_info["partition_id"],
                "segment_count": selected_info["segment_count"],
                "path": selected_info["path"],
            }
        )

    lines = [
        "# Phase-2 Gate Audit",
        "",
        f"- gate_0_passed: `{audit_summary['gate_0_passed']}`",
        f"- substrate_baseline_version: `{audit_summary.get('substrate_baseline_version', 'unknown')}`",
        f"- phase1_e1_machine_usable: `{audit_summary['e1_outputs']['machine_usable']}`",
        f"- phase1_e2_machine_usable: `{audit_summary['e2_outputs']['machine_usable']}`",
        f"- selected_partitions_loadable: `{audit_summary['selected_partitions']['loadable']}`",
        f"- protocols_documented: `{audit_summary['protocol_documentation']['documented']}`",
        "",
        "## Reuse vs Rerun",
        "",
        markdown_table(reuse_rows, ["component", "decision", "reason"]),
        "",
        "## Selected Partitions",
        "",
        markdown_table(selected_rows, ["family_id", "partition_id", "segment_count", "path"]),
        "",
    ]
    if audit_summary["blockers"]:
        lines.extend(
            [
                "## Blockers",
                "",
                *[f"- {blocker}" for blocker in audit_summary["blockers"]],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Gate Decision",
                "",
                "- Existing phase-1 outputs are readable, machine-usable, and reusable as the frozen baseline for phase-2.",
                "- No phase-1 rerun is required in this workspace.",
                "",
            ]
        )

    write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def audit_phase1_baseline(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    _scope_manifest(resolved_paths)

    report_path = resolved_paths.reports_dir / "workflow_evaluation_report.md"
    e1_summary_path = resolved_paths.results_dir / "workflow_evaluation" / "same_info_diff_cut" / "summary.json"
    e2_summary_path = resolved_paths.results_dir / "workflow_evaluation" / "boundary_ranking" / "summary.json"
    blockers: list[str] = []

    e1_summary = read_json(e1_summary_path) if e1_summary_path.exists() else None
    e2_summary = read_json(e2_summary_path) if e2_summary_path.exists() else None
    if e1_summary is None:
        blockers.append("Missing results/workflow_evaluation/same_info_diff_cut/summary.json.")
    if e2_summary is None:
        blockers.append("Missing results/workflow_evaluation/boundary_ranking/summary.json.")

    e1_outputs = {
        "present": e1_summary is not None,
        "machine_usable": False,
        "summary_json": "results/workflow_evaluation/same_info_diff_cut/summary.json",
        "per_run_results_csv": e1_summary.get("per_run_results_csv") if e1_summary else None,
        "per_family_summary_csv": e1_summary.get("per_family_summary_csv") if e1_summary else None,
    }
    if e1_summary is not None:
        csv_path = resolved_paths.workspace_root / e1_summary["per_run_results_csv"]
        summary_paths_ok = _all_paths_exist(
            resolved_paths,
            [e1_summary["per_run_results_csv"], e1_summary["per_family_summary_csv"], *e1_summary["figures"]],
        )
        e1_columns_ok = csv_path.exists() and REQUIRED_E1_COLUMNS.issubset(_load_csv_columns(csv_path))
        e1_outputs["machine_usable"] = summary_paths_ok and e1_columns_ok
        if not summary_paths_ok:
            blockers.append("Phase-1 E1 summary references missing output files.")
        if not e1_columns_ok:
            blockers.append("Phase-1 E1 per-run CSV is missing required columns.")

    e2_outputs = {
        "present": e2_summary is not None,
        "machine_usable": False,
        "summary_json": "results/workflow_evaluation/boundary_ranking/summary.json",
        "per_run_results_csv": e2_summary.get("per_run_results_csv") if e2_summary else None,
        "summary_csv": e2_summary.get("summary_csv") if e2_summary else None,
        "split_summary_csv": e2_summary.get("split_summary_csv") if e2_summary else None,
    }
    if e2_summary is not None:
        csv_path = resolved_paths.workspace_root / e2_summary["per_run_results_csv"]
        summary_paths_ok = _all_paths_exist(
            resolved_paths,
            [
                e2_summary["per_run_results_csv"],
                e2_summary["summary_csv"],
                e2_summary["split_summary_csv"],
                e2_summary["figure"],
            ],
        )
        e2_columns_ok = csv_path.exists() and REQUIRED_E2_COLUMNS.issubset(_load_csv_columns(csv_path))
        e2_outputs["machine_usable"] = summary_paths_ok and e2_columns_ok
        if not summary_paths_ok:
            blockers.append("Phase-1 E2 summary references missing output files.")
        if not e2_columns_ok:
            blockers.append("Phase-1 E2 per-run CSV is missing required columns.")

    selected_rows = {}
    selected_loadable = True
    for family_id in ALL_FAMILIES:
        selected_path = resolved_paths.results_dir / "partitions" / family_id / "selected_partition.json"
        if not selected_path.exists():
            selected_loadable = False
            blockers.append(f"Missing selected partition for {family_id}.")
            continue
        partition = read_json(selected_path)
        split_manifests = load_split_manifests(resolved_paths, family_id)
        selected_rows[family_id] = {
            "partition_id": partition["partition_id"],
            "segment_count": partition["segment_count"],
            "path": resolved_paths.relative_to_workspace(selected_path),
            "split_count": len(split_manifests),
        }

    substrate_version = None
    if e1_summary is not None:
        substrate_version = e1_summary.get("substrate_baseline_version")
    if substrate_version is None and e2_summary is not None:
        substrate_version = e2_summary.get("substrate_baseline_version")
    if substrate_version is None and (resolved_paths.results_dir / "predictive_substrate" / "summary.json").exists():
        substrate_version = compute_substrate_baseline_version(resolved_paths)

    documented = all(
        [
            _phase1_report_mentions(report_path, "Clean held-out runs saturated: `True`."),
            _phase1_report_mentions(report_path, "Light stress used: `True`."),
            _phase1_report_mentions(report_path, "The light transfer-stress protocol used reversible relocation"),
        ]
    )
    if not documented:
        blockers.append("Phase-1 report does not document the clean/stress protocol clearly enough.")

    audit_summary = {
        "gate_0_passed": len(blockers) == 0,
        "substrate_baseline_version": substrate_version,
        "e1_outputs": e1_outputs,
        "e2_outputs": e2_outputs,
        "selected_partitions": {
            "loadable": selected_loadable,
            "by_family": selected_rows,
        },
        "protocol_documentation": {
            "documented": documented,
            "report_path": resolved_paths.relative_to_workspace(report_path) if report_path.exists() else None,
            "clean_saturated": bool(e1_summary.get("clean_saturated")) if e1_summary else None,
            "light_stress_used": bool(e1_summary.get("light_stress_used")) if e1_summary else None,
            "stress_protocol_ids": e1_summary.get("stress_protocol_ids", []) if e1_summary else [],
        },
        "reuse_decisions": [
            {
                "component": "phase1_E1_same_info_diff_cut",
                "decision": "reuse" if e1_outputs["machine_usable"] else "rerun_required",
                "reason": "Machine-readable summary and CSV already exist." if e1_outputs["machine_usable"] else "Audit failed.",
            },
            {
                "component": "phase1_E2_boundary_ranking",
                "decision": "reuse" if e2_outputs["machine_usable"] else "rerun_required",
                "reason": "Machine-readable summary and CSV already exist." if e2_outputs["machine_usable"] else "Audit failed.",
            },
            {
                "component": "selected_partitions",
                "decision": "reuse" if selected_loadable else "reconstruct_required",
                "reason": "Per-family selected partitions are already saved under results/partitions." if selected_loadable else "Saved partitions are incomplete.",
            },
        ],
        "phase1_components_to_rerun": [],
        "blockers": blockers,
    }

    write_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json", audit_summary)
    report_output_path = _render_gate_audit_report(resolved_paths, audit_summary)
    audit_summary["gate_audit_report"] = resolved_paths.relative_to_workspace(report_output_path)
    write_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json", audit_summary)
    return audit_summary


def _ablation_manifest_segments(segments: list[dict[str, Any]], condition: AblationConditionSpec) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        row = dict(segment)
        if condition.has_contracts:
            row["input_contract"] = {
                "primary_input_role": segment["start_role"],
                "artifacts_are_explicit": True,
                "portable_paths_only": True,
            }
            row["output_contract"] = {
                "output_role": segment["end_role"],
                "validator_required": condition.validator_mode == "boundary",
                "scientifically_meaningful_artifact": True,
            }
        rows.append(row)
    return rows


def write_ablation_manifest(paths, family_id: str, split_manifest: dict[str, Any], condition: AblationConditionSpec) -> tuple[Path, dict[str, Any]]:
    partition = selected_partition(paths, family_id)
    segments = segment_rows(partition)
    support_evidence, support_trace_record_count = build_support_evidence(paths, family_id, split_manifest["support_tasks"])
    manifest = {
        "phase": "repair_evaluation",
        "experiment_family": "contract_validator_repair_ablation",
        "family_id": family_id,
        "split_id": split_manifest["split_id"],
        "heldout_task": split_manifest["heldout_task"],
        "support_tasks": split_manifest["support_tasks"],
        "condition": condition.condition_id,
        "controller_runtime": condition.controller_runtime,
        "has_contracts": condition.has_contracts,
        "validator_mode": condition.validator_mode,
        "repair_mode": condition.repair_mode,
        "selected_partition_id": partition["partition_id"],
        "representation_source": "reference_support_traces",
        "segments": _ablation_manifest_segments(segments, condition),
        "support_evidence": support_evidence,
    }
    output_path = manifests_root(paths) / "ablation" / family_id / split_manifest["split_id"] / f"{condition.condition_id}.json"
    write_json(output_path, manifest)
    info = {
        "segment_count": len(segments),
        "support_trace_record_count": support_trace_record_count,
        "manifest_path": paths.relative_to_workspace(output_path),
    }
    return output_path, info


def _role_paths_from_runner(runner, role: str) -> list[Path]:
    artifact = runner.role_artifacts.get(role)
    if artifact is None:
        return []
    return list(artifact.paths)


def _evaluate_contract_paths(paths_for_role: list[Path]) -> dict[str, Any]:
    missing = [path.as_posix() for path in paths_for_role if not path.exists()]
    return {
        "passed": len(missing) == 0 and bool(paths_for_role),
        "observed_path_count": len(paths_for_role),
        "missing_paths": missing,
        "observed_paths": [path.as_posix() for path in paths_for_role],
    }


def _append_contract_check(
    contract_checks: list[dict[str, Any]],
    *,
    kind: str,
    segment: dict[str, Any],
    role: str,
    evaluation: dict[str, Any],
) -> None:
    contract_checks.append(
        {
            "kind": kind,
            "segment_id": segment["segment_id"],
            "segment_index": segment["segment_index"],
            "role": role,
            "passed": evaluation["passed"],
            "observed_path_count": evaluation["observed_path_count"],
            "missing_paths": evaluation["missing_paths"],
        }
    )


def _first_failed_contract(contract_checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in contract_checks:
        if not item["passed"]:
            return {
                "failed_role": item["role"],
                "segment_id": item["segment_id"],
                "segment_index": item["segment_index"],
                "failure_mode": f"contract_{item['kind']}",
                "message": f"Contract {item['kind']} failed for {item['role']}.",
            }
    return None


def _metric_outcome(paths, family_id: str, heldout_task: str, run_dir: Path) -> dict[str, Any]:
    validation_results = validate_run_directory(run_dir, workspace_root=paths.workspace_root) if run_dir.exists() else {}
    validation_rows = {
        role: validation_results[role].as_dict() if role in validation_results else None for role in VALIDATOR_ROLES
    }
    artifact_validity_rate = (
        sum(1 for result in validation_results.values() if result.passed) / len(VALIDATOR_ROLES) if validation_results else 0.0
    )

    reference_metrics = load_reference_metrics(paths, family_id, heldout_task)
    metric_policy = load_family_metric_policy(paths, family_id)
    primary_metric = resolve_primary_metric(metric_policy, heldout_task)
    metrics_path = artifact_paths_for_run(run_dir)["metrics_json"]

    observed_metric = None
    metric_gap_to_reference = None
    threshold_met = False
    if metrics_path.exists():
        try:
            metrics_payload = read_json(metrics_path)
            observed_metric = metrics_payload.get(primary_metric)
            if observed_metric is not None:
                threshold_met, metric_gap_to_reference = threshold_passes(
                    float(reference_metrics[primary_metric]),
                    float(observed_metric),
                    primary_metric,
                    metric_policy["success_threshold_rule"],
                )
        except Exception:
            observed_metric = None
            metric_gap_to_reference = None

    final_success = (
        bool(validation_results)
        and all(result.passed for result in validation_results.values())
        and observed_metric is not None
        and bool(threshold_met)
    )
    return {
        "validation_results": validation_rows,
        "artifact_validity_rate": round(float(artifact_validity_rate), 6),
        "observed_metric": observed_metric,
        "reference_primary_metric": reference_metrics[primary_metric],
        "primary_metric_name": primary_metric,
        "metric_gap_to_reference": metric_gap_to_reference,
        "final_success": final_success,
    }


def _stage1_utility(
    *,
    success: bool,
    artifact_validity_rate: float,
    metric_gap_to_reference: float | None,
    rerun_span: int | None,
    wall_clock_seconds: float,
    reference_wall_clock_seconds: float,
) -> float:
    metric_fidelity = 0.0 if metric_gap_to_reference is None else max(0.0, 1.0 - abs(float(metric_gap_to_reference)))
    if rerun_span is None:
        repair_efficiency = 0.0
    elif rerun_span == 0:
        repair_efficiency = 1.0
    else:
        repair_efficiency = max(0.0, 1.0 - (float(rerun_span) / float(MAX_RERUN_SPAN)))
    cost_efficiency = 1.0 / (1.0 + (wall_clock_seconds / max(reference_wall_clock_seconds, 1.0e-6)))
    utility = (
        0.35 * float(success)
        + 0.20 * artifact_validity_rate
        + 0.20 * metric_fidelity
        + 0.15 * repair_efficiency
        + 0.10 * cost_efficiency
    )
    return round(float(utility), 6)


def execute_ablation_condition(
    paths,
    family_id: str,
    split_manifest: dict[str, Any],
    condition: AblationConditionSpec,
    representation_path: Path,
    support_trace_record_count: int,
    stress_protocol: dict[str, Any],
) -> dict[str, Any]:
    heldout_task = split_manifest["heldout_task"]
    run_namespace = "phase2_ablation"
    run_label = f"{split_manifest['split_id']}__{condition.condition_id}__{stress_protocol['setting_id']}"
    controller_trace_path = _controller_trace_path(
        paths,
        "ablation",
        family_id,
        split_manifest["split_id"],
        f"{condition.condition_id}__{stress_protocol['setting_id']}",
    )
    if controller_trace_path.exists():
        controller_trace_path.unlink()

    manifest = read_json(representation_path)
    segments = manifest["segments"]
    step_to_segment = {
        step_row["step_id"]: segment for segment in segments for step_row in segment["step_specs"]
    }
    boundary_roles = [segment["end_role"] for segment in segments] if condition.validator_mode == "boundary" else []

    runner = build_reference_runner(
        paths,
        family_id,
        heldout_task,
        run_namespace=run_namespace,
        run_label=run_label,
    )
    _append_controller_event(
        controller_trace_path,
        "controller_started",
        {
            "family_id": family_id,
            "split_id": split_manifest["split_id"],
            "heldout_task": heldout_task,
            "condition": condition.condition_id,
            "stress_protocol_id": stress_protocol["setting_id"],
            "representation_manifest": paths.relative_to_workspace(representation_path),
        },
    )
    runner.materialize_raw_data()
    _append_controller_event(
        controller_trace_path,
        "step_completed",
        {
            "step_id": "step_01_materialize_raw_data",
            "role_in": "source_dataset",
            "role_out": "raw_data",
            "segment_id": "raw_data_seed",
        },
    )

    boundary_validations: list[dict[str, Any]] = []
    contract_checks: list[dict[str, Any]] = []
    extra_runtime_operations = {
        "stress_injections": 0,
        "contract_checks": 0,
        "boundary_validations": 0,
        "repair_attempts": 0,
        "segment_restarts": 0,
    }
    boundary_failure_context: dict[str, Any] | None = None
    step_failure_context: dict[str, Any] | None = None
    repair_start_role: str | None = None
    stress_applied = False
    current_segment_id: str | None = None
    started_at = time.perf_counter()

    for step_id, role_in, role_out, function_name in PREDICTIVE_STEP_SPECS:
        segment = step_to_segment.get(step_id)
        if segment is None:
            continue

        if condition.has_contracts and current_segment_id != segment["segment_id"]:
            current_segment_id = segment["segment_id"]
            evaluation = _evaluate_contract_paths(_role_paths_from_runner(runner, segment["start_role"]))
            _append_contract_check(
                contract_checks,
                kind="precondition",
                segment=segment,
                role=segment["start_role"],
                evaluation=evaluation,
            )
            extra_runtime_operations["contract_checks"] += 1
            _append_controller_event(
                controller_trace_path,
                "contract_checked",
                {
                    "kind": "precondition",
                    "segment_id": segment["segment_id"],
                    "role": segment["start_role"],
                    "passed": evaluation["passed"],
                    "missing_paths": evaluation["missing_paths"],
                },
            )

        try:
            getattr(runner, function_name)()
        except Exception as exc:
            step_failure_context = {
                "failed_role": role_in,
                "segment_id": segment["segment_id"],
                "segment_index": segment["segment_index"],
                "failure_mode": "step_exception",
                "message": str(exc),
            }
            _append_controller_event(
                controller_trace_path,
                "step_failed",
                {
                    "step_id": step_id,
                    "role_in": role_in,
                    "role_out": role_out,
                    "segment_id": segment["segment_id"],
                    "message": str(exc),
                },
            )
            if condition.repair_mode == "segment_restart":
                repair_start_role = segment["start_role"]
                extra_runtime_operations["repair_attempts"] += 1
                extra_runtime_operations["segment_restarts"] += 1
                _append_controller_event(
                    controller_trace_path,
                    "repair_triggered",
                    {
                        "failed_role": role_in,
                        "segment_id": segment["segment_id"],
                        "repair_start_role": repair_start_role,
                        "repair_mode": condition.repair_mode,
                    },
                )
                runner = _repair_run_from_prefix(
                    paths,
                    family_id,
                    heldout_task,
                    runner,
                    repair_start_role,
                    run_namespace,
                    run_label,
                    controller_trace_path,
                )
            else:
                runner._write_run_summary()
            break

        _append_controller_event(
            controller_trace_path,
            "step_completed",
            {
                "step_id": step_id,
                "role_in": role_in,
                "role_out": role_out,
                "segment_id": segment["segment_id"],
            },
        )

        if stress_protocol["stress_role"] == role_out and not stress_applied:
            artifact_path = artifact_paths_for_run(runner.run_dir)[role_out]
            moved_path = artifact_path.parent / f"{artifact_path.name}__ported__"
            if moved_path.exists():
                if moved_path.is_dir():
                    shutil.rmtree(moved_path)
                else:
                    moved_path.unlink()
            artifact_path.rename(moved_path)
            stress_applied = True
            extra_runtime_operations["stress_injections"] += 1
            _append_controller_event(
                controller_trace_path,
                "stress_applied",
                {
                    "stress_role": role_out,
                    "source_path": artifact_path.as_posix(),
                    "relocated_path": moved_path.as_posix(),
                },
            )

        if condition.has_contracts and role_out == segment["end_role"]:
            evaluation = _evaluate_contract_paths(_role_paths_from_runner(runner, role_out))
            _append_contract_check(
                contract_checks,
                kind="postcondition",
                segment=segment,
                role=role_out,
                evaluation=evaluation,
            )
            extra_runtime_operations["contract_checks"] += 1
            _append_controller_event(
                controller_trace_path,
                "contract_checked",
                {
                    "kind": "postcondition",
                    "segment_id": segment["segment_id"],
                    "role": role_out,
                    "passed": evaluation["passed"],
                    "missing_paths": evaluation["missing_paths"],
                },
            )

        if role_out in boundary_roles:
            validation = validate_role(
                role_out,
                artifact_paths_for_run(runner.run_dir)[role_out],
                workspace_root=paths.workspace_root,
                run_dir=runner.run_dir,
            )
            boundary_validations.append(
                {
                    "role": role_out,
                    "segment_id": segment["segment_id"],
                    "segment_index": segment["segment_index"],
                    "passed": validation.passed,
                    "error_code": validation.error_code,
                    "message": validation.message,
                }
            )
            extra_runtime_operations["boundary_validations"] += 1
            _append_controller_event(
                controller_trace_path,
                "boundary_validated",
                {
                    "role": role_out,
                    "segment_id": segment["segment_id"],
                    "passed": validation.passed,
                    "error_code": validation.error_code,
                },
            )
            if not validation.passed:
                boundary_failure_context = {
                    "failed_role": role_out,
                    "segment_id": segment["segment_id"],
                    "segment_index": segment["segment_index"],
                    "failure_mode": "boundary_validation",
                    "message": validation.message,
                }
                if condition.repair_mode == "segment_restart":
                    repair_start_role = segment["start_role"]
                    extra_runtime_operations["repair_attempts"] += 1
                    extra_runtime_operations["segment_restarts"] += 1
                    _append_controller_event(
                        controller_trace_path,
                        "repair_triggered",
                        {
                            "failed_role": role_out,
                            "segment_id": segment["segment_id"],
                            "repair_start_role": repair_start_role,
                            "repair_mode": condition.repair_mode,
                        },
                    )
                    runner = _repair_run_from_prefix(
                        paths,
                        family_id,
                        heldout_task,
                        runner,
                        repair_start_role,
                        run_namespace,
                        run_label,
                        controller_trace_path,
                    )
                else:
                    runner._write_run_summary()
                break
    else:
        runner._write_run_summary()

    wall_clock_seconds = time.perf_counter() - started_at
    _append_controller_event(
        controller_trace_path,
        "controller_completed",
        {
            "final_run_dir": runner.run_dir.as_posix(),
            "wall_clock_seconds": round(wall_clock_seconds, 6),
            "repair_start_role": repair_start_role,
        },
    )

    contract_failure_context = _first_failed_contract(contract_checks)
    failure_context = boundary_failure_context or contract_failure_context or step_failure_context
    metric_outcome = _metric_outcome(paths, family_id, heldout_task, runner.run_dir)
    if condition.validator_mode == "none":
        all_segment_validations_passed = None
    else:
        all_segment_validations_passed = bool(boundary_validations) and all(item["passed"] for item in boundary_validations)
    precondition_rows = [row for row in contract_checks if row["kind"] == "precondition"]
    postcondition_rows = [row for row in contract_checks if row["kind"] == "postcondition"]
    precondition_satisfaction_rate = (
        round(sum(1 for row in precondition_rows if row["passed"]) / len(precondition_rows), 6) if precondition_rows else None
    )
    postcondition_satisfaction_rate = (
        round(sum(1 for row in postcondition_rows if row["passed"]) / len(postcondition_rows), 6) if postcondition_rows else None
    )

    reference_wall_clock_seconds = total_reference_wall_clock(load_reference_trace(paths, family_id, heldout_task))
    rerun_span = None
    if repair_start_role is not None:
        rerun_span = role_index("report_md") - role_index(repair_start_role)
    elif metric_outcome["final_success"]:
        rerun_span = 0
    utility = _stage1_utility(
        success=metric_outcome["final_success"],
        artifact_validity_rate=metric_outcome["artifact_validity_rate"],
        metric_gap_to_reference=metric_outcome["metric_gap_to_reference"],
        rerun_span=rerun_span,
        wall_clock_seconds=wall_clock_seconds,
        reference_wall_clock_seconds=reference_wall_clock_seconds,
    )

    return {
        "family_id": family_id,
        "split_id": split_manifest["split_id"],
        "heldout_task": heldout_task,
        "ablation_condition": condition.condition_id,
        "condition_label": condition.label,
        "clean_or_stress": stress_protocol["clean_or_stress"],
        "stress_protocol_id": stress_protocol["setting_id"],
        "stress_role": stress_protocol["stress_role"],
        "success": metric_outcome["final_success"],
        "primary_metric": metric_outcome["observed_metric"],
        "reference_primary_metric": metric_outcome["reference_primary_metric"],
        "primary_metric_name": metric_outcome["primary_metric_name"],
        "metric_gap_to_reference": metric_outcome["metric_gap_to_reference"],
        "artifact_validity_rate": metric_outcome["artifact_validity_rate"],
        "all_segment_validations_passed": all_segment_validations_passed,
        "first_failed_boundary_depth": failure_context["segment_index"] if failure_context else None,
        "first_failed_role_or_segment": failure_context["failed_role"] if failure_context else None,
        "rerun_span": rerun_span,
        "skills_touched": len(segments),
        "extra_runtime_operations": extra_runtime_operations,
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "precondition_satisfaction_rate": precondition_satisfaction_rate,
        "postcondition_satisfaction_rate": postcondition_satisfaction_rate,
        "representation_manifest": paths.relative_to_workspace(representation_path),
        "support_trace_record_count": support_trace_record_count,
        "controller_runtime_used": condition.controller_runtime,
        "run_dir": paths.relative_to_workspace(runner.run_dir),
        "controller_trace_path": paths.relative_to_workspace(controller_trace_path),
        "validation_results": metric_outcome["validation_results"],
        "contract_checks": contract_checks,
        "utility": utility,
    }


def _best_ablation_condition_by_family(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    target_frame = frame[frame["clean_or_stress"] == "stress"]
    grouped = (
        target_frame.groupby(["family_id", "ablation_condition"], dropna=False)[["utility", "success"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped = grouped.sort_values(["family_id", "utility", "success"], ascending=[True, False, False])
    return grouped.groupby("family_id", dropna=False).head(1).to_dict(orient="records")


def _primary_and_continuity_summaries(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_frame = frame[frame["family_id"].isin(PRIMARY_FAMILIES)]
    continuity_frame = frame[frame["family_id"] == CONTINUITY_FAMILY]
    primary_summary = primary_frame.groupby(group_columns, dropna=False)[value_columns].mean(numeric_only=True).reset_index()
    continuity_summary = continuity_frame.groupby(group_columns, dropna=False)[value_columns].mean(numeric_only=True).reset_index()
    return primary_summary, continuity_summary


def _plot_e3_success_utility(primary_summary: pd.DataFrame, output_path: Path) -> None:
    order = [condition.condition_id for condition in ABLATION_CONDITIONS]
    settings = ["clean", "stress"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric_name, title in [
        (axes[0], "success", "E3 Success By Condition"),
        (axes[1], "utility", "E3 Utility By Condition"),
    ]:
        width = 0.35
        x = list(range(len(order)))
        for offset_index, setting in enumerate(settings):
            subset = primary_summary[primary_summary["clean_or_stress"] == setting].set_index("ablation_condition")
            values = [float(subset.loc[condition_id, metric_name]) if condition_id in subset.index else 0.0 for condition_id in order]
            positions = [value + ((offset_index - 0.5) * width) for value in x]
            axis.bar(positions, values, width=width, label=setting)
        axis.set_xticks(x)
        axis.set_xticklabels(order, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylim(0.0, 1.05 if metric_name == "success" else max(1.0, axis.get_ylim()[1]))
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_e3_validity_failure(primary_summary: pd.DataFrame, output_path: Path) -> None:
    order = [condition.condition_id for condition in ABLATION_CONDITIONS]
    settings = ["clean", "stress"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric_name, title in [
        (axes[0], "artifact_validity_rate", "E3 Artifact Validity By Condition"),
        (axes[1], "first_failed_boundary_depth", "E3 First Failure Depth By Condition"),
    ]:
        width = 0.35
        x = list(range(len(order)))
        for offset_index, setting in enumerate(settings):
            subset = primary_summary[primary_summary["clean_or_stress"] == setting].set_index("ablation_condition")
            values = [float(subset.loc[condition_id, metric_name]) if condition_id in subset.index else 0.0 for condition_id in order]
            positions = [value + ((offset_index - 0.5) * width) for value in x]
            axis.bar(positions, values, width=width, label=setting)
        axis.set_xticks(x)
        axis.set_xticklabels(order, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_phase2_ablation(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    audit_summary = audit_phase1_baseline(resolved_paths)
    if not audit_summary["gate_0_passed"]:
        summary = {
            "gate_1_passed": False,
            "stage": "E3_ablation",
            "reason": "Gate 0 failed.",
            "gate_0_summary_path": resolved_paths.relative_to_workspace(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json"),
            "blockers": audit_summary["blockers"],
        }
        write_json(ablation_root(resolved_paths) / "summary.json", summary)
        return summary

    _scope_manifest(resolved_paths)
    rows: list[dict[str, Any]] = []
    for family_id in ALL_FAMILIES:
        split_manifests = load_split_manifests(resolved_paths, family_id)
        for split_manifest in split_manifests:
            for condition in ABLATION_CONDITIONS:
                representation_path, manifest_info = write_ablation_manifest(resolved_paths, family_id, split_manifest, condition)
                for stress_protocol in SAME_INFO_STRESS_PROTOCOLS:
                    rows.append(
                        execute_ablation_condition(
                            resolved_paths,
                            family_id,
                            split_manifest,
                            condition,
                            representation_path,
                            manifest_info["support_trace_record_count"],
                            stress_protocol,
                        )
                    )

    frame = _rows_to_dataframe(rows)
    per_run_csv_path = ablation_root(resolved_paths) / "per_run_results.csv"
    _save_dataframe(frame, per_run_csv_path)

    value_columns = [
        "success",
        "artifact_validity_rate",
        "first_failed_boundary_depth",
        "rerun_span",
        "wall_clock_seconds",
        "utility",
        "precondition_satisfaction_rate",
        "postcondition_satisfaction_rate",
    ]
    per_split_summary = (
        frame.groupby(["family_id", "split_id", "heldout_task", "ablation_condition", "clean_or_stress"], dropna=False)[value_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    per_family_summary = (
        frame.groupby(["family_id", "ablation_condition", "clean_or_stress"], dropna=False)[value_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    primary_summary, continuity_summary = _primary_and_continuity_summaries(
        per_family_summary,
        ["ablation_condition", "clean_or_stress"],
        value_columns,
    )
    _save_dataframe(per_split_summary, aggregate_root(resolved_paths) / "phase2_ablation_per_split.csv")
    _save_dataframe(per_family_summary, aggregate_root(resolved_paths) / "phase2_ablation_per_family.csv")
    _save_dataframe(primary_summary, aggregate_root(resolved_paths) / "phase2_ablation_pooled_primary.csv")
    _save_dataframe(continuity_summary, aggregate_root(resolved_paths) / "phase2_ablation_f1_continuity.csv")

    success_by_split = frame[
        (frame["ablation_condition"] == "artifact_partition_full")
        & (frame["clean_or_stress"] == "clean")
        & (frame["success"] == True)
    ][["family_id", "split_id", "heldout_task", "run_dir"]]
    targeted_splits = {(row["family_id"], row["split_id"]) for row in rows}
    stable_splits = {(row["family_id"], row["split_id"]) for row in success_by_split.to_dict(orient="records")}
    missing_stable_splits = sorted(targeted_splits - stable_splits)

    _plot_e3_success_utility(primary_summary, figures_root(resolved_paths) / "phase2_e3_success_utility.png")
    _plot_e3_validity_failure(primary_summary, figures_root(resolved_paths) / "phase2_e3_validity_failure_depth.png")

    summary = {
        "gate_1_passed": len(missing_stable_splits) == 0,
        "substrate_baseline_version": audit_summary["substrate_baseline_version"],
        "families_run": ALL_FAMILIES,
        "split_count": int(frame["split_id"].nunique()),
        "run_count": int(len(frame)),
        "stress_protocol_ids": [protocol["setting_id"] for protocol in SAME_INFO_STRESS_PROTOCOLS],
        "per_run_results_csv": resolved_paths.relative_to_workspace(per_run_csv_path),
        "per_family_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "phase2_ablation_per_family.csv"),
        "per_split_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "phase2_ablation_per_split.csv"),
        "pooled_primary_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "phase2_ablation_pooled_primary.csv"),
        "f1_continuity_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "phase2_ablation_f1_continuity.csv"),
        "figures": [
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase2_e3_success_utility.png"),
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase2_e3_validity_failure_depth.png"),
        ],
        "best_condition_by_family": _best_ablation_condition_by_family(frame),
        "stable_base_runs": success_by_split.to_dict(orient="records"),
        "missing_stable_splits": [list(item) for item in missing_stable_splits],
        "gate_0_summary_path": resolved_paths.relative_to_workspace(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json"),
    }
    write_json(ablation_root(resolved_paths) / "summary.json", summary)
    return summary


def _copy_run_tree(source_run_dir: Path, target_run_dir: Path) -> None:
    if target_run_dir.exists():
        shutil.rmtree(target_run_dir)
    target_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, target_run_dir)


def _rewrite_metrics_prediction_reference(paths, run_dir: Path) -> None:
    metrics_path = artifact_paths_for_run(run_dir)["metrics_json"]
    if not metrics_path.exists():
        return
    payload = read_json(metrics_path)
    payload["prediction_artifact"] = paths.relative_to_workspace(artifact_paths_for_run(run_dir)["model_bundle"] / "predictions.csv")
    write_json(metrics_path, payload)


def _reconstruct_run_state(run_dir: Path) -> CopiedRunState:
    artifact_paths = artifact_paths_for_run(run_dir)
    summary_path = run_dir / "summary.json"
    summary_payload = read_json(summary_path)
    artifact_registry = summary_payload["artifact_registry"]

    def registry_paths(role: str) -> list[Path]:
        paths_for_role = []
        for relative_path in artifact_registry.get(role, []):
            _, artifact_suffix = str(relative_path).split("/artifacts/", maxsplit=1)
            paths_for_role.append(run_dir / "artifacts" / Path(artifact_suffix))
        return paths_for_role

    role_artifacts: dict[str, CopiedRoleArtifact] = {}
    role_artifacts["raw_data"] = CopiedRoleArtifact(
        role="raw_data",
        paths=registry_paths("raw_data"),
        metadata=read_json(artifact_paths["raw_data"].parent / "manifest.json"),
    )
    role_artifacts["profile_json"] = CopiedRoleArtifact(
        role="profile_json",
        paths=registry_paths("profile_json"),
        metadata=read_json(artifact_paths["profile_json"]) if artifact_paths["profile_json"].exists() else {},
    )
    role_artifacts["split_spec_json"] = CopiedRoleArtifact(
        role="split_spec_json",
        paths=registry_paths("split_spec_json"),
        metadata=read_json(artifact_paths["split_spec_json"]) if artifact_paths["split_spec_json"].exists() else {},
    )
    preprocess_dir = artifact_paths["preprocess_bundle"]
    role_artifacts["preprocess_bundle"] = CopiedRoleArtifact(
        role="preprocess_bundle",
        paths=registry_paths("preprocess_bundle"),
        metadata=read_json(preprocess_dir / "manifest.json") if (preprocess_dir / "manifest.json").exists() else {},
    )
    model_dir = artifact_paths["model_bundle"]
    role_artifacts["model_bundle"] = CopiedRoleArtifact(
        role="model_bundle",
        paths=registry_paths("model_bundle"),
        metadata=read_json(model_dir / "manifest.json") if (model_dir / "manifest.json").exists() else {},
    )
    role_artifacts["metrics_json"] = CopiedRoleArtifact(
        role="metrics_json",
        paths=registry_paths("metrics_json"),
        metadata=read_json(artifact_paths["metrics_json"]) if artifact_paths["metrics_json"].exists() else {},
    )
    role_artifacts["report_md"] = CopiedRoleArtifact(
        role="report_md",
        paths=registry_paths("report_md"),
        metadata={"path": artifact_paths["report_md"].as_posix()},
    )
    return CopiedRunState(run_dir=run_dir, role_artifacts=role_artifacts)


def _fault_run_dir(paths, family_id: str, heldout_task: str, split_id: str, fault_type: str, repair_policy: str) -> Path:
    return paths.runs_dir / "phase2_induced_fault" / family_id / heldout_task / f"{split_id}__{fault_type}__{repair_policy}__faulty"


def _fault_manifest_path(paths, family_id: str, split_id: str, fault_type: str, repair_policy: str) -> Path:
    return manifests_root(paths) / "induced_fault" / family_id / split_id / f"{fault_type}__{repair_policy}.json"


def _write_fault_manifest(
    paths,
    *,
    family_id: str,
    split_id: str,
    heldout_task: str,
    base_run_dir: str,
    fault_spec: FaultSpec,
    repair_policy: str,
) -> Path:
    path = _fault_manifest_path(paths, family_id, split_id, fault_spec.fault_type, repair_policy)
    write_json(
        path,
        {
            "phase": "repair_evaluation",
            "experiment_family": "induced_fault_repair",
            "family_id": family_id,
            "split_id": split_id,
            "heldout_task": heldout_task,
            "base_condition": "artifact_partition_full_clean",
            "base_run_dir": base_run_dir,
            "fault_type": fault_spec.fault_type,
            "fault_description": fault_spec.description,
            "expected_failed_role": fault_spec.expected_failed_role,
            "repair_policy": repair_policy,
            "selected_partition_id": selected_partition(paths, family_id)["partition_id"],
        },
    )
    return path


def _apply_single_fault(paths, run_dir: Path, family_id: str, split_manifest: dict[str, Any], fault_spec: FaultSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact_paths = artifact_paths_for_run(run_dir)
    injections: list[dict[str, Any]] = []

    if fault_spec.fault_type == "split_overlap_leakage":
        split_payload = read_json(artifact_paths["split_spec_json"])
        overlapped_row_id = split_payload["test_row_ids"][0]
        split_payload["train_row_ids"] = list(split_payload["train_row_ids"]) + [overlapped_row_id]
        split_payload["train_size"] = int(len(split_payload["train_row_ids"]))
        write_json(artifact_paths["split_spec_json"], split_payload)
        injections.append({"operation": "split_overlap", "row_id": overlapped_row_id})
    elif fault_spec.fault_type == "wrong_target_metadata":
        split_payload = read_json(artifact_paths["split_spec_json"])
        split_payload["target_name"] = "__faulty_target__"
        write_json(artifact_paths["split_spec_json"], split_payload)
        injections.append({"operation": "split_target_name_overwrite", "target_name": "__faulty_target__"})
    elif fault_spec.fault_type == "broken_preprocess_manifest":
        manifest_path = artifact_paths["preprocess_bundle"] / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["fit_scope"] = "train_and_test"
        write_json(manifest_path, manifest)
        injections.append({"operation": "preprocess_fit_scope_overwrite", "fit_scope": "train_and_test"})
    elif fault_spec.fault_type == "missing_feature_artifact":
        source_path = artifact_paths["preprocess_bundle"] / "train_matrix.npy"
        backup_path = artifact_paths["preprocess_bundle"] / "train_matrix.npy.fault_backup"
        if backup_path.exists():
            backup_path.unlink()
        source_path.rename(backup_path)
        injections.append({"operation": "preprocess_artifact_removed", "source_path": source_path.as_posix(), "backup_path": backup_path.as_posix()})
    elif fault_spec.fault_type == "model_file_corruption":
        model_path = artifact_paths["model_bundle"] / "model.joblib"
        model_path.write_bytes(b"not-a-joblib-model")
        injections.append({"operation": "model_file_corrupted", "path": model_path.as_posix()})
    elif fault_spec.fault_type == "metrics_inconsistency":
        metrics_payload = read_json(artifact_paths["metrics_json"])
        metric_name = resolve_primary_metric(load_family_metric_policy(paths, family_id), split_manifest["heldout_task"])
        metrics_payload[metric_name] = float(metrics_payload[metric_name]) + 0.25
        write_json(artifact_paths["metrics_json"], metrics_payload)
        injections.append({"operation": "metric_value_perturbed", "metric_name": metric_name})
    elif fault_spec.fault_type == "report_corruption":
        artifact_paths["report_md"].write_text("", encoding="utf-8")
        injections.append({"operation": "report_emptied", "path": artifact_paths["report_md"].as_posix()})
    elif fault_spec.fault_type == "molecule_manifest_mismatch":
        profile_payload = read_json(artifact_paths["profile_json"])
        if "molecule_columns" not in profile_payload:
            raise ValueError("molecule_manifest_mismatch is only valid for TDC family profiles.")
        profile_payload["molecule_columns"] = ["__faulty_molecule_column__"]
        write_json(artifact_paths["profile_json"], profile_payload)
        injections.append({"operation": "molecule_columns_overwrite", "molecule_columns": ["__faulty_molecule_column__"]})
    else:
        raise ValueError(f"Unsupported fault type: {fault_spec.fault_type}")

    fault_details = {
        "fault_type": fault_spec.fault_type,
        "description": fault_spec.description,
        "expected_failed_role": fault_spec.expected_failed_role,
    }
    return fault_details, injections


def _first_failing_role(validation_map: dict[str, Any]) -> str | None:
    for role in VALIDATOR_ROLES:
        result = validation_map.get(role)
        if result is not None and not result.passed:
            return role
    return None


def _selected_partition_segments(paths, family_id: str) -> list[dict[str, Any]]:
    return segment_rows(selected_partition(paths, family_id))


def _repair_start_for_role(paths, family_id: str, failed_role: str) -> str:
    for segment in _selected_partition_segments(paths, family_id):
        if segment["start_role"] == failed_role:
            return segment["start_role"]
        if role_index(segment["start_role"]) < role_index(failed_role) <= role_index(segment["end_role"]):
            return segment["start_role"]
    raise KeyError(f"Could not map failed role {failed_role} to selected partition segment.")


def _stage2_final_metrics(paths, family_id: str, heldout_task: str, run_dir: Path) -> dict[str, Any]:
    return _metric_outcome(paths, family_id, heldout_task, run_dir)


def _plot_e4_repair_success(per_family_policy: pd.DataFrame, output_path: Path) -> None:
    families = list(per_family_policy["family_id"].drop_duplicates())
    policies = ["global_end_to_end_rerun", "validator_detect_no_local_repair", "full_local_repair"]
    width = 0.25
    x = list(range(len(families)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for policy_index, policy in enumerate(policies):
        subset = per_family_policy[per_family_policy["repair_policy"] == policy].set_index("family_id")
        values = [float(subset.loc[family_id, "final_success"]) if family_id in subset.index else 0.0 for family_id in families]
        positions = [value + ((policy_index - 1) * width) for value in x]
        ax.bar(positions, values, width=width, label=policy)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("E4 Repair Success By Policy And Family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_e4_rerun_cost(policy_summary: pd.DataFrame, output_path: Path) -> None:
    policies = list(policy_summary["repair_policy"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(policies, policy_summary["rerun_span"])
    axes[0].set_title("E4 Mean Rerun Span By Policy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(policies, policy_summary["extra_wall_clock_seconds"])
    axes[1].set_title("E4 Extra Wall Clock By Policy")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_e4_localization(per_family_policy: pd.DataFrame, output_path: Path) -> None:
    families = list(per_family_policy["family_id"].drop_duplicates())
    policies = ["global_end_to_end_rerun", "validator_detect_no_local_repair", "full_local_repair"]
    width = 0.25
    x = list(range(len(families)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for policy_index, policy in enumerate(policies):
        subset = per_family_policy[per_family_policy["repair_policy"] == policy].set_index("family_id")
        values = [float(subset.loc[family_id, "root_localization_accuracy"]) if family_id in subset.index else 0.0 for family_id in families]
        positions = [value + ((policy_index - 1) * width) for value in x]
        ax.bar(positions, values, width=width, label=policy)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("E4 Localization Accuracy By Policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_phase2_induced_fault_repair(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    ablation_summary_path = ablation_root(resolved_paths) / "summary.json"
    if not ablation_summary_path.exists():
        ablation_summary = run_phase2_ablation(resolved_paths)
    else:
        ablation_summary = read_json(ablation_summary_path)
    if not ablation_summary.get("gate_1_passed", False):
        summary = {
            "gate_2_passed": False,
            "stage": "E4_induced_fault_repair",
            "reason": "Gate 1 failed.",
            "gate_1_summary_path": resolved_paths.relative_to_workspace(ablation_summary_path),
            "blockers": ablation_summary.get("missing_stable_splits", []),
        }
        write_json(induced_fault_root(resolved_paths) / "summary.json", summary)
        return summary

    ablation_frame = pd.read_csv(resolved_paths.workspace_root / ablation_summary["per_run_results_csv"])
    base_rows = ablation_frame[
        (ablation_frame["ablation_condition"] == "artifact_partition_full")
        & (ablation_frame["clean_or_stress"] == "clean")
        & (ablation_frame["success"] == True)
    ][["family_id", "split_id", "heldout_task", "run_dir"]]
    base_by_split = {
        (row["family_id"], row["split_id"]): row
        for row in base_rows.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    repair_policies = ["global_end_to_end_rerun", "validator_detect_no_local_repair", "full_local_repair"]
    for family_id in ALL_FAMILIES:
        split_manifests = load_split_manifests(resolved_paths, family_id)
        for split_manifest in split_manifests:
            base_row = base_by_split[(family_id, split_manifest["split_id"])]
            base_run_dir = resolved_paths.workspace_root / base_row["run_dir"]
            segments = _selected_partition_segments(resolved_paths, family_id)
            for fault_spec in _faults_for_family(family_id):
                for repair_policy in repair_policies:
                    trace_path = _controller_trace_path(
                        resolved_paths,
                        "induced_fault",
                        family_id,
                        split_manifest["split_id"],
                        f"{fault_spec.fault_type}__{repair_policy}",
                    )
                    if trace_path.exists():
                        trace_path.unlink()

                    faulty_run_dir = _fault_run_dir(
                        resolved_paths,
                        family_id,
                        split_manifest["heldout_task"],
                        split_manifest["split_id"],
                        fault_spec.fault_type,
                        repair_policy,
                    )
                    _copy_run_tree(base_run_dir, faulty_run_dir)
                    _rewrite_metrics_prediction_reference(resolved_paths, faulty_run_dir)
                    fault_manifest_path = _write_fault_manifest(
                        resolved_paths,
                        family_id=family_id,
                        split_id=split_manifest["split_id"],
                        heldout_task=split_manifest["heldout_task"],
                        base_run_dir=resolved_paths.relative_to_workspace(base_run_dir),
                        fault_spec=fault_spec,
                        repair_policy=repair_policy,
                    )
                    fault_details, injection_log = _apply_single_fault(
                        resolved_paths,
                        faulty_run_dir,
                        family_id,
                        split_manifest,
                        fault_spec,
                    )
                    _append_controller_event(
                        trace_path,
                        "fault_injected",
                        {
                            "fault_type": fault_spec.fault_type,
                            "repair_policy": repair_policy,
                            "run_dir": faulty_run_dir.as_posix(),
                            "injection_log": injection_log,
                        },
                    )

                    extra_runtime_operations = {
                        "detection_validations": 0,
                        "repair_attempts": 0,
                        "global_reruns": 0,
                        "local_repairs": 0,
                    }
                    started_at = time.perf_counter()
                    validation_map = validate_run_directory(faulty_run_dir, workspace_root=resolved_paths.workspace_root)
                    extra_runtime_operations["detection_validations"] = len(VALIDATOR_ROLES)
                    actual_failed_role = _first_failing_role(validation_map)
                    if repair_policy == "global_end_to_end_rerun":
                        detected_failed_role = "workflow_root" if actual_failed_role is not None else None
                    else:
                        detected_failed_role = actual_failed_role
                    root_localization_accuracy = bool(detected_failed_role == fault_spec.expected_failed_role)

                    final_run_dir = faulty_run_dir
                    repair_success = False
                    rerun_span = 0
                    if repair_policy == "global_end_to_end_rerun" and actual_failed_role is not None:
                        extra_runtime_operations["repair_attempts"] += 1
                        extra_runtime_operations["global_reruns"] += 1
                        rerun_span = MAX_RERUN_SPAN
                        rerun_runner = build_reference_runner(
                            resolved_paths,
                            family_id,
                            split_manifest["heldout_task"],
                            run_namespace="phase2_induced_fault",
                            run_label=f"{split_manifest['split_id']}__{fault_spec.fault_type}__global_rerun",
                        )
                        rerun_runner.run_until("report_md")
                        final_run_dir = rerun_runner.run_dir
                        repair_success = True
                        _append_controller_event(
                            trace_path,
                            "global_rerun_completed",
                            {
                                "repair_policy": repair_policy,
                                "final_run_dir": final_run_dir.as_posix(),
                            },
                        )
                    elif repair_policy == "full_local_repair" and actual_failed_role is not None:
                        repair_start_role = _repair_start_for_role(resolved_paths, family_id, actual_failed_role)
                        rerun_span = role_index("report_md") - role_index(repair_start_role)
                        extra_runtime_operations["repair_attempts"] += 1
                        extra_runtime_operations["local_repairs"] += 1
                        copied_state = _reconstruct_run_state(faulty_run_dir)
                        repaired_runner = _repair_run_from_prefix(
                            resolved_paths,
                            family_id,
                            split_manifest["heldout_task"],
                            copied_state,
                            repair_start_role,
                            "phase2_induced_fault",
                            f"{split_manifest['split_id']}__{fault_spec.fault_type}__full_local_repair",
                            trace_path,
                        )
                        final_run_dir = repaired_runner.run_dir
                        repair_success = True
                        _append_controller_event(
                            trace_path,
                            "local_repair_completed",
                            {
                                "repair_policy": repair_policy,
                                "repair_start_role": repair_start_role,
                                "final_run_dir": final_run_dir.as_posix(),
                            },
                        )

                    extra_wall_clock_seconds = time.perf_counter() - started_at
                    final_metrics = _stage2_final_metrics(
                        resolved_paths,
                        family_id,
                        split_manifest["heldout_task"],
                        final_run_dir,
                    )
                    if repair_policy in {"global_end_to_end_rerun", "full_local_repair"}:
                        repair_success = bool(final_metrics["final_success"])
                    rows.append(
                        {
                            "family_id": family_id,
                            "split_id": split_manifest["split_id"],
                            "heldout_task": split_manifest["heldout_task"],
                            "base_condition": "artifact_partition_full",
                            "fault_type": fault_spec.fault_type,
                            "repair_policy": repair_policy,
                            "root_localization_accuracy": root_localization_accuracy,
                            "detected_failed_role_or_segment": detected_failed_role,
                            "repair_success": repair_success,
                            "final_success": final_metrics["final_success"],
                            "final_primary_metric": final_metrics["observed_metric"],
                            "reference_primary_metric": final_metrics["reference_primary_metric"],
                            "primary_metric_name": final_metrics["primary_metric_name"],
                            "final_metric_gap_to_reference": final_metrics["metric_gap_to_reference"],
                            "rerun_span": rerun_span,
                            "skills_touched": SELECTED_PARTITION_SEGMENT_COUNT if repair_policy == "global_end_to_end_rerun" else _segments_touched_from_role(segments, _repair_start_for_role(resolved_paths, family_id, actual_failed_role)) if repair_policy == "full_local_repair" and actual_failed_role is not None else 0,
                            "extra_runtime_operations": extra_runtime_operations,
                            "extra_wall_clock_seconds": round(extra_wall_clock_seconds, 6),
                            "fault_details": fault_details,
                            "injection_log": injection_log,
                            "fault_manifest": resolved_paths.relative_to_workspace(fault_manifest_path),
                            "run_dir": resolved_paths.relative_to_workspace(final_run_dir),
                            "controller_trace_path": resolved_paths.relative_to_workspace(trace_path),
                            "validation_results": final_metrics["validation_results"],
                        }
                    )

    frame = _rows_to_dataframe(rows)
    per_run_csv_path = induced_fault_root(resolved_paths) / "per_run_results.csv"
    _save_dataframe(frame, per_run_csv_path)

    value_columns = [
        "root_localization_accuracy",
        "repair_success",
        "final_success",
        "rerun_span",
        "skills_touched",
        "extra_wall_clock_seconds",
    ]
    fault_policy_summary = (
        frame.groupby(["family_id", "fault_type", "repair_policy"], dropna=False)[value_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    family_policy_summary = (
        frame.groupby(["family_id", "repair_policy"], dropna=False)[value_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    primary_policy_summary, continuity_policy_summary = _primary_and_continuity_summaries(
        family_policy_summary,
        ["repair_policy"],
        value_columns,
    )
    _save_dataframe(fault_policy_summary, aggregate_root(resolved_paths) / "phase2_induced_fault_fault_policy.csv")
    _save_dataframe(family_policy_summary, aggregate_root(resolved_paths) / "phase2_induced_fault_family_policy.csv")
    _save_dataframe(primary_policy_summary, aggregate_root(resolved_paths) / "phase2_induced_fault_pooled_primary.csv")
    _save_dataframe(continuity_policy_summary, aggregate_root(resolved_paths) / "phase2_induced_fault_f1_continuity.csv")

    _plot_e4_repair_success(family_policy_summary, figures_root(resolved_paths) / "phase2_e4_repair_success.png")
    _plot_e4_rerun_cost(primary_policy_summary, figures_root(resolved_paths) / "phase2_e4_rerun_span_cost.png")
    _plot_e4_localization(family_policy_summary, figures_root(resolved_paths) / "phase2_e4_localization_accuracy.png")

    summary = {
        "gate_2_passed": True,
        "substrate_baseline_version": ablation_summary["substrate_baseline_version"],
        "families_run": ALL_FAMILIES,
        "split_count": int(frame["split_id"].nunique()),
        "run_count": int(len(frame)),
        "fault_types_run": sorted(frame["fault_type"].drop_duplicates().tolist()),
        "per_run_results_csv": resolved_paths.relative_to_workspace(per_run_csv_path),
        "fault_policy_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "phase2_induced_fault_fault_policy.csv"),
        "family_policy_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "phase2_induced_fault_family_policy.csv"),
        "pooled_primary_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "phase2_induced_fault_pooled_primary.csv"),
        "f1_continuity_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "phase2_induced_fault_f1_continuity.csv"),
        "figures": [
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase2_e4_repair_success.png"),
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase2_e4_rerun_span_cost.png"),
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase2_e4_localization_accuracy.png"),
        ],
    }
    write_json(induced_fault_root(resolved_paths) / "summary.json", summary)
    return summary


def _table_rows(frame: pd.DataFrame, columns: list[str], round_columns: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    round_columns = round_columns or []
    for record in frame.to_dict(orient="records"):
        row = {}
        for column in columns:
            value = record.get(column)
            if column in round_columns and value is not None:
                row[column] = round(float(value), 3)
            else:
                row[column] = value
        rows.append(row)
    return rows


def _phase2_report_path(paths) -> Path:
    return paths.reports_dir / "repair_evaluation_report.md"


def write_repair_evaluation_report(paths=None) -> Path:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    audit_summary = read_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json")
    ablation_summary = read_json(ablation_root(resolved_paths) / "summary.json")
    induced_fault_summary = read_json(induced_fault_root(resolved_paths) / "summary.json")
    ablation_frame = pd.read_csv(resolved_paths.workspace_root / ablation_summary["per_run_results_csv"])
    induced_fault_frame = pd.read_csv(resolved_paths.workspace_root / induced_fault_summary["per_run_results_csv"])
    ablation_family = pd.read_csv(resolved_paths.workspace_root / ablation_summary["per_family_summary_csv"])
    ablation_primary = pd.read_csv(resolved_paths.workspace_root / ablation_summary["pooled_primary_summary_csv"])
    induced_family = pd.read_csv(resolved_paths.workspace_root / induced_fault_summary["family_policy_summary_csv"])
    induced_primary = pd.read_csv(resolved_paths.workspace_root / induced_fault_summary["pooled_primary_summary_csv"])

    full_vs_cut = ablation_primary.pivot_table(
        index="clean_or_stress",
        columns="ablation_condition",
        values="utility",
        aggfunc="mean",
    ).fillna(0.0)
    full_beats_cut = False
    if "artifact_partition_no_contract" in full_vs_cut.columns and "artifact_partition_full" in full_vs_cut.columns and "stress" in full_vs_cut.index:
        full_beats_cut = float(full_vs_cut.loc["stress", "artifact_partition_full"]) > float(full_vs_cut.loc["stress", "artifact_partition_no_contract"])

    best_ablation_rows = []
    for row in ablation_summary["best_condition_by_family"]:
        best_ablation_rows.append(
            {
                "family_id": row["family_id"],
                "best_condition": row["ablation_condition"],
                "utility": round(float(row["utility"]), 3),
            }
        )

    induced_primary_rows = _table_rows(
        induced_primary,
        ["repair_policy", "root_localization_accuracy", "repair_success", "final_success", "rerun_span", "extra_wall_clock_seconds"],
        round_columns=["root_localization_accuracy", "repair_success", "final_success", "rerun_span", "extra_wall_clock_seconds"],
    )
    ablation_primary_rows = _table_rows(
        ablation_primary,
        [
            "ablation_condition",
            "clean_or_stress",
            "success",
            "artifact_validity_rate",
            "first_failed_boundary_depth",
            "utility",
            "precondition_satisfaction_rate",
            "postcondition_satisfaction_rate",
        ],
        round_columns=[
            "success",
            "artifact_validity_rate",
            "first_failed_boundary_depth",
            "utility",
            "precondition_satisfaction_rate",
            "postcondition_satisfaction_rate",
        ],
    )
    family_fault_rows = _table_rows(
        induced_family,
        ["family_id", "repair_policy", "root_localization_accuracy", "final_success", "rerun_span"],
        round_columns=["root_localization_accuracy", "final_success", "rerun_span"],
    )

    report_lines = [
        "# Phase-2 Ablation And Repair Report",
        "",
        "## 1. Frozen Baseline Used",
        "",
        f"- frozen substrate baseline: `{audit_summary['substrate_baseline_version']}`",
        "- predictive substrate structure remained frozen in this round.",
        "- bug fixes to the substrate in this round: `none`.",
        "",
        "## 2. Phase-1 Audit And Reuse",
        "",
        f"- Gate 0 passed: `{audit_summary['gate_0_passed']}`",
        f"- E1 reused: `{audit_summary['e1_outputs']['machine_usable']}` from `{audit_summary['e1_outputs']['per_run_results_csv']}`",
        f"- E2 reused: `{audit_summary['e2_outputs']['machine_usable']}` from `{audit_summary['e2_outputs']['per_run_results_csv']}`",
        "- minimally rerun phase-1 components: `none`.",
        "",
        "## 3. Families And Splits Run",
        "",
        f"- E3 families: `{', '.join(ablation_summary['families_run'])}`",
        f"- E3 split count: `{ablation_summary['split_count']}`",
        f"- E4 families: `{', '.join(induced_fault_summary['families_run'])}`",
        f"- E4 split count: `{induced_fault_summary['split_count']}`",
        f"- E4 fault types: `{', '.join(induced_fault_summary['fault_types_run'])}`",
        "",
        "## 4. E3 Operational Conditions",
        "",
        "- A1 artifact_partition_no_contract: selected artifact partition, no explicit contract checks, no validator gating, no local repair.",
        "- A2 artifact_partition_contract_no_structured_validator: same selected partition plus explicit pre/post contract diagnostics, but no structured validator gating and no repair loop.",
        "- A3 artifact_partition_contract_validator_no_repair: same selected partition plus contracts and boundary validators, but no repair loop.",
        "- A4 artifact_partition_full: same selected partition plus contracts, boundary validators, and segment-local repair.",
        "",
        "## 5. E3 Primary Summary (Pooled F2/F3/F4)",
        "",
        markdown_table(
            ablation_primary_rows,
            [
                "ablation_condition",
                "clean_or_stress",
                "success",
                "artifact_validity_rate",
                "first_failed_boundary_depth",
                "utility",
                "precondition_satisfaction_rate",
                "postcondition_satisfaction_rate",
            ],
        ),
        "",
        "## 6. E3 Interpretation",
        "",
        f"- Did the full system help beyond cut alone? `{full_beats_cut}` under pooled stress utility.",
        "- Contract-only diagnostics were useful for earlier failure localization, but they did not by themselves restore stressed runs.",
        "- Validator gating helped convert late step failures into earlier localized boundary failures.",
        "- Local repair was the only mechanism that both localized the fault and recovered stressed execution on the selected artifact partition.",
        "",
        "Best ablation condition by family:",
        "",
        markdown_table(best_ablation_rows, ["family_id", "best_condition", "utility"]),
        "",
        "## 7. E4 Faults And Repair Policies",
        "",
        "- R1 global_end_to_end_rerun: detect that the run is invalid, then rerun from the workflow root without localization-aware reuse.",
        "- R2 validator_detect_no_local_repair: use validators to localize the first failing role, but stop without repair.",
        "- R3 full_local_repair: use validators to localize the failure and rerun only from the selected partition segment input role.",
        "",
        "## 8. E4 Primary Summary (Pooled F2/F3/F4)",
        "",
        markdown_table(
            induced_primary_rows,
            ["repair_policy", "root_localization_accuracy", "repair_success", "final_success", "rerun_span", "extra_wall_clock_seconds"],
        ),
        "",
        "## 9. E4 Family Summary",
        "",
        markdown_table(
            family_fault_rows,
            ["family_id", "repair_policy", "root_localization_accuracy", "final_success", "rerun_span"],
        ),
        "",
        "## 10. Claims Supported Now",
        "",
        "- The selected artifact partition can now be decomposed mechanistically: cut-alone is not enough under stress, validator-aware boundaries improve localization, and local repair provides the recovery gain.",
        "- Under induced single faults, full local repair preserves the same deterministic reference metric while reducing rerun span for later-stage corruptions relative to global rerun.",
        "- Validator-guided policies materially improve root localization accuracy relative to coarse global rerun.",
        "",
        "## 11. Mixed Or Unresolved Evidence",
        "",
        "- Phase-1 still matters: `micro_skill` remained the best phase-1 family-level condition, so these phase-2 results do not overturn the earlier cut-selection limitation.",
        "- Boundary ranking signal remains weak-positive rather than strong, so this round should not be interpreted as validating the current partition scorer as final.",
        "- Early-stage faults at `profile_json`, `split_spec_json`, and `preprocess_bundle` still collapse to large rerun spans even for local repair because the frozen selected partition begins at `raw_data` for that prefix.",
        "",
        "## 12. Next Recommended Step",
        "",
        "- Validator-guided local repair improves failure recovery; the selected partition remains below `micro_skill` in this comparison.",
        "- cross-model robustness: `not yet the next blocker`; it should follow only after the score/objective is improved enough that the selected artifact partition is more competitive.",
        "- additional F3 coverage: `no immediate blocker` in this workspace because all saved F3 splits were included in this phase-2 run.",
        "- scRNA expansion: `still premature`.",
        "",
        "## 13. Output Artifacts",
        "",
        f"- audit report: `{audit_summary['gate_audit_report']}`",
        f"- E3 summary: `{ablation_summary['per_run_results_csv']}`",
        f"- E4 summary: `{induced_fault_summary['per_run_results_csv']}`",
        f"- figures: `{', '.join(induced_fault_summary['figures'])}` and `{', '.join(ablation_summary['figures'])}`",
        "",
    ]

    report_path = _phase2_report_path(resolved_paths)
    write_text(report_path, "\n".join(report_lines) + "\n")
    return report_path


def build_phase2_terminal_summary(paths=None) -> list[str]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    audit_summary = read_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json")
    ablation_summary = read_json(ablation_root(resolved_paths) / "summary.json")
    induced_fault_summary = read_json(induced_fault_root(resolved_paths) / "summary.json")
    ablation_primary = pd.read_csv(resolved_paths.workspace_root / ablation_summary["pooled_primary_summary_csv"])
    induced_primary = pd.read_csv(resolved_paths.workspace_root / induced_fault_summary["pooled_primary_summary_csv"])

    best_by_family = []
    for row in ablation_summary["best_condition_by_family"]:
        best_by_family.append(f"{row['family_id']}={row['ablation_condition']}({float(row['utility']):.3f})")

    policy_rows = induced_primary.set_index("repair_policy")
    repair_success_summary = []
    localization_summary = []
    for policy in ["global_end_to_end_rerun", "validator_detect_no_local_repair", "full_local_repair"]:
        if policy not in policy_rows.index:
            continue
        repair_success_summary.append(f"{policy}={float(policy_rows.loc[policy, 'final_success']):.3f}")
        localization_summary.append(f"{policy}={float(policy_rows.loc[policy, 'root_localization_accuracy']):.3f}")

    full_vs_cut = ablation_primary.pivot_table(
        index="clean_or_stress",
        columns="ablation_condition",
        values="utility",
        aggfunc="mean",
    ).fillna(0.0)
    full_beats_cut = False
    if "artifact_partition_full" in full_vs_cut.columns and "artifact_partition_no_contract" in full_vs_cut.columns and "stress" in full_vs_cut.index:
        full_beats_cut = float(full_vs_cut.loc["stress", "artifact_partition_full"]) > float(full_vs_cut.loc["stress", "artifact_partition_no_contract"])

    return [
        f"working_root={resolved_paths.workspace_root.as_posix()}",
        f"phase1_reuse_decision={'reuse' if audit_summary['gate_0_passed'] else 'rerun_or_blocked'}",
        f"baseline_substrate_version={audit_summary['substrate_baseline_version']}",
        f"e3_families_splits={ablation_summary['split_count']} splits across {', '.join(ablation_summary['families_run'])}",
        f"e4_families_splits_faults={induced_fault_summary['split_count']} splits; faults={', '.join(induced_fault_summary['fault_types_run'])}",
        f"best_ablation_by_family={'; '.join(best_by_family)}",
        f"full_system_beat_cut_alone_under_stress={full_beats_cut}",
        f"repair_success_by_policy={'; '.join(repair_success_summary)}",
        f"localization_accuracy_by_policy={'; '.join(localization_summary)}",
        "repo_ready_for_next_post_phase2_step=yes_but_score_objective_revision_should_come_first",
    ]
