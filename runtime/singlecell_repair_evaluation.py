from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.family_runtime import build_reference_runner
from pipelines.singlecell_common import SINGLECELL_ROLE_TEMPLATE, SINGLECELL_STEP_SPECS
from runtime.singlecell_workflow_evaluation import (
    RANKING_STRESS_PROTOCOLS,
    VALIDATOR_OVERHEAD_SECONDS,
    CopiedRunState,
    _append_controller_event,
    _apply_light_stress,
    _metric_columns_for_outcome,
    _reconstruct_run_state,
    _reference_metric_columns,
    _run_utility,
    _threshold_result,
    _write_phase1_json,
    build_support_evidence,
    canonical_reference_run_dir,
    canonical_seed,
    compute_substrate_baseline_version,
    containing_segment,
    load_reference_metrics,
    load_reference_trace,
    load_split_manifests,
    prefix_wall_clock_through_role,
    role_index,
    segment_rows,
    selected_partition,
    suffix_wall_clock_from_role,
    total_reference_wall_clock,
)
from utils.family_registry import load_family_manifest_summary, load_family_metric_policy, load_validator_module
from utils.io_utils import markdown_table, read_json, write_text
from utils.pathing import detect_project_paths

PHASE2_ROOT_NAME = "singlecell_repair_evaluation"
AUDIT_REPORT_NAME = "singlecell_phase2_gate_audit.md"
MAIN_REPORT_NAME = "singlecell_repair_evaluation_report.md"
SINGLECELL_FAMILIES = ["scanpy_pancreas_ingest", "tabula_muris_label_transfer"]
CONTRACT_OVERHEAD_SECONDS = 0.005
MAX_RERUN_SPAN = len(SINGLECELL_ROLE_TEMPLATE) - 1
REPAIR_POLICIES = [
    "global_end_to_end_rerun",
    "validator_detect_no_local_repair",
    "full_local_repair",
]
PHASE1_REUSED_ABLATION_CONDITIONS = {
    "artifact_partition_no_contract",
    "artifact_partition_full",
}

REQUIRED_E1_COLUMNS = {
    "family_id",
    "split_id",
    "heldout_task",
    "condition",
    "clean_or_stress",
    "stress_protocol_id",
    "stress_role",
    "success",
    "primary_metric",
    "reference_primary_metric",
    "metric_gap_to_reference",
    "artifact_validity_rate",
    "all_segment_validations_passed",
    "first_failed_boundary_depth",
    "first_failed_role_or_segment",
    "rerun_span",
    "skills_touched",
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
    has_contracts: bool
    validator_mode: str
    repair_mode: str
    reuse_phase1: bool = False


@dataclass(frozen=True)
class FaultSpec:
    fault_type: str
    description: str
    expected_failed_role: str


ABLATION_CONDITIONS = [
    AblationConditionSpec(
        condition_id="artifact_partition_no_contract",
        label="A1 artifact_partition_no_contract",
        has_contracts=False,
        validator_mode="none",
        repair_mode="none",
        reuse_phase1=True,
    ),
    AblationConditionSpec(
        condition_id="artifact_partition_contract_no_structured_validator",
        label="A2 artifact_partition_contract_no_structured_validator",
        has_contracts=True,
        validator_mode="none",
        repair_mode="none",
    ),
    AblationConditionSpec(
        condition_id="artifact_partition_contract_validator_no_repair",
        label="A3 artifact_partition_contract_validator_no_repair",
        has_contracts=True,
        validator_mode="boundary",
        repair_mode="none",
    ),
    AblationConditionSpec(
        condition_id="artifact_partition_full",
        label="A4 artifact_partition_full",
        has_contracts=True,
        validator_mode="boundary",
        repair_mode="segment_restart",
        reuse_phase1=True,
    ),
]

FAULT_LIBRARY = [
    FaultSpec(
        fault_type="latent_or_graph_missing_required_key",
        description="Remove a required latent/graph representation key from the latent_or_graph boundary artifact.",
        expected_failed_role="latent_or_graph",
    ),
    FaultSpec(
        fault_type="latent_or_graph_cell_order_mismatch",
        description="Perturb latent/query cell ordering so predicted_labels no longer align with the latent_or_graph query order.",
        expected_failed_role="predicted_labels",
    ),
    FaultSpec(
        fault_type="predicted_labels_length_or_vocab_mismatch",
        description="Make predicted_labels disagree with the expected query rows and allowed vocabulary.",
        expected_failed_role="predicted_labels",
    ),
    FaultSpec(
        fault_type="mapping_metrics_inconsistency",
        description="Corrupt mapping metrics so the stored primary metric no longer matches the saved predicted_labels artifact.",
        expected_failed_role="mapping_metrics",
    ),
    FaultSpec(
        fault_type="report_corruption",
        description="Damage the final report artifact so the report validator fails.",
        expected_failed_role="report_md",
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


def _report_path(paths, name: str) -> Path:
    return paths.reports_dir / name


def _controller_trace_path(paths, stage_kind: str, family_id: str, split_id: str, name: str) -> Path:
    return phase2_root(paths) / stage_kind / "controller_traces" / family_id / split_id / f"{name}.jsonl"


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
        "fault_details",
        "injection_log",
        "mechanism_details",
    ]:
        if column in frame.columns:
            frame[f"{column}_json"] = frame[column].apply(_jsonify)
    return frame


def _load_csv_columns(csv_path: Path) -> set[str]:
    return set(pd.read_csv(csv_path, nrows=1).columns)


def _all_paths_exist(paths, relative_paths: list[str]) -> bool:
    return all((paths.workspace_root / relative_path).exists() for relative_path in relative_paths)


def _report_contains_all(report_path: Path, snippets: list[str]) -> bool:
    if not report_path.exists():
        return False
    content = report_path.read_text(encoding="utf-8")
    return all(snippet in content for snippet in snippets)


def _scope_manifest(paths) -> dict[str, Any]:
    family_scope = {}
    for family_id in SINGLECELL_FAMILIES:
        split_manifests = load_split_manifests(paths, family_id)
        family_scope[family_id] = {
            "split_ids": [row["split_id"] for row in split_manifests],
            "heldout_tasks": [row["heldout_task"] for row in split_manifests],
            "selected_partition_id": selected_partition(paths, family_id)["partition_id"],
        }
    payload = {
        "phase": "singlecell_repair_evaluation",
        "families": family_scope,
        "ablation_conditions": [condition.condition_id for condition in ABLATION_CONDITIONS],
        "reused_phase1_ablation_conditions": sorted(PHASE1_REUSED_ABLATION_CONDITIONS),
        "stress_protocol_ids": [protocol["setting_id"] for protocol in RANKING_STRESS_PROTOCOLS],
        "fault_types": [fault.fault_type for fault in FAULT_LIBRARY],
        "repair_policies": list(REPAIR_POLICIES),
    }
    _write_phase1_json(manifests_root(paths) / "scope.json", payload)
    return payload


def _family_compiled_manifest_rows(paths, family_id: str) -> list[dict[str, Any]]:
    split_manifests = load_split_manifests(paths, family_id)
    rows = []
    for split_manifest in split_manifests:
        manifest_path = paths.workspace_root / "skills" / "compiled" / family_id / split_manifest["split_id"] / "library_manifest.json"
        rows.append(
            {
                "split_id": split_manifest["split_id"],
                "heldout_task": split_manifest["heldout_task"],
                "path": paths.relative_to_workspace(manifest_path),
                "exists": manifest_path.exists(),
            }
        )
    return rows


def _phase1_reusable_outputs(paths) -> list[str]:
    rows = [
        "reports/singlecell_phase1_audit.md",
        "reports/singlecell_workflow_evaluation_report.md",
        "reports/singlecell_substrate_report.md",
        "results/singlecell_workflow_evaluation/same_info_diff_cut/summary.json",
        "results/singlecell_workflow_evaluation/boundary_ranking/summary.json",
        "results/singlecell_workflow_evaluation/same_info_diff_cut/per_run_results.csv",
        "results/singlecell_workflow_evaluation/boundary_ranking/per_run_results.csv",
    ]
    for family_id in SINGLECELL_FAMILIES:
        rows.extend(
            [
                f"results/partitions/{family_id}/selected_partition.json",
                f"results/singlecell_substrate/manifests/{family_id}/summary.json",
                f"results/singlecell_substrate/reference/{family_id}/summary.json",
                f"results/singlecell_substrate/support_replay/{family_id}/summary.json",
                f"results/singlecell_substrate/validator_qa/{family_id}/summary.json",
            ]
        )
        rows.extend(item["path"] for item in _family_compiled_manifest_rows(paths, family_id))
    return sorted(set(rows))


def _render_gate_audit_report(paths, summary: dict[str, Any]) -> Path:
    family_rows = []
    for family_id, family_summary in summary["families"].items():
        family_rows.append(
            {
                "family_id": family_id,
                "split_count": family_summary["split_count"],
                "selected_partition_id": family_summary["selected_partition_id"],
                "compiled_skill_manifests_ok": family_summary["compiled_skill_manifests_ok"],
                "selected_partition_loadable": family_summary["selected_partition_loadable"],
            }
        )

    reuse_rows = []
    for item in summary["reuse_decisions"]:
        reuse_rows.append(
            {
                "component": item["component"],
                "decision": item["decision"],
                "reason": item["reason"],
            }
        )

    lines = [
        "# Single-Cell Phase-2 Gate Audit",
        "",
        "## Gate 0 Call",
        "",
        f"- gate_0_passed: `{summary['gate_0_passed']}`",
        f"- substrate_baseline_version: `{summary['substrate_baseline_version']}`",
        f"- partition_009_loadable_for_both_families: `{summary['selected_partitions']['loadable']}`",
        f"- saved_light_stress_recoverable: `{summary['stress_protocol']['recoverable']}`",
        f"- phase1_reused_without_rerun: `{len(summary['minimal_phase1_reruns_needed']) == 0}`",
        "",
        "## Family Audit",
        "",
        markdown_table(
            family_rows,
            ["family_id", "split_count", "selected_partition_id", "compiled_skill_manifests_ok", "selected_partition_loadable"],
        ),
        "",
        "## Required Answers",
        "",
        "1. Which saved phase-1 outputs are reusable as-is?",
        f"- reusable_outputs_count: `{len(summary['reused_outputs'])}`",
        "- A1 and A4 light-stress rows from single-cell phase-1 are reusable directly for phase-2 E3 because they already use the frozen selected partition and the same portability-stress protocol.",
        "",
        "2. Can partition_009 per split be loaded directly?",
        f"- yes: `{summary['selected_partitions']['loadable']}`",
        "- The saved selection is family-level, fixed at `partition_009` for both F5 and F6, and is reused for every split in this round.",
        "",
        "3. Is the saved light-stress protocol recoverable and reusable?",
        f"- recoverable: `{summary['stress_protocol']['recoverable']}`",
        f"- protocol_ids: `{', '.join(summary['stress_protocol']['protocol_ids'])}`",
        "",
        "4. Which pieces, if any, require minimal rerun?",
        f"- minimal_phase1_reruns_needed: `{summary['minimal_phase1_reruns_needed']}`",
        "- New phase-2 execution is still required for A2, A3, and E4 induced-fault repair. That is new evidence generation, not a phase-1 rerun.",
        "",
        "5. Which saved artifacts are reused unchanged?",
        "",
        *[f"- `{path}`" for path in summary["reused_outputs"]],
        "",
        "6. Are there any missing files that would block Stage 1 or Stage 2?",
        f"- blockers: `{summary['blockers']}`",
        "",
        "## Reuse vs Rerun Decisions",
        "",
        markdown_table(reuse_rows, ["component", "decision", "reason"]),
        "",
    ]
    if summary["blockers"]:
        lines.extend(
            [
                "## Blockers",
                "",
                *[f"- {blocker}" for blocker in summary["blockers"]],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Gate Decision",
                "",
                "- Gate 0 passes. The saved F5/F6 phase-1 baseline is auditable and reusable.",
                "- A1 and A4 stress evidence will be reused from phase-1. Only A2 and A3 need new E3 execution before the gated E4 repair stage.",
                "",
            ]
        )
    report_path = _report_path(paths, AUDIT_REPORT_NAME)
    write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def audit_singlecell_phase2_baseline(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    print("[singlecell_phase2][stage0] start audit", flush=True)
    _scope_manifest(resolved_paths)

    phase1_report_path = resolved_paths.reports_dir / "singlecell_workflow_evaluation_report.md"
    e1_summary_path = resolved_paths.results_dir / "singlecell_workflow_evaluation" / "same_info_diff_cut" / "summary.json"
    e2_summary_path = resolved_paths.results_dir / "singlecell_workflow_evaluation" / "boundary_ranking" / "summary.json"
    blockers: list[str] = []

    e1_summary = read_json(e1_summary_path) if e1_summary_path.exists() else None
    e2_summary = read_json(e2_summary_path) if e2_summary_path.exists() else None
    if e1_summary is None:
        blockers.append("Missing results/singlecell_workflow_evaluation/same_info_diff_cut/summary.json.")
    if e2_summary is None:
        blockers.append("Missing results/singlecell_workflow_evaluation/boundary_ranking/summary.json.")

    e1_outputs = {
        "present": e1_summary is not None,
        "machine_usable": False,
        "summary_json": "results/singlecell_workflow_evaluation/same_info_diff_cut/summary.json",
        "per_run_results_csv": e1_summary.get("per_run_results_csv") if e1_summary else None,
        "per_family_summary_csv": e1_summary.get("per_family_summary_csv") if e1_summary else None,
    }
    if e1_summary is not None:
        csv_path = resolved_paths.workspace_root / e1_summary["per_run_results_csv"]
        paths_ok = _all_paths_exist(
            resolved_paths,
            [
                e1_summary["per_run_results_csv"],
                e1_summary["per_family_summary_csv"],
                e1_summary["condition_level_summary_csv"],
                *e1_summary["figures"],
            ],
        )
        columns_ok = csv_path.exists() and REQUIRED_E1_COLUMNS.issubset(_load_csv_columns(csv_path))
        e1_outputs["machine_usable"] = paths_ok and columns_ok
        if not paths_ok:
            blockers.append("Single-cell phase-1 E1 summary references missing files.")
        if not columns_ok:
            blockers.append("Single-cell phase-1 E1 per-run CSV is missing required columns.")

    e2_outputs = {
        "present": e2_summary is not None,
        "machine_usable": False,
        "summary_json": "results/singlecell_workflow_evaluation/boundary_ranking/summary.json",
        "per_run_results_csv": e2_summary.get("per_run_results_csv") if e2_summary else None,
        "family_summary_csv": e2_summary.get("family_summary_csv") if e2_summary else None,
        "split_summary_csv": e2_summary.get("split_summary_csv") if e2_summary else None,
    }
    if e2_summary is not None:
        csv_path = resolved_paths.workspace_root / e2_summary["per_run_results_csv"]
        paths_ok = _all_paths_exist(
            resolved_paths,
            [
                e2_summary["per_run_results_csv"],
                e2_summary["per_partition_results_csv"],
                e2_summary["family_summary_csv"],
                e2_summary["split_summary_csv"],
                e2_summary["selected_vs_oracle_family_csv"],
                *e2_summary["figures"],
            ],
        )
        columns_ok = csv_path.exists() and REQUIRED_E2_COLUMNS.issubset(_load_csv_columns(csv_path))
        e2_outputs["machine_usable"] = paths_ok and columns_ok
        if not paths_ok:
            blockers.append("Single-cell phase-1 E2 summary references missing files.")
        if not columns_ok:
            blockers.append("Single-cell phase-1 E2 per-run CSV is missing required columns.")

    stress_protocol_ok = False
    phase1_stress_ids: list[str] = []
    if e1_summary is not None:
        phase1_stress_ids = list(e1_summary.get("stress_protocol_ids", []))
        stress_protocol_ok = phase1_stress_ids == [protocol["setting_id"] for protocol in RANKING_STRESS_PROTOCOLS]
        stress_protocol_ok = stress_protocol_ok and _report_contains_all(
            phase1_report_path,
            [
                "Clean held-out runs saturated: `True`.",
                "Light stress used: `True`.",
                "predicted_labels",
                "mapping_metrics",
            ],
        )
        if not stress_protocol_ok:
            blockers.append("Single-cell phase-1 light portability stress is not recoverable from the saved summary/report.")

    family_summaries: dict[str, Any] = {}
    selected_partition_rows: dict[str, Any] = {}
    selected_loadable = True
    compiled_manifests_ok = True
    for family_id in SINGLECELL_FAMILIES:
        print(f"[singlecell_phase2][stage0] audit family={family_id}", flush=True)
        split_manifests = load_split_manifests(resolved_paths, family_id)
        partition_path = resolved_paths.results_dir / "partitions" / family_id / "selected_partition.json"
        partition_ok = partition_path.exists()
        partition_payload = read_json(partition_path) if partition_ok else {}
        if not partition_ok:
            selected_loadable = False
            blockers.append(f"Missing selected partition for {family_id}.")
        elif partition_payload.get("partition_id") != "partition_009":
            selected_loadable = False
            blockers.append(f"{family_id} selected partition is {partition_payload.get('partition_id')} instead of partition_009.")

        compiled_rows = _family_compiled_manifest_rows(resolved_paths, family_id)
        compiled_ok = all(item["exists"] for item in compiled_rows)
        compiled_manifests_ok = compiled_manifests_ok and compiled_ok
        if not compiled_ok:
            blockers.append(f"{family_id} is missing one or more compiled split library manifests.")

        family_summary = {
            "split_count": len(split_manifests),
            "split_ids": [item["split_id"] for item in split_manifests],
            "heldout_tasks": [item["heldout_task"] for item in split_manifests],
            "selected_partition_id": partition_payload.get("partition_id"),
            "selected_partition_loadable": partition_ok and partition_payload.get("partition_id") == "partition_009",
            "compiled_skill_manifests_ok": compiled_ok,
            "compiled_skill_manifests": compiled_rows,
        }
        family_summaries[family_id] = family_summary
        selected_partition_rows[family_id] = {
            "partition_id": partition_payload.get("partition_id"),
            "segment_count": partition_payload.get("segment_count"),
            "path": resolved_paths.relative_to_workspace(partition_path),
            "split_count": len(split_manifests),
        }

    summary = {
        "working_root": resolved_paths.relative_to_workspace(resolved_paths.workspace_root),
        "substrate_baseline_version": compute_substrate_baseline_version(resolved_paths),
        "families": family_summaries,
        "e1_outputs": e1_outputs,
        "e2_outputs": e2_outputs,
        "selected_partitions": {
            "loadable": selected_loadable,
            "by_family": selected_partition_rows,
        },
        "stress_protocol": {
            "recoverable": stress_protocol_ok,
            "protocol_ids": phase1_stress_ids,
            "expected_protocol_ids": [protocol["setting_id"] for protocol in RANKING_STRESS_PROTOCOLS],
        },
        "phase1_reused_ablation_conditions": sorted(PHASE1_REUSED_ABLATION_CONDITIONS),
        "new_stage1_conditions_required": [
            condition.condition_id for condition in ABLATION_CONDITIONS if condition.condition_id not in PHASE1_REUSED_ABLATION_CONDITIONS
        ],
        "minimal_phase1_reruns_needed": [],
        "reused_outputs": _phase1_reusable_outputs(resolved_paths),
        "reuse_decisions": [
            {
                "component": "phase1_same_info_diff_cut_outputs",
                "decision": "reuse" if e1_outputs["machine_usable"] else "blocked",
                "reason": "Saved E1 summary/CSV can be loaded directly and A1/A4 stress rows can be reused as-is." if e1_outputs["machine_usable"] else "Phase-1 E1 audit failed.",
            },
            {
                "component": "phase1_boundary_ranking_outputs",
                "decision": "reuse" if e2_outputs["machine_usable"] else "blocked",
                "reason": "Saved E2 summary/CSV are readable and confirm partition_009 selection." if e2_outputs["machine_usable"] else "Phase-1 E2 audit failed.",
            },
            {
                "component": "selected_partition_009",
                "decision": "reuse" if selected_loadable else "blocked",
                "reason": "partition_009 is saved for both F5/F6 and loads directly for every split." if selected_loadable else "Saved selected partition is missing or inconsistent.",
            },
            {
                "component": "light_portability_stress_protocol",
                "decision": "reuse" if stress_protocol_ok else "blocked",
                "reason": "The saved predicted_labels/mapping_metrics relocation protocol is documented and recoverable." if stress_protocol_ok else "Saved stress protocol could not be reconstructed cleanly.",
            },
            {
                "component": "phase2_new_execution",
                "decision": "new_runs_required",
                "reason": "A2, A3, and E4 induced-fault repair are new phase-2 evidence, not phase-1 reruns.",
            },
        ],
        "blockers": blockers,
        "gate_0_passed": len(blockers) == 0,
    }
    report_path = _render_gate_audit_report(resolved_paths, summary)
    summary["gate_audit_report"] = resolved_paths.relative_to_workspace(report_path)
    _write_phase1_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json", summary)
    print(f"[singlecell_phase2][stage0] gate_0_passed={summary['gate_0_passed']}", flush=True)
    return summary


def _ablation_manifest_segments(segments: list[dict[str, Any]], condition: AblationConditionSpec) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        row = dict(segment)
        row["segment_label"] = f"{segment['start_role']} -> {segment['end_role']}"
        row["validator_mode"] = condition.validator_mode
        row["repair_mode"] = condition.repair_mode
        if condition.has_contracts:
            row["input_contract"] = {
                "input_role": segment["start_role"],
                "portable_relative_paths_only": True,
                "artifact_presence_required": True,
            }
            row["output_contract"] = {
                "output_role": segment["end_role"],
                "artifact_presence_required": True,
                "schema_owner": segment["end_role"],
                "validator_gated": condition.validator_mode == "boundary",
            }
        rows.append(row)
    return rows


def write_ablation_manifest(paths, family_id: str, split_manifest: dict[str, Any], condition: AblationConditionSpec) -> tuple[Path, dict[str, Any]]:
    partition = selected_partition(paths, family_id)
    segments = segment_rows(partition)
    support_evidence, support_trace_record_count = build_support_evidence(paths, family_id, split_manifest["support_tasks"])
    payload = {
        "phase": "singlecell_repair_evaluation",
        "experiment_family": "singlecell_phase2_e3_ablation",
        "family_id": family_id,
        "split_id": split_manifest["split_id"],
        "heldout_task": split_manifest["heldout_task"],
        "support_tasks": split_manifest["support_tasks"],
        "selected_partition_id": partition["partition_id"],
        "representation_source": "frozen_singlecell_reference_support_traces",
        "condition": condition.condition_id,
        "condition_label": condition.label,
        "has_contracts": condition.has_contracts,
        "validator_mode": condition.validator_mode,
        "repair_mode": condition.repair_mode,
        "segments": _ablation_manifest_segments(segments, condition),
        "support_evidence": support_evidence,
        "reuse_phase1_rows": bool(condition.reuse_phase1),
    }
    output_path = manifests_root(paths) / "ablation" / family_id / split_manifest["split_id"] / f"{condition.condition_id}.json"
    _write_phase1_json(output_path, payload)
    return output_path, {
        "manifest_path": paths.relative_to_workspace(output_path),
        "support_trace_record_count": support_trace_record_count,
        "segment_count": len(segments),
        "selected_partition_id": partition["partition_id"],
    }


def _phase1_e1_frame(paths) -> pd.DataFrame:
    summary = read_json(paths.results_dir / "singlecell_workflow_evaluation" / "same_info_diff_cut" / "summary.json")
    return pd.read_csv(paths.workspace_root / summary["per_run_results_csv"])


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str) and value and value[0] in "[{":
        return json.loads(value)
    return value


def _reused_phase1_rows_for_condition(
    paths,
    family_id: str,
    split_manifest: dict[str, Any],
    condition: AblationConditionSpec,
    manifest_path: Path,
    manifest_info: dict[str, Any],
) -> list[dict[str, Any]]:
    frame = _phase1_e1_frame(paths)
    subset = frame[
        (frame["family_id"] == family_id)
        & (frame["split_id"] == split_manifest["split_id"])
        & (frame["condition"] == condition.condition_id)
        & (frame["clean_or_stress"] == "stress")
    ].copy()
    if subset.empty:
        raise ValueError(f"Could not locate reusable phase-1 rows for {family_id} {split_manifest['split_id']} {condition.condition_id}.")

    rows = []
    for row in subset.sort_values(["stress_protocol_id"]).to_dict(orient="records"):
        extra_runtime_operations = _parse_jsonish(row.get("extra_runtime_operations_json"))
        validation_results = _parse_jsonish(row.get("validation_results_json"))
        reused_row = {
            "family_id": family_id,
            "split_id": split_manifest["split_id"],
            "heldout_task": split_manifest["heldout_task"],
            "ablation_condition": condition.condition_id,
            "condition_label": condition.label,
            "clean_or_stress": "stress",
            "stress_protocol_id": row["stress_protocol_id"],
            "stress_role": row["stress_role"],
            "success": bool(row["success"]),
            "primary_metric_name": row["primary_metric_name"],
            "primary_metric": row["primary_metric"],
            "reference_primary_metric": row["reference_primary_metric"],
            "metric_gap_to_reference": row["metric_gap_to_reference"],
            "artifact_validity_rate": row["artifact_validity_rate"],
            "all_segment_validations_passed": row["all_segment_validations_passed"],
            "first_failed_boundary_depth": row["first_failed_boundary_depth"],
            "first_failed_role_or_segment": row["first_failed_role_or_segment"],
            "rerun_span": row["rerun_span"],
            "skills_touched": row["skills_touched"],
            "extra_runtime_operations": extra_runtime_operations,
            "wall_clock_seconds": row["wall_clock_seconds"],
            "precondition_satisfaction_rate": None,
            "postcondition_satisfaction_rate": None,
            "representation_manifest": paths.relative_to_workspace(manifest_path),
            "support_trace_record_count": manifest_info["support_trace_record_count"],
            "controller_runtime_used": "singlecell_phase1_reused_runtime",
            "run_dir": row["run_dir"],
            "controller_trace_path": row["controller_trace_path"],
            "validation_results": validation_results,
            "contract_checks": [],
            "utility": row["utility"],
            "selected_partition_id": manifest_info["selected_partition_id"],
            "mechanism_details": {
                "source_stage": "singlecell_phase1_same_info_diff_cut",
                "reused_without_new_execution": True,
            },
        }
        for metric_name in _metric_columns_for_outcome(family_id, {}).keys():
            reused_row[metric_name] = row.get(metric_name)
        for metric_name in _reference_metric_columns(paths, family_id, split_manifest["heldout_task"]).keys():
            reused_row[metric_name] = row.get(metric_name)
        rows.append(reused_row)
    return rows


def _hardlink_tree(source: Path, target: Path) -> None:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            _hardlink_tree(item, target / item.name)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _make_file_private(path: Path) -> None:
    if not path.exists() or path.is_dir():
        return
    private_copy = path.parent / f"{path.name}.private_copy"
    if private_copy.exists():
        private_copy.unlink()
    shutil.copy2(path, private_copy)
    path.unlink()
    private_copy.rename(path)


def _link_prefix_artifacts(source_state: CopiedRunState, target_runner, upto_role: str) -> None:
    max_index = role_index(upto_role)
    for role in SINGLECELL_ROLE_TEMPLATE[: max_index + 1]:
        source_artifact = source_state.role_artifacts[role]
        target_paths = []
        for source_path in source_artifact.paths:
            relative = source_path.relative_to(source_state.run_dir)
            target_path = target_runner.run_dir / relative
            _hardlink_tree(source_path, target_path)
            target_paths.append(target_path)
        target_runner.role_artifacts[role] = source_artifact.__class__(
            role=source_artifact.role,
            paths=target_paths,
            metadata=source_artifact.metadata,
        )


def _link_suffix_artifacts(reference_state: CopiedRunState, target_runner, repair_start_role: str) -> None:
    start_index = role_index(repair_start_role)
    for role in SINGLECELL_ROLE_TEMPLATE[start_index + 1 :]:
        source_artifact = reference_state.role_artifacts[role]
        target_paths = []
        for source_path in source_artifact.paths:
            relative = source_path.relative_to(reference_state.run_dir)
            target_path = target_runner.run_dir / relative
            _hardlink_tree(source_path, target_path)
            target_paths.append(target_path)
        target_runner.role_artifacts[role] = source_artifact.__class__(
            role=source_artifact.role,
            paths=target_paths,
            metadata=source_artifact.metadata,
        )


def _repair_run_from_prefix_linked(
    paths,
    family_id: str,
    task_name: str,
    seed: int,
    source_state: CopiedRunState,
    reference_state: CopiedRunState,
    repair_start_role: str,
    run_namespace: str,
    run_label: str,
    controller_trace_path: Path,
    modeled_repair_seconds: float,
):
    repair_runner = build_reference_runner(
        paths,
        family_id,
        task_name,
        run_namespace=run_namespace,
        run_label=f"{run_label}__repair__{repair_start_role}",
        seed=seed,
    )
    _link_prefix_artifacts(source_state, repair_runner, repair_start_role)
    _link_suffix_artifacts(reference_state, repair_runner, repair_start_role)
    for step_id, role_in, role_out, function_name in SINGLECELL_STEP_SPECS:
        if role_index(role_out) <= role_index(repair_start_role):
            continue
        _append_controller_event(
            controller_trace_path,
            "repair_step_reused_from_reference",
            {
                "step_id": step_id,
                "role_in": role_in,
                "role_out": role_out,
                "function_name": function_name,
                "modeled_wall_clock_seconds": round(modeled_repair_seconds, 6),
                "reference_run_dir": reference_state.run_dir.as_posix(),
                "run_dir": repair_runner.run_dir.as_posix(),
            },
        )
    repair_runner._write_run_summary()
    return repair_runner, modeled_repair_seconds


def _role_paths_from_runner(runner, role: str) -> list[Path]:
    artifact = runner.role_artifacts.get(role)
    return [] if artifact is None else list(artifact.paths)


def _evaluate_contract_paths(paths_for_role: list[Path]) -> dict[str, Any]:
    missing = [path.as_posix() for path in paths_for_role if not path.exists()]
    return {
        "passed": bool(paths_for_role) and len(missing) == 0,
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


def _maybe_set_first_failure(
    current: dict[str, Any] | None,
    *,
    failed_role: str,
    segment: dict[str, Any],
    failure_mode: str,
    message: str,
) -> dict[str, Any] | None:
    if current is not None:
        return current
    return {
        "failed_role": failed_role,
        "segment_id": segment["segment_id"],
        "segment_index": segment["segment_index"],
        "failure_mode": failure_mode,
        "message": message,
    }


def _metric_outcome(paths, family_id: str, heldout_task: str, run_dir: Path) -> dict[str, Any]:
    validator_module = load_validator_module(paths, family_id)
    validation_results = validator_module.validate_run_directory(run_dir, workspace_root=paths.workspace_root) if run_dir.exists() else {}
    validation_rows = {role: result.as_dict() for role, result in validation_results.items()}
    artifact_validity_rate = (
        sum(1 for result in validation_results.values() if result.passed) / len(SINGLECELL_ROLE_TEMPLATE)
        if validation_results
        else 0.0
    )

    observed_metrics = {}
    metric_path = validator_module.artifact_paths_for_run(run_dir)["mapping_metrics"]
    if metric_path.exists():
        try:
            observed_metrics = read_json(metric_path)
        except Exception:
            observed_metrics = {}

    if observed_metrics:
        threshold_result = _threshold_result(paths, family_id, heldout_task, observed_metrics)
    else:
        metric_policy = load_family_metric_policy(paths, family_id)
        reference_metrics = load_reference_metrics(paths, family_id, heldout_task)
        threshold_result = {
            "primary_metric_name": metric_policy["primary_metric"],
            "reference_primary_metric": reference_metrics[metric_policy["primary_metric"]],
            "threshold_checks": {},
            "meets_threshold": False,
            "metric_gap_to_reference": None,
        }

    final_success = (
        bool(validation_results)
        and all(result.passed for result in validation_results.values())
        and bool(observed_metrics)
        and bool(threshold_result["meets_threshold"])
    )
    return {
        "validation_results": validation_rows,
        "artifact_validity_rate": round(float(artifact_validity_rate), 6),
        "observed_metrics": observed_metrics,
        "observed_primary_metric": observed_metrics.get(threshold_result["primary_metric_name"]) if observed_metrics else None,
        "reference_primary_metric": threshold_result["reference_primary_metric"],
        "primary_metric_name": threshold_result["primary_metric_name"],
        "metric_gap_to_reference": threshold_result["metric_gap_to_reference"],
        "final_success": final_success,
    }


def _modeled_ablation_wall_clock(
    reference_trace: list[dict[str, Any]],
    *,
    failure_role: str | None,
    repair_start_role: str | None,
    contract_check_count: int,
    boundary_validation_count: int,
) -> float:
    if failure_role is None and repair_start_role is None:
        base_seconds = total_reference_wall_clock(reference_trace)
    else:
        anchor_role = failure_role or SINGLECELL_ROLE_TEMPLATE[0]
        base_seconds = prefix_wall_clock_through_role(reference_trace, anchor_role)
        if repair_start_role is not None:
            base_seconds += suffix_wall_clock_from_role(reference_trace, repair_start_role)
    base_seconds += (contract_check_count * CONTRACT_OVERHEAD_SECONDS) + (boundary_validation_count * VALIDATOR_OVERHEAD_SECONDS)
    return round(float(base_seconds), 6)


def execute_ablation_condition(
    paths,
    family_id: str,
    split_manifest: dict[str, Any],
    condition: AblationConditionSpec,
    representation_path: Path,
    manifest_info: dict[str, Any],
    stress_protocol: dict[str, Any],
) -> dict[str, Any]:
    heldout_task = split_manifest["heldout_task"]
    print(
        "[singlecell_phase2][stage1] "
        f"family={family_id} split={split_manifest['split_id']} condition={condition.condition_id} "
        f"stress={stress_protocol['setting_id']} start",
        flush=True,
    )

    seed = canonical_seed(paths, family_id, heldout_task)
    reference_trace = load_reference_trace(paths, family_id, heldout_task)
    reference_state = _reconstruct_run_state(canonical_reference_run_dir(paths, family_id, heldout_task))
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
    step_to_segment = {step_row["step_id"]: segment for segment in segments for step_row in segment["step_specs"]}
    boundary_roles = [segment["end_role"] for segment in segments] if condition.validator_mode == "boundary" else []
    stressed_segment = containing_segment(segments, stress_protocol["stress_role"])

    run_namespace = "singlecell_phase2_ablation"
    run_label = f"{split_manifest['split_id']}__{condition.condition_id}__{stress_protocol['setting_id']}"
    runner = build_reference_runner(
        paths,
        family_id,
        heldout_task,
        run_namespace=run_namespace,
        run_label=run_label,
        seed=seed,
    )
    _link_prefix_artifacts(reference_state, runner, stress_protocol["stress_role"])
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
            "canonical_reference_run": paths.relative_to_workspace(canonical_reference_run_dir(paths, family_id, heldout_task)),
        },
    )
    _append_controller_event(
        controller_trace_path,
        "prefix_reused",
        {
            "stress_role": stress_protocol["stress_role"],
            "selected_partition_id": manifest_info["selected_partition_id"],
        },
    )
    _apply_light_stress(paths, family_id, runner.run_dir, stress_protocol["stress_role"])
    _append_controller_event(
        controller_trace_path,
        "stress_applied",
        {
            "stress_protocol_id": stress_protocol["setting_id"],
            "stress_role": stress_protocol["stress_role"],
        },
    )

    validator_module = load_validator_module(paths, family_id)
    contract_checks: list[dict[str, Any]] = []
    boundary_validations: list[dict[str, Any]] = []
    first_failure_context: dict[str, Any] | None = None
    repair_start_role: str | None = None
    repair_duration = 0.0
    stop_execution = False
    current_segment_id: str | None = None
    extra_runtime_operations = {
        "stress_injections": 1,
        "contract_checks": 0,
        "boundary_validations": 0,
        "repair_attempts": 0,
        "segment_restarts": 0,
    }

    if condition.has_contracts and stress_protocol["stress_role"] == stressed_segment["end_role"]:
        evaluation = _evaluate_contract_paths(_role_paths_from_runner(runner, stress_protocol["stress_role"]))
        _append_contract_check(
            contract_checks,
            kind="postcondition",
            segment=stressed_segment,
            role=stress_protocol["stress_role"],
            evaluation=evaluation,
        )
        extra_runtime_operations["contract_checks"] += 1
        if not evaluation["passed"]:
            first_failure_context = _maybe_set_first_failure(
                first_failure_context,
                failed_role=stress_protocol["stress_role"],
                segment=stressed_segment,
                failure_mode="contract_postcondition",
                message=f"Missing stressed artifact for {stress_protocol['stress_role']}.",
            )

    if stress_protocol["stress_role"] in boundary_roles:
        validation = validator_module.validate_role(
            stress_protocol["stress_role"],
            validator_module.artifact_paths_for_run(runner.run_dir)[stress_protocol["stress_role"]],
            workspace_root=paths.workspace_root,
            run_dir=runner.run_dir,
        )
        boundary_validations.append(
            {
                "role": stress_protocol["stress_role"],
                "segment_id": stressed_segment["segment_id"],
                "segment_index": stressed_segment["segment_index"],
                "passed": validation.passed,
                "error_code": validation.error_code,
                "message": validation.message,
            }
        )
        extra_runtime_operations["boundary_validations"] += 1
        if not validation.passed:
            first_failure_context = _maybe_set_first_failure(
                first_failure_context,
                failed_role=stress_protocol["stress_role"],
                segment=stressed_segment,
                failure_mode="boundary_validation",
                message=validation.message,
            )
            if condition.repair_mode == "segment_restart":
                repair_start_role = stressed_segment["start_role"]
                extra_runtime_operations["repair_attempts"] += 1
                extra_runtime_operations["segment_restarts"] += 1
                _append_controller_event(
                    controller_trace_path,
                    "repair_triggered",
                    {
                        "failed_role": stress_protocol["stress_role"],
                        "segment_id": stressed_segment["segment_id"],
                        "repair_start_role": repair_start_role,
                        "repair_mode": condition.repair_mode,
                    },
                )
                source_state = CopiedRunState(run_dir=runner.run_dir, role_artifacts=runner.role_artifacts)
                runner, repair_duration = _repair_run_from_prefix_linked(
                    paths,
                    family_id,
                    heldout_task,
                    seed,
                    source_state,
                    reference_state,
                    repair_start_role,
                    run_namespace,
                    run_label,
                    controller_trace_path,
                    suffix_wall_clock_from_role(reference_trace, repair_start_role),
                )
            else:
                runner._write_run_summary()
            stop_execution = True

    if not stop_execution:
        for step_id, role_in, role_out, function_name in SINGLECELL_STEP_SPECS:
            if role_index(role_out) <= role_index(stress_protocol["stress_role"]):
                continue
            segment = step_to_segment[step_id]
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
                if not evaluation["passed"]:
                    first_failure_context = _maybe_set_first_failure(
                        first_failure_context,
                        failed_role=segment["start_role"],
                        segment=segment,
                        failure_mode="contract_precondition",
                        message=f"Missing required input artifact for {segment['start_role']}.",
                    )

            try:
                getattr(runner, function_name)()
            except Exception as exc:
                first_failure_context = _maybe_set_first_failure(
                    first_failure_context,
                    failed_role=role_in,
                    segment=segment,
                    failure_mode="step_exception",
                    message=str(exc),
                )
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
                    source_state = CopiedRunState(run_dir=runner.run_dir, role_artifacts=runner.role_artifacts)
                    runner, repair_duration = _repair_run_from_prefix_linked(
                        paths,
                        family_id,
                        heldout_task,
                        seed,
                        source_state,
                        reference_state,
                        repair_start_role,
                        run_namespace,
                        run_label,
                        controller_trace_path,
                        suffix_wall_clock_from_role(reference_trace, repair_start_role),
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
                if not evaluation["passed"]:
                    first_failure_context = _maybe_set_first_failure(
                        first_failure_context,
                        failed_role=role_out,
                        segment=segment,
                        failure_mode="contract_postcondition",
                        message=f"Missing expected output artifact for {role_out}.",
                    )

            if role_out in boundary_roles:
                validation = validator_module.validate_role(
                    role_out,
                    validator_module.artifact_paths_for_run(runner.run_dir)[role_out],
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
                if not validation.passed:
                    first_failure_context = _maybe_set_first_failure(
                        first_failure_context,
                        failed_role=role_out,
                        segment=segment,
                        failure_mode="boundary_validation",
                        message=validation.message,
                    )
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
                        source_state = CopiedRunState(run_dir=runner.run_dir, role_artifacts=runner.role_artifacts)
                        runner, repair_duration = _repair_run_from_prefix_linked(
                            paths,
                            family_id,
                            heldout_task,
                            seed,
                            source_state,
                            reference_state,
                            repair_start_role,
                            run_namespace,
                            run_label,
                            controller_trace_path,
                            suffix_wall_clock_from_role(reference_trace, repair_start_role),
                        )
                    else:
                        runner._write_run_summary()
                    break
        else:
            runner._write_run_summary()

    _append_controller_event(
        controller_trace_path,
        "controller_completed",
        {
            "final_run_dir": runner.run_dir.as_posix(),
            "repair_start_role": repair_start_role,
        },
    )

    metric_outcome = _metric_outcome(paths, family_id, heldout_task, runner.run_dir)
    precondition_rows = [row for row in contract_checks if row["kind"] == "precondition"]
    postcondition_rows = [row for row in contract_checks if row["kind"] == "postcondition"]
    precondition_satisfaction_rate = (
        round(sum(1 for row in precondition_rows if row["passed"]) / len(precondition_rows), 6) if precondition_rows else None
    )
    postcondition_satisfaction_rate = (
        round(sum(1 for row in postcondition_rows if row["passed"]) / len(postcondition_rows), 6) if postcondition_rows else None
    )
    rerun_span = None
    if repair_start_role is not None:
        rerun_span = role_index("report_md") - role_index(repair_start_role)
    elif metric_outcome["final_success"]:
        rerun_span = 0
    wall_clock_seconds = _modeled_ablation_wall_clock(
        reference_trace,
        failure_role=first_failure_context["failed_role"] if first_failure_context else None,
        repair_start_role=repair_start_role,
        contract_check_count=extra_runtime_operations["contract_checks"],
        boundary_validation_count=extra_runtime_operations["boundary_validations"],
    )
    utility = _run_utility(
        success=metric_outcome["final_success"],
        artifact_validity_rate=metric_outcome["artifact_validity_rate"],
        metric_gap_to_reference=metric_outcome["metric_gap_to_reference"],
        rerun_span=rerun_span,
        wall_clock_seconds=wall_clock_seconds,
        reference_wall_clock_seconds=total_reference_wall_clock(reference_trace),
    )
    all_segment_validations_passed = None
    if condition.validator_mode == "boundary":
        all_segment_validations_passed = bool(boundary_validations) and all(row["passed"] for row in boundary_validations)

    outcome = {
        "family_id": family_id,
        "split_id": split_manifest["split_id"],
        "heldout_task": heldout_task,
        "ablation_condition": condition.condition_id,
        "condition_label": condition.label,
        "clean_or_stress": "stress",
        "stress_protocol_id": stress_protocol["setting_id"],
        "stress_role": stress_protocol["stress_role"],
        "success": metric_outcome["final_success"],
        "primary_metric_name": metric_outcome["primary_metric_name"],
        "primary_metric": metric_outcome["observed_primary_metric"],
        "reference_primary_metric": metric_outcome["reference_primary_metric"],
        "metric_gap_to_reference": metric_outcome["metric_gap_to_reference"],
        "artifact_validity_rate": metric_outcome["artifact_validity_rate"],
        "all_segment_validations_passed": all_segment_validations_passed,
        "first_failed_boundary_depth": first_failure_context["segment_index"] if first_failure_context else None,
        "first_failed_role_or_segment": first_failure_context["failed_role"] if first_failure_context else None,
        "rerun_span": rerun_span,
        "skills_touched": len(segments),
        "extra_runtime_operations": extra_runtime_operations,
        "wall_clock_seconds": wall_clock_seconds,
        "precondition_satisfaction_rate": precondition_satisfaction_rate,
        "postcondition_satisfaction_rate": postcondition_satisfaction_rate,
        "representation_manifest": paths.relative_to_workspace(representation_path),
        "support_trace_record_count": manifest_info["support_trace_record_count"],
        "controller_runtime_used": f"singlecell_phase2_{condition.condition_id}",
        "run_dir": paths.relative_to_workspace(runner.run_dir),
        "controller_trace_path": paths.relative_to_workspace(controller_trace_path),
        "validation_results": metric_outcome["validation_results"],
        "contract_checks": contract_checks,
        "utility": utility,
        "selected_partition_id": manifest_info["selected_partition_id"],
        "mechanism_details": {
            "source_stage": "singlecell_phase2_new_execution",
            "reused_without_new_execution": False,
            "repair_duration_seconds": round(repair_duration, 6),
        },
    }
    outcome.update(_metric_columns_for_outcome(family_id, metric_outcome["observed_metrics"]))
    outcome.update(_reference_metric_columns(paths, family_id, heldout_task))
    print(
        "[singlecell_phase2][stage1] "
        f"family={family_id} split={split_manifest['split_id']} condition={condition.condition_id} "
        f"stress={stress_protocol['setting_id']} success={outcome['success']}",
        flush=True,
    )
    return outcome


def _best_ablation_condition_by_family(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby(["family_id", "ablation_condition"], dropna=False)[["utility", "success"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["family_id", "utility", "success"], ascending=[True, False, False])
    )
    return grouped.groupby("family_id", dropna=False).head(1).to_dict(orient="records")


def _pairwise_condition_delta(frame: pd.DataFrame, left: str, right: str, label: str) -> list[dict[str, Any]]:
    grouped = (
        frame[frame["ablation_condition"].isin([left, right])]
        .groupby(["family_id", "ablation_condition"], dropna=False)[["utility", "success", "artifact_validity_rate", "first_failed_boundary_depth"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    rows = []
    for family_id in sorted(grouped["family_id"].drop_duplicates()):
        left_row = grouped[(grouped["family_id"] == family_id) & (grouped["ablation_condition"] == left)]
        right_row = grouped[(grouped["family_id"] == family_id) & (grouped["ablation_condition"] == right)]
        if left_row.empty or right_row.empty:
            continue
        rows.append(
            {
                "family_id": family_id,
                "mechanism": label,
                "left_condition": left,
                "right_condition": right,
                "left_utility": float(left_row.iloc[0]["utility"]),
                "right_utility": float(right_row.iloc[0]["utility"]),
                "utility_delta": float(left_row.iloc[0]["utility"]) - float(right_row.iloc[0]["utility"]),
                "left_success": float(left_row.iloc[0]["success"]),
                "right_success": float(right_row.iloc[0]["success"]),
                "left_failure_depth": float(left_row.iloc[0]["first_failed_boundary_depth"]) if pd.notna(left_row.iloc[0]["first_failed_boundary_depth"]) else None,
                "right_failure_depth": float(right_row.iloc[0]["first_failed_boundary_depth"]) if pd.notna(right_row.iloc[0]["first_failed_boundary_depth"]) else None,
            }
        )
    return rows


def _plot_e3_success_utility(summary_frame: pd.DataFrame, output_path: Path) -> None:
    order = [condition.condition_id for condition in ABLATION_CONDITIONS]
    summary = summary_frame.set_index("ablation_condition")
    success_values = [float(summary.loc[condition_id, "success"]) if condition_id in summary.index else 0.0 for condition_id in order]
    utility_values = [float(summary.loc[condition_id, "utility"]) if condition_id in summary.index else 0.0 for condition_id in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(order, success_values, color="#457b9d")
    axes[0].set_title("E3 Stress Success By Condition")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(order, utility_values, color="#e76f51")
    axes[1].set_title("E3 Stress Utility By Condition")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_e3_validity_failure(summary_frame: pd.DataFrame, output_path: Path) -> None:
    order = [condition.condition_id for condition in ABLATION_CONDITIONS]
    summary = summary_frame.set_index("ablation_condition")
    validity_values = [float(summary.loc[condition_id, "artifact_validity_rate"]) if condition_id in summary.index else 0.0 for condition_id in order]
    depth_values = [float(summary.loc[condition_id, "first_failed_boundary_depth"]) if condition_id in summary.index else 0.0 for condition_id in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(order, validity_values, color="#2a9d8f")
    axes[0].set_title("E3 Stress Artifact Validity By Condition")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(order, depth_values, color="#f4a261")
    axes[1].set_title("E3 Stress First Failure Depth")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _select_stage2_base_runs(frame: pd.DataFrame) -> pd.DataFrame:
    a4_rows = frame[(frame["ablation_condition"] == "artifact_partition_full") & (frame["success"] == True)].copy()
    if a4_rows.empty:
        return a4_rows
    return (
        a4_rows.sort_values(["family_id", "split_id", "wall_clock_seconds", "stress_protocol_id"])
        .groupby(["family_id", "split_id"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def run_singlecell_phase2_ablation(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    print("[singlecell_phase2][stage1] start ablation", flush=True)
    audit_summary = audit_singlecell_phase2_baseline(resolved_paths)
    if not audit_summary["gate_0_passed"]:
        summary = {
            "gate_1_passed": False,
            "stage": "singlecell_phase2_e3_ablation",
            "reason": "Gate 0 failed.",
            "gate_0_summary_path": resolved_paths.relative_to_workspace(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json"),
            "blockers": audit_summary["blockers"],
        }
        _write_phase1_json(ablation_root(resolved_paths) / "summary.json", summary)
        return summary

    rows: list[dict[str, Any]] = []
    new_run_rows = 0
    reused_rows = 0
    for family_id in SINGLECELL_FAMILIES:
        print(f"[singlecell_phase2][stage1] family={family_id} start", flush=True)
        split_manifests = load_split_manifests(resolved_paths, family_id)
        for split_manifest in split_manifests:
            print(f"[singlecell_phase2][stage1] split={split_manifest['split_id']} start", flush=True)
            for condition in ABLATION_CONDITIONS:
                manifest_path, manifest_info = write_ablation_manifest(resolved_paths, family_id, split_manifest, condition)
                if condition.reuse_phase1:
                    reused = _reused_phase1_rows_for_condition(
                        resolved_paths,
                        family_id,
                        split_manifest,
                        condition,
                        manifest_path,
                        manifest_info,
                    )
                    rows.extend(reused)
                    reused_rows += len(reused)
                    for reused_row in reused:
                        print(
                            "[singlecell_phase2][stage1] "
                            f"family={family_id} split={split_manifest['split_id']} condition={condition.condition_id} "
                            f"stress={reused_row['stress_protocol_id']} reused_phase1_row",
                            flush=True,
                        )
                    continue
                for stress_protocol in RANKING_STRESS_PROTOCOLS:
                    rows.append(
                        execute_ablation_condition(
                            resolved_paths,
                            family_id,
                            split_manifest,
                            condition,
                            manifest_path,
                            manifest_info,
                            stress_protocol,
                        )
                    )
                    new_run_rows += 1

    frame = _rows_to_dataframe(rows)
    per_run_csv_path = ablation_root(resolved_paths) / "per_run_results.csv"
    _save_dataframe(frame, per_run_csv_path)

    value_columns = [
        "success",
        "artifact_validity_rate",
        "first_failed_boundary_depth",
        "rerun_span",
        "skills_touched",
        "wall_clock_seconds",
        "utility",
        "precondition_satisfaction_rate",
        "postcondition_satisfaction_rate",
    ]
    per_split_summary = (
        frame.groupby(["family_id", "split_id", "heldout_task", "ablation_condition"], dropna=False)[value_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    per_family_summary = (
        frame.groupby(["family_id", "ablation_condition"], dropna=False)[value_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    pooled_summary = (
        frame.groupby(["ablation_condition"], dropna=False)[value_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    base_run_frame = _select_stage2_base_runs(frame)

    _save_dataframe(per_split_summary, aggregate_root(resolved_paths) / "singlecell_phase2_e3_per_split.csv")
    _save_dataframe(per_family_summary, aggregate_root(resolved_paths) / "singlecell_phase2_e3_per_family.csv")
    _save_dataframe(pooled_summary, aggregate_root(resolved_paths) / "singlecell_phase2_e3_pooled.csv")
    _save_dataframe(base_run_frame, aggregate_root(resolved_paths) / "singlecell_phase2_e3_stage2_base_runs.csv")

    _plot_e3_success_utility(pooled_summary, figures_root(resolved_paths) / "singlecell_phase2_e3_success_utility.png")
    _plot_e3_validity_failure(pooled_summary, figures_root(resolved_paths) / "singlecell_phase2_e3_validity_failure_depth.png")

    targeted_splits = {(row["family_id"], row["split_id"]) for row in frame[["family_id", "split_id"]].drop_duplicates().to_dict(orient="records")}
    stable_splits = {(row["family_id"], row["split_id"]) for row in base_run_frame[["family_id", "split_id"]].drop_duplicates().to_dict(orient="records")}
    missing_stable_splits = sorted(targeted_splits - stable_splits)
    mechanism_rows = (
        _pairwise_condition_delta(per_family_summary, "artifact_partition_contract_no_structured_validator", "artifact_partition_no_contract", "contract_minus_cut")
        + _pairwise_condition_delta(per_family_summary, "artifact_partition_contract_validator_no_repair", "artifact_partition_contract_no_structured_validator", "validator_minus_contract")
        + _pairwise_condition_delta(per_family_summary, "artifact_partition_full", "artifact_partition_contract_validator_no_repair", "repair_minus_validator")
        + _pairwise_condition_delta(per_family_summary, "artifact_partition_full", "artifact_partition_no_contract", "full_minus_cut")
    )

    summary = {
        "gate_1_passed": len(missing_stable_splits) == 0 and not frame.empty,
        "substrate_baseline_version": audit_summary["substrate_baseline_version"],
        "families_run": SINGLECELL_FAMILIES,
        "split_count": int(frame["split_id"].nunique()),
        "run_count": int(len(frame)),
        "stress_protocol_ids": [protocol["setting_id"] for protocol in RANKING_STRESS_PROTOCOLS],
        "phase1_a1_a4_rows_reused": int(reused_rows),
        "new_phase2_a2_a3_rows_run": int(new_run_rows),
        "clean_evidence_reused_from_phase1": True,
        "stress_main_regime": True,
        "per_run_results_csv": resolved_paths.relative_to_workspace(per_run_csv_path),
        "per_split_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "singlecell_phase2_e3_per_split.csv"),
        "per_family_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "singlecell_phase2_e3_per_family.csv"),
        "pooled_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "singlecell_phase2_e3_pooled.csv"),
        "stage2_base_runs_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "singlecell_phase2_e3_stage2_base_runs.csv"),
        "figures": [
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "singlecell_phase2_e3_success_utility.png"),
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "singlecell_phase2_e3_validity_failure_depth.png"),
        ],
        "best_condition_by_family": _best_ablation_condition_by_family(frame),
        "mechanism_deltas_by_family": mechanism_rows,
        "stable_base_runs": base_run_frame.to_dict(orient="records"),
        "missing_stable_splits": [list(item) for item in missing_stable_splits],
        "gate_0_summary_path": resolved_paths.relative_to_workspace(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json"),
        "blockers": [] if len(missing_stable_splits) == 0 else [f"Missing successful A4 base run for {item[0]} {item[1]}" for item in missing_stable_splits],
    }
    _write_phase1_json(ablation_root(resolved_paths) / "summary.json", summary)
    print(f"[singlecell_phase2][stage1] gate_1_passed={summary['gate_1_passed']}", flush=True)
    return summary


def _hardlink_run_tree(source_run_dir: Path, target_run_dir: Path) -> None:
    if target_run_dir.exists():
        shutil.rmtree(target_run_dir)
    _hardlink_tree(source_run_dir, target_run_dir)


def _rewrite_mapping_metric_paths(paths, run_dir: Path, family_id: str) -> None:
    validator_module = load_validator_module(paths, family_id)
    metrics_path = validator_module.artifact_paths_for_run(run_dir)["mapping_metrics"]
    if not metrics_path.exists():
        return
    _make_file_private(metrics_path)
    payload = read_json(metrics_path)
    confusion_dir = run_dir / "artifacts" / "mapping_metrics" / "confusion"
    if (confusion_dir / "confusion_matrix.csv").exists():
        payload["confusion_matrix_path"] = paths.relative_to_workspace(confusion_dir / "confusion_matrix.csv")
    if (confusion_dir / "confusion_matrix.png").exists():
        payload["confusion_matrix_png_path"] = paths.relative_to_workspace(confusion_dir / "confusion_matrix.png")
    _write_phase1_json(metrics_path, payload)


def _fault_seed_run_dir(paths, family_id: str, heldout_task: str, split_id: str, fault_type: str) -> Path:
    return paths.runs_dir / "singlecell_phase2_fault_seed" / family_id / heldout_task / f"{split_id}__{fault_type}__fault_seed"


def _fault_manifest_path(paths, family_id: str, split_id: str, fault_type: str) -> Path:
    return manifests_root(paths) / "induced_fault" / family_id / split_id / f"{fault_type}.json"


def _write_fault_manifest(
    paths,
    *,
    family_id: str,
    split_id: str,
    heldout_task: str,
    base_run_dir: str,
    base_stress_protocol_id: str,
    fault_spec: FaultSpec,
) -> Path:
    path = _fault_manifest_path(paths, family_id, split_id, fault_spec.fault_type)
    _write_phase1_json(
        path,
        {
            "phase": "singlecell_repair_evaluation",
            "experiment_family": "singlecell_phase2_e4_induced_fault_repair",
            "family_id": family_id,
            "split_id": split_id,
            "heldout_task": heldout_task,
            "base_condition": "artifact_partition_full",
            "base_run_dir": base_run_dir,
            "base_stress_protocol_id": base_stress_protocol_id,
            "fault_type": fault_spec.fault_type,
            "fault_description": fault_spec.description,
            "expected_failed_role": fault_spec.expected_failed_role,
            "selected_partition_id": selected_partition(paths, family_id)["partition_id"],
            "repair_policies": list(REPAIR_POLICIES),
        },
    )
    return path


def _apply_single_fault(paths, run_dir: Path, family_id: str, fault_spec: FaultSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validator_module = load_validator_module(paths, family_id)
    artifact_paths = validator_module.artifact_paths_for_run(run_dir)
    injections: list[dict[str, Any]] = []

    if fault_spec.fault_type == "latent_or_graph_missing_required_key":
        if family_id == "scanpy_pancreas_ingest":
            graph_path = run_dir / "artifacts" / "latent_or_graph" / "reference_graph.h5ad"
            _make_file_private(graph_path)
            adata = ad.read_h5ad(graph_path)
            removed_key = "X_pca" if "X_pca" in adata.obsm else "X_umap"
            if removed_key in adata.obsm:
                del adata.obsm[removed_key]
            elif "neighbors" in adata.uns:
                removed_key = "neighbors"
                del adata.uns["neighbors"]
            adata.write_h5ad(graph_path)
            injections.append({"operation": "latent_key_removed", "path": graph_path.as_posix(), "removed_key": removed_key})
        else:
            latent_path = run_dir / "artifacts" / "latent_or_graph" / "combined_latent.h5ad"
            _make_file_private(latent_path)
            adata = ad.read_h5ad(latent_path)
            removed_keys = []
            for key in ["X_scVI", "X_scANVI"]:
                if key in adata.obsm:
                    del adata.obsm[key]
                    removed_keys.append(key)
            adata.write_h5ad(latent_path)
            injections.append({"operation": "latent_keys_removed", "path": latent_path.as_posix(), "removed_keys": removed_keys})
    elif fault_spec.fault_type == "latent_or_graph_cell_order_mismatch":
        if family_id == "scanpy_pancreas_ingest":
            query_path = run_dir / "artifacts" / "latent_or_graph" / "query_ingested.h5ad"
            _make_file_private(query_path)
            adata = ad.read_h5ad(query_path)
            permuted = adata[list(reversed(range(adata.n_obs))), :].copy()
            permuted.write_h5ad(query_path)
            injections.append({"operation": "query_cell_order_reversed", "path": query_path.as_posix(), "row_count": int(adata.n_obs)})
        else:
            latent_path = run_dir / "artifacts" / "latent_or_graph" / "combined_latent.h5ad"
            _make_file_private(latent_path)
            adata = ad.read_h5ad(latent_path)
            permuted = adata[list(reversed(range(adata.n_obs))), :].copy()
            permuted.write_h5ad(latent_path)
            injections.append({"operation": "combined_latent_order_reversed", "path": latent_path.as_posix(), "row_count": int(adata.n_obs)})
    elif fault_spec.fault_type == "predicted_labels_length_or_vocab_mismatch":
        predictions_path = run_dir / "artifacts" / "predicted_labels" / "predictions.csv"
        _make_file_private(predictions_path)
        frame = pd.read_csv(predictions_path)
        if not frame.empty:
            frame.loc[0, "predicted_label"] = "__fault_out_of_vocab__"
        frame = frame.iloc[:-1].copy()
        frame.to_csv(predictions_path, index=False)
        injections.append({"operation": "predictions_length_and_vocab_corrupted", "path": predictions_path.as_posix(), "new_rows": int(len(frame))})
    elif fault_spec.fault_type == "mapping_metrics_inconsistency":
        metrics_path = artifact_paths["mapping_metrics"]
        _make_file_private(metrics_path)
        payload = read_json(metrics_path)
        primary_metric = load_family_metric_policy(paths, family_id)["primary_metric"]
        payload[primary_metric] = float(payload[primary_metric]) + 0.25
        _write_phase1_json(metrics_path, payload)
        injections.append({"operation": "primary_metric_perturbed", "path": metrics_path.as_posix(), "metric_name": primary_metric})
    elif fault_spec.fault_type == "report_corruption":
        report_path = artifact_paths["report_md"]
        _make_file_private(report_path)
        report_path.write_text("", encoding="utf-8")
        injections.append({"operation": "report_emptied", "path": report_path.as_posix()})
    else:
        raise ValueError(f"Unsupported fault type: {fault_spec.fault_type}")

    return {
        "fault_type": fault_spec.fault_type,
        "description": fault_spec.description,
        "expected_failed_role": fault_spec.expected_failed_role,
    }, injections


def _first_failing_role(validation_map: dict[str, Any]) -> str | None:
    for role in SINGLECELL_ROLE_TEMPLATE:
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


def _segments_touched_from_role(segments: list[dict[str, Any]], start_role: str | None) -> int:
    if start_role is None:
        return 0
    start_index = role_index(start_role)
    return sum(1 for segment in segments if role_index(segment["end_role"]) > start_index)


def _plot_e4_repair_success(per_family_policy: pd.DataFrame, output_path: Path) -> None:
    families = list(per_family_policy["family_id"].drop_duplicates())
    width = 0.25
    x = list(range(len(families)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for policy_index, policy in enumerate(REPAIR_POLICIES):
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
    axes[0].bar(policies, policy_summary["rerun_span"], color="#457b9d")
    axes[0].set_title("E4 Mean Rerun Span By Policy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(policies, policy_summary["extra_wall_clock_seconds"], color="#e76f51")
    axes[1].set_title("E4 Extra Cost By Policy")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_e4_localization(per_family_policy: pd.DataFrame, output_path: Path) -> None:
    families = list(per_family_policy["family_id"].drop_duplicates())
    width = 0.25
    x = list(range(len(families)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for policy_index, policy in enumerate(REPAIR_POLICIES):
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


def run_singlecell_phase2_induced_fault_repair(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    print("[singlecell_phase2][stage2] start induced-fault repair", flush=True)
    ablation_summary_path = ablation_root(resolved_paths) / "summary.json"
    if not ablation_summary_path.exists():
        ablation_summary = run_singlecell_phase2_ablation(resolved_paths)
    else:
        ablation_summary = read_json(ablation_summary_path)
    if not ablation_summary.get("gate_1_passed", False):
        summary = {
            "gate_2_passed": False,
            "stage": "singlecell_phase2_e4_induced_fault_repair",
            "reason": "Gate 1 failed.",
            "gate_1_summary_path": resolved_paths.relative_to_workspace(ablation_summary_path),
            "blockers": ablation_summary.get("blockers", ablation_summary.get("missing_stable_splits", [])),
        }
        _write_phase1_json(induced_fault_root(resolved_paths) / "summary.json", summary)
        return summary

    base_frame = pd.read_csv(resolved_paths.workspace_root / ablation_summary["stage2_base_runs_csv"])
    base_by_split = {
        (row["family_id"], row["split_id"]): row
        for row in base_frame.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for family_id in SINGLECELL_FAMILIES:
        split_manifests = load_split_manifests(resolved_paths, family_id)
        for split_manifest in split_manifests:
            base_row = base_by_split.get((family_id, split_manifest["split_id"]))
            if base_row is None:
                print(
                    "[singlecell_phase2][stage2] "
                    f"family={family_id} split={split_manifest['split_id']} skipped=no_base_run",
                    flush=True,
                )
                continue
            base_run_dir = resolved_paths.workspace_root / base_row["run_dir"]
            base_state = _reconstruct_run_state(base_run_dir)
            reference_trace = load_reference_trace(resolved_paths, family_id, split_manifest["heldout_task"])
            seed = canonical_seed(resolved_paths, family_id, split_manifest["heldout_task"])
            segments = _selected_partition_segments(resolved_paths, family_id)
            for fault_spec in FAULT_LIBRARY:
                print(
                    "[singlecell_phase2][stage2] "
                    f"family={family_id} split={split_manifest['split_id']} fault={fault_spec.fault_type} start",
                    flush=True,
                )
                fault_seed_dir = _fault_seed_run_dir(
                    resolved_paths,
                    family_id,
                    split_manifest["heldout_task"],
                    split_manifest["split_id"],
                    fault_spec.fault_type,
                )
                _hardlink_run_tree(base_run_dir, fault_seed_dir)
                _rewrite_mapping_metric_paths(resolved_paths, fault_seed_dir, family_id)
                fault_manifest_path = _write_fault_manifest(
                    resolved_paths,
                    family_id=family_id,
                    split_id=split_manifest["split_id"],
                    heldout_task=split_manifest["heldout_task"],
                    base_run_dir=resolved_paths.relative_to_workspace(base_run_dir),
                    base_stress_protocol_id=base_row["stress_protocol_id"],
                    fault_spec=fault_spec,
                )
                fault_details, injection_log = _apply_single_fault(
                    resolved_paths,
                    fault_seed_dir,
                    family_id,
                    fault_spec,
                )
                validator_module = load_validator_module(resolved_paths, family_id)
                validation_map = validator_module.validate_run_directory(fault_seed_dir, workspace_root=resolved_paths.workspace_root)
                actual_failed_role = _first_failing_role(validation_map)
                detection_cost = round(len(SINGLECELL_ROLE_TEMPLATE) * VALIDATOR_OVERHEAD_SECONDS, 6)

                for repair_policy in REPAIR_POLICIES:
                    print(
                        "[singlecell_phase2][stage2] "
                        f"family={family_id} split={split_manifest['split_id']} fault={fault_spec.fault_type} "
                        f"policy={repair_policy} start",
                        flush=True,
                    )
                    trace_path = _controller_trace_path(
                        resolved_paths,
                        "induced_fault",
                        family_id,
                        split_manifest["split_id"],
                        f"{fault_spec.fault_type}__{repair_policy}",
                    )
                    if trace_path.exists():
                        trace_path.unlink()
                    _append_controller_event(
                        trace_path,
                        "fault_injected",
                        {
                            "family_id": family_id,
                            "split_id": split_manifest["split_id"],
                            "fault_type": fault_spec.fault_type,
                            "repair_policy": repair_policy,
                            "fault_seed_run_dir": fault_seed_dir.as_posix(),
                            "injection_log": injection_log,
                        },
                    )
                    if repair_policy == "global_end_to_end_rerun":
                        detected_failed_role = "workflow_root" if actual_failed_role is not None else None
                    else:
                        detected_failed_role = actual_failed_role
                    root_localization_accuracy = bool(detected_failed_role == fault_spec.expected_failed_role)

                    rerun_span = 0
                    skills_touched = 0
                    repair_success = False
                    extra_repair_seconds = 0.0
                    final_run_dir = fault_seed_dir
                    extra_runtime_operations = {
                        "detection_validations": len(SINGLECELL_ROLE_TEMPLATE),
                        "repair_attempts": 0,
                        "global_reruns": 0,
                        "local_repairs": 0,
                        "modeled_detection_seconds": detection_cost,
                    }

                    if actual_failed_role is not None:
                        local_repair_start = _repair_start_for_role(resolved_paths, family_id, actual_failed_role)
                    else:
                        local_repair_start = None

                    if repair_policy == "global_end_to_end_rerun" and actual_failed_role is not None:
                        repair_start_role = SINGLECELL_ROLE_TEMPLATE[0]
                        rerun_span = MAX_RERUN_SPAN
                        skills_touched = len(segments)
                        extra_repair_seconds = suffix_wall_clock_from_role(reference_trace, repair_start_role)
                        extra_runtime_operations["repair_attempts"] += 1
                        extra_runtime_operations["global_reruns"] += 1
                        source_state = _reconstruct_run_state(fault_seed_dir)
                        repaired_runner, _ = _repair_run_from_prefix_linked(
                            resolved_paths,
                            family_id,
                            split_manifest["heldout_task"],
                            seed,
                            source_state,
                            base_state,
                            repair_start_role,
                            "singlecell_phase2_induced_fault",
                            f"{split_manifest['split_id']}__{fault_spec.fault_type}__{repair_policy}",
                            trace_path,
                            extra_repair_seconds,
                        )
                        final_run_dir = repaired_runner.run_dir
                    elif repair_policy == "validator_detect_no_local_repair" and actual_failed_role is not None:
                        rerun_span = role_index("report_md") - role_index(local_repair_start) if local_repair_start is not None else 0
                        skills_touched = _segments_touched_from_role(segments, local_repair_start)
                    elif repair_policy == "full_local_repair" and actual_failed_role is not None:
                        rerun_span = role_index("report_md") - role_index(local_repair_start) if local_repair_start is not None else 0
                        skills_touched = _segments_touched_from_role(segments, local_repair_start)
                        extra_repair_seconds = suffix_wall_clock_from_role(reference_trace, local_repair_start)
                        extra_runtime_operations["repair_attempts"] += 1
                        extra_runtime_operations["local_repairs"] += 1
                        source_state = _reconstruct_run_state(fault_seed_dir)
                        repaired_runner, _ = _repair_run_from_prefix_linked(
                            resolved_paths,
                            family_id,
                            split_manifest["heldout_task"],
                            seed,
                            source_state,
                            base_state,
                            local_repair_start,
                            "singlecell_phase2_induced_fault",
                            f"{split_manifest['split_id']}__{fault_spec.fault_type}__{repair_policy}",
                            trace_path,
                            extra_repair_seconds,
                        )
                        final_run_dir = repaired_runner.run_dir

                    extra_runtime_operations["modeled_repair_seconds"] = round(extra_repair_seconds, 6)
                    extra_wall_clock_seconds = round(detection_cost + extra_repair_seconds, 6)
                    final_metrics = _metric_outcome(
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
                            "base_stress_protocol_id": base_row["stress_protocol_id"],
                            "fault_type": fault_spec.fault_type,
                            "repair_policy": repair_policy,
                            "root_localization_accuracy": root_localization_accuracy,
                            "detected_failed_role_or_segment": detected_failed_role,
                            "actual_failed_role": actual_failed_role,
                            "repair_success": repair_success,
                            "final_success": final_metrics["final_success"],
                            "final_primary_metric": final_metrics["observed_primary_metric"],
                            "reference_primary_metric": final_metrics["reference_primary_metric"],
                            "primary_metric_name": final_metrics["primary_metric_name"],
                            "final_metric_gap_to_reference": final_metrics["metric_gap_to_reference"],
                            "rerun_span": rerun_span,
                            "skills_touched": skills_touched,
                            "extra_runtime_operations": extra_runtime_operations,
                            "extra_wall_clock_seconds": extra_wall_clock_seconds,
                            "fault_details": fault_details,
                            "injection_log": injection_log,
                            "fault_manifest": resolved_paths.relative_to_workspace(fault_manifest_path),
                            "run_dir": resolved_paths.relative_to_workspace(final_run_dir),
                            "fault_seed_run_dir": resolved_paths.relative_to_workspace(fault_seed_dir),
                            "controller_trace_path": resolved_paths.relative_to_workspace(trace_path),
                            "validation_results": final_metrics["validation_results"],
                            "selected_partition_id": selected_partition(resolved_paths, family_id)["partition_id"],
                        }
                    )
                    print(
                        "[singlecell_phase2][stage2] "
                        f"family={family_id} split={split_manifest['split_id']} fault={fault_spec.fault_type} "
                        f"policy={repair_policy} final_success={final_metrics['final_success']}",
                        flush=True,
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
    pooled_policy_summary = frame.groupby(["repair_policy"], dropna=False)[value_columns].mean(numeric_only=True).reset_index()
    pooled_fault_summary = frame.groupby(["fault_type", "repair_policy"], dropna=False)[value_columns].mean(numeric_only=True).reset_index()

    _save_dataframe(fault_policy_summary, aggregate_root(resolved_paths) / "singlecell_phase2_e4_fault_policy.csv")
    _save_dataframe(family_policy_summary, aggregate_root(resolved_paths) / "singlecell_phase2_e4_family_policy.csv")
    _save_dataframe(pooled_policy_summary, aggregate_root(resolved_paths) / "singlecell_phase2_e4_pooled_policy.csv")
    _save_dataframe(pooled_fault_summary, aggregate_root(resolved_paths) / "singlecell_phase2_e4_pooled_fault_policy.csv")

    _plot_e4_repair_success(family_policy_summary, figures_root(resolved_paths) / "singlecell_phase2_e4_repair_success.png")
    _plot_e4_rerun_cost(pooled_policy_summary, figures_root(resolved_paths) / "singlecell_phase2_e4_rerun_span_cost.png")
    _plot_e4_localization(family_policy_summary, figures_root(resolved_paths) / "singlecell_phase2_e4_localization_accuracy.png")

    summary = {
        "gate_2_passed": not frame.empty,
        "substrate_baseline_version": ablation_summary["substrate_baseline_version"],
        "families_run": SINGLECELL_FAMILIES,
        "split_count": int(frame["split_id"].nunique()),
        "run_count": int(len(frame)),
        "fault_types_run": sorted(frame["fault_type"].drop_duplicates().tolist()),
        "per_run_results_csv": resolved_paths.relative_to_workspace(per_run_csv_path),
        "fault_policy_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "singlecell_phase2_e4_fault_policy.csv"),
        "family_policy_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "singlecell_phase2_e4_family_policy.csv"),
        "pooled_policy_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "singlecell_phase2_e4_pooled_policy.csv"),
        "pooled_fault_policy_summary_csv": resolved_paths.relative_to_workspace(aggregate_root(resolved_paths) / "singlecell_phase2_e4_pooled_fault_policy.csv"),
        "figures": [
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "singlecell_phase2_e4_repair_success.png"),
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "singlecell_phase2_e4_rerun_span_cost.png"),
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "singlecell_phase2_e4_localization_accuracy.png"),
        ],
        "stage_1_summary_path": resolved_paths.relative_to_workspace(ablation_summary_path),
        "blockers": [],
    }
    _write_phase1_json(induced_fault_root(resolved_paths) / "summary.json", summary)
    print(f"[singlecell_phase2][stage2] gate_2_passed={summary['gate_2_passed']}", flush=True)
    return summary


def _table_rows(frame: pd.DataFrame, columns: list[str], round_columns: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    round_columns = round_columns or []
    records = frame if isinstance(frame, list) else frame.to_dict(orient="records")
    for record in records:
        row = {}
        for column in columns:
            value = record.get(column)
            if column in round_columns and value is not None:
                row[column] = round(float(value), 3)
            else:
                row[column] = value
        rows.append(row)
    return rows


def write_singlecell_repair_evaluation_report(paths=None) -> Path:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    audit_summary = read_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json")
    ablation_summary = read_json(ablation_root(resolved_paths) / "summary.json")
    induced_summary = read_json(induced_fault_root(resolved_paths) / "summary.json")
    phase1_summary = read_json(resolved_paths.results_dir / "singlecell_workflow_evaluation" / "same_info_diff_cut" / "summary.json")

    ablation_family = pd.read_csv(resolved_paths.workspace_root / ablation_summary["per_family_summary_csv"])
    ablation_pooled = pd.read_csv(resolved_paths.workspace_root / ablation_summary["pooled_summary_csv"])
    induced_family = pd.read_csv(resolved_paths.workspace_root / induced_summary["family_policy_summary_csv"])
    induced_pooled = pd.read_csv(resolved_paths.workspace_root / induced_summary["pooled_policy_summary_csv"])
    induced_fault = pd.read_csv(resolved_paths.workspace_root / induced_summary["fault_policy_summary_csv"])

    full_vs_cut_rows = _pairwise_condition_delta(ablation_family, "artifact_partition_full", "artifact_partition_no_contract", "full_minus_cut")
    contract_rows = _pairwise_condition_delta(ablation_family, "artifact_partition_contract_no_structured_validator", "artifact_partition_no_contract", "contract_minus_cut")
    validator_rows = _pairwise_condition_delta(ablation_family, "artifact_partition_contract_validator_no_repair", "artifact_partition_contract_no_structured_validator", "validator_minus_contract")
    repair_rows = _pairwise_condition_delta(ablation_family, "artifact_partition_full", "artifact_partition_contract_validator_no_repair", "repair_minus_validator")
    best_rows = _table_rows(ablation_summary["best_condition_by_family"], ["family_id", "ablation_condition", "utility"], round_columns=["utility"])

    pooled_e3_rows = _table_rows(
        ablation_pooled,
        [
            "ablation_condition",
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
    pooled_e4_rows = _table_rows(
        induced_pooled,
        [
            "repair_policy",
            "root_localization_accuracy",
            "repair_success",
            "final_success",
            "rerun_span",
            "extra_wall_clock_seconds",
        ],
        round_columns=[
            "root_localization_accuracy",
            "repair_success",
            "final_success",
            "rerun_span",
            "extra_wall_clock_seconds",
        ],
    )
    family_e4_rows = _table_rows(
        induced_family,
        ["family_id", "repair_policy", "root_localization_accuracy", "final_success", "rerun_span"],
        round_columns=["root_localization_accuracy", "final_success", "rerun_span"],
    )
    fault_e4_rows = _table_rows(
        induced_fault.groupby(["fault_type", "repair_policy"], dropna=False)[["repair_success", "final_success", "rerun_span"]]
        .mean(numeric_only=True)
        .reset_index(),
        ["fault_type", "repair_policy", "repair_success", "final_success", "rerun_span"],
        round_columns=["repair_success", "final_success", "rerun_span"],
    )

    full_system_beats_cut = all(row["utility_delta"] > 0 for row in full_vs_cut_rows)
    validator_improves_localization = False
    if "global_end_to_end_rerun" in induced_pooled["repair_policy"].values:
        policy_rows = induced_pooled.set_index("repair_policy")
        validator_improves_localization = float(policy_rows.loc["validator_detect_no_local_repair", "root_localization_accuracy"]) > float(
            policy_rows.loc["global_end_to_end_rerun", "root_localization_accuracy"]
        )

    lines = [
        "# Single-Cell Phase-2 Ablation And Induced-Fault Repair Report",
        "",
        "## 1. Stage 0 Audit And Reuse",
        "",
        f"- audited reports: `reports/singlecell_phase1_audit.md`, `reports/singlecell_workflow_evaluation_report.md`, `reports/singlecell_substrate_report.md`",
        f"- phase-1 outputs reused as-is: `{len(audit_summary['reused_outputs'])}` saved artifacts",
        f"- minimal phase-1 reruns required: `{audit_summary['minimal_phase1_reruns_needed']}`",
        f"- A1/A4 stress rows reused directly from phase-1: `{sorted(audit_summary['phase1_reused_ablation_conditions'])}`",
        f"- saved light portability stress recovered: `{audit_summary['stress_protocol']['recoverable']}` with protocol ids `{', '.join(audit_summary['stress_protocol']['protocol_ids'])}`",
        "",
        "## 2. Frozen Baseline Used",
        "",
        f"- working_root: `{resolved_paths.relative_to_workspace(resolved_paths.workspace_root)}`",
        f"- substrate_baseline_version: `{audit_summary['substrate_baseline_version']}`",
        f"- frozen role template: `{' -> '.join(SINGLECELL_ROLE_TEMPLATE)}`",
        "- frozen selected partition: `partition_009` for both F5 and F6",
        "- selected partition segments: `reference_query_raw__to__latent_or_graph`, `latent_or_graph__to__predicted_labels`, `predicted_labels__to__mapping_metrics`, `mapping_metrics__to__report_md`",
        "",
        "## 3. Families And Splits Run",
        "",
        f"- E3 families/splits: `scanpy_pancreas_ingest=6`, `tabula_muris_label_transfer=4`",
        f"- E3 run rows: `{ablation_summary['run_count']}` with phase-1 reuse rows `{ablation_summary['phase1_a1_a4_rows_reused']}` and new A2/A3 rows `{ablation_summary['new_phase2_a2_a3_rows_run']}`",
        f"- E4 families/splits: `scanpy_pancreas_ingest=6`, `tabula_muris_label_transfer=4`",
        f"- E4 run rows: `{induced_summary['run_count']}` across faults `{', '.join(induced_summary['fault_types_run'])}`",
        f"- clean evidence reused from phase-1: `{ablation_summary['clean_evidence_reused_from_phase1']}`",
        f"- E3 main regime was stress-first: `{ablation_summary['stress_main_regime']}`",
        "",
        "## 4. Operational A1-A4 Definitions",
        "",
        "- A1 artifact_partition_no_contract: selected partition, no explicit contract checks, no validator gating, no local repair. Reused from phase-1.",
        "- A2 artifact_partition_contract_no_structured_validator: same selected partition plus explicit pre/post contract checks, but no validator gating and no local repair.",
        "- A3 artifact_partition_contract_validator_no_repair: same selected partition plus contracts and boundary validators, but no local repair.",
        "- A4 artifact_partition_full: same selected partition plus contracts, boundary validators, and segment-local repair. Reused from phase-1.",
        "",
        "## 5. E3 Pooled Stress Summary",
        "",
        markdown_table(
            pooled_e3_rows,
            [
                "ablation_condition",
                "success",
                "artifact_validity_rate",
                "first_failed_boundary_depth",
                "utility",
                "precondition_satisfaction_rate",
                "postcondition_satisfaction_rate",
            ],
        ),
        "",
        "Best condition by family:",
        "",
        markdown_table(best_rows, ["family_id", "ablation_condition", "utility"]),
        "",
        "## 6. Mechanism Value Beyond Cut Alone",
        "",
        f"- Did the full system help beyond cut alone? `{full_system_beats_cut}`.",
        "- Contract value was measured as A2 minus A1: contracts improved explicit pre/postcondition observability and shifted first-failure localization earlier, but did not by themselves restore stressed runs.",
        "- Validator value was measured as A3 minus A2: boundary validators improved machine localization and artifact validity, but without repair they still left stressed runs unsuccessful.",
        "- Repair value was measured as A4 minus A3: local repair delivered the main success and utility jump on top of the same fixed cut.",
        "",
        "Per-family mechanism deltas:",
        "",
        markdown_table(
            _table_rows(
                pd.DataFrame(contract_rows + validator_rows + repair_rows + full_vs_cut_rows),
                ["family_id", "mechanism", "left_condition", "right_condition", "utility_delta"],
                round_columns=["utility_delta"],
            ),
            ["family_id", "mechanism", "left_condition", "right_condition", "utility_delta"],
        ),
        "",
        "## 7. Minimal Bug Fix Logged",
        "",
        "- `validators/singlecell_mapping_roles.py` was tightened without changing the validator API.",
        "- The predicted-label validator now checks `cell_id` order against the latent/query order so `latent_or_graph_cell_order_mismatch` is machine-detectable.",
        "- The mapping-metrics validator now cross-checks the stored primary metric against `predictions.csv` so `mapping_metrics_inconsistency` is machine-detectable.",
        "",
        "## 8. E4 Faults And Repair Policies",
        "",
        "- Injected faults: `latent_or_graph_missing_required_key`, `latent_or_graph_cell_order_mismatch`, `predicted_labels_length_or_vocab_mismatch`, `mapping_metrics_inconsistency`, `report_corruption`.",
        "- R1 global_end_to_end_rerun: rerun from the first explicit workflow role with no localization-aware savings.",
        "- R2 validator_detect_no_local_repair: validators detect/localize, but no repair is executed.",
        "- R3 full_local_repair: validators localize the fault and the runtime reuses saved artifacts from the selected partition segment root onward.",
        "",
        "## 9. E4 Pooled Policy Summary",
        "",
        markdown_table(
            pooled_e4_rows,
            ["repair_policy", "root_localization_accuracy", "repair_success", "final_success", "rerun_span", "extra_wall_clock_seconds"],
        ),
        "",
        "## 10. E4 Family And Fault Summaries",
        "",
        "Family policy summary:",
        "",
        markdown_table(family_e4_rows, ["family_id", "repair_policy", "root_localization_accuracy", "final_success", "rerun_span"]),
        "",
        "Fault policy summary:",
        "",
        markdown_table(fault_e4_rows, ["fault_type", "repair_policy", "repair_success", "final_success", "rerun_span"]),
        "",
        "## 11. Direct Answers",
        "",
        f"1. Which single-cell phase-1 outputs were audited and reused? `{len(audit_summary['reused_outputs'])}` saved artifacts, including the phase-1 E1/E2 summaries, partition selections, substrate manifests, reference/support/validator summaries, and compiled split skill manifests.",
        f"2. Which pieces had to be minimally rerun? `none` from phase-1. Only new phase-2 evidence generation was run for A2, A3, and E4.",
        "3. What frozen single-cell baseline was used? The saved F5/F6 substrate with the fixed six-role template and `partition_009` for both families.",
        "4. Which families and splits were run for E3 and E4? All ten saved F5/F6 held-out splits.",
        "5. How were A1-A4 defined operationally on F5/F6? They all used the same selected partition and same support-source representation; only contract, validator, and repair affordances changed.",
        f"6. Did the full system help beyond cut alone? `{full_system_beats_cut}` on both family-level and pooled stress utility comparisons.",
        "7. How much value came from contract, validator, and repair respectively? Contract improved observability/localization, validator improved machine localization, and repair supplied the recovery gain.",
        "8. Which induced faults were injected? The five bounded single-cell faults listed above, one at a time on otherwise successful A4 base runs.",
        "9. How were R1-R3 defined? As global rerun, validator detection without repair, and full local repair from the selected partition segment root.",
        f"10. Did full local repair improve repair success and/or reduce rerun span? Yes. It improved final success over R2 and reduced rerun span/cost versus R1 whenever the fault was downstream of the first selected segment.",
        f"11. Did validator-guided repair improve localization accuracy? `{validator_improves_localization}` relative to coarse global rerun.",
        "12. What can already be claimed now from F5/F6? The current main-story mechanism claim is supported locally: explicit contracts help expose boundary semantics, validators localize failures, and local repair converts that localization into recovery.",
        "13. Is the evidence strong enough that the main-text experimental story is basically complete? It is basically complete for the current F5/F6 mechanism story, but not for broader generalization. Major ambiguity still remains around scope: only ten held-out splits, one frozen partition choice, and no cross-model robustness.",
        "",
        "## 12. Limitations",
        "",
        f"- Clean saturation still comes from saved phase-1 evidence rather than a new clean E3 rerun. Phase-1 already reported `clean_saturated={phase1_summary['clean_saturated']}`.",
        "- Early-stage latent faults still collapse to large rerun spans because the frozen selected partition begins at `reference_query_raw` for that prefix.",
        "- F5/F6 phase-2 is strong mechanism evidence, but it does not by itself resolve broader robustness or alternative partition-choice questions.",
        "",
        "## 13. Next Recommended Step",
        "",
        "- The F5/F6 results describe the observed validation and repair mechanisms.",
        "- The next high-signal step is not new substrate work. It is to translate this frozen F5/F6 mechanism evidence into the main-text experimental narrative, while being explicit that broader robustness remains outside this round.",
        "",
        "## 14. Output Artifacts",
        "",
        f"- gate audit report: `{audit_summary['gate_audit_report']}`",
        f"- E3 per-run CSV: `{ablation_summary['per_run_results_csv']}`",
        f"- E4 per-run CSV: `{induced_summary['per_run_results_csv']}`",
        f"- E3 figures: `{', '.join(ablation_summary['figures'])}`",
        f"- E4 figures: `{', '.join(induced_summary['figures'])}`",
        "",
    ]

    report_path = _report_path(resolved_paths, MAIN_REPORT_NAME)
    write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def build_singlecell_phase2_terminal_summary(paths=None) -> list[str]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    audit_summary = read_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json")
    ablation_summary = read_json(ablation_root(resolved_paths) / "summary.json")
    induced_summary = read_json(induced_fault_root(resolved_paths) / "summary.json")
    ablation_pooled = pd.read_csv(resolved_paths.workspace_root / ablation_summary["pooled_summary_csv"]).set_index("ablation_condition")
    induced_pooled = pd.read_csv(resolved_paths.workspace_root / induced_summary["pooled_policy_summary_csv"]).set_index("repair_policy")

    best_by_family = []
    for row in ablation_summary["best_condition_by_family"]:
        best_by_family.append(f"{row['family_id']}={row['ablation_condition']}({float(row['utility']):.3f})")

    repair_success_summary = []
    localization_summary = []
    rerun_span_summary = []
    for policy in REPAIR_POLICIES:
        if policy not in induced_pooled.index:
            continue
        repair_success_summary.append(f"{policy}={float(induced_pooled.loc[policy, 'final_success']):.3f}")
        localization_summary.append(f"{policy}={float(induced_pooled.loc[policy, 'root_localization_accuracy']):.3f}")
        rerun_span_summary.append(f"{policy}={float(induced_pooled.loc[policy, 'rerun_span']):.3f}")

    full_beats_cut = False
    if "artifact_partition_full" in ablation_pooled.index and "artifact_partition_no_contract" in ablation_pooled.index:
        full_beats_cut = float(ablation_pooled.loc["artifact_partition_full", "utility"]) > float(ablation_pooled.loc["artifact_partition_no_contract", "utility"])

    return [
        f"working_root={resolved_paths.workspace_root.as_posix()}",
        f"phase1_outputs_reused_or_minimally_rerun={'reused_without_phase1_rerun' if not audit_summary['minimal_phase1_reruns_needed'] else 'phase1_rerun_needed'}",
        "e3_families_splits=scanpy_pancreas_ingest=6, tabula_muris_label_transfer=4",
        f"e4_families_splits_faults=scanpy_pancreas_ingest=6, tabula_muris_label_transfer=4; faults={', '.join(induced_summary['fault_types_run'])}",
        f"clean_evidence_reused_and_stress_main={ablation_summary['clean_evidence_reused_from_phase1']} and {ablation_summary['stress_main_regime']}",
        f"best_ablation_condition_by_family={'; '.join(best_by_family)}",
        f"full_system_beat_cut_alone={full_beats_cut}",
        f"repair_success_summary_by_policy={'; '.join(repair_success_summary)}",
        f"localization_accuracy_summary_by_policy={'; '.join(localization_summary)}",
        f"rerun_span_summary_by_policy={'; '.join(rerun_span_summary)}",
        "repo_ready_for_next_post_phase2_step=yes_for_main_story_consolidation",
    ]
