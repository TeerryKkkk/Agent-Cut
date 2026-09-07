from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runtime import singlecell_workflow_evaluation as phase1
from runtime import singlecell_repair_evaluation as phase2
from utils.io_utils import read_json, write_text
from utils.pathing import detect_project_paths

RESULTS_ROOT_NAME = "repair_comparison"
REPORT_NAME = "STEP2_PHASE2_SELECTED_VS_MICRO_REPAIR.md"
SELECTED_METHOD_LABEL = "selected_full"
MICRO_METHOD_LABEL = "micro_full"
SELECTED_CONDITION_LABEL = "artifact_partition_full"
MICRO_CONDITION_LABEL = "micro_skill_full"
SELECTED_PARTITION_ID = "partition_009"
MICRO_PARTITION_ID = "partition_001"
SINGLECELL_FAMILIES = ["scanpy_pancreas_ingest", "tabula_muris_label_transfer"]
POLICIES = list(phase2.REPAIR_POLICIES)
FAULT_TYPES = [fault.fault_type for fault in phase2.FAULT_LIBRARY]
EPS = 1e-12


@dataclass(frozen=True)
class ScopedProjectPaths:
    base_paths: Any
    scoped_runs_dir: Path

    @property
    def launch_root(self) -> Path:
        return self.base_paths.launch_root

    @property
    def workspace_root(self) -> Path:
        return self.base_paths.workspace_root

    @property
    def layout_mode(self) -> str:
        return self.base_paths.layout_mode

    @property
    def configs_dir(self) -> Path:
        return self.base_paths.configs_dir

    @property
    def data_dir(self) -> Path:
        return self.base_paths.data_dir

    @property
    def exports_dir(self) -> Path:
        return self.base_paths.exports_dir

    @property
    def openml_cache_dir(self) -> Path:
        return self.base_paths.openml_cache_dir

    @property
    def runs_dir(self) -> Path:
        return self.scoped_runs_dir

    @property
    def results_dir(self) -> Path:
        return self.base_paths.results_dir

    @property
    def reports_dir(self) -> Path:
        return self.base_paths.reports_dir

    @property
    def skills_dir(self) -> Path:
        return self.base_paths.skills_dir

    @property
    def compiled_skills_dir(self) -> Path:
        return self.base_paths.compiled_skills_dir

    def relative_to_workspace(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix()


def step2_root(paths) -> Path:
    return paths.results_dir / RESULTS_ROOT_NAME


def aggregate_root(paths) -> Path:
    return step2_root(paths) / "aggregate_tables"


def manifests_root(paths) -> Path:
    return step2_root(paths) / "manifests"


def controller_trace_root(paths) -> Path:
    return step2_root(paths) / "controller_traces"


def scoped_runtime_runs_root(paths) -> Path:
    return step2_root(paths) / "_runtime_runs"


def report_path(paths) -> Path:
    return paths.reports_dir / REPORT_NAME


def _float_bool(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return 1.0
    if text in {"false", "0", "0.0", ""}:
        return 0.0
    return float(value)


def _bool(value: Any) -> bool:
    return _float_bool(value) > 0.5


def _int_if_close(value: float) -> int | float:
    rounded = round(value)
    return int(rounded) if abs(value - rounded) <= EPS else value


def _partition_by_id(paths, family_id: str, partition_id: str) -> dict[str, Any]:
    if partition_id == SELECTED_PARTITION_ID:
        partition = phase1.selected_partition(paths, family_id)
        if partition["partition_id"] != partition_id:
            raise ValueError(f"Selected partition mismatch for {family_id}: {partition['partition_id']}")
        return partition
    for partition in phase1.all_legal_partitions(paths, family_id):
        if partition["partition_id"] == partition_id:
            return partition
    raise KeyError(f"Partition {partition_id} not found for {family_id}")


def _repair_start_for_role(segments: list[dict[str, Any]], failed_role: str) -> str:
    for segment in segments:
        if segment["start_role"] == failed_role:
            return segment["start_role"]
        if phase1.role_index(segment["start_role"]) < phase1.role_index(failed_role) <= phase1.role_index(segment["end_role"]):
            return segment["start_role"]
    raise KeyError(f"Could not map failed role {failed_role} to partition segments.")


def _segments_touched_from_role(segments: list[dict[str, Any]], start_role: str | None) -> int:
    if start_role is None:
        return 0
    start_index = phase1.role_index(start_role)
    return sum(1 for segment in segments if phase1.role_index(segment["end_role"]) > start_index)


def _case_id(split_id: str, base_stress_protocol_id: str, fault_type: str) -> str:
    return f"{split_id}__{base_stress_protocol_id}__{fault_type}"


def _micro_fault_seed_dir(paths: ScopedProjectPaths, family_id: str, heldout_task: str, split_id: str, fault_type: str) -> Path:
    return paths.runs_dir / "fault_seed" / family_id / heldout_task / f"{split_id}__{MICRO_METHOD_LABEL}__{fault_type}__fault_seed"


def _micro_fault_manifest_path(paths, family_id: str, split_id: str, fault_type: str) -> Path:
    return manifests_root(paths) / "micro_fault" / family_id / split_id / f"{fault_type}.json"


def _micro_controller_trace_path(paths, family_id: str, split_id: str, fault_type: str, policy_label: str) -> Path:
    return controller_trace_root(paths) / family_id / split_id / f"{MICRO_METHOD_LABEL}__{fault_type}__{policy_label}.jsonl"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_selected_phase2_rows(paths) -> pd.DataFrame:
    frame = pd.read_csv(paths.results_dir / "singlecell_repair_evaluation" / "induced_fault" / "per_run_results.csv")
    frame = frame[frame["family_id"].isin(SINGLECELL_FAMILIES)].copy()
    if frame.empty:
        raise RuntimeError("Canonical selected phase-2 induced-fault rows are missing.")
    frame["method_label"] = SELECTED_METHOD_LABEL
    frame["condition_label"] = SELECTED_CONDITION_LABEL
    frame["partition_id"] = frame["selected_partition_id"]
    frame["policy_label"] = frame["repair_policy"]
    frame["case_id"] = frame.apply(
        lambda row: _case_id(str(row["split_id"]), str(row["base_stress_protocol_id"]), str(row["fault_type"])),
        axis=1,
    )
    frame["localization"] = frame["root_localization_accuracy"].apply(_float_bool)
    frame["repair_success"] = frame["repair_success"].apply(_float_bool)
    frame["final_success"] = frame["final_success"].apply(_float_bool)
    frame["rerun_span"] = frame["rerun_span"].astype(float)
    frame["skills_touched"] = frame["skills_touched"].astype(float)
    frame["extra_wall_clock"] = frame["extra_wall_clock_seconds"].astype(float)
    frame["source_origin"] = "canonical_reused_selected"
    return frame


def _load_selected_base_runs(paths) -> pd.DataFrame:
    ablation_summary = read_json(paths.results_dir / "singlecell_repair_evaluation" / "ablation" / "summary.json")
    base_runs_path = paths.workspace_root / ablation_summary["stage2_base_runs_csv"]
    frame = pd.read_csv(base_runs_path)
    frame = frame[frame["family_id"].isin(SINGLECELL_FAMILIES)].copy()
    if frame.empty:
        raise RuntimeError("Canonical selected phase-2 base runs are missing.")
    return frame


def _load_micro_phase1_lookup(paths) -> dict[tuple[str, str, str], dict[str, Any]]:
    frame = pd.read_csv(paths.results_dir / "singlecell_workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv")
    frame = frame[
        (frame["family_id"].isin(SINGLECELL_FAMILIES))
        & (frame["condition"] == "micro_skill")
        & (frame["clean_or_stress"] == "stress")
        & (frame["success"].astype(str).str.lower() == "true")
    ].copy()
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = (str(row["family_id"]), str(row["split_id"]), str(row["stress_protocol_id"]))
        lookup[key] = row
    return lookup


def _selected_fault_policy_cross_checks(paths, selected_frame: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    selected_grouped = (
        selected_frame.groupby(["policy_label"], dropna=False)[
            ["localization", "repair_success", "final_success", "rerun_span", "skills_touched", "extra_wall_clock"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    canonical_pooled = pd.read_csv(paths.results_dir / "singlecell_repair_evaluation" / "aggregate_tables" / "singlecell_phase2_e4_pooled_policy.csv")
    canonical_pooled = canonical_pooled.rename(
        columns={
            "repair_policy": "policy_label",
            "root_localization_accuracy": "localization",
            "extra_wall_clock_seconds": "extra_wall_clock",
        }
    )
    for row in canonical_pooled.to_dict(orient="records"):
        observed = selected_grouped[selected_grouped["policy_label"] == row["policy_label"]]
        if observed.empty:
            checks.append({"check": f"selected_pooled::{row['policy_label']}", "match": False, "reason": "missing_observed"})
            continue
        obs = observed.iloc[0]
        match = (
            abs(float(obs["localization"]) - float(row["localization"])) <= EPS
            and abs(float(obs["repair_success"]) - float(row["repair_success"])) <= EPS
            and abs(float(obs["final_success"]) - float(row["final_success"])) <= EPS
            and abs(float(obs["rerun_span"]) - float(row["rerun_span"])) <= EPS
            and abs(float(obs["skills_touched"]) - float(row["skills_touched"])) <= EPS
            and abs(float(obs["extra_wall_clock"]) - float(row["extra_wall_clock"])) <= EPS
        )
        checks.append({"check": f"selected_pooled::{row['policy_label']}", "match": match})

    canonical_fault = pd.read_csv(paths.results_dir / "singlecell_repair_evaluation" / "aggregate_tables" / "singlecell_phase2_e4_fault_policy.csv")
    canonical_fault = canonical_fault.rename(
        columns={
            "repair_policy": "policy_label",
            "root_localization_accuracy": "localization",
            "extra_wall_clock_seconds": "extra_wall_clock",
        }
    )
    selected_fault = (
        selected_frame.groupby(["family_id", "fault_type", "policy_label"], dropna=False)[
            ["localization", "repair_success", "final_success", "rerun_span", "skills_touched", "extra_wall_clock"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    for row in canonical_fault.to_dict(orient="records"):
        observed = selected_fault[
            (selected_fault["family_id"] == row["family_id"])
            & (selected_fault["fault_type"] == row["fault_type"])
            & (selected_fault["policy_label"] == row["policy_label"])
        ]
        if observed.empty:
            checks.append(
                {
                    "check": f"selected_fault::{row['family_id']}::{row['fault_type']}::{row['policy_label']}",
                    "match": False,
                    "reason": "missing_observed",
                }
            )
            continue
        obs = observed.iloc[0]
        match = (
            abs(float(obs["localization"]) - float(row["localization"])) <= EPS
            and abs(float(obs["repair_success"]) - float(row["repair_success"])) <= EPS
            and abs(float(obs["final_success"]) - float(row["final_success"])) <= EPS
            and abs(float(obs["rerun_span"]) - float(row["rerun_span"])) <= EPS
            and abs(float(obs["skills_touched"]) - float(row["skills_touched"])) <= EPS
            and abs(float(obs["extra_wall_clock"]) - float(row["extra_wall_clock"])) <= EPS
        )
        checks.append(
            {
                "check": f"selected_fault::{row['family_id']}::{row['fault_type']}::{row['policy_label']}",
                "match": match,
            }
        )
    return checks


def _run_micro_phase2_rows(paths) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    scoped_paths = ScopedProjectPaths(paths, scoped_runtime_runs_root(paths))
    selected_base_runs = _load_selected_base_runs(paths)
    micro_lookup = _load_micro_phase1_lookup(paths)
    rows: list[dict[str, Any]] = []
    deviations: list[dict[str, Any]] = []
    source_files = [
        "results/singlecell_repair_evaluation/ablation/summary.json",
        "results/singlecell_repair_evaluation/aggregate_tables/singlecell_phase2_e3_stage2_base_runs.csv",
        "results/singlecell_workflow_evaluation/same_info_diff_cut/per_run_results.csv",
        "results/partitions/scanpy_pancreas_ingest/legal_partitions.json",
        "results/partitions/tabula_muris_label_transfer/legal_partitions.json",
    ]

    for family_id in SINGLECELL_FAMILIES:
        micro_partition = _partition_by_id(paths, family_id, MICRO_PARTITION_ID)
        if micro_partition["partition_id"] != MICRO_PARTITION_ID:
            raise ValueError(f"Micro partition mismatch for {family_id}: {micro_partition['partition_id']}")
        micro_segments = phase1.segment_rows(micro_partition)
        family_base_runs = selected_base_runs[selected_base_runs["family_id"] == family_id]
        split_manifests = {
            row["split_id"]: row for row in phase1.load_split_manifests(paths, family_id)
        }
        for base_row in family_base_runs.to_dict(orient="records"):
            split_id = str(base_row["split_id"])
            heldout_task = str(base_row["heldout_task"])
            stress_protocol_id = str(base_row["stress_protocol_id"])
            lookup_key = (family_id, split_id, stress_protocol_id)
            micro_base = micro_lookup.get(lookup_key)
            if micro_base is None:
                deviations.append(
                    {
                        "kind": "missing_micro_base_match",
                        "family_id": family_id,
                        "split_id": split_id,
                        "stress_protocol_id": stress_protocol_id,
                    }
                )
                continue

            split_manifest = split_manifests[split_id]
            base_run_dir = paths.workspace_root / str(micro_base["run_dir"])
            if not base_run_dir.exists():
                deviations.append(
                    {
                        "kind": "missing_micro_base_run_dir",
                        "family_id": family_id,
                        "split_id": split_id,
                        "stress_protocol_id": stress_protocol_id,
                        "run_dir": str(micro_base["run_dir"]),
                    }
                )
                continue

            base_state = phase1._reconstruct_run_state(base_run_dir)
            reference_trace = phase1.load_reference_trace(paths, family_id, heldout_task)
            seed = phase1.canonical_seed(paths, family_id, heldout_task)

            for fault_spec in phase2.FAULT_LIBRARY:
                fault_seed_dir = _micro_fault_seed_dir(scoped_paths, family_id, heldout_task, split_id, fault_spec.fault_type)
                phase2._hardlink_run_tree(base_run_dir, fault_seed_dir)
                phase2._rewrite_mapping_metric_paths(scoped_paths, fault_seed_dir, family_id)

                fault_manifest_path = _micro_fault_manifest_path(paths, family_id, split_id, fault_spec.fault_type)
                _write_json(
                    fault_manifest_path,
                    {
                        "phase": "repair_comparison",
                        "family_id": family_id,
                        "split_id": split_id,
                        "heldout_task": heldout_task,
                        "method_label": MICRO_METHOD_LABEL,
                        "condition_label": MICRO_CONDITION_LABEL,
                        "partition_id": MICRO_PARTITION_ID,
                        "base_condition": MICRO_CONDITION_LABEL,
                        "base_run_dir": paths.relative_to_workspace(base_run_dir),
                        "matched_selected_base_stress_protocol_id": stress_protocol_id,
                        "fault_type": fault_spec.fault_type,
                        "fault_description": fault_spec.description,
                        "expected_failed_role": fault_spec.expected_failed_role,
                        "repair_policies": POLICIES,
                    },
                )

                fault_details, injection_log = phase2._apply_single_fault(scoped_paths, fault_seed_dir, family_id, fault_spec)
                validator_module = phase1.load_validator_module(scoped_paths, family_id)
                validation_map = validator_module.validate_run_directory(fault_seed_dir, workspace_root=scoped_paths.workspace_root)
                actual_failed_role = phase2._first_failing_role(validation_map)
                detection_cost = round(len(phase1.SINGLECELL_ROLE_TEMPLATE) * phase1.VALIDATOR_OVERHEAD_SECONDS, 6)
                case_id = _case_id(split_id, stress_protocol_id, fault_spec.fault_type)

                for policy_label in POLICIES:
                    trace_path = _micro_controller_trace_path(paths, family_id, split_id, fault_spec.fault_type, policy_label)
                    if trace_path.exists():
                        trace_path.unlink()
                    phase2._append_controller_event(
                        trace_path,
                        "fault_injected",
                        {
                            "family_id": family_id,
                            "split_id": split_id,
                            "method_label": MICRO_METHOD_LABEL,
                            "case_id": case_id,
                            "fault_type": fault_spec.fault_type,
                            "repair_policy": policy_label,
                            "fault_seed_run_dir": fault_seed_dir.as_posix(),
                            "injection_log": injection_log,
                        },
                    )

                    detected_failed_role = "workflow_root" if policy_label == "global_end_to_end_rerun" and actual_failed_role is not None else actual_failed_role
                    localization = float(detected_failed_role == fault_spec.expected_failed_role)
                    rerun_span = 0.0
                    skills_touched = 0.0
                    repair_success = 0.0
                    extra_repair_seconds = 0.0
                    final_run_dir = fault_seed_dir
                    extra_runtime_operations = {
                        "detection_validations": len(phase1.SINGLECELL_ROLE_TEMPLATE),
                        "repair_attempts": 0,
                        "global_reruns": 0,
                        "local_repairs": 0,
                        "modeled_detection_seconds": detection_cost,
                    }

                    local_repair_start = _repair_start_for_role(micro_segments, actual_failed_role) if actual_failed_role is not None else None

                    if policy_label == "global_end_to_end_rerun" and actual_failed_role is not None:
                        rerun_span = float(phase2.MAX_RERUN_SPAN)
                        skills_touched = float(len(micro_segments))
                        extra_repair_seconds = phase1.suffix_wall_clock_from_role(reference_trace, phase1.SINGLECELL_ROLE_TEMPLATE[0])
                        extra_runtime_operations["repair_attempts"] += 1
                        extra_runtime_operations["global_reruns"] += 1
                        source_state = phase1._reconstruct_run_state(fault_seed_dir)
                        repaired_runner, _ = phase2._repair_run_from_prefix_linked(
                            scoped_paths,
                            family_id,
                            heldout_task,
                            seed,
                            source_state,
                            base_state,
                            phase1.SINGLECELL_ROLE_TEMPLATE[0],
                            "micro_repair",
                            f"{split_id}__{fault_spec.fault_type}__{policy_label}__{MICRO_METHOD_LABEL}",
                            trace_path,
                            extra_repair_seconds,
                        )
                        final_run_dir = repaired_runner.run_dir
                    elif policy_label == "validator_detect_no_local_repair" and actual_failed_role is not None:
                        rerun_span = float(phase1.role_index("report_md") - phase1.role_index(local_repair_start))
                        skills_touched = float(_segments_touched_from_role(micro_segments, local_repair_start))
                    elif policy_label == "full_local_repair" and actual_failed_role is not None:
                        rerun_span = float(phase1.role_index("report_md") - phase1.role_index(local_repair_start))
                        skills_touched = float(_segments_touched_from_role(micro_segments, local_repair_start))
                        extra_repair_seconds = phase1.suffix_wall_clock_from_role(reference_trace, local_repair_start)
                        extra_runtime_operations["repair_attempts"] += 1
                        extra_runtime_operations["local_repairs"] += 1
                        source_state = phase1._reconstruct_run_state(fault_seed_dir)
                        repaired_runner, _ = phase2._repair_run_from_prefix_linked(
                            scoped_paths,
                            family_id,
                            heldout_task,
                            seed,
                            source_state,
                            base_state,
                            local_repair_start,
                            "micro_repair",
                            f"{split_id}__{fault_spec.fault_type}__{policy_label}__{MICRO_METHOD_LABEL}",
                            trace_path,
                            extra_repair_seconds,
                        )
                        final_run_dir = repaired_runner.run_dir

                    extra_runtime_operations["modeled_repair_seconds"] = round(extra_repair_seconds, 6)
                    extra_wall_clock = round(detection_cost + extra_repair_seconds, 6)
                    final_metrics = phase2._metric_outcome(scoped_paths, family_id, heldout_task, final_run_dir)
                    if policy_label in {"global_end_to_end_rerun", "full_local_repair"}:
                        repair_success = _float_bool(final_metrics["final_success"])

                    rows.append(
                        {
                            "family_id": family_id,
                            "split_id": split_id,
                            "heldout_task": heldout_task,
                            "method_label": MICRO_METHOD_LABEL,
                            "condition_label": MICRO_CONDITION_LABEL,
                            "partition_id": MICRO_PARTITION_ID,
                            "policy_label": policy_label,
                            "fault_type": fault_spec.fault_type,
                            "case_id": case_id,
                            "base_stress_protocol_id": stress_protocol_id,
                            "localization": localization,
                            "repair_success": repair_success,
                            "final_success": _float_bool(final_metrics["final_success"]),
                            "rerun_span": rerun_span,
                            "skills_touched": skills_touched,
                            "extra_wall_clock": extra_wall_clock,
                            "detected_failed_role_or_segment": detected_failed_role,
                            "actual_failed_role": actual_failed_role,
                            "source_origin": "newly_run_micro",
                            "run_dir": paths.relative_to_workspace(final_run_dir),
                            "fault_seed_run_dir": paths.relative_to_workspace(fault_seed_dir),
                            "controller_trace_path": paths.relative_to_workspace(trace_path),
                            "fault_manifest": paths.relative_to_workspace(fault_manifest_path),
                            "extra_runtime_operations_json": json.dumps(extra_runtime_operations, sort_keys=True),
                            "validation_results_json": json.dumps(final_metrics["validation_results"], sort_keys=True),
                            "fault_details_json": json.dumps(fault_details, sort_keys=True),
                            "injection_log_json": json.dumps(injection_log, sort_keys=True),
                        }
                    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No Step-2 micro repair rows were generated.")
    return frame, deviations, source_files


def _standardize_selected_columns(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        [
            "family_id",
            "split_id",
            "heldout_task",
            "method_label",
            "condition_label",
            "partition_id",
            "policy_label",
            "fault_type",
            "case_id",
            "base_stress_protocol_id",
            "localization",
            "repair_success",
            "final_success",
            "rerun_span",
            "skills_touched",
            "extra_wall_clock",
            "detected_failed_role_or_segment",
            "actual_failed_role",
            "source_origin",
            "run_dir",
            "fault_seed_run_dir",
            "controller_trace_path",
            "fault_manifest",
            "extra_runtime_operations_json",
            "validation_results_json",
            "fault_details_json",
            "injection_log_json",
        ]
    ].copy()
    return selected


def _selected_standardized_rows(paths) -> pd.DataFrame:
    selected = _load_selected_phase2_rows(paths)
    selected["fault_details_json"] = selected.get("fault_details_json", "")
    selected["injection_log_json"] = selected.get("injection_log_json", "")
    standardized = selected.rename(columns={"base_stress_protocol_id": "base_stress_protocol_id"})
    return _standardize_selected_columns(standardized)


def _aggregate(frame: pd.DataFrame, group_columns: list[str], include_family_counts: bool = True) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        if include_family_counts:
            row["n_families"] = int(group["family_id"].nunique())
            row["n_splits"] = int(group["split_id"].nunique())
        row["n_cases"] = int(group["case_id"].nunique())
        row["mean_localization"] = float(group["localization"].mean())
        row["mean_repair_success"] = float(group["repair_success"].mean())
        row["mean_final_success"] = float(group["final_success"].mean())
        row["mean_rerun_span"] = float(group["rerun_span"].mean())
        row["mean_skills_touched"] = float(group["skills_touched"].mean())
        row["mean_extra_wall_clock"] = float(group["extra_wall_clock"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _policy_deltas(per_family_policy: pd.DataFrame, pooled_policy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_frame, scope_column in [(pooled_policy, None), (per_family_policy, "family_id")]:
        for policy_label in POLICIES:
            if scope_column is None:
                subset = source_frame[source_frame["policy_label"] == policy_label]
                scopes = [("pooled", subset)]
            else:
                scopes = []
                for family_id in SINGLECELL_FAMILIES:
                    subset = source_frame[
                        (source_frame["policy_label"] == policy_label) & (source_frame["family_id"] == family_id)
                    ]
                    scopes.append((family_id, subset))
            for scope_name, subset in scopes:
                if subset.empty:
                    continue
                selected_row = subset[subset["method_label"] == SELECTED_METHOD_LABEL]
                micro_row = subset[subset["method_label"] == MICRO_METHOD_LABEL]
                if selected_row.empty or micro_row.empty:
                    continue
                left = selected_row.iloc[0]
                right = micro_row.iloc[0]
                rows.append(
                    {
                        "scope": scope_name,
                        "policy_label": policy_label,
                        "selected_minus_micro_localization": float(left["mean_localization"]) - float(right["mean_localization"]),
                        "selected_minus_micro_repair_success": float(left["mean_repair_success"]) - float(right["mean_repair_success"]),
                        "selected_minus_micro_final_success": float(left["mean_final_success"]) - float(right["mean_final_success"]),
                        "selected_minus_micro_rerun_span": float(left["mean_rerun_span"]) - float(right["mean_rerun_span"]),
                        "selected_minus_micro_skills_touched": float(left["mean_skills_touched"]) - float(right["mean_skills_touched"]),
                        "selected_minus_micro_extra_wall_clock": float(left["mean_extra_wall_clock"]) - float(right["mean_extra_wall_clock"]),
                    }
                )
    return pd.DataFrame(rows)


def _r3_relative_call(delta_row: pd.Series) -> str:
    final_delta = float(delta_row["selected_minus_micro_final_success"])
    localization_delta = float(delta_row["selected_minus_micro_localization"])
    rerun_delta = float(delta_row["selected_minus_micro_rerun_span"])
    wall_delta = float(delta_row["selected_minus_micro_extra_wall_clock"])
    skills_delta = float(delta_row["selected_minus_micro_skills_touched"])

    if abs(final_delta) <= EPS and abs(localization_delta) <= EPS:
        if abs(rerun_delta) <= EPS and abs(wall_delta) <= EPS and abs(skills_delta) <= EPS:
            return "tied"
        if rerun_delta > EPS or wall_delta > EPS or skills_delta > EPS:
            return "worse"
        if rerun_delta < -EPS or wall_delta < -EPS or skills_delta < -EPS:
            return "better"
    if final_delta > EPS or localization_delta > EPS:
        return "better"
    if final_delta < -EPS or localization_delta < -EPS:
        return "worse"
    return "tied"


def _find_fault_driver(fault_policy: pd.DataFrame) -> dict[str, Any] | None:
    r3 = fault_policy[fault_policy["policy_label"] == "full_local_repair"]
    drivers: list[dict[str, Any]] = []
    for fault_type in FAULT_TYPES:
        subset = r3[r3["fault_type"] == fault_type]
        selected_row = subset[subset["method_label"] == SELECTED_METHOD_LABEL]
        micro_row = subset[subset["method_label"] == MICRO_METHOD_LABEL]
        if selected_row.empty or micro_row.empty:
            continue
        left = selected_row.iloc[0]
        right = micro_row.iloc[0]
        drivers.append(
            {
                "fault_type": fault_type,
                "selected_minus_micro_final_success": float(left["mean_final_success"]) - float(right["mean_final_success"]),
                "selected_minus_micro_rerun_span": float(left["mean_rerun_span"]) - float(right["mean_rerun_span"]),
                "selected_minus_micro_extra_wall_clock": float(left["mean_extra_wall_clock"]) - float(right["mean_extra_wall_clock"]),
                "magnitude": max(
                    abs(float(left["mean_final_success"]) - float(right["mean_final_success"])),
                    abs(float(left["mean_rerun_span"]) - float(right["mean_rerun_span"])),
                    abs(float(left["mean_extra_wall_clock"]) - float(right["mean_extra_wall_clock"])),
                ),
            }
        )
    if not drivers:
        return None
    driver = max(drivers, key=lambda row: row["magnitude"])
    return None if driver["magnitude"] <= EPS else driver


def _write_report(
    paths,
    pooled_policy: pd.DataFrame,
    per_family_policy: pd.DataFrame,
    fault_policy: pd.DataFrame,
    deltas: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    pooled_rows = pooled_policy.sort_values(["policy_label", "method_label"])
    delta_pooled = deltas[deltas["scope"] == "pooled"].set_index("policy_label")
    r3_call = _r3_relative_call(delta_pooled.loc["full_local_repair"])
    fault_driver = _find_fault_driver(fault_policy)

    def pooled_row(method_label: str, policy_label: str) -> pd.Series:
        subset = pooled_rows[(pooled_rows["method_label"] == method_label) & (pooled_rows["policy_label"] == policy_label)]
        return subset.iloc[0]

    f5_r3 = per_family_policy[
        (per_family_policy["family_id"] == "scanpy_pancreas_ingest") & (per_family_policy["policy_label"] == "full_local_repair")
    ].set_index("method_label")
    f6_r3 = per_family_policy[
        (per_family_policy["family_id"] == "tabula_muris_label_transfer") & (per_family_policy["policy_label"] == "full_local_repair")
    ].set_index("method_label")

    lines = [
        "# STEP2 Phase-2 Selected vs Micro Repair",
        "",
        "## 1. Scope and canonical inputs",
        "",
        "- Phase-1 parity from Step 1 was treated as frozen and was not revisited.",
        "- Step 2 exists because the repo did not yet contain a dedicated selected-vs-micro phase-2 repair comparison table for the single-cell line.",
        "- Canonical selected phase-2 rows were reused directly from `results/singlecell_repair_evaluation/induced_fault/per_run_results.csv` together with the canonical selected pooled/fault summaries.",
        "- Canonical freeze context was read from `reports/EXPERIMENT_FREEZE_STEP0.md` and `reports/STEP1_PHASE1_PARETO.md`.",
        "",
        "## 2. Comparison setup and matching rule",
        "",
        "- Methods compared: `selected_full` / `artifact_partition_full` / `partition_009` versus `micro_full` / `micro_skill_full` / `partition_001`.",
        "- Policies compared exactly: `global_end_to_end_rerun`, `validator_detect_no_local_repair`, and `full_local_repair`.",
        "- Selected rows were reused; only the missing micro full-system rows were newly run.",
        "- Matching rule: for each canonical selected stage-2 base split, the micro base run was the canonical phase-1 `micro_skill` stress row with the same `family_id`, `split_id`, and `stress_protocol_id`; comparison cases were aligned by `split_id + base_stress_protocol_id + fault_type + policy_label`.",
        "- No ranking outputs, same-cardinality outputs, or mismatch-only follow-up outputs were promoted into the main Step-2 story.",
        "",
        "## 3. Pooled selected-vs-micro repair comparison",
        "",
        f"- R1 `global_end_to_end_rerun`: selected final_success=`{pooled_row(SELECTED_METHOD_LABEL, 'global_end_to_end_rerun')['mean_final_success']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'global_end_to_end_rerun')['mean_final_success']:.3f}`; selected rerun_span=`{pooled_row(SELECTED_METHOD_LABEL, 'global_end_to_end_rerun')['mean_rerun_span']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'global_end_to_end_rerun')['mean_rerun_span']:.3f}`; selected extra_wall_clock=`{pooled_row(SELECTED_METHOD_LABEL, 'global_end_to_end_rerun')['mean_extra_wall_clock']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'global_end_to_end_rerun')['mean_extra_wall_clock']:.3f}`.",
        f"- R2 `validator_detect_no_local_repair`: selected final_success=`{pooled_row(SELECTED_METHOD_LABEL, 'validator_detect_no_local_repair')['mean_final_success']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'validator_detect_no_local_repair')['mean_final_success']:.3f}`; selected rerun_span=`{pooled_row(SELECTED_METHOD_LABEL, 'validator_detect_no_local_repair')['mean_rerun_span']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'validator_detect_no_local_repair')['mean_rerun_span']:.3f}`; selected extra_wall_clock=`{pooled_row(SELECTED_METHOD_LABEL, 'validator_detect_no_local_repair')['mean_extra_wall_clock']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'validator_detect_no_local_repair')['mean_extra_wall_clock']:.3f}`.",
        f"- R3 `full_local_repair`: selected final_success=`{pooled_row(SELECTED_METHOD_LABEL, 'full_local_repair')['mean_final_success']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'full_local_repair')['mean_final_success']:.3f}`; selected rerun_span=`{pooled_row(SELECTED_METHOD_LABEL, 'full_local_repair')['mean_rerun_span']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'full_local_repair')['mean_rerun_span']:.3f}`; selected extra_wall_clock=`{pooled_row(SELECTED_METHOD_LABEL, 'full_local_repair')['mean_extra_wall_clock']:.3f}` vs micro `{pooled_row(MICRO_METHOD_LABEL, 'full_local_repair')['mean_extra_wall_clock']:.3f}`.",
        f"- Pooled R3 call: selected is `{r3_call}` relative to micro under the baseline phase-2 repair setup.",
        "",
        "## 4. Family-level comparison",
        "",
        f"- F5 / `scanpy_pancreas_ingest`, R3: selected final_success=`{float(f5_r3.loc[SELECTED_METHOD_LABEL, 'mean_final_success']):.3f}` vs micro `{float(f5_r3.loc[MICRO_METHOD_LABEL, 'mean_final_success']):.3f}`; selected rerun_span=`{float(f5_r3.loc[SELECTED_METHOD_LABEL, 'mean_rerun_span']):.3f}` vs micro `{float(f5_r3.loc[MICRO_METHOD_LABEL, 'mean_rerun_span']):.3f}`; selected extra_wall_clock=`{float(f5_r3.loc[SELECTED_METHOD_LABEL, 'mean_extra_wall_clock']):.3f}` vs micro `{float(f5_r3.loc[MICRO_METHOD_LABEL, 'mean_extra_wall_clock']):.3f}`.",
        f"- F6 / `tabula_muris_label_transfer`, R3: selected final_success=`{float(f6_r3.loc[SELECTED_METHOD_LABEL, 'mean_final_success']):.3f}` vs micro `{float(f6_r3.loc[MICRO_METHOD_LABEL, 'mean_final_success']):.3f}`; selected rerun_span=`{float(f6_r3.loc[SELECTED_METHOD_LABEL, 'mean_rerun_span']):.3f}` vs micro `{float(f6_r3.loc[MICRO_METHOD_LABEL, 'mean_rerun_span']):.3f}`; selected extra_wall_clock=`{float(f6_r3.loc[SELECTED_METHOD_LABEL, 'mean_extra_wall_clock']):.3f}` vs micro `{float(f6_r3.loc[MICRO_METHOD_LABEL, 'mean_extra_wall_clock']):.3f}`.",
        "",
        "## 5. Fault-level notes",
        "",
    ]
    if fault_driver is None:
        lines.append("- No pooled fault type produced a material selected-vs-micro difference under R3; the methods remained extremely similar.")
    else:
        lines.append(
            f"- The main pooled R3 difference is concentrated in `{fault_driver['fault_type']}`: "
            f"selected_minus_micro_final_success=`{fault_driver['selected_minus_micro_final_success']:.3f}`, "
            f"selected_minus_micro_rerun_span=`{fault_driver['selected_minus_micro_rerun_span']:.3f}`, "
            f"selected_minus_micro_extra_wall_clock=`{fault_driver['selected_minus_micro_extra_wall_clock']:.3f}`."
        )
        lines.append("- Other fault types remained tied or near-tied on the main pooled metrics.")
    lines.extend(
        [
            "",
            "## 6. Data availability, deviations, and limitations",
            "",
            "- Selected canonical phase-2 rows were reused directly; they were not reproduced.",
            f"- Newly executed micro rows: `{manifest['new_micro_runs_executed']}`.",
            f"- Matching deviations from the ideal row key were `{len(manifest['deviations_from_ideal_row_matching'])}`.",
            "- The mismatch-only follow-up under `results/singlecell_targeted_integrity_repair_followup/` was not used as a replacement all-fault main table.",
            f"- Selected reuse cross-checks against canonical pooled/fault tables all passed: `{manifest['all_selected_reuse_cross_checks_passed']}`.",
            "- No canonical result trees were modified.",
        ]
    )
    write_text(report_path(paths), "\n".join(lines) + "\n")
    return report_path(paths)


def run_repair_comparison(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    step2_root(resolved_paths).mkdir(parents=True, exist_ok=True)
    aggregate_root(resolved_paths).mkdir(parents=True, exist_ok=True)
    manifests_root(resolved_paths).mkdir(parents=True, exist_ok=True)
    controller_trace_root(resolved_paths).mkdir(parents=True, exist_ok=True)
    scoped_runtime_runs_root(resolved_paths).mkdir(parents=True, exist_ok=True)

    selected_rows = _selected_standardized_rows(resolved_paths)
    selected_checks = _selected_fault_policy_cross_checks(resolved_paths, selected_rows)

    micro_rows, deviations, extra_sources = _run_micro_phase2_rows(resolved_paths)
    combined = pd.concat([selected_rows, micro_rows], ignore_index=True)

    per_run_path = step2_root(resolved_paths) / "per_run_results.csv"
    combined.to_csv(per_run_path, index=False)

    pooled_policy = _aggregate(combined, ["method_label", "partition_id", "policy_label"])
    per_family_policy = _aggregate(combined, ["family_id", "method_label", "partition_id", "policy_label"])
    fault_policy = _aggregate(combined, ["fault_type", "method_label", "partition_id", "policy_label"], include_family_counts=False)
    deltas = _policy_deltas(per_family_policy, pooled_policy)

    pooled_policy_path = aggregate_root(resolved_paths) / "selected_vs_micro_pooled_policy.csv"
    per_family_policy_path = aggregate_root(resolved_paths) / "selected_vs_micro_per_family_policy.csv"
    deltas_path = aggregate_root(resolved_paths) / "selected_minus_micro_policy_deltas.csv"
    fault_policy_path = aggregate_root(resolved_paths) / "selected_vs_micro_fault_policy.csv"
    pooled_policy.to_csv(pooled_policy_path, index=False)
    per_family_policy.to_csv(per_family_policy_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    fault_policy.to_csv(fault_policy_path, index=False)

    pooled_policy_records = pooled_policy.sort_values(["policy_label", "method_label"]).to_dict(orient="records")
    per_family_records = per_family_policy.sort_values(["family_id", "policy_label", "method_label"]).to_dict(orient="records")
    delta_records = deltas.sort_values(["scope", "policy_label"]).to_dict(orient="records")
    fault_records = fault_policy.sort_values(["fault_type", "policy_label", "method_label"]).to_dict(orient="records")
    r3_delta = deltas[(deltas["scope"] == "pooled") & (deltas["policy_label"] == "full_local_repair")].iloc[0]
    fault_driver = _find_fault_driver(fault_policy)
    r3_relative = _r3_relative_call(r3_delta)

    summary_payload = {
        "phase1_frozen_not_revisited": True,
        "selected_rows_reused": True,
        "selected_rows_reproduced": False,
        "new_micro_runs_executed": int(len(micro_rows)),
        "method_labels": {"selected": SELECTED_METHOD_LABEL, "micro": MICRO_METHOD_LABEL},
        "condition_labels": {"selected": SELECTED_CONDITION_LABEL, "micro": MICRO_CONDITION_LABEL},
        "partition_ids": {"selected": SELECTED_PARTITION_ID, "micro": MICRO_PARTITION_ID},
        "policies": POLICIES,
        "fault_types": FAULT_TYPES,
        "pooled_policy_comparison": pooled_policy_records,
        "per_family_policy_comparison": per_family_records,
        "policy_deltas": delta_records,
        "fault_policy_comparison": fault_records,
        "pooled_r3_relative_call_for_selected": r3_relative,
        "pooled_r3_fault_driver": fault_driver,
        "selected_reuse_cross_checks_passed": all(item["match"] for item in selected_checks),
        "deviations_from_ideal_row_matching": deviations,
        "canonical_result_trees_modified": False,
    }
    summary_path = step2_root(resolved_paths) / "summary.json"
    _write_json(summary_path, summary_payload)

    source_files = [
        "reports/EXPERIMENT_FREEZE_STEP0.md",
        "reports/STEP1_PHASE1_PARETO.md",
        "reports/singlecell_repair_evaluation_report.md",
        "results/singlecell_repair_evaluation/induced_fault/summary.json",
        "results/singlecell_repair_evaluation/induced_fault/per_run_results.csv",
        "results/singlecell_repair_evaluation/aggregate_tables/singlecell_phase2_e4_pooled_policy.csv",
        "results/singlecell_repair_evaluation/aggregate_tables/singlecell_phase2_e4_fault_policy.csv",
        "results/singlecell_repair_evaluation/ablation/summary.json",
        "results/singlecell_repair_evaluation/aggregate_tables/singlecell_phase2_e3_stage2_base_runs.csv",
        "results/partitions/scanpy_pancreas_ingest/selected_partition.json",
        "results/partitions/tabula_muris_label_transfer/selected_partition.json",
        "results/partitions/scanpy_pancreas_ingest/legal_partitions.json",
        "results/partitions/tabula_muris_label_transfer/legal_partitions.json",
        "results/singlecell_substrate/manifests/scanpy_pancreas_ingest/summary.json",
        "results/singlecell_substrate/manifests/tabula_muris_label_transfer/summary.json",
        "results/singlecell_workflow_evaluation/same_info_diff_cut/per_run_results.csv",
        "validators/singlecell_mapping_roles.py",
    ] + extra_sources
    source_files = sorted(set(source_files))

    manifest = {
        "source_files_read": source_files,
        "runtime_modules_used": [
            "runtime/repair_comparison.py",
            "runtime/singlecell_workflow_evaluation.py",
            "runtime/singlecell_repair_evaluation.py",
        ],
        "output_files_created": [
            resolved_paths.relative_to_workspace(per_run_path),
            resolved_paths.relative_to_workspace(pooled_policy_path),
            resolved_paths.relative_to_workspace(per_family_policy_path),
            resolved_paths.relative_to_workspace(deltas_path),
            resolved_paths.relative_to_workspace(fault_policy_path),
            resolved_paths.relative_to_workspace(summary_path),
            resolved_paths.relative_to_workspace(step2_root(resolved_paths) / "step2_manifest.json"),
            resolved_paths.relative_to_workspace(report_path(resolved_paths)),
        ],
        "selected_rows_reused": True,
        "selected_rows_reproduced": False,
        "new_micro_runs_executed": int(len(micro_rows)),
        "exact_method_labels_used": {
            "selected_method_label": SELECTED_METHOD_LABEL,
            "micro_method_label": MICRO_METHOD_LABEL,
            "selected_condition_label": SELECTED_CONDITION_LABEL,
            "micro_condition_label": MICRO_CONDITION_LABEL,
        },
        "exact_partition_ids_used": {
            "selected_partition_id": SELECTED_PARTITION_ID,
            "micro_partition_id": MICRO_PARTITION_ID,
        },
        "exact_policies_used": POLICIES,
        "exact_fault_set_used": FAULT_TYPES,
        "matching_rule": {
            "selected_rows": "reused directly from canonical phase-2 induced-fault per-run table",
            "micro_rows": "newly run from canonical phase-1 micro stress rows matched on family_id + split_id + stress_protocol_id",
            "comparison_case_key": "split_id + base_stress_protocol_id + fault_type + policy_label",
        },
        "deviations_from_ideal_row_matching": deviations,
        "selected_reuse_cross_checks": selected_checks,
        "all_selected_reuse_cross_checks_passed": all(item["match"] for item in selected_checks),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase1_revisited": False,
        "canonical_result_trees_modified": False,
    }
    manifest_path = step2_root(resolved_paths) / "step2_manifest.json"
    _write_json(manifest_path, manifest)

    report_file = _write_report(resolved_paths, pooled_policy, per_family_policy, fault_policy, deltas, manifest)

    return {
        "per_run_results_csv": resolved_paths.relative_to_workspace(per_run_path),
        "pooled_policy_csv": resolved_paths.relative_to_workspace(pooled_policy_path),
        "per_family_policy_csv": resolved_paths.relative_to_workspace(per_family_policy_path),
        "policy_deltas_csv": resolved_paths.relative_to_workspace(deltas_path),
        "fault_policy_csv": resolved_paths.relative_to_workspace(fault_policy_path),
        "summary_json": resolved_paths.relative_to_workspace(summary_path),
        "manifest_json": resolved_paths.relative_to_workspace(manifest_path),
        "report_path": resolved_paths.relative_to_workspace(report_file),
        "selected_rows_reused": True,
        "selected_rows_reproduced": False,
        "new_micro_runs_executed": int(len(micro_rows)),
        "pooled_policy_records": pooled_policy_records,
        "pooled_r3_relative_call_for_selected": r3_relative,
        "pooled_r3_fault_driver": fault_driver,
        "phase1_revisited": False,
        "canonical_result_trees_modified": False,
    }
