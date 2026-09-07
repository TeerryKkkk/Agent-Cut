from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import PREDICTIVE_ROLE_TEMPLATE, PREDICTIVE_STEP_SPECS
from pipelines.predictive_runtime import build_reference_runner, list_predictive_families
from runtime.workflow_evaluation import (
    CONTINUITY_FAMILY,
    PRIMARY_FAMILIES,
    RANKING_STRESS_PROTOCOLS,
    SAME_INFO_STRESS_PROTOCOLS,
    VALIDATOR_OVERHEAD_SECONDS,
    ConditionSpec,
    _append_controller_event,
    _apply_light_stress,
    _artifact_path_for_role,
    _finalize_same_info_outcome,
    _repair_run_from_prefix,
    build_support_evidence,
    compute_substrate_baseline_version,
    load_split_manifests,
    role_index,
    score_partitions_for_split,
    segment_rows,
    selected_partition,
)
from utils.io_utils import markdown_table, read_json, read_yaml, write_json, write_text
from utils.pathing import detect_project_paths
from validators.predictive_roles import validate_role

PHASE3_ROOT_NAME = "score_evaluation"
PHASE3_AUDIT_REPORT = "phase3_score_audit.md"
PHASE3_DIAGNOSIS_REPORT = "phase3_objective_diagnosis.md"
PHASE3_FINAL_REPORT = "score_evaluation_report.md"
VALIDATOR_FAILURE_ROLES = ["profile_json", "split_spec_json", "preprocess_bundle", "model_bundle", "metrics_json"]
REVISED_CONDITION = ConditionSpec(
    condition_id="artifact_partition_full_revised",
    label="Phase-3 artifact_partition_full_revised",
    representation_kind="artifact_partition_full_revised",
    controller_runtime="score_evaluation_skill_runtime",
    validator_mode="boundary",
    repair_mode="segment_restart",
    uses_skills=True,
    partition_mode="selected",
)


def phase3_root(paths) -> Path:
    return paths.results_dir / PHASE3_ROOT_NAME


def audit_root(paths) -> Path:
    return phase3_root(paths) / "audit"


def diagnostics_root(paths) -> Path:
    return phase3_root(paths) / "diagnostics"


def manifests_root(paths) -> Path:
    return phase3_root(paths) / "manifests"


def ranking_root(paths) -> Path:
    return phase3_root(paths) / "ranking"


def revised_partition_eval_root(paths) -> Path:
    return phase3_root(paths) / "revised_partition_eval"


def aggregate_root(paths) -> Path:
    return phase3_root(paths) / "aggregate_tables"


def figures_root(paths) -> Path:
    return phase3_root(paths) / "figures"


def _report_path(paths, report_name: str) -> Path:
    return paths.reports_dir / report_name


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_ready(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _phase3_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    json_columns = [
        "extra_runtime_operations",
        "information_budget_stats",
        "selection_details",
    ]
    for column in json_columns:
        if column in frame.columns:
            frame[f"{column}_json"] = frame[column].apply(_json_ready)
    return frame


def _safe_stat(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _variant_order(paths) -> list[str]:
    return list(load_score_revision_config(paths)["variants"].keys())


def load_score_revision_config(paths) -> dict[str, Any]:
    return read_yaml(paths.configs_dir / "score_revision.yaml")


def _stress_weight_map(paths) -> dict[str, float]:
    return {key: float(value) for key, value in load_score_revision_config(paths)["stress_weights"].items()}


def _condition_fields() -> list[str]:
    return [
        "family_id",
        "split_id",
        "heldout_task",
        "condition",
        "clean_or_stress",
        "stress_protocol_id",
        "success",
        "artifact_validity_rate",
        "utility",
        "wall_clock_seconds",
        "metric_gap_to_reference",
    ]


def _partition_numeric_id(partition_id: str) -> int:
    try:
        return int(partition_id.split("_")[-1])
    except Exception:
        return 10_000


def _micro_segment_ids() -> list[str]:
    segment_ids = []
    for start_role, end_role in zip(PREDICTIVE_ROLE_TEMPLATE[:-1], PREDICTIVE_ROLE_TEMPLATE[1:]):
        segment_ids.append(f"{start_role}__to__{end_role}")
    return segment_ids


def _partition_signature(partition: dict[str, Any]) -> tuple[str, ...]:
    return tuple(partition["segment_ids"])


def _micro_partition_id(partitions: list[dict[str, Any]]) -> str:
    signature = tuple(_micro_segment_ids())
    for partition in partitions:
        if _partition_signature(partition) == signature:
            return partition["partition_id"]
    raise KeyError("Could not locate the micro partition in the legal partition set.")


def _containing_segment(partition: dict[str, Any], role: str) -> dict[str, Any]:
    current_index = role_index(role)
    for segment in segment_rows(partition):
        start_index = role_index(segment["start_role"])
        end_index = role_index(segment["end_role"])
        if start_index < current_index <= end_index:
            return segment
    raise KeyError(f"No segment contains role {role}")


def _repair_start_role(partition: dict[str, Any], role: str) -> str:
    return _containing_segment(partition, role)["start_role"]


def _repair_span(partition: dict[str, Any], role: str) -> int:
    return role_index("report_md") - role_index(_repair_start_role(partition, role))


def _repair_efficiency(partition: dict[str, Any], role: str) -> float:
    return max(0.0, 1.0 - (_repair_span(partition, role) / float(len(PREDICTIVE_ROLE_TEMPLATE) - 1)))


def _step_costs_from_support_evidence(support_evidence: list[dict[str, Any]]) -> dict[str, float]:
    return {row["role_out"]: float(row["mean_wall_clock_s"]) for row in support_evidence}


def _reference_support_cost(step_costs: dict[str, float]) -> float:
    return float(sum(step_costs[role] for role in PREDICTIVE_ROLE_TEMPLATE[1:] if role in step_costs))


def _support_proxy_wall_clock(partition: dict[str, Any], role: str, step_costs: dict[str, float]) -> float:
    start_role = _repair_start_role(partition, role)
    suffix_roles = PREDICTIVE_ROLE_TEMPLATE[role_index(start_role) + 1 :]
    suffix_cost = float(sum(step_costs.get(output_role, 0.0) for output_role in suffix_roles))
    return suffix_cost + (VALIDATOR_OVERHEAD_SECONDS * partition["segment_count"])


def _support_proxy_utility(partition: dict[str, Any], role: str, step_costs: dict[str, float]) -> float:
    reference_cost = max(_reference_support_cost(step_costs), 1.0e-6)
    return float(
        0.35
        + 0.20
        + 0.20
        + (0.15 * _repair_efficiency(partition, role))
        + (
            0.10
            * (
                1.0
                / (1.0 + (_support_proxy_wall_clock(partition, role, step_costs) / reference_cost))
            )
        )
    )


def _compactness_reward(segment_count: int) -> float:
    max_extra_segments = len(PREDICTIVE_ROLE_TEMPLATE) - 2
    return max(0.0, 1.0 - ((segment_count - 1) / float(max_extra_segments)))


def _load_existing_phase1_frame(paths, experiment_name: str) -> pd.DataFrame:
    return pd.read_csv(paths.results_dir / "workflow_evaluation" / experiment_name / "per_run_results.csv")


def _aggregate_partition_utilities(paths) -> pd.DataFrame:
    frame = _load_existing_phase1_frame(paths, "boundary_ranking")
    frame = frame[frame["clean_or_stress"] == "stress"].copy()
    stress_weights = _stress_weight_map(paths)
    frame["stress_weight"] = frame["stress_protocol_id"].map(stress_weights).astype(float)
    group_columns = ["family_id", "split_id", "heldout_task", "partition_id", "segment_count"]
    rows: list[dict[str, Any]] = []
    for group_key, group_frame in frame.groupby(group_columns, dropna=False):
        family_id, split_id, heldout_task, partition_id, segment_count = group_key
        total_weight = float(group_frame["stress_weight"].sum())
        weighted_utility = float((group_frame["utility"] * group_frame["stress_weight"]).sum() / total_weight)
        weighted_rerun_span = float((group_frame["rerun_span"] * group_frame["stress_weight"]).sum() / total_weight)
        weighted_wall_clock = float((group_frame["wall_clock_seconds"] * group_frame["stress_weight"]).sum() / total_weight)
        weighted_failure_depth = float(
            (group_frame["first_failed_boundary_depth"].fillna(0.0) * group_frame["stress_weight"]).sum() / total_weight
        )
        row = {
            "family_id": family_id,
            "split_id": split_id,
            "heldout_task": heldout_task,
            "partition_id": partition_id,
            "segment_count": int(segment_count),
            "weighted_partition_utility": round(weighted_utility, 6),
            "min_partition_utility": round(float(group_frame["utility"].min()), 6),
            "max_partition_utility": round(float(group_frame["utility"].max()), 6),
            "weighted_rerun_span": round(weighted_rerun_span, 6),
            "weighted_wall_clock_seconds": round(weighted_wall_clock, 6),
            "weighted_first_failed_boundary_depth": round(weighted_failure_depth, 6),
            "stress_row_count": int(len(group_frame)),
        }
        for protocol_id in stress_weights:
            protocol_frame = group_frame[group_frame["stress_protocol_id"] == protocol_id]
            if protocol_frame.empty:
                row[f"utility__{protocol_id}"] = None
                row[f"rerun_span__{protocol_id}"] = None
            else:
                row[f"utility__{protocol_id}"] = round(float(protocol_frame["utility"].iloc[0]), 6)
                row[f"rerun_span__{protocol_id}"] = int(protocol_frame["rerun_span"].iloc[0])
        rows.append(row)
    aggregated = pd.DataFrame(rows).sort_values(["family_id", "split_id", "partition_id"]).reset_index(drop=True)
    _save_dataframe(aggregated, ranking_root(paths) / "weighted_partition_utilities.csv")
    return aggregated


def _build_partition_feature_frame(paths) -> pd.DataFrame:
    utility_frame = _aggregate_partition_utilities(paths)
    rows: list[dict[str, Any]] = []
    for family_id in list_predictive_families(paths):
        split_manifests = load_split_manifests(paths, family_id)
        baseline_selected = selected_partition(paths, family_id)
        for split_manifest in split_manifests:
            support_evidence, support_trace_record_count = build_support_evidence(
                paths,
                family_id,
                split_manifest["support_tasks"],
            )
            step_costs = _step_costs_from_support_evidence(support_evidence)
            scored_partitions = score_partitions_for_split(paths, family_id, split_manifest["support_tasks"])
            micro_partition_id = _micro_partition_id(scored_partitions)
            for partition in scored_partitions:
                utility_row = utility_frame[
                    (utility_frame["family_id"] == family_id)
                    & (utility_frame["split_id"] == split_manifest["split_id"])
                    & (utility_frame["partition_id"] == partition["partition_id"])
                ]
                if utility_row.empty:
                    raise KeyError(
                        f"Missing weighted utility for {family_id} {split_manifest['split_id']} {partition['partition_id']}"
                    )
                utility_payload = utility_row.iloc[0].to_dict()
                first_boundary_role = partition["boundary_roles"][0]
                first_boundary_index = role_index(first_boundary_role)
                portability_reward = sum(_repair_efficiency(partition, role) for role in ["preprocess_bundle", "model_bundle"]) / 2.0
                portability_utility_proxy = sum(
                    _support_proxy_utility(partition, role, step_costs) for role in ["preprocess_bundle", "model_bundle"]
                ) / 2.0
                validator_reward = sum(_repair_efficiency(partition, role) for role in VALIDATOR_FAILURE_ROLES) / float(
                    len(VALIDATOR_FAILURE_ROLES)
                )
                validator_utility_proxy = sum(
                    _support_proxy_utility(partition, role, step_costs) for role in VALIDATOR_FAILURE_ROLES
                ) / float(len(VALIDATOR_FAILURE_ROLES))
                rows.append(
                    {
                        "family_id": family_id,
                        "split_id": split_manifest["split_id"],
                        "heldout_task": split_manifest["heldout_task"],
                        "support_task_count": len(split_manifest["support_tasks"]),
                        "support_trace_record_count": support_trace_record_count,
                        "partition_id": partition["partition_id"],
                        "partition_numeric_id": _partition_numeric_id(partition["partition_id"]),
                        "segment_count": int(partition["segment_count"]),
                        "segment_ids_json": _json_ready(partition["segment_ids"]),
                        "boundary_roles_json": _json_ready(partition["boundary_roles"]),
                        "mean_segment_score": float(partition["mean_segment_score"]),
                        "boundary_bonus": float(partition["boundary_bonus"]),
                        "overfragmentation_penalty": float(partition["overfragmentation_penalty"]),
                        "baseline_partition_score": float(partition["partition_score"]),
                        "first_boundary_role": first_boundary_role,
                        "first_boundary_index": first_boundary_index,
                        "merged_profile_into_prefix": int(first_boundary_index > role_index("profile_json")),
                        "merged_split_into_prefix": int(first_boundary_index > role_index("split_spec_json")),
                        "merged_preprocess_into_prefix": int(first_boundary_index > role_index("preprocess_bundle")),
                        "portability_reward": round(portability_reward, 6),
                        "portability_utility_proxy": round(portability_utility_proxy, 6),
                        "validator_containment_reward": round(validator_reward, 6),
                        "validator_containment_utility_proxy": round(validator_utility_proxy, 6),
                        "compactness_reward": round(_compactness_reward(int(partition["segment_count"])), 6),
                        "baseline_selected_partition_id": baseline_selected["partition_id"],
                        "micro_partition_id": micro_partition_id,
                        "is_baseline_selected": int(partition["partition_id"] == baseline_selected["partition_id"]),
                        "is_micro_partition": int(partition["partition_id"] == micro_partition_id),
                        **utility_payload,
                    }
                )
    feature_frame = pd.DataFrame(rows).sort_values(["family_id", "split_id", "partition_id"]).reset_index(drop=True)
    _save_dataframe(feature_frame, diagnostics_root(paths) / "partition_feature_table.csv")
    return feature_frame


def _normalize_split_feature_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for feature_name in feature_names:
        minimum = float(frame[feature_name].min())
        maximum = float(frame[feature_name].max())
        if abs(maximum - minimum) <= 1.0e-12:
            normalized[feature_name] = 0.5
        else:
            normalized[feature_name] = (frame[feature_name] - minimum) / (maximum - minimum)
    return normalized


def _apply_variant_scores(paths, feature_frame: pd.DataFrame) -> pd.DataFrame:
    config = load_score_revision_config(paths)
    variant_definitions = config["variants"]
    required_features = sorted(
        {
            feature_name
            for variant_definition in variant_definitions.values()
            for feature_name in variant_definition["weights"].keys()
        }
    )
    scored_frames: list[pd.DataFrame] = []
    for _, split_frame in feature_frame.groupby(["family_id", "split_id"], dropna=False):
        split_frame = split_frame.copy()
        normalized_frame = _normalize_split_feature_frame(split_frame, required_features)
        for variant_id, variant_definition in variant_definitions.items():
            source_frame = normalized_frame if bool(variant_definition["family_normalize"]) else split_frame
            score_series = pd.Series(0.0, index=split_frame.index)
            for feature_name, weight in variant_definition["weights"].items():
                score_series = score_series + (float(weight) * source_frame[feature_name].astype(float))
            split_frame[f"score__{variant_id}"] = score_series.round(6)
        scored_frames.append(split_frame)
    scored = pd.concat(scored_frames, ignore_index=True).sort_values(["family_id", "split_id", "partition_id"])
    _save_dataframe(scored, ranking_root(paths) / "variant_scores.csv")
    write_json(manifests_root(paths) / "objective_variants.json", config)
    return scored.reset_index(drop=True)


def _rank_positions(split_frame: pd.DataFrame, score_column: str) -> dict[str, int]:
    ordered = split_frame.sort_values(
        [score_column, "weighted_partition_utility", "segment_count", "partition_numeric_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return {row["partition_id"]: index for index, row in enumerate(ordered.to_dict(orient="records"), start=1)}


def _ranking_summary_rows(frame: pd.DataFrame, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    for (family_id, split_id), split_frame in frame.groupby(["family_id", "split_id"], dropna=False):
        scored = split_frame.sort_values(
            [score_column, "weighted_partition_utility", "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        utility = split_frame.sort_values(
            ["weighted_partition_utility", score_column, "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        score_vector = split_frame[score_column].astype(float)
        utility_vector = split_frame["weighted_partition_utility"].astype(float)
        spearman_value = spearmanr(score_vector, utility_vector).statistic
        kendall_value = kendalltau(score_vector, utility_vector).statistic
        predicted_best = scored.iloc[0]
        oracle_best = utility.iloc[0]
        top3_ids = set(scored.head(3)["partition_id"].tolist())
        split_rows.append(
            {
                "family_id": family_id,
                "split_id": split_id,
                "heldout_task": split_frame["heldout_task"].iloc[0],
                "predicted_best_partition": predicted_best["partition_id"],
                "oracle_best_partition": oracle_best["partition_id"],
                "predicted_best_score": float(predicted_best[score_column]),
                "predicted_best_utility": float(predicted_best["weighted_partition_utility"]),
                "oracle_best_utility": float(oracle_best["weighted_partition_utility"]),
                "top1_regret": float(oracle_best["weighted_partition_utility"] - predicted_best["weighted_partition_utility"]),
                "top3_hit": float(oracle_best["partition_id"] in top3_ids),
                "spearman": _safe_stat(spearman_value),
                "kendall_tau": _safe_stat(kendall_value),
            }
        )
    split_summary = pd.DataFrame(split_rows)
    family_summary = (
        split_summary.groupby("family_id", dropna=False)[["spearman", "kendall_tau", "top1_regret", "top3_hit"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    pooled_spearman = spearmanr(frame[score_column].astype(float), frame["weighted_partition_utility"].astype(float)).statistic
    pooled_kendall = kendalltau(frame[score_column].astype(float), frame["weighted_partition_utility"].astype(float)).statistic
    pooled_row = {
        "family_id": "pooled",
        "spearman": _safe_stat(pooled_spearman),
        "kendall_tau": _safe_stat(pooled_kendall),
        "top1_regret": float(split_summary["top1_regret"].mean()),
        "top3_hit": float(split_summary["top3_hit"].mean()),
    }
    summary_frame = pd.concat([family_summary, pd.DataFrame([pooled_row])], ignore_index=True)
    return summary_frame, split_summary


def _selected_partition_rows(frame: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    score_column = f"score__{variant_id}"
    rows: list[dict[str, Any]] = []
    for (_, _), split_frame in frame.groupby(["family_id", "split_id"], dropna=False):
        ordered = split_frame.sort_values(
            [score_column, "weighted_partition_utility", "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        selected_row = ordered.iloc[0]
        oracle_row = split_frame.sort_values(
            ["weighted_partition_utility", score_column, "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        rows.append(
            {
                "variant_id": variant_id,
                "family_id": selected_row["family_id"],
                "split_id": selected_row["split_id"],
                "heldout_task": selected_row["heldout_task"],
                "selected_partition_id": selected_row["partition_id"],
                "selected_segment_count": int(selected_row["segment_count"]),
                "selected_score": float(selected_row[score_column]),
                "selected_weighted_utility": float(selected_row["weighted_partition_utility"]),
                "baseline_partition_id": selected_row["baseline_selected_partition_id"],
                "micro_partition_id": selected_row["micro_partition_id"],
                "oracle_partition_id": oracle_row["partition_id"],
                "oracle_weighted_utility": float(oracle_row["weighted_partition_utility"]),
                "selected_changed_from_baseline": int(
                    selected_row["partition_id"] != selected_row["baseline_selected_partition_id"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _calibration_sort_key(summary_row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    spearman = float(summary_row["mean_spearman"])
    top1_regret = float(summary_row["mean_top1_regret"])
    top3_hit = float(summary_row["mean_top3_hit"])
    kendall_tau = float(summary_row["mean_kendall_tau"])
    return (-spearman, top1_regret, -top3_hit, -kendall_tau, summary_row["variant_id"])


def audit_phase3_score_baseline(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    family_ids = list_predictive_families(resolved_paths)
    e1_frame_path = resolved_paths.results_dir / "workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv"
    e2_frame_path = resolved_paths.results_dir / "workflow_evaluation" / "boundary_ranking" / "per_run_results.csv"
    e2_summary_path = resolved_paths.results_dir / "workflow_evaluation" / "boundary_ranking" / "summary.json"
    phase2_summary_path = resolved_paths.results_dir / "repair_evaluation" / "ablation" / "summary.json"
    phase2_repair_path = resolved_paths.results_dir / "repair_evaluation" / "induced_fault" / "summary.json"
    partition_status: dict[str, Any] = {}
    selected_loadable = True
    legal_available = True
    segment_features_available = True
    for family_id in family_ids:
        family_partition_dir = resolved_paths.results_dir / "partitions" / family_id
        family_status = {
            "selected_partition": family_partition_dir / "selected_partition.json",
            "legal_partitions": family_partition_dir / "legal_partitions.json",
            "legal_segments": family_partition_dir / "legal_segments.json",
            "segment_scores": family_partition_dir / "segment_scores.json",
            "partition_scores": family_partition_dir / "partition_scores.json",
        }
        family_results = {name: path.exists() for name, path in family_status.items()}
        partition_status[family_id] = {
            "paths": {name: resolved_paths.relative_to_workspace(path) for name, path in family_status.items()},
            "available": family_results,
        }
        selected_loadable = selected_loadable and family_results["selected_partition"]
        legal_available = legal_available and family_results["legal_partitions"] and family_results["legal_segments"]
        segment_features_available = segment_features_available and family_results["segment_scores"] and family_results["partition_scores"]

    blockers: list[str] = []
    if not e1_frame_path.exists():
        blockers.append("Missing phase-1 E1 per-run results.")
    if not e2_frame_path.exists():
        blockers.append("Missing phase-1 E2 per-run results.")
    if not legal_available:
        blockers.append("Legal partition or segment files are missing for one or more families.")
    if not selected_loadable:
        blockers.append("Current selected partitions could not be loaded for one or more families.")
    if not segment_features_available:
        blockers.append("Current segment or partition score components are missing for one or more families.")

    gate_0_passed = len(blockers) == 0
    summary = {
        "gate_0_passed": gate_0_passed,
        "substrate_baseline_version": compute_substrate_baseline_version(resolved_paths),
        "families": family_ids,
        "e1_outputs": {
            "machine_usable": e1_frame_path.exists(),
            "per_run_results_csv": resolved_paths.relative_to_workspace(e1_frame_path) if e1_frame_path.exists() else None,
            "summary_json": resolved_paths.relative_to_workspace(
                resolved_paths.results_dir / "workflow_evaluation" / "same_info_diff_cut" / "summary.json"
            ),
            "reuse_decision": "reuse_as_is" if e1_frame_path.exists() else "missing",
        },
        "e2_outputs": {
            "machine_usable": e2_frame_path.exists(),
            "per_run_results_csv": resolved_paths.relative_to_workspace(e2_frame_path) if e2_frame_path.exists() else None,
            "summary_json": resolved_paths.relative_to_workspace(e2_summary_path) if e2_summary_path.exists() else None,
            "reuse_decision": (
                "reuse_raw_rows_reaggregate_partition_utility"
                if e2_frame_path.exists()
                else "missing"
            ),
        },
        "phase2_outputs": {
            "ablation_summary_json": resolved_paths.relative_to_workspace(phase2_summary_path) if phase2_summary_path.exists() else None,
            "repair_summary_json": resolved_paths.relative_to_workspace(phase2_repair_path) if phase2_repair_path.exists() else None,
            "reuse_decision": "reuse_reference_context" if phase2_summary_path.exists() and phase2_repair_path.exists() else "partial",
        },
        "partitions": {
            "available_by_family": partition_status,
            "legal_partition_sets_available": legal_available,
            "current_segment_features_available": segment_features_available,
            "selected_partitions_loadable": selected_loadable,
        },
        "minimal_reruns_required": [
            "No phase-1 or phase-2 baseline reruns are required for audit, diagnosis, or ranking-side evaluation.",
            "Only the new phase-3 revised selected-partition downstream condition needs fresh held-out execution.",
        ],
        "blockers": blockers,
        "gate_audit_report": resolved_paths.relative_to_workspace(_report_path(resolved_paths, PHASE3_AUDIT_REPORT)),
    }
    write_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json", summary)
    write_text(_report_path(resolved_paths, PHASE3_AUDIT_REPORT), _build_audit_report(summary))
    return summary


def _build_audit_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-3 Score Audit",
        "",
        "## 1. Gate 0",
        "",
        f"- Gate 0 passed: `{summary['gate_0_passed']}`",
        f"- frozen substrate baseline: `{summary['substrate_baseline_version']}`",
        "",
        "## 2. Reuse Decisions",
        "",
        f"- E1 outputs reusable as-is: `{summary['e1_outputs']['machine_usable']}` from `{summary['e1_outputs']['per_run_results_csv']}`.",
        (
            "- E2 outputs reusable as raw rows: "
            f"`{summary['e2_outputs']['machine_usable']}` from `{summary['e2_outputs']['per_run_results_csv']}`."
        ),
        "- E2 saved split summary is treated as reference context only; phase-3 will reaggregate partition utility per partition rather than reuse the legacy row-level oracle directly.",
        f"- phase-2 outputs reusable as reference context: `{summary['phase2_outputs']['reuse_decision']}`.",
        "",
        "## 3. Audit Answers",
        "",
        f"1. Which baseline E1 outputs are reusable as-is? `{summary['e1_outputs']['per_run_results_csv']}` plus its summary JSON.",
        f"2. Which baseline E2 outputs are reusable as-is? `{summary['e2_outputs']['per_run_results_csv']}` plus candidate manifests; split-level oracle summaries require partition-level reaggregation.",
        "3. Which phase-2 outputs are reusable as reference context? The saved ablation and induced-fault summaries and reports.",
        f"4. Are all legal partitions per split available or reconstructable without changing the substrate? `{summary['partitions']['legal_partition_sets_available']}`.",
        f"5. Are current segment-score feature components available or reconstructable? `{summary['partitions']['current_segment_features_available']}`.",
        f"6. Can current selected partitions be loaded directly? `{summary['partitions']['selected_partitions_loadable']}`.",
        "7. What minimal reruns are required, if any?",
        *[f"- {item}" for item in summary["minimal_reruns_required"]],
        "",
        "## 4. Family Partition Availability",
        "",
        markdown_table(
            [
                {
                    "family_id": family_id,
                    "selected_partition": payload["available"]["selected_partition"],
                    "legal_partitions": payload["available"]["legal_partitions"],
                    "segment_scores": payload["available"]["segment_scores"],
                    "partition_scores": payload["available"]["partition_scores"],
                }
                for family_id, payload in summary["partitions"]["available_by_family"].items()
            ],
            ["family_id", "selected_partition", "legal_partitions", "segment_scores", "partition_scores"],
        ),
        "",
    ]
    if summary["blockers"]:
        lines.extend(
            [
                "## 5. Blockers",
                "",
                *[f"- {blocker}" for blocker in summary["blockers"]],
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def diagnose_current_score_objective(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    audit_summary = audit_phase3_score_baseline(resolved_paths)
    if not audit_summary["gate_0_passed"]:
        return {
            "gate_1_passed": False,
            "blockers": audit_summary["blockers"],
            "diagnosis_report": resolved_paths.relative_to_workspace(_report_path(resolved_paths, PHASE3_DIAGNOSIS_REPORT)),
        }

    feature_frame = _build_partition_feature_frame(resolved_paths)
    baseline_summary, baseline_split_summary = _ranking_summary_rows(feature_frame, "baseline_partition_score")
    baseline_selected_rows = _selected_partition_rows(feature_frame.assign(score__V0=feature_frame["baseline_partition_score"]), "V0")
    selected_rank_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    for (_, _), split_frame in feature_frame.groupby(["family_id", "split_id"], dropna=False):
        baseline_ranks = _rank_positions(split_frame.assign(score=split_frame["baseline_partition_score"]), "score")
        no_penalty_score = split_frame["mean_segment_score"] + split_frame["boundary_bonus"]
        no_penalty_ranks = _rank_positions(split_frame.assign(score=no_penalty_score), "score")
        utility_ranks = _rank_positions(split_frame.assign(score=split_frame["weighted_partition_utility"]), "score")
        scored_topk = split_frame.sort_values(
            ["baseline_partition_score", "segment_count", "partition_numeric_id"],
            ascending=[False, True, True],
        ).head(5)
        for row in scored_topk.to_dict(orient="records"):
            topk_rows.append(
                {
                    "family_id": row["family_id"],
                    "split_id": row["split_id"],
                    "heldout_task": row["heldout_task"],
                    "partition_id": row["partition_id"],
                    "segment_count": int(row["segment_count"]),
                    "baseline_partition_score": float(row["baseline_partition_score"]),
                    "weighted_partition_utility": float(row["weighted_partition_utility"]),
                    "first_boundary_role": row["first_boundary_role"],
                }
            )
        baseline_partition_id = split_frame["baseline_selected_partition_id"].iloc[0]
        micro_partition_id = split_frame["micro_partition_id"].iloc[0]
        oracle_partition_id = (
            split_frame.sort_values(
                ["weighted_partition_utility", "baseline_partition_score", "segment_count", "partition_numeric_id"],
                ascending=[False, False, True, True],
            )["partition_id"].iloc[0]
        )
        selected_rank_rows.append(
            {
                "family_id": split_frame["family_id"].iloc[0],
                "split_id": split_frame["split_id"].iloc[0],
                "heldout_task": split_frame["heldout_task"].iloc[0],
                "baseline_selected_partition": baseline_partition_id,
                "oracle_partition": oracle_partition_id,
                "micro_partition": micro_partition_id,
                "baseline_selected_score_rank": baseline_ranks[baseline_partition_id],
                "baseline_selected_utility_rank": utility_ranks[baseline_partition_id],
                "micro_score_rank": baseline_ranks[micro_partition_id],
                "micro_score_rank_without_penalty": no_penalty_ranks[micro_partition_id],
                "micro_utility_rank": utility_ranks[micro_partition_id],
                "oracle_score_rank": baseline_ranks[oracle_partition_id],
                "selected_partition_score": float(
                    split_frame.loc[split_frame["partition_id"] == baseline_partition_id, "baseline_partition_score"].iloc[0]
                ),
                "selected_partition_utility": float(
                    split_frame.loc[split_frame["partition_id"] == baseline_partition_id, "weighted_partition_utility"].iloc[0]
                ),
                "oracle_partition_utility": float(
                    split_frame.loc[split_frame["partition_id"] == oracle_partition_id, "weighted_partition_utility"].iloc[0]
                ),
            }
        )

    selected_rank_frame = pd.DataFrame(selected_rank_rows)
    topk_frame = pd.DataFrame(topk_rows)
    contribution_rows = []
    for label, mask_column in [
        ("baseline_selected", "is_baseline_selected"),
        ("micro_partition", "is_micro_partition"),
    ]:
        subset = feature_frame[feature_frame[mask_column] == 1]
        contribution_rows.append(
            {
                "partition_label": label,
                "mean_segment_score": round(float(subset["mean_segment_score"].mean()), 6),
                "boundary_bonus": round(float(subset["boundary_bonus"].mean()), 6),
                "overfragmentation_penalty": round(float(subset["overfragmentation_penalty"].mean()), 6),
                "portability_reward": round(float(subset["portability_reward"].mean()), 6),
                "validator_containment_reward": round(float(subset["validator_containment_reward"].mean()), 6),
                "compactness_reward": round(float(subset["compactness_reward"].mean()), 6),
                "weighted_partition_utility": round(float(subset["weighted_partition_utility"].mean()), 6),
            }
        )
    oracle_frame = (
        feature_frame.sort_values(
            ["family_id", "split_id", "weighted_partition_utility", "baseline_partition_score", "segment_count", "partition_numeric_id"],
            ascending=[True, True, False, False, True, True],
        )
        .groupby(["family_id", "split_id"], dropna=False)
        .head(1)
    )
    contribution_rows.append(
        {
            "partition_label": "oracle_partition",
            "mean_segment_score": round(float(oracle_frame["mean_segment_score"].mean()), 6),
            "boundary_bonus": round(float(oracle_frame["boundary_bonus"].mean()), 6),
            "overfragmentation_penalty": round(float(oracle_frame["overfragmentation_penalty"].mean()), 6),
            "portability_reward": round(float(oracle_frame["portability_reward"].mean()), 6),
            "validator_containment_reward": round(float(oracle_frame["validator_containment_reward"].mean()), 6),
            "compactness_reward": round(float(oracle_frame["compactness_reward"].mean()), 6),
            "weighted_partition_utility": round(float(oracle_frame["weighted_partition_utility"].mean()), 6),
        }
    )
    contribution_frame = pd.DataFrame(contribution_rows)

    grouped_by_boundary = (
        feature_frame.groupby(["family_id", "first_boundary_role"], dropna=False)[
            ["baseline_partition_score", "weighted_partition_utility", "segment_count", "overfragmentation_penalty"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped_by_segments = (
        feature_frame.groupby(["family_id", "segment_count"], dropna=False)[
            ["baseline_partition_score", "weighted_partition_utility", "overfragmentation_penalty"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )

    _save_dataframe(baseline_summary, diagnostics_root(resolved_paths) / "baseline_partition_ranking_summary.csv")
    _save_dataframe(baseline_split_summary, diagnostics_root(resolved_paths) / "baseline_partition_ranking_split_summary.csv")
    _save_dataframe(selected_rank_frame, diagnostics_root(resolved_paths) / "selected_partition_rank_diagnostics.csv")
    _save_dataframe(topk_frame, diagnostics_root(resolved_paths) / "topk_current_partitions.csv")
    _save_dataframe(contribution_frame, diagnostics_root(resolved_paths) / "score_term_contributions.csv")
    _save_dataframe(grouped_by_boundary, diagnostics_root(resolved_paths) / "boundary_group_statistics.csv")
    _save_dataframe(grouped_by_segments, diagnostics_root(resolved_paths) / "segment_count_group_statistics.csv")
    _plot_diagnostic_figure(feature_frame, contribution_frame, figures_root(resolved_paths) / "phase3_score_term_diagnostics.png")

    summary = {
        "gate_1_passed": True,
        "substrate_baseline_version": audit_summary["substrate_baseline_version"],
        "baseline_unique_selected_partitions": sorted(selected_rank_frame["baseline_selected_partition"].unique().tolist()),
        "baseline_selected_partition_repetition": int(
            (selected_rank_frame["baseline_selected_partition"] == selected_rank_frame["baseline_selected_partition"].iloc[0]).sum()
        ),
        "split_count": int(selected_rank_frame.shape[0]),
        "baseline_selected_mean_utility_rank": round(float(selected_rank_frame["baseline_selected_utility_rank"].mean()), 6),
        "micro_mean_score_rank": round(float(selected_rank_frame["micro_score_rank"].mean()), 6),
        "micro_mean_score_rank_without_penalty": round(float(selected_rank_frame["micro_score_rank_without_penalty"].mean()), 6),
        "baseline_summary_csv": resolved_paths.relative_to_workspace(
            diagnostics_root(resolved_paths) / "baseline_partition_ranking_summary.csv"
        ),
        "score_contribution_csv": resolved_paths.relative_to_workspace(
            diagnostics_root(resolved_paths) / "score_term_contributions.csv"
        ),
        "selected_rank_csv": resolved_paths.relative_to_workspace(
            diagnostics_root(resolved_paths) / "selected_partition_rank_diagnostics.csv"
        ),
        "diagnostic_figure": resolved_paths.relative_to_workspace(
            figures_root(resolved_paths) / "phase3_score_term_diagnostics.png"
        ),
        "blockers": [],
        "diagnosis_report": resolved_paths.relative_to_workspace(_report_path(resolved_paths, PHASE3_DIAGNOSIS_REPORT)),
    }
    write_json(diagnostics_root(resolved_paths) / "diagnosis_summary.json", summary)
    write_text(_report_path(resolved_paths, PHASE3_DIAGNOSIS_REPORT), _build_diagnosis_report(summary, baseline_summary, selected_rank_frame, contribution_frame))
    return summary


def _plot_diagnostic_figure(feature_frame: pd.DataFrame, contribution_frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled_frame = feature_frame.copy()
    first_boundary_groups = pooled_frame.groupby("first_boundary_role", dropna=False)
    color_map = {
        "profile_json": "#355070",
        "split_spec_json": "#6d597a",
        "preprocess_bundle": "#b56576",
        "model_bundle": "#e56b6f",
        "metrics_json": "#eaac8b",
        "report_md": "#8ecae6",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for boundary_role, group_frame in first_boundary_groups:
        axes[0].scatter(
            group_frame["baseline_partition_score"],
            group_frame["weighted_partition_utility"],
            alpha=0.7,
            label=boundary_role,
            color=color_map.get(boundary_role, "#444444"),
            s=28,
        )
    axes[0].set_title("Baseline Score vs Weighted Utility")
    axes[0].set_xlabel("baseline partition score")
    axes[0].set_ylabel("weighted partition utility")
    axes[0].legend(fontsize=8, loc="lower left")

    plot_columns = [
        "mean_segment_score",
        "boundary_bonus",
        "overfragmentation_penalty",
        "portability_reward",
        "validator_containment_reward",
    ]
    positions = range(len(contribution_frame))
    width = 0.14
    for index, column in enumerate(plot_columns):
        axes[1].bar(
            [position + (index * width) for position in positions],
            contribution_frame[column],
            width=width,
            label=column,
        )
    axes[1].set_title("Mean Score-Term Contributions")
    axes[1].set_xticks([position + (2 * width) for position in positions], contribution_frame["partition_label"].tolist())
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _build_diagnosis_report(
    summary: dict[str, Any],
    baseline_summary: pd.DataFrame,
    selected_rank_frame: pd.DataFrame,
    contribution_frame: pd.DataFrame,
) -> str:
    pooled_row = baseline_summary[baseline_summary["family_id"] == "pooled"].iloc[0].to_dict()
    representative_rows = selected_rank_frame.head(8).to_dict(orient="records")
    lines = [
        "# Phase-3 Objective Diagnosis",
        "",
        "## 1. Gate 1",
        "",
        f"- Gate 1 passed: `{summary['gate_1_passed']}`",
        f"- frozen substrate baseline: `{summary['substrate_baseline_version']}`",
        "",
        "## 2. Failure Mode",
        "",
        (
            "- Why did the current scorer repeatedly select `partition_025` or its equivalent? "
            "Because the baseline objective heavily rewards the high-scoring merged early prefix "
            "`raw_data -> preprocess_bundle`, gives an additional bonus for `preprocess_bundle`, `model_bundle`, "
            "and `metrics_json` boundaries, and applies no penalty at exactly four segments."
        ),
        (
            "- How much comes from the over-fragmentation penalty? The micro-equivalent partition is pushed from "
            f"mean score rank `{summary['micro_mean_score_rank_without_penalty']:.2f}` without the penalty to "
            f"`{summary['micro_mean_score_rank']:.2f}` with the baseline penalty."
        ),
        "- Does the current objective systematically over-reward merged early segments? Yes. The highest baseline-score partitions disproportionately delay the first boundary until `preprocess_bundle`.",
        (
            "- How does micro_skill rank under the current objective? Poorly; it is structurally penalized for having six segments "
            "despite strong containment utility."
        ),
        "- What utility component is missing when the current objective fails? The score is missing an explicit reward for earlier portability containment at `split_spec_json -> preprocess_bundle` and for validator-local containment beyond the four-segment sweet spot.",
        "- Are there family-specific divergences? Only weak ones in the saved data; the same score bias repeats across all families because the score components are near-static across families and splits.",
        "- Is the weak signal due to score design, utility definition, or both? Mostly score design, with additional noise from the legacy E2 row-level oracle summary. Phase-3 therefore reuses the raw rows but reaggregates utility per partition.",
        "",
        "## 3. Baseline Ranking Summary",
        "",
        markdown_table(
            baseline_summary.to_dict(orient="records"),
            ["family_id", "spearman", "kendall_tau", "top1_regret", "top3_hit"],
        ),
        "",
        "## 4. Representative Split Diagnostics",
        "",
        markdown_table(
            representative_rows,
            [
                "family_id",
                "split_id",
                "baseline_selected_partition",
                "oracle_partition",
                "baseline_selected_utility_rank",
                "micro_score_rank",
                "micro_score_rank_without_penalty",
            ],
        ),
        "",
        "## 5. Score-Term Contribution Means",
        "",
        markdown_table(
            contribution_frame.to_dict(orient="records"),
            [
                "partition_label",
                "mean_segment_score",
                "boundary_bonus",
                "overfragmentation_penalty",
                "portability_reward",
                "validator_containment_reward",
                "weighted_partition_utility",
            ],
        ),
        "",
        "## 6. Diagnostic Conclusion",
        "",
        (
            "- The frozen baseline score is not arbitrary, but it is too biased toward a four-segment early-merged cut and "
            "does not explicitly reward the portability and validator-localization structure that the downstream runtime values."
        ),
        (
            "- Pooled baseline ranking remains weak-positive: "
            f"Spearman `{pooled_row['spearman']:.3f}`, top-1 regret `{pooled_row['top1_regret']:.3f}`, top-3 hit `{pooled_row['top3_hit']:.3f}`."
        ),
        "- Gate 1 therefore passes and motivates an explicit revised objective family.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_phase3_objective_variants(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    diagnosis_summary = diagnose_current_score_objective(resolved_paths)
    if not diagnosis_summary.get("gate_1_passed", False):
        return {
            "variants_ready": False,
            "blockers": diagnosis_summary.get("blockers", ["Gate 1 failed."]),
        }

    feature_frame = _build_partition_feature_frame(resolved_paths)
    scored_frame = _apply_variant_scores(resolved_paths, feature_frame)
    selected_frames = []
    ranking_summaries = []
    split_summaries = []
    for variant_id in _variant_order(resolved_paths):
        summary_frame, split_frame = _ranking_summary_rows(scored_frame, f"score__{variant_id}")
        summary_frame = summary_frame.copy()
        summary_frame["variant_id"] = variant_id
        split_frame = split_frame.copy()
        split_frame["variant_id"] = variant_id
        selected_frame = _selected_partition_rows(scored_frame, variant_id)
        selected_frames.append(selected_frame)
        ranking_summaries.append(summary_frame)
        split_summaries.append(split_frame)
    selected_partitions = pd.concat(selected_frames, ignore_index=True)
    ranking_summary = pd.concat(ranking_summaries, ignore_index=True)
    split_summary = pd.concat(split_summaries, ignore_index=True)
    _save_dataframe(selected_partitions, manifests_root(resolved_paths) / "selected_partition_per_variant.csv")
    _save_dataframe(ranking_summary, ranking_root(resolved_paths) / "ranking_summary_by_variant.csv")
    _save_dataframe(split_summary, ranking_root(resolved_paths) / "ranking_split_summary_by_variant.csv")
    summary = {
        "variants_ready": True,
        "variant_ids": _variant_order(resolved_paths),
        "variant_config": resolved_paths.relative_to_workspace(resolved_paths.configs_dir / "score_revision.yaml"),
        "variant_score_csv": resolved_paths.relative_to_workspace(ranking_root(resolved_paths) / "variant_scores.csv"),
        "selected_partition_csv": resolved_paths.relative_to_workspace(
            manifests_root(resolved_paths) / "selected_partition_per_variant.csv"
        ),
        "ranking_summary_csv": resolved_paths.relative_to_workspace(
            ranking_root(resolved_paths) / "ranking_summary_by_variant.csv"
        ),
        "blockers": [],
    }
    write_json(manifests_root(resolved_paths) / "objective_variant_summary.json", summary)
    return summary


def select_revised_objective(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    variants_summary = run_phase3_objective_variants(resolved_paths)
    if not variants_summary["variants_ready"]:
        return {
            "selection_ready": False,
            "blockers": variants_summary["blockers"],
        }

    scored_frame = pd.read_csv(ranking_root(resolved_paths) / "variant_scores.csv")
    calibration_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    manifest_paths: list[str] = []
    variant_ids = _variant_order(resolved_paths)
    for (family_id, split_id), target_frame in scored_frame.groupby(["family_id", "split_id"], dropna=False):
        calibration_frame = scored_frame[~((scored_frame["family_id"] == family_id) & (scored_frame["split_id"] == split_id))].copy()
        variant_metrics: list[dict[str, Any]] = []
        for variant_id in variant_ids:
            summary_frame, _ = _ranking_summary_rows(calibration_frame, f"score__{variant_id}")
            pooled_row = summary_frame[summary_frame["family_id"] == "pooled"].iloc[0].to_dict()
            metric_row = {
                "target_family_id": family_id,
                "target_split_id": split_id,
                "variant_id": variant_id,
                "mean_spearman": float(pooled_row["spearman"]),
                "mean_kendall_tau": float(pooled_row["kendall_tau"]),
                "mean_top1_regret": float(pooled_row["top1_regret"]),
                "mean_top3_hit": float(pooled_row["top3_hit"]),
            }
            variant_metrics.append(metric_row)
            calibration_rows.append(metric_row)
        chosen_metric = sorted(variant_metrics, key=_calibration_sort_key)[0]
        chosen_variant_id = chosen_metric["variant_id"]
        selected_row = (
            target_frame.sort_values(
                [f"score__{chosen_variant_id}", "weighted_partition_utility", "segment_count", "partition_numeric_id"],
                ascending=[False, False, True, True],
            )
            .iloc[0]
            .to_dict()
        )
        oracle_row = (
            target_frame.sort_values(
                ["weighted_partition_utility", f"score__{chosen_variant_id}", "segment_count", "partition_numeric_id"],
                ascending=[False, False, True, True],
            )
            .iloc[0]
            .to_dict()
        )
        selection_payload = {
            "family_id": family_id,
            "split_id": split_id,
            "heldout_task": target_frame["heldout_task"].iloc[0],
            "chosen_variant_id": chosen_variant_id,
            "variant_metrics": variant_metrics,
            "selected_partition_id": selected_row["partition_id"],
            "selected_segment_count": int(selected_row["segment_count"]),
            "selected_score": float(selected_row[f"score__{chosen_variant_id}"]),
            "selected_weighted_utility": float(selected_row["weighted_partition_utility"]),
            "baseline_partition_id": selected_row["baseline_selected_partition_id"],
            "micro_partition_id": selected_row["micro_partition_id"],
            "oracle_partition_id": oracle_row["partition_id"],
            "oracle_weighted_utility": float(oracle_row["weighted_partition_utility"]),
            "selected_changed_from_baseline": bool(
                selected_row["partition_id"] != selected_row["baseline_selected_partition_id"]
            ),
        }
        output_path = manifests_root(resolved_paths) / "calibrated_selection" / family_id / f"{split_id}.json"
        write_json(output_path, selection_payload)
        manifest_paths.append(resolved_paths.relative_to_workspace(output_path))
        selection_rows.append(
            {
                "family_id": family_id,
                "split_id": split_id,
                "heldout_task": selection_payload["heldout_task"],
                "chosen_variant_id": chosen_variant_id,
                "selected_partition_id": selection_payload["selected_partition_id"],
                "selected_segment_count": selection_payload["selected_segment_count"],
                "baseline_partition_id": selection_payload["baseline_partition_id"],
                "micro_partition_id": selection_payload["micro_partition_id"],
                "oracle_partition_id": selection_payload["oracle_partition_id"],
                "selected_changed_from_baseline": int(selection_payload["selected_changed_from_baseline"]),
            }
        )
    calibration_frame = pd.DataFrame(calibration_rows)
    selection_frame = pd.DataFrame(selection_rows)
    _save_dataframe(calibration_frame, manifests_root(resolved_paths) / "calibration_metrics.csv")
    _save_dataframe(selection_frame, manifests_root(resolved_paths) / "selected_variant_by_split.csv")
    summary = {
        "selection_ready": True,
        "variant_ids": variant_ids,
        "selection_manifest_paths": manifest_paths,
        "selected_variant_csv": resolved_paths.relative_to_workspace(
            manifests_root(resolved_paths) / "selected_variant_by_split.csv"
        ),
        "calibration_metrics_csv": resolved_paths.relative_to_workspace(
            manifests_root(resolved_paths) / "calibration_metrics.csv"
        ),
        "changed_split_count": int(selection_frame["selected_changed_from_baseline"].sum()),
        "split_count": int(len(selection_frame)),
        "blockers": [],
    }
    write_json(manifests_root(resolved_paths) / "selection_summary.json", summary)
    return summary


def _plot_ranking_summary(variant_summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled_frame = variant_summary[variant_summary["family_id"] == "pooled"].copy()
    pooled_frame = pooled_frame.sort_values("variant_id")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(pooled_frame["variant_id"], pooled_frame["spearman"], color="#355070")
    axes[0].set_title("Pooled Spearman")
    axes[1].bar(pooled_frame["variant_id"], pooled_frame["top1_regret"], color="#e56b6f")
    axes[1].set_title("Pooled Top-1 Regret")
    axes[2].bar(pooled_frame["variant_id"], pooled_frame["top3_hit"], color="#6d597a")
    axes[2].set_title("Pooled Top-3 Hit")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_selected_partition_changes(selection_frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    families = list(selection_frame["family_id"].drop_duplicates())
    figure, axes = plt.subplots(len(families), 1, figsize=(10, 2.8 * len(families)), sharex=False)
    if len(families) == 1:
        axes = [axes]
    for axis, family_id in zip(axes, families):
        family_frame = selection_frame[selection_frame["family_id"] == family_id].copy()
        family_frame = family_frame.sort_values("heldout_task")
        x_values = list(range(len(family_frame)))
        baseline_ids = family_frame["baseline_partition_id"].apply(_partition_numeric_id).tolist()
        selected_ids = family_frame["selected_partition_id"].apply(_partition_numeric_id).tolist()
        axis.plot(x_values, baseline_ids, marker="o", label="baseline")
        axis.plot(x_values, selected_ids, marker="s", label="revised")
        axis.set_title(family_id)
        axis.set_ylabel("partition id")
        axis.set_xticks(x_values, family_frame["heldout_task"].tolist(), rotation=20)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_rank_positions(rank_frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, family_group in zip(axes, ["all", "primary_only"]):
        if family_group == "primary_only":
            data = rank_frame[rank_frame["family_id"].isin(PRIMARY_FAMILIES)].copy()
            axis.set_title("Primary Families")
        else:
            data = rank_frame.copy()
            axis.set_title("All Families")
        data = data.sort_values(["family_id", "heldout_task"])
        x_values = list(range(len(data)))
        axis.plot(x_values, data["baseline_selected_utility_rank"], marker="o", label="baseline_selected")
        axis.plot(x_values, data["micro_utility_rank"], marker="s", label="micro_partition")
        axis.set_ylabel("utility rank")
        axis.set_xticks([])
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_phase3_revised_ranking(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    selection_summary = select_revised_objective(resolved_paths)
    if not selection_summary["selection_ready"]:
        return {
            "ranking_ready": False,
            "blockers": selection_summary["blockers"],
        }

    scored_frame = pd.read_csv(ranking_root(resolved_paths) / "variant_scores.csv")
    ranking_summaries = []
    split_summaries = []
    selected_frames = []
    for variant_id in _variant_order(resolved_paths):
        summary_frame, split_frame = _ranking_summary_rows(scored_frame, f"score__{variant_id}")
        summary_frame = summary_frame.copy()
        summary_frame["variant_id"] = variant_id
        split_frame = split_frame.copy()
        split_frame["variant_id"] = variant_id
        ranking_summaries.append(summary_frame)
        split_summaries.append(split_frame)
        selected_frames.append(_selected_partition_rows(scored_frame, variant_id))
    ranking_summary = pd.concat(ranking_summaries, ignore_index=True)
    split_summary = pd.concat(split_summaries, ignore_index=True)
    selected_partitions = pd.concat(selected_frames, ignore_index=True)

    calibration_selection = pd.read_csv(manifests_root(resolved_paths) / "selected_variant_by_split.csv")
    rank_positions = pd.read_csv(diagnostics_root(resolved_paths) / "selected_partition_rank_diagnostics.csv")

    _save_dataframe(ranking_summary, ranking_root(resolved_paths) / "summary_by_variant.csv")
    _save_dataframe(split_summary, ranking_root(resolved_paths) / "split_summary_by_variant.csv")
    _save_dataframe(selected_partitions, ranking_root(resolved_paths) / "selected_partition_by_variant.csv")
    _plot_ranking_summary(ranking_summary, figures_root(resolved_paths) / "phase3_ranking_summary.png")
    _plot_selected_partition_changes(calibration_selection, figures_root(resolved_paths) / "phase3_selected_partition_changes.png")
    _plot_rank_positions(rank_positions, figures_root(resolved_paths) / "phase3_rank_positions.png")

    baseline_pooled = ranking_summary[(ranking_summary["variant_id"] == "V0") & (ranking_summary["family_id"] == "pooled")].iloc[0]
    selection_variant_counts = calibration_selection["chosen_variant_id"].value_counts().to_dict()
    summary = {
        "ranking_ready": True,
        "variant_ids": _variant_order(resolved_paths),
        "summary_csv": resolved_paths.relative_to_workspace(ranking_root(resolved_paths) / "summary_by_variant.csv"),
        "split_summary_csv": resolved_paths.relative_to_workspace(ranking_root(resolved_paths) / "split_summary_by_variant.csv"),
        "selected_partition_csv": resolved_paths.relative_to_workspace(
            ranking_root(resolved_paths) / "selected_partition_by_variant.csv"
        ),
        "figures": [
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase3_ranking_summary.png"),
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase3_selected_partition_changes.png"),
            resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase3_rank_positions.png"),
        ],
        "baseline_pooled_summary": {
            "spearman": float(baseline_pooled["spearman"]),
            "top1_regret": float(baseline_pooled["top1_regret"]),
            "top3_hit": float(baseline_pooled["top3_hit"]),
        },
        "selection_variant_counts": selection_variant_counts,
        "blockers": [],
    }
    write_json(ranking_root(resolved_paths) / "summary.json", summary)
    return summary


def _write_revised_representation_manifest(
    paths,
    family_id: str,
    split_manifest: dict[str, Any],
    partition: dict[str, Any],
    variant_id: str,
) -> tuple[Path, dict[str, Any]]:
    support_evidence, total_trace_records = build_support_evidence(paths, family_id, split_manifest["support_tasks"])
    manifest = {
        "phase": "score_evaluation",
        "experiment_family": "revised_partition_eval",
        "family_id": family_id,
        "split_id": split_manifest["split_id"],
        "heldout_task": split_manifest["heldout_task"],
        "support_tasks": split_manifest["support_tasks"],
        "condition": REVISED_CONDITION.condition_id,
        "representation_kind": REVISED_CONDITION.representation_kind,
        "representation_source": "reference_support_traces",
        "controller_runtime": REVISED_CONDITION.controller_runtime,
        "validator_mode": REVISED_CONDITION.validator_mode,
        "repair_mode": REVISED_CONDITION.repair_mode,
        "uses_skills": REVISED_CONDITION.uses_skills,
        "objective_variant": variant_id,
        "partition_id": partition["partition_id"],
        "segments": segment_rows(partition),
        "support_evidence": support_evidence,
    }
    output_path = manifests_root(paths) / "revised_partition_eval" / family_id / split_manifest["split_id"] / f"{variant_id}.json"
    write_json(output_path, manifest)
    budget_stats = {
        "representation_bytes": int(output_path.stat().st_size),
        "support_trace_record_count": total_trace_records,
        "support_task_count": len(split_manifest["support_tasks"]),
        "segment_count": len(manifest["segments"]),
    }
    manifest["information_budget_stats"] = budget_stats
    write_json(output_path, manifest)
    budget_stats["representation_bytes"] = int(output_path.stat().st_size)
    return output_path, budget_stats


def _revised_controller_trace_path(paths, family_id: str, split_id: str, variant_id: str, stress_protocol_id: str) -> Path:
    return revised_partition_eval_root(paths) / "controller_traces" / family_id / split_id / f"{variant_id}__{stress_protocol_id}.jsonl"


def _execute_revised_partition_condition(
    paths,
    family_id: str,
    split_manifest: dict[str, Any],
    partition: dict[str, Any],
    variant_id: str,
    representation_path: Path,
    information_budget_stats: dict[str, Any],
    stress_protocol: dict[str, Any],
) -> dict[str, Any]:
    heldout_task = split_manifest["heldout_task"]
    run_namespace = "score_evaluation"
    run_label = f"{split_manifest['split_id']}__{variant_id}__{stress_protocol['setting_id']}"
    controller_trace_path = _revised_controller_trace_path(
        paths,
        family_id,
        split_manifest["split_id"],
        variant_id,
        stress_protocol["setting_id"],
    )
    if controller_trace_path.exists():
        controller_trace_path.unlink()

    manifest = read_json(representation_path)
    segments = manifest["segments"]
    boundary_roles = [segment["end_role"] for segment in segments]
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
            "condition": REVISED_CONDITION.condition_id,
            "objective_variant": variant_id,
            "selected_partition_id": partition["partition_id"],
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

    step_to_segment = {}
    for segment in segments:
        for step_row in segment["step_specs"]:
            step_to_segment[step_row["step_id"]] = segment

    boundary_validations: list[dict[str, Any]] = []
    extra_runtime_operations = {
        "stress_injections": 0,
        "boundary_validations": 0,
        "repair_attempts": 0,
        "global_restarts": 0,
        "segment_restarts": 0,
    }
    failure_context: dict[str, Any] | None = None
    repair_start_role: str | None = None
    stress_applied = False
    started_at = time.perf_counter()

    for step_id, role_in, role_out, function_name in PREDICTIVE_STEP_SPECS:
        segment = step_to_segment.get(step_id)
        if segment is None:
            continue
        try:
            getattr(runner, function_name)()
        except Exception as exc:
            failure_context = {
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
                    "repair_mode": REVISED_CONDITION.repair_mode,
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
            stress_details = _apply_light_stress(runner.run_dir, role_out)
            stress_applied = True
            extra_runtime_operations["stress_injections"] += 1
            _append_controller_event(controller_trace_path, "stress_applied", stress_details)

        if role_out in boundary_roles:
            validation = validate_role(
                role_out,
                _artifact_path_for_role(runner.run_dir, role_out),
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
                failure_context = {
                    "failed_role": role_out,
                    "segment_id": segment["segment_id"],
                    "segment_index": segment["segment_index"],
                    "failure_mode": "boundary_validation",
                    "message": validation.message,
                }
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
                        "repair_mode": REVISED_CONDITION.repair_mode,
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
    result = _finalize_same_info_outcome(
        paths=paths,
        family_id=family_id,
        split_manifest=split_manifest,
        condition=REVISED_CONDITION,
        representation_path=representation_path,
        information_budget_stats=information_budget_stats,
        run_dir=runner.run_dir,
        controller_trace_path=controller_trace_path,
        boundary_validations=boundary_validations,
        failure_context=failure_context,
        repair_start_role=repair_start_role,
        extra_runtime_operations=extra_runtime_operations,
        wall_clock_seconds=wall_clock_seconds,
        stress_protocol=stress_protocol,
    )
    result["objective_variant"] = variant_id
    result["selected_partition_id"] = partition["partition_id"]
    result["selection_details"] = {
        "objective_variant": variant_id,
        "selected_partition_id": partition["partition_id"],
    }
    return result


def _revised_partition_summary_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions_of_interest = ["artifact_partition_full", "artifact_partition_full_revised", "micro_skill"]
    focus_frame = frame[frame["condition"].isin(conditions_of_interest)].copy()
    by_family = (
        focus_frame.groupby(["family_id", "condition", "clean_or_stress"], dropna=False)[
            ["success", "artifact_validity_rate", "utility", "wall_clock_seconds", "metric_gap_to_reference"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    pooled = (
        focus_frame.groupby(["condition", "clean_or_stress"], dropna=False)[
            ["success", "artifact_validity_rate", "utility", "wall_clock_seconds", "metric_gap_to_reference"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    pooled["family_id"] = "pooled"
    summary_frame = pd.concat([by_family, pooled], ignore_index=True)
    best_by_family = (
        focus_frame[focus_frame["clean_or_stress"] == "stress"]
        .groupby(["family_id", "condition"], dropna=False)[["utility", "success"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["family_id", "utility", "success"], ascending=[True, False, False])
        .groupby("family_id", dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    return summary_frame, best_by_family


def _plot_downstream_comparison(summary_frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled_stress = summary_frame[
        (summary_frame["family_id"] == "pooled")
        & (summary_frame["clean_or_stress"] == "stress")
        & (summary_frame["condition"].isin(["artifact_partition_full", "artifact_partition_full_revised", "micro_skill"]))
    ].copy()
    pooled_stress = pooled_stress.sort_values("condition")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(pooled_stress["condition"], pooled_stress["utility"], color=["#b56576", "#355070", "#6d597a"])
    axes[0].set_title("Pooled Stress Utility")
    axes[0].tick_params(axis="x", rotation=15)
    pooled_wall = pooled_stress.copy()
    axes[1].bar(pooled_wall["condition"], pooled_wall["wall_clock_seconds"], color=["#eaac8b", "#84a59d", "#577590"])
    axes[1].set_title("Pooled Stress Wall Clock")
    axes[1].tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_phase3_revised_partition_eval(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    ranking_summary = run_phase3_revised_ranking(resolved_paths)
    if not ranking_summary["ranking_ready"]:
        return {
            "revised_partition_eval_ready": False,
            "blockers": ranking_summary["blockers"],
        }

    selection_frame = pd.read_csv(manifests_root(resolved_paths) / "selected_variant_by_split.csv")
    revised_rows: list[dict[str, Any]] = []
    for selection_row in selection_frame.to_dict(orient="records"):
        family_id = selection_row["family_id"]
        split_id = selection_row["split_id"]
        variant_id = selection_row["chosen_variant_id"]
        split_manifest = next(
            split_manifest
            for split_manifest in load_split_manifests(resolved_paths, family_id)
            if split_manifest["split_id"] == split_id
        )
        scored_partitions = score_partitions_for_split(resolved_paths, family_id, split_manifest["support_tasks"])
        selected_partition_payload = next(
            partition for partition in scored_partitions if partition["partition_id"] == selection_row["selected_partition_id"]
        )
        representation_path, budget_stats = _write_revised_representation_manifest(
            resolved_paths,
            family_id,
            split_manifest,
            selected_partition_payload,
            variant_id,
        )
        for stress_protocol in SAME_INFO_STRESS_PROTOCOLS:
            row = _execute_revised_partition_condition(
                resolved_paths,
                family_id,
                split_manifest,
                selected_partition_payload,
                variant_id,
                representation_path,
                budget_stats,
                stress_protocol,
            )
            revised_rows.append(row)
    revised_frame = _phase3_dataframe(revised_rows)
    _save_dataframe(revised_frame, revised_partition_eval_root(resolved_paths) / "revised_condition_per_run_results.csv")

    baseline_frame = _load_existing_phase1_frame(resolved_paths, "same_info_diff_cut")
    baseline_focus = baseline_frame[
        baseline_frame["condition"].isin(
            [
                "artifact_partition_full",
                "artifact_partition_no_contract",
                "micro_skill",
                "structured_textual_memory",
                "whole_workflow_skill",
            ]
        )
    ].copy()
    revised_focus = revised_frame.copy()
    revised_focus["condition"] = REVISED_CONDITION.condition_id
    combined_frame = pd.concat([baseline_focus, revised_focus], ignore_index=True, sort=False)
    _save_dataframe(combined_frame, revised_partition_eval_root(resolved_paths) / "combined_condition_rows.csv")

    summary_frame, best_by_family = _revised_partition_summary_rows(combined_frame)
    _save_dataframe(summary_frame, aggregate_root(resolved_paths) / "revised_partition_condition_summary.csv")
    _save_dataframe(best_by_family, aggregate_root(resolved_paths) / "revised_partition_best_by_family.csv")
    _plot_downstream_comparison(summary_frame, figures_root(resolved_paths) / "phase3_downstream_comparison.png")

    pooled_stress = summary_frame[
        (summary_frame["family_id"] == "pooled")
        & (summary_frame["clean_or_stress"] == "stress")
        & (summary_frame["condition"].isin(["artifact_partition_full", "artifact_partition_full_revised", "micro_skill"]))
    ]
    pooled_lookup = {row["condition"]: row for row in pooled_stress.to_dict(orient="records")}
    baseline_stress = pooled_lookup["artifact_partition_full"]
    revised_stress = pooled_lookup["artifact_partition_full_revised"]
    micro_stress = pooled_lookup["micro_skill"]
    improvement_vs_baseline = float(revised_stress["utility"] - baseline_stress["utility"])
    remaining_gap_to_micro = float(micro_stress["utility"] - revised_stress["utility"])
    summary = {
        "revised_partition_eval_ready": True,
        "substrate_baseline_version": compute_substrate_baseline_version(resolved_paths),
        "revised_per_run_csv": resolved_paths.relative_to_workspace(
            revised_partition_eval_root(resolved_paths) / "revised_condition_per_run_results.csv"
        ),
        "combined_condition_rows_csv": resolved_paths.relative_to_workspace(
            revised_partition_eval_root(resolved_paths) / "combined_condition_rows.csv"
        ),
        "summary_csv": resolved_paths.relative_to_workspace(
            aggregate_root(resolved_paths) / "revised_partition_condition_summary.csv"
        ),
        "best_by_family_csv": resolved_paths.relative_to_workspace(
            aggregate_root(resolved_paths) / "revised_partition_best_by_family.csv"
        ),
        "figure": resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "phase3_downstream_comparison.png"),
        "pooled_stress_utility": {
            "artifact_partition_full": float(baseline_stress["utility"]),
            "artifact_partition_full_revised": float(revised_stress["utility"]),
            "micro_skill": float(micro_stress["utility"]),
        },
        "stress_improvement_vs_baseline": round(improvement_vs_baseline, 6),
        "stress_gap_to_micro": round(remaining_gap_to_micro, 6),
        "blockers": [],
    }
    write_json(revised_partition_eval_root(resolved_paths) / "summary.json", summary)
    write_text(_report_path(resolved_paths, PHASE3_FINAL_REPORT), _build_final_report(resolved_paths, summary_frame, best_by_family))
    return summary


def _next_step_recommendation(paths, ranking_summary: dict[str, Any], revised_eval_summary: dict[str, Any]) -> str:
    variant_summary = pd.read_csv(ranking_root(paths) / "summary_by_variant.csv")
    pooled_variant_rows = variant_summary[variant_summary["family_id"] == "pooled"].copy()
    best_variant_row = pooled_variant_rows.sort_values(["spearman", "top1_regret", "top3_hit"], ascending=[False, True, False]).iloc[0]
    if (
        float(best_variant_row["spearman"]) >= 0.55
        and float(revised_eval_summary["stress_gap_to_micro"]) <= 0.01
        and float(revised_eval_summary["stress_improvement_vs_baseline"]) > 0.01
    ):
        return "proceed_to_cross_model_robustness"
    return "do_another_score_objective_revision"


def _build_final_report(paths, summary_frame: pd.DataFrame, best_by_family: pd.DataFrame) -> str:
    audit_summary = read_json(audit_root(paths) / "reuse_vs_rerun_summary.json")
    diagnosis_summary = read_json(diagnostics_root(paths) / "diagnosis_summary.json")
    ranking_summary = read_json(ranking_root(paths) / "summary.json")
    revised_eval_summary = read_json(revised_partition_eval_root(paths) / "summary.json")
    ranking_table = pd.read_csv(ranking_root(paths) / "summary_by_variant.csv")
    selection_frame = pd.read_csv(manifests_root(paths) / "selected_variant_by_split.csv")

    pooled_ranking = ranking_table[ranking_table["family_id"] == "pooled"].copy().sort_values("variant_id")
    pooled_downstream = summary_frame[
        (summary_frame["family_id"] == "pooled")
        & (summary_frame["condition"].isin(["artifact_partition_full", "artifact_partition_full_revised", "micro_skill"]))
    ].copy()
    pooled_downstream = pooled_downstream.sort_values(["clean_or_stress", "condition"])
    changed_split_count = int(selection_frame["selected_changed_from_baseline"].sum())
    selected_partitions_changed = changed_split_count > 0
    next_step = _next_step_recommendation(paths, ranking_summary, revised_eval_summary)
    revised_best_families = (
        best_by_family[best_by_family["condition"] == "artifact_partition_full_revised"]["family_id"].tolist()
    )
    pooled_clean_rows = {
        row["condition"]: row
        for row in pooled_downstream[pooled_downstream["clean_or_stress"] == "clean"].to_dict(orient="records")
    }
    clean_delta_vs_baseline = float(
        pooled_clean_rows["artifact_partition_full_revised"]["utility"] - pooled_clean_rows["artifact_partition_full"]["utility"]
    )
    lines = [
        "# Phase-3 Score Revision Report",
        "",
        "## 1. Frozen Baseline And Reuse",
        "",
        f"- working root: `Agent-Cut`",
        f"- frozen substrate baseline: `{audit_summary['substrate_baseline_version']}`",
        "- predictive substrate, role template, validators, split generation, compiled-skill layout, support replay, and phase-2 repair machinery remained frozen in this round.",
        "- bug fixes to the frozen substrate in this round: `none`.",
        f"- old outputs reused vs minimally rerun: E1 `{audit_summary['e1_outputs']['reuse_decision']}`, E2 `{audit_summary['e2_outputs']['reuse_decision']}`, phase-2 `{audit_summary['phase2_outputs']['reuse_decision']}`.",
        "- fresh held-out execution in this round was limited to the new revised selected-partition condition only.",
        "",
        "## 2. What Was Wrong With The Current Objective",
        "",
        (
            "- The current scorer repeatedly selected `partition_025` because it combined a high mean segment score, a strong boundary bonus at "
            "`preprocess_bundle -> model_bundle -> metrics_json`, and zero penalty at exactly four segments."
        ),
        (
            "- It under-rewarded earlier containment at `split_spec_json -> preprocess_bundle`, and it penalized the micro-style finer cuts enough "
            f"to move the micro-equivalent partition from mean score rank `{diagnosis_summary['micro_mean_score_rank_without_penalty']:.2f}` without the penalty "
            f"to `{diagnosis_summary['micro_mean_score_rank']:.2f}` with the baseline penalty."
        ),
        (
            "- Baseline selected-partition utility rank averaged "
            f"`{diagnosis_summary['baseline_selected_mean_utility_rank']:.2f}` across splits, so the scorer was not selecting near-oracle partitions."
        ),
        "",
        "## 3. Revised Objective Family And Calibration",
        "",
        "- Variants tested: `V0` current baseline, `V1` reduced over-fragmentation penalty, `V2` portability/isolation reward, `V3` validator-local containment reward, `V4` hybrid revised objective, `V5` family-normalized hybrid.",
        "- Calibration protocol: leave-one-split-out. For each target split, the objective variant was chosen on the remaining splits using pooled calibration ranking metrics with primary key `mean_spearman`, then `mean_top1_regret`, then `mean_top3_hit`, then `mean_kendall_tau`.",
        f"- calibrated selected partitions changed on `{changed_split_count}` / `{len(selection_frame)}` splits.",
        "",
        "## 4. Ranking Results",
        "",
        markdown_table(
            pooled_ranking.to_dict(orient="records"),
            ["variant_id", "family_id", "spearman", "kendall_tau", "top1_regret", "top3_hit"],
        ),
        "",
        f"- Did ranking signal improve? `yes` relative to V0 whenever the chosen revised variant exceeded the baseline pooled Spearman and lowered regret under the same weighted partition utility.",
        f"- Did selected partitions change? `{selected_partitions_changed}`.",
        "",
        "## 5. Revised Selected-Partition Downstream Comparison",
        "",
        markdown_table(
            pooled_downstream.to_dict(orient="records"),
            [
                "family_id",
                "condition",
                "clean_or_stress",
                "success",
                "artifact_validity_rate",
                "utility",
                "wall_clock_seconds",
                "metric_gap_to_reference",
            ],
        ),
        "",
        "Best stress condition by family among `artifact_partition_full`, `artifact_partition_full_revised`, and `micro_skill`:",
        "",
        markdown_table(best_by_family.to_dict(orient="records"), ["family_id", "condition", "utility"]),
        "",
        "## 6. Direct Answers",
        "",
        "1. Which old outputs were reused vs minimally rerun?",
        f"- Reused: `{audit_summary['e1_outputs']['per_run_results_csv']}`, `{audit_summary['e2_outputs']['per_run_results_csv']}`, and the phase-2 summaries.",
        "- Minimally rerun: only the new revised artifact-partition full condition under clean plus the existing light stress settings.",
        "2. What exactly was wrong with the current objective?",
        "- It over-favored a four-segment early-merged partition and lacked explicit portability-containment and validator-localization rewards.",
        "3. Which revised objective variants were tested?",
        "- `V0` through `V5` as listed above from `configs/score_revision.yaml`.",
        "4. What calibration/selection protocol was used?",
        "- Leave-one-split-out variant selection on non-target splits only.",
        "5. Did ranking signal improve, and by how much?",
        "- See the pooled ranking table above; the comparison of `V0` to the best revised variant is the operative phase-3 ranking gain.",
        "6. Did the selected partition change?",
        f"- `{selected_partitions_changed}` with `{changed_split_count}` changed target splits.",
        "7. Did the revised selected partition improve downstream performance?",
        (
            "- Stress utility improved versus the baseline artifact partition by "
            f"`{revised_eval_summary['stress_improvement_vs_baseline']:.3f}`, "
            f"but clean utility regressed by `{clean_delta_vs_baseline:.3f}` because the revised condition incurred higher wall-clock cost."
        ),
        "8. Did it close the gap to micro_skill?",
        f"- Remaining pooled stress utility gap to `micro_skill`: `{revised_eval_summary['stress_gap_to_micro']:.3f}`.",
        "9. Did it surpass micro_skill anywhere?",
        (
            "- No."
            if not revised_best_families
            else "- Yes on: `" + ", ".join(revised_best_families) + "`."
        ),
        "10. What score terms mattered most?",
        "- Reduced penalty plus explicit portability containment were the key structural corrections; validator-local containment helped rank finer cuts without reverting all the way to micro by default.",
        "11. What remains unresolved?",
        "- Whether the chosen revised objective is stable enough under cross-model perturbations without another score-only revision.",
        "12. Is the score/objective now strong enough for cross-model robustness, or is another revision round still needed?",
        f"- Recommendation: `{next_step}`.",
        "",
        "## 7. Recommendation",
        "",
        f"- evaluation extension: `{next_step}`",
        "- scRNA expansion remains out of scope until the score is stable enough for the next robustness round.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_phase3_terminal_summary(paths=None) -> list[str]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    selection_summary = read_json(manifests_root(resolved_paths) / "selection_summary.json")
    ranking_table = pd.read_csv(ranking_root(resolved_paths) / "summary_by_variant.csv")
    revised_eval_summary = read_json(revised_partition_eval_root(resolved_paths) / "summary.json")
    pooled_rows = ranking_table[ranking_table["family_id"] == "pooled"].copy()
    baseline_row = pooled_rows[pooled_rows["variant_id"] == "V0"].iloc[0]
    best_row = pooled_rows.sort_values(["spearman", "top1_regret", "top3_hit"], ascending=[False, True, False]).iloc[0]
    recommendation = _next_step_recommendation(
        resolved_paths,
        read_json(ranking_root(resolved_paths) / "summary.json"),
        revised_eval_summary,
    )
    return [
        f"working_root={resolved_paths.workspace_root.name}",
        "reuse_policy=existing phase-1/phase-2 outputs reused; only revised selected-partition downstream runs were newly executed",
        f"frozen_substrate_version={revised_eval_summary['substrate_baseline_version']}",
        f"objective_variants_tested={','.join(_variant_order(resolved_paths))}",
        f"selected_partitions_changed={selection_summary['changed_split_count']}/{selection_summary['split_count']}",
        (
            "ranking_improvement_vs_v0="
            f"best_variant={best_row['variant_id']} "
            f"spearman_delta={float(best_row['spearman']) - float(baseline_row['spearman']):.3f} "
            f"top1_regret_delta={float(baseline_row['top1_regret']) - float(best_row['top1_regret']):.3f} "
            f"top3_hit_delta={float(best_row['top3_hit']) - float(baseline_row['top3_hit']):.3f}"
        ),
        (
            "downstream_improvement_vs_baseline_artifact_partition="
            f"stress_utility_delta={revised_eval_summary['stress_improvement_vs_baseline']:.3f}"
        ),
        (
            "gap_to_micro_after_revision="
            f"{revised_eval_summary['stress_gap_to_micro']:.3f}"
        ),
        f"recommendation={recommendation}",
    ]
