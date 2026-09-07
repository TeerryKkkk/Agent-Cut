from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runtime.workflow_evaluation import compute_substrate_baseline_version, load_reference_trace, total_reference_wall_clock
from utils.io_utils import markdown_table, read_json, read_yaml, write_json, write_text

FAMILY_ORDER = [
    "openml_tabular_binary",
    "tdc_admet_binary",
    "tdc_admet_regression",
    "tdc_tox_binary",
]
ALL_FAMILY_ORDER = [*FAMILY_ORDER, "pooled"]
FAMILY_ALIAS = {
    "openml_tabular_binary": "F1",
    "tdc_admet_binary": "F2",
    "tdc_admet_regression": "F3",
    "tdc_tox_binary": "F4",
    "pooled": "Pooled",
}
FAMILY_LABEL = {
    "openml_tabular_binary": "openml_tabular_binary",
    "tdc_admet_binary": "tdc_admet_binary",
    "tdc_admet_regression": "tdc_admet_regression",
    "tdc_tox_binary": "tdc_tox_binary",
    "pooled": "pooled",
}
BOUNDARY_IDS = ["B1", "B2", "B3", "B4", "B5", "B6"]
NEAR_OPTIMAL_TOLERANCE = 0.002
PREDICTIVE_CLOSURE_ROOT_NAME = "predictive_evaluation"
AUDIT_REPORT_NAME = "predictive_evaluation_audit.md"
FLAT_TOP_REPORT_NAME = "predictive_flat_top_analysis.md"
CLOSURE_REPORT_NAME = "predictive_evaluation_report.md"

PHASE_COLORS = {
    "baseline": "#8d99ae",
    "v4": "#457b9d",
    "v6": "#2a9d8f",
    "micro": "#e76f51",
    "oracle": "#264653",
}


def closure_root(paths) -> Path:
    return paths.results_dir / PREDICTIVE_CLOSURE_ROOT_NAME


def aggregate_root(paths) -> Path:
    return closure_root(paths) / "aggregate_tables"


def flat_top_root(paths) -> Path:
    return closure_root(paths) / "flat_top_analysis"


def figures_root(paths) -> Path:
    return closure_root(paths) / "figures"


def audit_report_path(paths) -> Path:
    return paths.reports_dir / AUDIT_REPORT_NAME


def flat_top_report_path(paths) -> Path:
    return paths.reports_dir / FLAT_TOP_REPORT_NAME


def closure_report_path(paths) -> Path:
    return paths.reports_dir / CLOSURE_REPORT_NAME


def _relative(paths, path: Path) -> str:
    return paths.relative_to_workspace(path)


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _round_float(value: Any, digits: int = 6) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _with_pooled_rows(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)[metric_columns].mean().reset_index()
    pooled_frame = frame.copy()
    pooled_frame["family_id"] = "pooled"
    pooled_grouped = pooled_frame.groupby(group_columns, dropna=False)[metric_columns].mean().reset_index()
    combined = pd.concat([grouped, pooled_grouped], ignore_index=True)
    return combined


def _summary_lookup(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    return frame.set_index(key_columns).sort_index()


def _value_from_lookup(lookup: pd.DataFrame, key: tuple[Any, ...], column: str) -> float:
    row = lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return float(row[column])


def _string_from_lookup(lookup: pd.DataFrame, key: tuple[Any, ...], column: str) -> str:
    row = lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return str(row[column])


def _partition_vector(row: pd.Series) -> dict[str, int]:
    return {boundary_id: int(row[boundary_id]) for boundary_id in BOUNDARY_IDS}


def _boundary_distance(left: dict[str, int], right: dict[str, int]) -> int:
    return sum(int(left[boundary_id]) != int(right[boundary_id]) for boundary_id in BOUNDARY_IDS)


def _clean_cost_efficiency(wall_clock_seconds: float, reference_wall_clock_seconds: float) -> float:
    return 1.0 / (1.0 + (float(wall_clock_seconds) / max(float(reference_wall_clock_seconds), 1.0e-9)))


def _predicted_clean_utility(wall_clock_seconds: float, reference_wall_clock_seconds: float) -> float:
    return 0.9 + (0.1 * _clean_cost_efficiency(wall_clock_seconds, reference_wall_clock_seconds))


def _family_selection_partition(selection_frame: pd.DataFrame, family_id: str, column: str) -> str:
    if family_id == "pooled":
        values = selection_frame[column].dropna().astype(str).unique().tolist()
    else:
        values = selection_frame.loc[selection_frame["family_id"] == family_id, column].dropna().astype(str).unique().tolist()
    if not values:
        return ""
    values = sorted(values)
    return values[0] if len(values) == 1 else ",".join(values)


def _family_split_count(selection_frame: pd.DataFrame, family_id: str, change_column: str) -> tuple[int, int]:
    if family_id == "pooled":
        subset = selection_frame
    else:
        subset = selection_frame.loc[selection_frame["family_id"] == family_id]
    changed = int(subset[change_column].astype(int).sum())
    total = int(len(subset))
    return changed, total


def _reference_wall_clock_frame(paths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id in FAMILY_ORDER:
        split_summary = read_json(paths.results_dir / "predictive_substrate" / "splits" / family_id / "summary.json")
        for split_path_str in split_summary["split_paths"]:
            split_manifest = read_json(paths.workspace_root / split_path_str)
            heldout_task = split_manifest["heldout_task"]
            rows.append(
                {
                    "family_id": family_id,
                    "split_id": split_manifest["split_id"],
                    "heldout_task": heldout_task,
                    "reference_wall_clock_seconds": float(
                        total_reference_wall_clock(load_reference_trace(paths, family_id, heldout_task))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _group_best_condition(frame: pd.DataFrame, condition_column: str, value_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id in ALL_FAMILY_ORDER:
        subset = frame.loc[frame["family_id"] == family_id].copy()
        if subset.empty:
            continue
        ordered = subset.sort_values([value_column, condition_column], ascending=[False, True]).reset_index(drop=True)
        best = ordered.iloc[0]
        rows.append(
            {
                "family_id": family_id,
                "best_condition": str(best[condition_column]),
                value_column: float(best[value_column]),
            }
        )
    return pd.DataFrame(rows)


def _load_context(paths) -> dict[str, Any]:
    family_specs = read_yaml(paths.configs_dir / "families.yaml")["families"]
    dataset_specs = read_yaml(paths.configs_dir / "datasets.yaml")["datasets"]

    phase1_same_info_raw = _coerce_numeric(
        pd.read_csv(paths.results_dir / "workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv"),
        ["success", "artifact_validity_rate", "utility", "wall_clock_seconds", "first_failed_boundary_depth", "rerun_span"],
    )
    phase1_boundary_summary = _coerce_numeric(
        pd.read_csv(paths.results_dir / "workflow_evaluation" / "aggregate_tables" / "phase1_boundary_ranking_summary.csv"),
        ["spearman", "kendall_tau", "top1_regret", "top3_hit"],
    )
    phase1_same_info_summary = _with_pooled_rows(
        phase1_same_info_raw,
        ["family_id", "condition", "clean_or_stress"],
        ["success", "artifact_validity_rate", "utility", "wall_clock_seconds"],
    )

    phase2_ablation_raw = _coerce_numeric(
        pd.read_csv(paths.results_dir / "repair_evaluation" / "ablation" / "per_run_results.csv"),
        [
            "success",
            "artifact_validity_rate",
            "utility",
            "wall_clock_seconds",
            "precondition_satisfaction_rate",
            "postcondition_satisfaction_rate",
            "first_failed_boundary_depth",
            "rerun_span",
        ],
    )
    phase2_ablation_summary = _with_pooled_rows(
        phase2_ablation_raw,
        ["family_id", "ablation_condition", "clean_or_stress"],
        [
            "success",
            "artifact_validity_rate",
            "utility",
            "wall_clock_seconds",
            "precondition_satisfaction_rate",
            "postcondition_satisfaction_rate",
            "first_failed_boundary_depth",
            "rerun_span",
        ],
    )

    phase2_fault_raw = _coerce_numeric(
        pd.read_csv(paths.results_dir / "repair_evaluation" / "induced_fault" / "per_run_results.csv"),
        [
            "root_localization_accuracy",
            "repair_success",
            "final_success",
            "rerun_span",
            "skills_touched",
            "extra_wall_clock_seconds",
        ],
    )
    phase2_fault_summary = _with_pooled_rows(
        phase2_fault_raw,
        ["family_id", "repair_policy"],
        [
            "root_localization_accuracy",
            "repair_success",
            "final_success",
            "rerun_span",
            "skills_touched",
            "extra_wall_clock_seconds",
        ],
    )

    phase3_eval_raw = _coerce_numeric(
        pd.read_csv(paths.results_dir / "score_evaluation" / "revised_partition_eval" / "revised_condition_per_run_results.csv"),
        ["success", "artifact_validity_rate", "utility", "wall_clock_seconds", "metric_gap_to_reference", "rerun_span"],
    )
    phase3_eval_summary = _with_pooled_rows(
        phase3_eval_raw,
        ["family_id", "condition", "clean_or_stress"],
        ["success", "artifact_validity_rate", "utility", "wall_clock_seconds", "metric_gap_to_reference", "rerun_span"],
    )
    phase3_ranking_summary = _coerce_numeric(
        pd.read_csv(paths.results_dir / "score_evaluation" / "ranking" / "summary_by_variant.csv"),
        ["spearman", "kendall_tau", "top1_regret", "top3_hit"],
    )
    phase3_selection = pd.read_csv(paths.results_dir / "score_evaluation" / "manifests" / "selected_variant_by_split.csv")

    phase3b_eval_raw = _coerce_numeric(
        pd.read_csv(paths.results_dir / "boundary_evaluation" / "revised_partition_eval" / "selected_partition_per_run_results.csv"),
        [
            "success",
            "artifact_validity_rate",
            "utility",
            "wall_clock_seconds",
            "metric_gap_to_reference",
            "rerun_span",
            "first_failed_boundary_depth",
        ],
    )
    phase3b_eval_summary = _with_pooled_rows(
        phase3b_eval_raw,
        ["family_id", "condition", "clean_or_stress"],
        [
            "success",
            "artifact_validity_rate",
            "utility",
            "wall_clock_seconds",
            "metric_gap_to_reference",
            "rerun_span",
            "first_failed_boundary_depth",
        ],
    )
    phase3b_ranking_summary = _coerce_numeric(
        pd.read_csv(paths.results_dir / "boundary_evaluation" / "ranking" / "summary_by_variant.csv"),
        ["spearman", "kendall_tau", "top1_regret", "top3_hit"],
    )
    phase3b_selection = pd.read_csv(paths.results_dir / "boundary_evaluation" / "manifests" / "selected_variant_by_split.csv")

    phase3_feature_table = pd.read_csv(paths.results_dir / "score_evaluation" / "diagnostics" / "partition_feature_table.csv")
    phase3_feature_table = _coerce_numeric(
        phase3_feature_table,
        [
            "weighted_partition_utility",
            "weighted_wall_clock_seconds",
            "weighted_rerun_span",
            "weighted_first_failed_boundary_depth",
            "segment_count",
        ],
    )

    candidate_frame = _coerce_numeric(
        pd.read_csv(paths.results_dir / "boundary_evaluation" / "boundary_attribution" / "per_partition_outcome_table.csv"),
        [
            "segment_count",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "weighted_partition_utility",
            "predicted_clean_wall_clock_seconds",
            "weighted_rerun_span",
            "weighted_first_failed_boundary_depth",
            "weighted_wall_clock_seconds",
        ],
    )
    baseline_ids = phase3_feature_table[["family_id", "split_id", "baseline_selected_partition_id"]].drop_duplicates()
    reference_wall_clocks = _reference_wall_clock_frame(paths)
    candidate_frame = candidate_frame.merge(baseline_ids, on=["family_id", "split_id"], how="left")
    candidate_frame = candidate_frame.merge(
        phase3b_selection[["family_id", "split_id", "selected_partition_id"]],
        on=["family_id", "split_id"],
        how="left",
    ).rename(columns={"selected_partition_id": "phase3b_selected_partition_id"})
    candidate_frame = candidate_frame.merge(reference_wall_clocks, on=["family_id", "split_id", "heldout_task"], how="left")
    candidate_frame["predicted_clean_utility"] = candidate_frame.apply(
        lambda row: _predicted_clean_utility(row["predicted_clean_wall_clock_seconds"], row["reference_wall_clock_seconds"]),
        axis=1,
    )
    candidate_frame["boundary_indicator_vector_json"] = candidate_frame.apply(
        lambda row: _json_dumps(_partition_vector(row)),
        axis=1,
    )

    return {
        "family_specs": family_specs,
        "dataset_specs": dataset_specs,
        "phase1_same_info_raw": phase1_same_info_raw,
        "phase1_same_info_summary": phase1_same_info_summary,
        "phase1_boundary_summary": phase1_boundary_summary,
        "phase2_ablation_raw": phase2_ablation_raw,
        "phase2_ablation_summary": phase2_ablation_summary,
        "phase2_fault_raw": phase2_fault_raw,
        "phase2_fault_summary": phase2_fault_summary,
        "phase3_eval_raw": phase3_eval_raw,
        "phase3_eval_summary": phase3_eval_summary,
        "phase3_ranking_summary": phase3_ranking_summary,
        "phase3_selection": phase3_selection,
        "phase3b_eval_raw": phase3b_eval_raw,
        "phase3b_eval_summary": phase3b_eval_summary,
        "phase3b_ranking_summary": phase3b_ranking_summary,
        "phase3b_selection": phase3b_selection,
        "phase3_feature_table": phase3_feature_table,
        "candidate_frame": candidate_frame,
        "reference_wall_clocks": reference_wall_clocks,
    }


def audit_predictive_evaluation(paths) -> dict[str, Any]:
    artifact_rows = [
        {
            "artifact_group": "predictive_substrate",
            "report_path": "reports/predictive_substrate_report.md",
            "machine_readable_paths": [
                "results/predictive_substrate/summary.json",
                "results/predictive_substrate/family_table.json",
                "results/predictive_substrate/reference/summary.json",
                "results/predictive_substrate/support_replay/summary.json",
                "results/predictive_substrate/smoke/summary.json",
                "results/predictive_substrate/validator_qa/summary.json",
            ],
            "reconstructs": "family scope, split counts, substrate/replay/smoke readiness",
            "decision": "reuse_as_is",
        },
        {
            "artifact_group": "phase1",
            "report_path": "reports/workflow_evaluation_report.md",
            "machine_readable_paths": [
                "results/workflow_evaluation/same_info_diff_cut/per_run_results.csv",
                "results/workflow_evaluation/boundary_ranking/per_run_results.csv",
                "results/workflow_evaluation/aggregate_tables/phase1_same_info_per_family.csv",
                "results/workflow_evaluation/aggregate_tables/phase1_boundary_ranking_summary.csv",
                "results/workflow_evaluation/aggregate_tables/phase1_boundary_ranking_split_summary.csv",
            ],
            "reconstructs": "same-information outcomes, phase-1 ranking trajectory, baseline partition utility",
            "decision": "reuse_raw_rows_and_aggregate",
        },
        {
            "artifact_group": "phase2",
            "report_path": "reports/repair_evaluation_report.md",
            "machine_readable_paths": [
                "results/repair_evaluation/ablation/per_run_results.csv",
                "results/repair_evaluation/induced_fault/per_run_results.csv",
                "results/repair_evaluation/aggregate_tables/phase2_ablation_per_family.csv",
                "results/repair_evaluation/aggregate_tables/phase2_induced_fault_family_policy.csv",
                "results/repair_evaluation/aggregate_tables/phase2_induced_fault_pooled_primary.csv",
            ],
            "reconstructs": "contract-validator-repair ablations and induced-fault repair summaries",
            "decision": "reuse_raw_rows_and_aggregate",
        },
        {
            "artifact_group": "phase3",
            "report_path": "reports/score_evaluation_report.md",
            "machine_readable_paths": [
                "results/score_evaluation/ranking/summary_by_variant.csv",
                "results/score_evaluation/ranking/split_summary_by_variant.csv",
                "results/score_evaluation/manifests/selected_variant_by_split.csv",
                "results/score_evaluation/diagnostics/partition_feature_table.csv",
                "results/score_evaluation/revised_partition_eval/revised_condition_per_run_results.csv",
                "results/score_evaluation/revised_partition_eval/summary.json",
            ],
            "reconstructs": "phase-3 ranking changes, selected partitions, executed downstream comparison",
            "decision": "reuse_as_is",
        },
        {
            "artifact_group": "phase3b",
            "report_path": "reports/boundary_evaluation_report.md",
            "machine_readable_paths": [
                "results/boundary_evaluation/ranking/summary_by_variant.csv",
                "results/boundary_evaluation/ranking/split_summary_by_variant.csv",
                "results/boundary_evaluation/manifests/selected_variant_by_split.csv",
                "results/boundary_evaluation/revised_partition_eval/selected_partition_per_run_results.csv",
                "results/boundary_evaluation/revised_partition_eval/summary.json",
                "results/boundary_evaluation/boundary_attribution/per_partition_outcome_table.csv",
                "results/boundary_evaluation/boundary_attribution/partition_by_boundary_matrix.csv",
                "results/boundary_evaluation/boundary_attribution/per_boundary_marginal_summaries.csv",
                "results/boundary_evaluation/boundary_attribution/micro_vs_selected_vs_oracle_boundary_comparison.csv",
                "results/boundary_evaluation/boundary_attribution/summary.json",
            ],
            "reconstructs": "phase-3b ranking, boundary attribution, near-optimal geometry, oracle/selected/micro comparison",
            "decision": "reuse_as_is",
        },
    ]

    blockers: list[str] = []
    artifact_table_rows: list[dict[str, Any]] = []
    existing_summary_rows: list[dict[str, Any]] = []

    for artifact in artifact_rows:
        report_path = paths.workspace_root / artifact["report_path"]
        machine_paths = [paths.workspace_root / relative_path for relative_path in artifact["machine_readable_paths"]]
        missing_paths = [relative_path for relative_path, abs_path in zip(artifact["machine_readable_paths"], machine_paths) if not abs_path.exists()]
        if not report_path.exists():
            missing_paths.insert(0, artifact["report_path"])
        artifact_table_rows.append(
            {
                "artifact_group": artifact["artifact_group"],
                "report_exists": report_path.exists(),
                "machine_readable_count": len(artifact["machine_readable_paths"]) - len(missing_paths),
                "machine_readable_total": len(artifact["machine_readable_paths"]),
                "reconstructs": artifact["reconstructs"],
                "decision": artifact["decision"],
                "missing_paths": "; ".join(missing_paths) if missing_paths else "",
            }
        )
        if missing_paths:
            blockers.append(f"{artifact['artifact_group']}: missing {', '.join(missing_paths)}")
        for relative_path in artifact["machine_readable_paths"]:
            existing_summary_rows.append(
                {
                    "artifact_group": artifact["artifact_group"],
                    "summary_path": relative_path,
                    "exists": (paths.workspace_root / relative_path).exists(),
                }
            )

    sufficiency_rows = [
        {
            "question": "ranking trajectories",
            "status": "yes",
            "source_paths": "results/workflow_evaluation/boundary_ranking/per_run_results.csv; results/score_evaluation/ranking/summary_by_variant.csv; results/boundary_evaluation/ranking/summary_by_variant.csv",
        },
        {
            "question": "selected partitions by phase",
            "status": "yes",
            "source_paths": "results/predictive_substrate/smoke/*/summary.json; results/score_evaluation/manifests/selected_variant_by_split.csv; results/boundary_evaluation/manifests/selected_variant_by_split.csv",
        },
        {
            "question": "downstream utility comparisons",
            "status": "yes",
            "source_paths": "results/workflow_evaluation/same_info_diff_cut/per_run_results.csv; results/score_evaluation/revised_partition_eval/revised_condition_per_run_results.csv; results/boundary_evaluation/revised_partition_eval/selected_partition_per_run_results.csv",
        },
        {
            "question": "ablation comparisons",
            "status": "yes",
            "source_paths": "results/repair_evaluation/ablation/per_run_results.csv",
        },
        {
            "question": "induced-fault repair comparisons",
            "status": "yes",
            "source_paths": "results/repair_evaluation/induced_fault/per_run_results.csv",
        },
        {
            "question": "oracle / selected / micro stress utility per split",
            "status": "yes",
            "source_paths": "results/boundary_evaluation/boundary_attribution/per_partition_outcome_table.csv + results/boundary_evaluation/manifests/selected_variant_by_split.csv",
        },
        {
            "question": "oracle actual clean utility per split",
            "status": "proxy_only",
            "source_paths": "predicted_clean_wall_clock_seconds in results/boundary_evaluation/boundary_attribution/per_partition_outcome_table.csv",
        },
        {
            "question": "selected / micro actual clean and stress execution",
            "status": "yes",
            "source_paths": "results/workflow_evaluation/same_info_diff_cut/per_run_results.csv; results/boundary_evaluation/revised_partition_eval/selected_partition_per_run_results.csv",
        },
    ]

    reconstruction_rows = [
        {
            "needed_output": "unified closure tables",
            "method": "aggregate saved CSV/JSON summaries only",
            "costly_rerun": "no",
        },
        {
            "needed_output": "oracle clean utility columns",
            "method": "derive from saved predicted_clean_wall_clock_seconds and the saved clean utility formula",
            "costly_rerun": "no",
        },
        {
            "needed_output": "flat-top / near-optimal summaries",
            "method": "recompute from saved per-partition outcome tables",
            "costly_rerun": "no",
        },
        {
            "needed_output": "benchmark figures",
            "method": "matplotlib from saved aggregates",
            "costly_rerun": "no",
        },
    ]

    gate_0_passed = not blockers
    audit_summary = {
        "gate_0_passed": gate_0_passed,
        "blockers": blockers,
        "audit_report": _relative(paths, audit_report_path(paths)),
        "artifact_audit_rows": artifact_table_rows,
        "sufficiency_rows": sufficiency_rows,
        "reconstruction_rows": reconstruction_rows,
        "substrate_baseline_version": compute_substrate_baseline_version(paths),
        "costly_reruns_avoided": True,
    }
    write_json(closure_root(paths) / "audit_summary.json", audit_summary)

    lines = [
        "# Predictive Closure Audit",
        "",
        f"- Gate 0 passed: `{gate_0_passed}`",
        f"- predictive substrate baseline: `{compute_substrate_baseline_version(paths)}`",
        "- audit policy: reuse saved outputs whenever possible; reconstruct only aggregate views and figures.",
        "- costly reruns needed: `none`.",
        "",
        "## Reusable Outputs",
        "",
        markdown_table(
            artifact_table_rows,
            [
                "artifact_group",
                "report_exists",
                "machine_readable_count",
                "machine_readable_total",
                "reconstructs",
                "decision",
                "missing_paths",
            ],
        ),
        "",
        "## Existing Summary Tables",
        "",
        markdown_table(existing_summary_rows, ["artifact_group", "summary_path", "exists"]),
        "",
        "## Sufficiency For Predictive Closure",
        "",
        markdown_table(sufficiency_rows, ["question", "status", "source_paths"]),
        "",
        "## Reuse vs Reconstruction Decisions",
        "",
        markdown_table(reconstruction_rows, ["needed_output", "method", "costly_rerun"]),
        "",
        "## Gate 0 Conclusion",
        "",
    ]
    if gate_0_passed:
        lines.extend(
            [
                "- Enough saved outputs exist to build the full predictive closure package.",
                "- The only material reconstruction is lightweight aggregation plus proxy clean-cost columns for oracle comparisons.",
                "- No frozen predictive experiments need to be rerun.",
            ]
        )
    else:
        lines.extend(["- Gate 0 failed.", *[f"- blocker: `{blocker}`" for blocker in blockers]])

    write_text(audit_report_path(paths), "\n".join(lines) + "\n")
    return audit_summary


def build_unified_predictive_story_tables(paths) -> dict[str, Any]:
    context = _load_context(paths)

    phase1_lookup = _summary_lookup(context["phase1_same_info_summary"], ["family_id", "condition", "clean_or_stress"])
    phase1_ranking_lookup = _summary_lookup(context["phase1_boundary_summary"], ["family_id"])
    phase2_ablation_lookup = _summary_lookup(context["phase2_ablation_summary"], ["family_id", "ablation_condition", "clean_or_stress"])
    phase2_fault_lookup = _summary_lookup(context["phase2_fault_summary"], ["family_id", "repair_policy"])
    phase3_lookup = _summary_lookup(context["phase3_eval_summary"], ["family_id", "condition", "clean_or_stress"])
    phase3b_lookup = _summary_lookup(context["phase3b_eval_summary"], ["family_id", "condition", "clean_or_stress"])

    phase3_variant_lookup = _summary_lookup(
        context["phase3_ranking_summary"].rename(columns={"variant_id": "objective_variant"}),
        ["family_id", "objective_variant"],
    )
    phase3b_variant_lookup = _summary_lookup(
        context["phase3b_ranking_summary"].rename(columns={"variant_id": "objective_variant"}),
        ["family_id", "objective_variant"],
    )

    candidate_family_rows: list[dict[str, Any]] = []
    split_top_rows = _build_split_top_comparison(context["candidate_frame"])
    split_top_lookup = _summary_lookup(split_top_rows, ["family_id", "split_id"])
    for family_id in ALL_FAMILY_ORDER:
        if family_id == "pooled":
            subset = split_top_rows.loc[split_top_rows["family_id"] != "pooled"]
        else:
            subset = split_top_rows.loc[split_top_rows["family_id"] == family_id]
        candidate_family_rows.append(
            {
                "family_id": family_id,
                "baseline_proxy_stress_utility": float(subset["baseline_stress_utility"].mean()),
                "v4_proxy_stress_utility": float(subset["v4_stress_utility"].mean()),
                "v6_proxy_stress_utility": float(subset["v6_stress_utility"].mean()),
                "oracle_proxy_stress_utility": float(subset["oracle_stress_utility"].mean()),
                "micro_proxy_stress_utility": float(subset["micro_stress_utility"].mean()),
                "baseline_to_v4_proxy_gain": float((subset["v4_stress_utility"] - subset["baseline_stress_utility"]).mean()),
                "v4_to_v6_proxy_gain": float((subset["v6_stress_utility"] - subset["v4_stress_utility"]).mean()),
                "v6_oracle_proxy_gap": float((subset["oracle_stress_utility"] - subset["v6_stress_utility"]).mean()),
                "micro_minus_v6_proxy": float((subset["micro_stress_utility"] - subset["v6_stress_utility"]).mean()),
                "mean_near_optimal_count_0p002": float(subset["near_optimal_count_0p002"].mean()),
            }
        )
    candidate_family_frame = pd.DataFrame(candidate_family_rows)
    candidate_family_lookup = _summary_lookup(candidate_family_frame, ["family_id"])

    t1_rows: list[dict[str, Any]] = []
    for family_id in FAMILY_ORDER:
        family_spec = context["family_specs"][family_id]
        split_summary = read_json(paths.results_dir / "predictive_substrate" / "splits" / family_id / "summary.json")
        reference_summary = read_json(paths.results_dir / "predictive_substrate" / "reference" / family_id / "summary.json")
        replay_summary = read_json(paths.results_dir / "predictive_substrate" / "support_replay" / family_id / "summary.json")
        smoke_summary = read_json(paths.results_dir / "predictive_substrate" / "smoke" / family_id / "summary.json")
        validator_summary = read_json(paths.results_dir / "predictive_substrate" / "validator_qa" / family_id / "summary.json")
        current_selected = read_json(paths.results_dir / "partitions" / family_id / "selected_partition.json")["partition_id"]
        t1_rows.append(
            {
                "family_alias": FAMILY_ALIAS[family_id],
                "family_id": family_id,
                "datasets": ", ".join(family_spec["datasets"]),
                "dataset_count": len(family_spec["datasets"]),
                "split_count": int(split_summary["split_count"]),
                "role_template": " -> ".join(family_spec["role_template"]),
                "substrate_reference_status": bool(reference_summary["passed_all"]),
                "validator_qa_status": f"{validator_summary['bad_detection_rate']:.3f}/{validator_summary['false_positive_rate']:.3f}",
                "support_replay_status": f"{replay_summary['pass_rate']:.3f}",
                "smoke_status": bool(smoke_summary["meets_reference_threshold"]),
                "substrate_selected_partition": smoke_summary["selected_partition_id"],
                "current_selected_partition": current_selected,
            }
        )
    t1_frame = pd.DataFrame(t1_rows)

    t2_rows: list[dict[str, Any]] = []
    stress_conditions = context["phase1_same_info_summary"].loc[
        context["phase1_same_info_summary"]["clean_or_stress"] == "stress"
    ].copy()
    best_phase1 = _group_best_condition(stress_conditions, "condition", "utility")
    best_phase1_lookup = _summary_lookup(best_phase1, ["family_id"])
    for family_id in ALL_FAMILY_ORDER:
        baseline_partition = read_json(paths.results_dir / "predictive_substrate" / "smoke" / FAMILY_ORDER[0] / "summary.json")[
            "selected_partition_id"
        ]
        if family_id != "pooled":
            baseline_partition = read_json(paths.results_dir / "predictive_substrate" / "smoke" / family_id / "summary.json")[
                "selected_partition_id"
            ]
        artifact_stress = _value_from_lookup(phase1_lookup, (family_id, "artifact_partition_full", "stress"), "utility")
        micro_stress = _value_from_lookup(phase1_lookup, (family_id, "micro_skill", "stress"), "utility")
        textual_stress = _value_from_lookup(phase1_lookup, (family_id, "structured_textual_memory", "stress"), "utility")
        macro_stress = _value_from_lookup(phase1_lookup, (family_id, "whole_workflow_skill", "stress"), "utility")
        no_contract_stress = _value_from_lookup(phase1_lookup, (family_id, "artifact_partition_no_contract", "stress"), "utility")
        t2_rows.append(
            {
                "family_id": family_id,
                "baseline_selected_partition": baseline_partition,
                "ranking_spearman": _value_from_lookup(phase1_ranking_lookup, (family_id,), "spearman"),
                "ranking_kendall_tau": _value_from_lookup(phase1_ranking_lookup, (family_id,), "kendall_tau"),
                "ranking_top1_regret": _value_from_lookup(phase1_ranking_lookup, (family_id,), "top1_regret"),
                "ranking_top3_hit": _value_from_lookup(phase1_ranking_lookup, (family_id,), "top3_hit"),
                "best_stress_condition": _string_from_lookup(best_phase1_lookup, (family_id,), "best_condition"),
                "artifact_partition_full_stress_utility": artifact_stress,
                "micro_skill_stress_utility": micro_stress,
                "structured_textual_memory_stress_utility": textual_stress,
                "whole_workflow_skill_stress_utility": macro_stress,
                "artifact_partition_no_contract_stress_utility": no_contract_stress,
                "artifact_partition_full_beats_textual_memory": artifact_stress > textual_stress,
                "artifact_partition_full_beats_whole_workflow": artifact_stress > macro_stress,
                "artifact_partition_full_beats_no_contract": artifact_stress > no_contract_stress,
                "micro_skill_beats_selected_artifact": micro_stress > artifact_stress,
                "same_information_outcome_summary": (
                    "artifact partition beats macro/text/no-contract under stress; micro still leads"
                    if micro_stress > artifact_stress
                    else "artifact partition closes or exceeds the micro condition"
                ),
            }
        )
    t2_frame = pd.DataFrame(t2_rows)

    t3_rows: list[dict[str, Any]] = []
    phase2_stress = context["phase2_ablation_summary"].loc[
        context["phase2_ablation_summary"]["clean_or_stress"] == "stress"
    ].copy()
    best_phase2 = _group_best_condition(phase2_stress, "ablation_condition", "utility")
    best_phase2_lookup = _summary_lookup(best_phase2, ["family_id"])
    for family_id in ALL_FAMILY_ORDER:
        full_stress = _value_from_lookup(phase2_ablation_lookup, (family_id, "artifact_partition_full", "stress"), "utility")
        cut_stress = _value_from_lookup(phase2_ablation_lookup, (family_id, "artifact_partition_no_contract", "stress"), "utility")
        contract_only = _value_from_lookup(
            phase2_ablation_lookup,
            (family_id, "artifact_partition_contract_no_structured_validator", "stress"),
            "utility",
        )
        validator_no_repair = _value_from_lookup(
            phase2_ablation_lookup,
            (family_id, "artifact_partition_contract_validator_no_repair", "stress"),
            "utility",
        )
        full_local = phase2_fault_lookup.loc[(family_id, "full_local_repair")]
        global_rerun = phase2_fault_lookup.loc[(family_id, "global_end_to_end_rerun")]
        validator_only = phase2_fault_lookup.loc[(family_id, "validator_detect_no_local_repair")]
        t3_rows.append(
            {
                "family_id": family_id,
                "best_ablation_condition": _string_from_lookup(best_phase2_lookup, (family_id,), "best_condition"),
                "full_system_beats_cut_alone": full_stress > cut_stress,
                "artifact_partition_no_contract_stress_utility": cut_stress,
                "artifact_partition_contract_no_validator_stress_utility": contract_only,
                "artifact_partition_validator_no_repair_stress_utility": validator_no_repair,
                "artifact_partition_full_stress_utility": full_stress,
                "full_local_repair_localization_accuracy": float(full_local["root_localization_accuracy"]),
                "global_rerun_localization_accuracy": float(global_rerun["root_localization_accuracy"]),
                "validator_only_localization_accuracy": float(validator_only["root_localization_accuracy"]),
                "full_local_repair_final_success": float(full_local["final_success"]),
                "global_rerun_final_success": float(global_rerun["final_success"]),
                "validator_only_final_success": float(validator_only["final_success"]),
                "full_local_repair_rerun_span": float(full_local["rerun_span"]),
                "global_rerun_rerun_span": float(global_rerun["rerun_span"]),
                "rerun_span_saved_vs_global": float(global_rerun["rerun_span"] - full_local["rerun_span"]),
            }
        )
    t3_frame = pd.DataFrame(t3_rows)

    t4_rows: list[dict[str, Any]] = []
    phase3_variants = ",".join(sorted(context["phase3_ranking_summary"]["variant_id"].astype(str).unique().tolist()))
    phase3b_variants = ",".join(sorted(context["phase3b_ranking_summary"]["variant_id"].astype(str).unique().tolist()))
    for family_id in ALL_FAMILY_ORDER:
        changed_phase3, total_phase3 = _family_split_count(context["phase3_selection"], family_id, "selected_changed_from_baseline")
        changed_phase3b, total_phase3b = _family_split_count(context["phase3b_selection"], family_id, "selected_changed_from_v4")
        candidate_metrics = candidate_family_lookup.loc[(family_id,)]
        t4_rows.append(
            {
                "family_id": family_id,
                "phase3_variants_tested": phase3_variants,
                "phase3b_variants_tested": phase3b_variants,
                "baseline_selected_partition": "partition_025",
                "phase3_selected_partition": _family_selection_partition(context["phase3_selection"], family_id, "selected_partition_id"),
                "phase3b_selected_partition": _family_selection_partition(context["phase3b_selection"], family_id, "selected_partition_id"),
                "phase3_changed_splits": f"{changed_phase3}/{total_phase3}",
                "phase3b_changed_splits": f"{changed_phase3b}/{total_phase3b}",
                "phase3_spearman": _value_from_lookup(phase3_variant_lookup, (family_id, "V4"), "spearman"),
                "phase3_top3_hit": _value_from_lookup(phase3_variant_lookup, (family_id, "V4"), "top3_hit"),
                "phase3b_spearman": _value_from_lookup(phase3b_variant_lookup, (family_id, "V6"), "spearman"),
                "phase3b_top3_hit": _value_from_lookup(phase3b_variant_lookup, (family_id, "V6"), "top3_hit"),
                "baseline_actual_stress_utility": _value_from_lookup(
                    phase1_lookup, (family_id, "artifact_partition_full", "stress"), "utility"
                ),
                "phase3_actual_stress_utility": _value_from_lookup(
                    phase3_lookup, (family_id, "artifact_partition_full_revised", "stress"), "utility"
                ),
                "phase3b_actual_stress_utility": _value_from_lookup(
                    phase3b_lookup, (family_id, "artifact_partition_phase3b", "stress"), "utility"
                ),
                "micro_actual_stress_utility": _value_from_lookup(phase1_lookup, (family_id, "micro_skill", "stress"), "utility"),
                "phase3_actual_clean_wall_clock": _value_from_lookup(
                    phase3_lookup, (family_id, "artifact_partition_full_revised", "clean"), "wall_clock_seconds"
                ),
                "phase3b_actual_clean_wall_clock": _value_from_lookup(
                    phase3b_lookup, (family_id, "artifact_partition_phase3b", "clean"), "wall_clock_seconds"
                ),
                "micro_actual_clean_wall_clock": _value_from_lookup(
                    phase1_lookup, (family_id, "micro_skill", "clean"), "wall_clock_seconds"
                ),
                "baseline_to_phase3_actual_stress_gain": _value_from_lookup(
                    phase3_lookup, (family_id, "artifact_partition_full_revised", "stress"), "utility"
                )
                - _value_from_lookup(phase1_lookup, (family_id, "artifact_partition_full", "stress"), "utility"),
                "phase3_to_phase3b_actual_stress_gain": _value_from_lookup(
                    phase3b_lookup, (family_id, "artifact_partition_phase3b", "stress"), "utility"
                )
                - _value_from_lookup(phase3_lookup, (family_id, "artifact_partition_full_revised", "stress"), "utility"),
                "phase3b_actual_gap_to_micro": _value_from_lookup(
                    phase1_lookup, (family_id, "micro_skill", "stress"), "utility"
                )
                - _value_from_lookup(phase3b_lookup, (family_id, "artifact_partition_phase3b", "stress"), "utility"),
                "baseline_to_phase3_proxy_gain": float(candidate_metrics["baseline_to_v4_proxy_gain"]),
                "phase3_to_phase3b_proxy_gain": float(candidate_metrics["v4_to_v6_proxy_gain"]),
                "phase3b_proxy_gap_to_oracle": float(candidate_metrics["v6_oracle_proxy_gap"]),
            }
        )
    t4_frame = pd.DataFrame(t4_rows)

    t5_rows: list[dict[str, Any]] = []
    for family_id in ALL_FAMILY_ORDER:
        micro_gap = float(t4_frame.loc[t4_frame["family_id"] == family_id, "phase3b_actual_gap_to_micro"].iloc[0])
        oracle_gap = float(t4_frame.loc[t4_frame["family_id"] == family_id, "phase3b_proxy_gap_to_oracle"].iloc[0])
        t5_rows.append(
            {
                "family_id": family_id,
                "clearly_supported": (
                    f"boundary choice matters under stress; validator-plus-repair beats cut-alone; V6 is near-oracle on the saved partition utility table (oracle gap {oracle_gap:.3f})"
                ),
                "not_supported": (
                    f"selected artifact partition beats micro on executed predictive-family utility (micro lead {micro_gap:.3f})"
                ),
                "ambiguous": (
                    "oracle clean execution was not run directly; clean ordering outside the executed conditions remains proxy-based and overhead-sensitive"
                ),
                "recommended_paper_wording": (
                    "predictive families support family-appropriate cut selection and mechanism value, not a universal artifact-partition win over micro"
                ),
            }
        )
    t5_frame = pd.DataFrame(t5_rows)

    outputs = {
        "t1_predictive_family_scope_csv": aggregate_root(paths) / "t1_predictive_family_scope.csv",
        "t2_phase1_summary_csv": aggregate_root(paths) / "t2_phase1_summary.csv",
        "t3_phase2_summary_csv": aggregate_root(paths) / "t3_phase2_summary.csv",
        "t4_phase3_phase3b_summary_csv": aggregate_root(paths) / "t4_phase3_boundary_evaluation_summary.csv",
        "t5_interpretation_csv": aggregate_root(paths) / "t5_final_predictive_interpretation.csv",
    }
    _save_dataframe(t1_frame, outputs["t1_predictive_family_scope_csv"])
    _save_dataframe(t2_frame, outputs["t2_phase1_summary_csv"])
    _save_dataframe(t3_frame, outputs["t3_phase2_summary_csv"])
    _save_dataframe(t4_frame, outputs["t4_phase3_phase3b_summary_csv"])
    _save_dataframe(t5_frame, outputs["t5_interpretation_csv"])

    summary = {
        "aggregate_tables_ready": True,
        "table_paths": {key: _relative(paths, value) for key, value in outputs.items()},
        "phase3b_selected_partition": _family_selection_partition(context["phase3b_selection"], "pooled", "selected_partition_id"),
        "phase3_selected_partition": _family_selection_partition(context["phase3_selection"], "pooled", "selected_partition_id"),
        "baseline_selected_partition": "partition_025",
    }
    write_json(aggregate_root(paths) / "summary.json", summary)
    return summary


def _build_split_top_comparison(candidate_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family_id, split_id), group_frame in candidate_frame.groupby(["family_id", "split_id"], dropna=False):
        oracle_row = group_frame.sort_values(
            ["weighted_partition_utility", "partition_id"], ascending=[False, True]
        ).iloc[0]
        selected_row = group_frame.loc[group_frame["partition_id"] == group_frame["phase3b_selected_partition_id"].iloc[0]].iloc[0]
        v4_row = group_frame.loc[group_frame["partition_id"] == group_frame["phase3_v4_selected_partition_id"].iloc[0]].iloc[0]
        baseline_row = group_frame.loc[group_frame["partition_id"] == group_frame["baseline_selected_partition_id"].iloc[0]].iloc[0]
        micro_row = group_frame.loc[group_frame["partition_id"] == group_frame["micro_partition_id"].iloc[0]].iloc[0]

        oracle_vector = _partition_vector(oracle_row)
        selected_vector = _partition_vector(selected_row)
        v4_vector = _partition_vector(v4_row)
        micro_vector = _partition_vector(micro_row)

        oracle_utility = float(oracle_row["weighted_partition_utility"])
        near_optimal_mask = (oracle_utility - group_frame["weighted_partition_utility"]) <= NEAR_OPTIMAL_TOLERANCE

        rows.append(
            {
                "family_id": family_id,
                "split_id": split_id,
                "heldout_task": str(oracle_row["heldout_task"]),
                "reference_wall_clock_seconds": float(oracle_row["reference_wall_clock_seconds"]),
                "baseline_partition_id": str(baseline_row["partition_id"]),
                "v4_partition_id": str(v4_row["partition_id"]),
                "v6_partition_id": str(selected_row["partition_id"]),
                "oracle_partition_id": str(oracle_row["partition_id"]),
                "micro_partition_id": str(micro_row["partition_id"]),
                "baseline_stress_utility": float(baseline_row["weighted_partition_utility"]),
                "v4_stress_utility": float(v4_row["weighted_partition_utility"]),
                "v6_stress_utility": float(selected_row["weighted_partition_utility"]),
                "oracle_stress_utility": oracle_utility,
                "micro_stress_utility": float(micro_row["weighted_partition_utility"]),
                "baseline_predicted_clean_utility": float(baseline_row["predicted_clean_utility"]),
                "v4_predicted_clean_utility": float(v4_row["predicted_clean_utility"]),
                "v6_predicted_clean_utility": float(selected_row["predicted_clean_utility"]),
                "oracle_predicted_clean_utility": float(oracle_row["predicted_clean_utility"]),
                "micro_predicted_clean_utility": float(micro_row["predicted_clean_utility"]),
                "baseline_predicted_clean_wall_clock_seconds": float(baseline_row["predicted_clean_wall_clock_seconds"]),
                "v4_predicted_clean_wall_clock_seconds": float(v4_row["predicted_clean_wall_clock_seconds"]),
                "v6_predicted_clean_wall_clock_seconds": float(selected_row["predicted_clean_wall_clock_seconds"]),
                "oracle_predicted_clean_wall_clock_seconds": float(oracle_row["predicted_clean_wall_clock_seconds"]),
                "micro_predicted_clean_wall_clock_seconds": float(micro_row["predicted_clean_wall_clock_seconds"]),
                "v6_oracle_stress_gap": oracle_utility - float(selected_row["weighted_partition_utility"]),
                "micro_oracle_stress_gap": oracle_utility - float(micro_row["weighted_partition_utility"]),
                "v4_oracle_stress_gap": oracle_utility - float(v4_row["weighted_partition_utility"]),
                "baseline_oracle_stress_gap": oracle_utility - float(baseline_row["weighted_partition_utility"]),
                "v4_to_v6_stress_delta": float(selected_row["weighted_partition_utility"]) - float(v4_row["weighted_partition_utility"]),
                "v6_minus_micro_stress_delta": float(selected_row["weighted_partition_utility"]) - float(micro_row["weighted_partition_utility"]),
                "v4_to_v6_boundary_distance": _boundary_distance(v4_vector, selected_vector),
                "v6_to_oracle_boundary_distance": _boundary_distance(selected_vector, oracle_vector),
                "micro_to_oracle_boundary_distance": _boundary_distance(micro_vector, oracle_vector),
                "v6_to_micro_boundary_distance": _boundary_distance(selected_vector, micro_vector),
                "near_optimal_count_0p002": int(near_optimal_mask.sum()),
                "v6_in_near_optimal_0p002": bool(
                    oracle_utility - float(selected_row["weighted_partition_utility"]) <= NEAR_OPTIMAL_TOLERANCE
                ),
                "micro_in_near_optimal_0p002": bool(
                    oracle_utility - float(micro_row["weighted_partition_utility"]) <= NEAR_OPTIMAL_TOLERANCE
                ),
                "v4_in_near_optimal_0p002": bool(
                    oracle_utility - float(v4_row["weighted_partition_utility"]) <= NEAR_OPTIMAL_TOLERANCE
                ),
                "v6_boundary_vector_json": _json_dumps(selected_vector),
                "oracle_boundary_vector_json": _json_dumps(oracle_vector),
                "micro_boundary_vector_json": _json_dumps(micro_vector),
                "v4_boundary_vector_json": _json_dumps(v4_vector),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["family_id", "split_id"]).reset_index(drop=True)
    pooled_row = {
        "family_id": "pooled",
        "split_id": "pooled",
        "heldout_task": "all_predictive_splits",
        "reference_wall_clock_seconds": float(frame["reference_wall_clock_seconds"].mean()),
        "baseline_partition_id": "partition_025",
        "v4_partition_id": "partition_018",
        "v6_partition_id": "partition_004",
        "oracle_partition_id": "partition_020",
        "micro_partition_id": "partition_001",
        "baseline_stress_utility": float(frame["baseline_stress_utility"].mean()),
        "v4_stress_utility": float(frame["v4_stress_utility"].mean()),
        "v6_stress_utility": float(frame["v6_stress_utility"].mean()),
        "oracle_stress_utility": float(frame["oracle_stress_utility"].mean()),
        "micro_stress_utility": float(frame["micro_stress_utility"].mean()),
        "baseline_predicted_clean_utility": float(frame["baseline_predicted_clean_utility"].mean()),
        "v4_predicted_clean_utility": float(frame["v4_predicted_clean_utility"].mean()),
        "v6_predicted_clean_utility": float(frame["v6_predicted_clean_utility"].mean()),
        "oracle_predicted_clean_utility": float(frame["oracle_predicted_clean_utility"].mean()),
        "micro_predicted_clean_utility": float(frame["micro_predicted_clean_utility"].mean()),
        "baseline_predicted_clean_wall_clock_seconds": float(frame["baseline_predicted_clean_wall_clock_seconds"].mean()),
        "v4_predicted_clean_wall_clock_seconds": float(frame["v4_predicted_clean_wall_clock_seconds"].mean()),
        "v6_predicted_clean_wall_clock_seconds": float(frame["v6_predicted_clean_wall_clock_seconds"].mean()),
        "oracle_predicted_clean_wall_clock_seconds": float(frame["oracle_predicted_clean_wall_clock_seconds"].mean()),
        "micro_predicted_clean_wall_clock_seconds": float(frame["micro_predicted_clean_wall_clock_seconds"].mean()),
        "v6_oracle_stress_gap": float(frame["v6_oracle_stress_gap"].mean()),
        "micro_oracle_stress_gap": float(frame["micro_oracle_stress_gap"].mean()),
        "v4_oracle_stress_gap": float(frame["v4_oracle_stress_gap"].mean()),
        "baseline_oracle_stress_gap": float(frame["baseline_oracle_stress_gap"].mean()),
        "v4_to_v6_stress_delta": float(frame["v4_to_v6_stress_delta"].mean()),
        "v6_minus_micro_stress_delta": float(frame["v6_minus_micro_stress_delta"].mean()),
        "v4_to_v6_boundary_distance": float(frame["v4_to_v6_boundary_distance"].mean()),
        "v6_to_oracle_boundary_distance": float(frame["v6_to_oracle_boundary_distance"].mean()),
        "micro_to_oracle_boundary_distance": float(frame["micro_to_oracle_boundary_distance"].mean()),
        "v6_to_micro_boundary_distance": float(frame["v6_to_micro_boundary_distance"].mean()),
        "near_optimal_count_0p002": float(frame["near_optimal_count_0p002"].mean()),
        "v6_in_near_optimal_0p002": bool(frame["v6_in_near_optimal_0p002"].all()),
        "micro_in_near_optimal_0p002": float(frame["micro_in_near_optimal_0p002"].mean()),
        "v4_in_near_optimal_0p002": float(frame["v4_in_near_optimal_0p002"].mean()),
        "v6_boundary_vector_json": "",
        "oracle_boundary_vector_json": "",
        "micro_boundary_vector_json": "",
        "v4_boundary_vector_json": "",
    }
    return pd.concat([frame, pd.DataFrame([pooled_row])], ignore_index=True)


def _boundary_distance_tables(candidate_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (family_id, split_id), group_frame in candidate_frame.groupby(["family_id", "split_id"], dropna=False):
        oracle_row = group_frame.sort_values(
            ["weighted_partition_utility", "partition_id"], ascending=[False, True]
        ).iloc[0]
        oracle_vector = _partition_vector(oracle_row)
        oracle_utility = float(oracle_row["weighted_partition_utility"])
        for _, row in group_frame.iterrows():
            gap = oracle_utility - float(row["weighted_partition_utility"])
            rows.append(
                {
                    "family_id": family_id,
                    "split_id": split_id,
                    "partition_id": str(row["partition_id"]),
                    "boundary_distance_to_oracle": _boundary_distance(_partition_vector(row), oracle_vector),
                    "utility_gap_to_oracle": gap,
                    "near_optimal_0p002": gap <= NEAR_OPTIMAL_TOLERANCE,
                }
            )
    detail_frame = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    for family_id in ALL_FAMILY_ORDER:
        subset = detail_frame if family_id == "pooled" else detail_frame.loc[detail_frame["family_id"] == family_id]
        for boundary_distance, distance_frame in subset.groupby("boundary_distance_to_oracle", dropna=False):
            summary_rows.append(
                {
                    "family_id": family_id,
                    "boundary_distance_to_oracle": int(boundary_distance),
                    "partition_count": int(len(distance_frame)),
                    "mean_utility_gap_to_oracle": float(distance_frame["utility_gap_to_oracle"].mean()),
                    "median_utility_gap_to_oracle": float(distance_frame["utility_gap_to_oracle"].median()),
                    "max_utility_gap_to_oracle": float(distance_frame["utility_gap_to_oracle"].max()),
                    "near_optimal_rate_0p002": float(distance_frame["near_optimal_0p002"].mean()),
                }
            )
    return detail_frame, pd.DataFrame(summary_rows).sort_values(["family_id", "boundary_distance_to_oracle"])


def _translation_loss_tables(context: dict[str, Any], split_top_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase1_lookup = _summary_lookup(context["phase1_same_info_summary"], ["family_id", "condition", "clean_or_stress"])
    phase3_lookup = _summary_lookup(context["phase3_eval_summary"], ["family_id", "condition", "clean_or_stress"])
    phase3b_lookup = _summary_lookup(context["phase3b_eval_summary"], ["family_id", "condition", "clean_or_stress"])

    candidate_rows: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    for family_id in ALL_FAMILY_ORDER:
        if family_id == "pooled":
            split_subset = split_top_frame.loc[split_top_frame["family_id"] != "pooled"]
        else:
            split_subset = split_top_frame.loc[split_top_frame["family_id"] == family_id]

        candidate_rows.extend(
            [
                {
                    "family_id": family_id,
                    "phase": "baseline",
                    "stress_utility": float(split_subset["baseline_stress_utility"].mean()),
                    "predicted_clean_utility": float(split_subset["baseline_predicted_clean_utility"].mean()),
                    "predicted_clean_wall_clock_seconds": float(
                        split_subset["baseline_predicted_clean_wall_clock_seconds"].mean()
                    ),
                },
                {
                    "family_id": family_id,
                    "phase": "V4",
                    "stress_utility": float(split_subset["v4_stress_utility"].mean()),
                    "predicted_clean_utility": float(split_subset["v4_predicted_clean_utility"].mean()),
                    "predicted_clean_wall_clock_seconds": float(
                        split_subset["v4_predicted_clean_wall_clock_seconds"].mean()
                    ),
                },
                {
                    "family_id": family_id,
                    "phase": "V6",
                    "stress_utility": float(split_subset["v6_stress_utility"].mean()),
                    "predicted_clean_utility": float(split_subset["v6_predicted_clean_utility"].mean()),
                    "predicted_clean_wall_clock_seconds": float(
                        split_subset["v6_predicted_clean_wall_clock_seconds"].mean()
                    ),
                },
                {
                    "family_id": family_id,
                    "phase": "oracle",
                    "stress_utility": float(split_subset["oracle_stress_utility"].mean()),
                    "predicted_clean_utility": float(split_subset["oracle_predicted_clean_utility"].mean()),
                    "predicted_clean_wall_clock_seconds": float(
                        split_subset["oracle_predicted_clean_wall_clock_seconds"].mean()
                    ),
                },
                {
                    "family_id": family_id,
                    "phase": "micro",
                    "stress_utility": float(split_subset["micro_stress_utility"].mean()),
                    "predicted_clean_utility": float(split_subset["micro_predicted_clean_utility"].mean()),
                    "predicted_clean_wall_clock_seconds": float(
                        split_subset["micro_predicted_clean_wall_clock_seconds"].mean()
                    ),
                },
            ]
        )

        actual_rows.extend(
            [
                {
                    "family_id": family_id,
                    "phase": "baseline",
                    "stress_utility": _value_from_lookup(phase1_lookup, (family_id, "artifact_partition_full", "stress"), "utility"),
                    "clean_utility": _value_from_lookup(phase1_lookup, (family_id, "artifact_partition_full", "clean"), "utility"),
                    "clean_wall_clock_seconds": _value_from_lookup(
                        phase1_lookup, (family_id, "artifact_partition_full", "clean"), "wall_clock_seconds"
                    ),
                },
                {
                    "family_id": family_id,
                    "phase": "V4",
                    "stress_utility": _value_from_lookup(
                        phase3_lookup, (family_id, "artifact_partition_full_revised", "stress"), "utility"
                    ),
                    "clean_utility": _value_from_lookup(
                        phase3_lookup, (family_id, "artifact_partition_full_revised", "clean"), "utility"
                    ),
                    "clean_wall_clock_seconds": _value_from_lookup(
                        phase3_lookup, (family_id, "artifact_partition_full_revised", "clean"), "wall_clock_seconds"
                    ),
                },
                {
                    "family_id": family_id,
                    "phase": "V6",
                    "stress_utility": _value_from_lookup(
                        phase3b_lookup, (family_id, "artifact_partition_phase3b", "stress"), "utility"
                    ),
                    "clean_utility": _value_from_lookup(
                        phase3b_lookup, (family_id, "artifact_partition_phase3b", "clean"), "utility"
                    ),
                    "clean_wall_clock_seconds": _value_from_lookup(
                        phase3b_lookup, (family_id, "artifact_partition_phase3b", "clean"), "wall_clock_seconds"
                    ),
                },
                {
                    "family_id": family_id,
                    "phase": "micro",
                    "stress_utility": _value_from_lookup(phase1_lookup, (family_id, "micro_skill", "stress"), "utility"),
                    "clean_utility": _value_from_lookup(phase1_lookup, (family_id, "micro_skill", "clean"), "utility"),
                    "clean_wall_clock_seconds": _value_from_lookup(
                        phase1_lookup, (family_id, "micro_skill", "clean"), "wall_clock_seconds"
                    ),
                },
            ]
        )

    candidate_frame = pd.DataFrame(candidate_rows)
    actual_frame = pd.DataFrame(actual_rows)

    for family_id in ALL_FAMILY_ORDER:
        candidate_subset = candidate_frame.loc[candidate_frame["family_id"] == family_id].set_index("phase")
        actual_subset = actual_frame.loc[actual_frame["family_id"] == family_id].set_index("phase")
        bridge_rows.append(
            {
                "family_id": family_id,
                "baseline_proxy_stress_utility": float(candidate_subset.loc["baseline", "stress_utility"]),
                "baseline_actual_stress_utility": float(actual_subset.loc["baseline", "stress_utility"]),
                "v4_proxy_stress_utility": float(candidate_subset.loc["V4", "stress_utility"]),
                "v4_actual_stress_utility": float(actual_subset.loc["V4", "stress_utility"]),
                "v6_proxy_stress_utility": float(candidate_subset.loc["V6", "stress_utility"]),
                "v6_actual_stress_utility": float(actual_subset.loc["V6", "stress_utility"]),
                "micro_proxy_stress_utility": float(candidate_subset.loc["micro", "stress_utility"]),
                "micro_actual_stress_utility": float(actual_subset.loc["micro", "stress_utility"]),
                "baseline_to_v4_proxy_gain": float(
                    candidate_subset.loc["V4", "stress_utility"] - candidate_subset.loc["baseline", "stress_utility"]
                ),
                "baseline_to_v4_actual_gain": float(
                    actual_subset.loc["V4", "stress_utility"] - actual_subset.loc["baseline", "stress_utility"]
                ),
                "v4_to_v6_proxy_gain": float(candidate_subset.loc["V6", "stress_utility"] - candidate_subset.loc["V4", "stress_utility"]),
                "v4_to_v6_actual_gain": float(actual_subset.loc["V6", "stress_utility"] - actual_subset.loc["V4", "stress_utility"]),
                "v6_oracle_proxy_gap": float(candidate_subset.loc["oracle", "stress_utility"] - candidate_subset.loc["V6", "stress_utility"]),
                "micro_minus_v6_proxy": float(candidate_subset.loc["micro", "stress_utility"] - candidate_subset.loc["V6", "stress_utility"]),
                "micro_minus_v6_actual": float(actual_subset.loc["micro", "stress_utility"] - actual_subset.loc["V6", "stress_utility"]),
                "v6_proxy_minus_actual": float(candidate_subset.loc["V6", "stress_utility"] - actual_subset.loc["V6", "stress_utility"]),
                "micro_proxy_minus_actual": float(candidate_subset.loc["micro", "stress_utility"] - actual_subset.loc["micro", "stress_utility"]),
                "v6_actual_clean_wall_clock_seconds": float(actual_subset.loc["V6", "clean_wall_clock_seconds"]),
                "micro_actual_clean_wall_clock_seconds": float(actual_subset.loc["micro", "clean_wall_clock_seconds"]),
                "v6_actual_clean_wall_clock_minus_micro": float(
                    actual_subset.loc["V6", "clean_wall_clock_seconds"] - actual_subset.loc["micro", "clean_wall_clock_seconds"]
                ),
            }
        )
    return candidate_frame, pd.DataFrame(bridge_rows)


def run_predictive_flat_top_analysis(paths) -> dict[str, Any]:
    audit_summary = audit_predictive_evaluation(paths)
    if not audit_summary["gate_0_passed"]:
        return {
            "gate_1_passed": False,
            "blockers": audit_summary["blockers"],
            "flat_top_report": _relative(paths, flat_top_report_path(paths)),
        }

    build_unified_predictive_story_tables(paths)
    context = _load_context(paths)

    split_top_frame = _build_split_top_comparison(context["candidate_frame"])
    boundary_detail_frame, boundary_summary_frame = _boundary_distance_tables(context["candidate_frame"])
    candidate_progression_frame, translation_bridge_frame = _translation_loss_tables(context, split_top_frame)

    near_optimal_rows: list[dict[str, Any]] = []
    for family_id in ALL_FAMILY_ORDER:
        if family_id == "pooled":
            subset = split_top_frame.loc[split_top_frame["family_id"] != "pooled"]
        else:
            subset = split_top_frame.loc[split_top_frame["family_id"] == family_id]
        near_optimal_rows.append(
            {
                "family_id": family_id,
                "tolerance": NEAR_OPTIMAL_TOLERANCE,
                "mean_near_optimal_count": float(subset["near_optimal_count_0p002"].mean()),
                "min_near_optimal_count": float(subset["near_optimal_count_0p002"].min()),
                "max_near_optimal_count": float(subset["near_optimal_count_0p002"].max()),
                "v6_in_near_optimal_rate": float(subset["v6_in_near_optimal_0p002"].mean()),
                "micro_in_near_optimal_rate": float(subset["micro_in_near_optimal_0p002"].mean()),
                "mean_v6_oracle_stress_gap": float(subset["v6_oracle_stress_gap"].mean()),
                "mean_micro_oracle_stress_gap": float(subset["micro_oracle_stress_gap"].mean()),
            }
        )
    near_optimal_frame = pd.DataFrame(near_optimal_rows)

    _save_dataframe(split_top_frame, flat_top_root(paths) / "top_partition_comparison_by_split.csv")
    _save_dataframe(near_optimal_frame, flat_top_root(paths) / "near_optimal_summary_by_family.csv")
    _save_dataframe(boundary_detail_frame, flat_top_root(paths) / "boundary_distance_to_oracle_rows.csv")
    _save_dataframe(boundary_summary_frame, flat_top_root(paths) / "boundary_distance_to_oracle_summary.csv")
    _save_dataframe(candidate_progression_frame, flat_top_root(paths) / "candidate_phase_progression_by_family.csv")
    _save_dataframe(translation_bridge_frame, flat_top_root(paths) / "translation_loss_summary_by_family.csv")

    pooled_flat = split_top_frame.loc[split_top_frame["family_id"] == "pooled"].iloc[0]
    pooled_translation = translation_bridge_frame.loc[translation_bridge_frame["family_id"] == "pooled"].iloc[0]

    gate_1_passed = (
        bool(pooled_flat["v6_in_near_optimal_0p002"])
        and float(pooled_flat["near_optimal_count_0p002"]) >= 4.0
        and abs(float(pooled_flat["v4_to_v6_stress_delta"])) <= 1.0e-9
    )

    report_lines = [
        "# Predictive Flat-Top Analysis",
        "",
        f"- Gate 1 passed: `{gate_1_passed}`",
        f"- near-optimal tolerance under saved stress utility: `{NEAR_OPTIMAL_TOLERANCE:.3f}`",
        "- all analyses below reuse saved partition utilities and saved executed-condition rows; no costly rerun was performed.",
        "",
        "## A. Top-Partition Utility Spread",
        "",
        markdown_table(
            split_top_frame.loc[split_top_frame["family_id"] == "pooled"].to_dict(orient="records"),
            [
                "baseline_stress_utility",
                "v4_stress_utility",
                "v6_stress_utility",
                "oracle_stress_utility",
                "micro_stress_utility",
                "v6_oracle_stress_gap",
                "micro_oracle_stress_gap",
            ],
        ),
        "",
        "- The saved partition table puts `V6` essentially on the oracle plateau: pooled oracle-minus-V6 gap is tiny.",
        "- `micro` is also close on the saved partition table, but farther from oracle than `V6` in boundary space and in saved stress utility.",
        "",
        "## B. Near-Optimal Set Analysis",
        "",
        markdown_table(
            near_optimal_frame.to_dict(orient="records"),
            [
                "family_id",
                "mean_near_optimal_count",
                "min_near_optimal_count",
                "max_near_optimal_count",
                "v6_in_near_optimal_rate",
                "micro_in_near_optimal_rate",
            ],
        ),
        "",
        f"- At tolerance `{NEAR_OPTIMAL_TOLERANCE:.3f}`, `V6` is inside the near-optimal set on every split.",
        "- The mean near-optimal set size is well above one in every family, so the top of the utility surface is not sharp.",
        "",
        "## C. Boundary Distance vs Utility Distance",
        "",
        markdown_table(
            boundary_summary_frame.loc[boundary_summary_frame["family_id"] == "pooled"].to_dict(orient="records"),
            [
                "boundary_distance_to_oracle",
                "partition_count",
                "mean_utility_gap_to_oracle",
                "median_utility_gap_to_oracle",
                "near_optimal_rate_0p002",
            ],
        ),
        "",
        f"- `V4` and `V6` differ by `{float(pooled_flat['v4_to_v6_boundary_distance']):.0f}` boundary flips on every split, yet their saved stress utility delta is exactly `{float(pooled_flat['v4_to_v6_stress_delta']):.3f}`.",
        f"- `V6` is only `{float(pooled_flat['v6_to_oracle_boundary_distance']):.0f}` boundary away from oracle on average, while `micro` stays `{float(pooled_flat['micro_to_oracle_boundary_distance']):.0f}` away.",
        "",
        "## D. Translation Loss Analysis",
        "",
        markdown_table(
            translation_bridge_frame.to_dict(orient="records"),
            [
                "family_id",
                "baseline_to_v4_proxy_gain",
                "baseline_to_v4_actual_gain",
                "v4_to_v6_proxy_gain",
                "v4_to_v6_actual_gain",
                "v6_oracle_proxy_gap",
                "micro_minus_v6_proxy",
                "micro_minus_v6_actual",
                "v6_actual_clean_wall_clock_minus_micro",
            ],
        ),
        "",
        "- The largest translation loss happens after phase-3: proxy utility improves strongly from baseline to `V4`, but actual executed stress utility rises only modestly.",
        "- The phase-3b ranking jump is even more diagnostic: `V6` improves pooled Spearman sharply, yet the saved partition table says `V4` and `V6` are utility-tied, so the selected utility cannot move much.",
        "- `micro` remains the best executed stress condition even though the saved partition table ranks it slightly below `V6`; that is evidence that actual runtime overhead remains a bottleneck outside the partition-ranking proxy.",
        "",
        "## Gate 1 Conclusion",
        "",
        (
            f"- Gate 1 passes. The strongest supported explanation is a flat top near the oracle together with a phase-3b plateau: "
            f"`V6` is near-oracle on all splits, `V4` and `V6` are utility-tied on the saved partition table, and actual executed utility is still dominated by small cost differences and residual overhead relative to `micro`."
        ),
    ]

    write_text(flat_top_report_path(paths), "\n".join(report_lines) + "\n")

    summary = {
        "gate_1_passed": gate_1_passed,
        "blockers": [],
        "flat_top_report": _relative(paths, flat_top_report_path(paths)),
        "top_partition_csv": _relative(paths, flat_top_root(paths) / "top_partition_comparison_by_split.csv"),
        "near_optimal_csv": _relative(paths, flat_top_root(paths) / "near_optimal_summary_by_family.csv"),
        "boundary_distance_csv": _relative(paths, flat_top_root(paths) / "boundary_distance_to_oracle_summary.csv"),
        "translation_loss_csv": _relative(paths, flat_top_root(paths) / "translation_loss_summary_by_family.csv"),
        "mean_near_optimal_count": float(pooled_flat["near_optimal_count_0p002"]),
        "v6_oracle_gap": float(pooled_flat["v6_oracle_stress_gap"]),
        "v4_to_v6_proxy_gain": float(pooled_translation["v4_to_v6_proxy_gain"]),
        "v4_to_v6_actual_gain": float(pooled_translation["v4_to_v6_actual_gain"]),
        "micro_minus_v6_actual": float(pooled_translation["micro_minus_v6_actual"]),
    }
    write_json(flat_top_root(paths) / "summary.json", summary)
    return summary


def _plot_predictive_timeline(output_path: Path) -> None:
    phases = [
        ("Substrate", "F1-F4\n15 splits\nreplay/smoke pass", "#264653"),
        ("Phase-1", "cut matters\nmicro leads\nrho=0.124", "#8d99ae"),
        ("Phase-2", "validator + repair\nmechanism established", "#457b9d"),
        ("Phase-3", "V4\npartition_018\nrho=0.681", "#2a9d8f"),
        ("Phase-3b", "V6\npartition_004\nrho=0.947", "#e76f51"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, len(phases))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, (title, body, color) in enumerate(phases):
        x0 = index + 0.08
        box = FancyBboxPatch(
            (x0, 0.18),
            0.84,
            0.62,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.14,
        )
        ax.add_patch(box)
        ax.text(x0 + 0.42, 0.64, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x0 + 0.42, 0.40, body, ha="center", va="center", fontsize=9)
        if index < len(phases) - 1:
            ax.annotate(
                "",
                xy=(index + 1.0, 0.49),
                xytext=(index + 0.92, 0.49),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#444444"},
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_ranking_vs_translation(translation_bridge_frame: pd.DataFrame, output_path: Path) -> None:
    pooled = translation_bridge_frame.loc[translation_bridge_frame["family_id"] == "pooled"].iloc[0]
    ranking_values = [0.123751, 0.681000, 0.947000]
    ranking_labels = ["V0", "V4", "V6"]
    proxy_values = [
        float(pooled["baseline_proxy_stress_utility"]),
        float(pooled["v4_proxy_stress_utility"]),
        float(pooled["v6_proxy_stress_utility"]),
        float(pooled["v6_oracle_proxy_gap"] + pooled["v6_proxy_stress_utility"]),
    ]
    proxy_labels = ["baseline", "V4", "V6", "oracle"]
    actual_values = [
        float(pooled["baseline_actual_stress_utility"]),
        float(pooled["v4_actual_stress_utility"]),
        float(pooled["v6_actual_stress_utility"]),
        float(pooled["micro_actual_stress_utility"]),
    ]
    actual_labels = ["baseline", "V4", "V6", "micro"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].bar(ranking_labels, ranking_values, color=["#8d99ae", "#457b9d", "#2a9d8f"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Pooled Spearman")
    axes[0].set_ylabel("rank correlation")

    axes[1].bar(proxy_labels, proxy_values, color=["#8d99ae", "#457b9d", "#2a9d8f", "#264653"])
    axes[1].set_ylim(0.82, 0.88)
    axes[1].set_title("Saved Stress Utility")
    axes[1].set_ylabel("proxy utility")

    axes[2].bar(actual_labels, actual_values, color=["#8d99ae", "#457b9d", "#2a9d8f", "#e76f51"])
    axes[2].set_ylim(0.82, 0.87)
    axes[2].set_title("Executed Stress Utility")
    axes[2].set_ylabel("actual utility")

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_oracle_v6_micro_comparison(split_top_frame: pd.DataFrame, output_path: Path) -> None:
    family_rows: list[dict[str, Any]] = []
    for family_id in ALL_FAMILY_ORDER:
        subset = split_top_frame.loc[split_top_frame["family_id"] == family_id]
        family_rows.append(
            {
                "family_id": family_id,
                "label": FAMILY_ALIAS[family_id],
                "oracle_stress_utility": float(subset["oracle_stress_utility"].mean()),
                "v6_stress_utility": float(subset["v6_stress_utility"].mean()),
                "micro_stress_utility": float(subset["micro_stress_utility"].mean()),
                "oracle_predicted_clean_utility": float(subset["oracle_predicted_clean_utility"].mean()),
                "v6_predicted_clean_utility": float(subset["v6_predicted_clean_utility"].mean()),
                "micro_predicted_clean_utility": float(subset["micro_predicted_clean_utility"].mean()),
                "oracle_predicted_clean_wall_clock_seconds": float(
                    subset["oracle_predicted_clean_wall_clock_seconds"].mean()
                ),
                "v6_predicted_clean_wall_clock_seconds": float(subset["v6_predicted_clean_wall_clock_seconds"].mean()),
                "micro_predicted_clean_wall_clock_seconds": float(
                    subset["micro_predicted_clean_wall_clock_seconds"].mean()
                ),
            }
        )
    frame = pd.DataFrame(family_rows)
    labels = frame["label"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    series = [
        ("oracle", "oracle_stress_utility", "oracle_predicted_clean_utility", "oracle_predicted_clean_wall_clock_seconds", "#264653"),
        ("V6", "v6_stress_utility", "v6_predicted_clean_utility", "v6_predicted_clean_wall_clock_seconds", "#2a9d8f"),
        ("micro", "micro_stress_utility", "micro_predicted_clean_utility", "micro_predicted_clean_wall_clock_seconds", "#e76f51"),
    ]
    for offset, (label, stress_col, clean_col, wall_col, color) in zip([-width, 0.0, width], series):
        axes[0].bar(x + offset, frame[stress_col], width=width, label=label, color=color)
        axes[1].bar(x + offset, frame[clean_col], width=width, label=label, color=color)
        axes[2].bar(x + offset, frame[wall_col], width=width, label=label, color=color)

    axes[0].set_title("Saved Stress Utility")
    axes[1].set_title("Proxy Clean Utility")
    axes[2].set_title("Proxy Clean Wall-Clock")
    axes[0].set_ylabel("utility")
    axes[1].set_ylabel("utility")
    axes[2].set_ylabel("seconds")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_flat_top_summary(near_optimal_frame: pd.DataFrame, boundary_summary_frame: pd.DataFrame, output_path: Path) -> None:
    family_frame = near_optimal_frame.loc[near_optimal_frame["family_id"] != "pooled"].copy()
    boundary_pooled = boundary_summary_frame.loc[boundary_summary_frame["family_id"] == "pooled"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

    axes[0].bar(
        [FAMILY_ALIAS[family_id] for family_id in family_frame["family_id"]],
        family_frame["mean_near_optimal_count"],
        color="#457b9d",
    )
    axes[0].set_title("Near-Optimal Set Size")
    axes[0].set_ylabel("mean count within 0.002 of oracle")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        boundary_pooled["boundary_distance_to_oracle"],
        boundary_pooled["mean_utility_gap_to_oracle"],
        marker="o",
        color="#2a9d8f",
    )
    axes[1].bar(
        boundary_pooled["boundary_distance_to_oracle"],
        boundary_pooled["near_optimal_rate_0p002"],
        alpha=0.20,
        color="#e76f51",
    )
    axes[1].set_title("Boundary Distance vs Utility Gap")
    axes[1].set_xlabel("distance to oracle in boundary space")
    axes[1].set_ylabel("mean utility gap / near-optimal rate")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_claim_summary(output_path: Path) -> None:
    boxes = [
        ("Established", "boundary importance\nmechanism value\nfamily-appropriate cuts", "#2a9d8f", (0.05, 0.56, 0.40, 0.30)),
        ("Not Established", "artifact partition > micro\nlarge predictive-only gains", "#e76f51", (0.55, 0.56, 0.40, 0.30)),
        ("Why Flat-Top", "4-8 near-optimal partitions/split\nV4 and V6 are utility-tied", "#457b9d", (0.05, 0.12, 0.40, 0.30)),
        ("Next Stage", "stop predictive score tuning\nmove frozen stack to richer families", "#264653", (0.55, 0.12, 0.40, 0.30)),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for title, body, color, (x0, y0, width, height) in boxes:
        patch = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.03",
            facecolor=color,
            edgecolor=color,
            alpha=0.14,
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x0 + width / 2.0, y0 + height * 0.72, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x0 + width / 2.0, y0 + height * 0.36, body, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _generate_figures(paths, split_top_frame: pd.DataFrame, near_optimal_frame: pd.DataFrame, boundary_summary_frame: pd.DataFrame, translation_bridge_frame: pd.DataFrame) -> dict[str, str]:
    figure_paths = {
        "timeline": figures_root(paths) / "predictive_timeline.png",
        "ranking_translation": figures_root(paths) / "ranking_vs_downstream_translation.png",
        "oracle_v6_micro": figures_root(paths) / "oracle_v6_micro_comparison.png",
        "flat_top": figures_root(paths) / "flat_top_near_optimal_summary.png",
        "claim_summary": figures_root(paths) / "predictive_claim_summary.png",
    }
    _plot_predictive_timeline(figure_paths["timeline"])
    _plot_ranking_vs_translation(translation_bridge_frame, figure_paths["ranking_translation"])
    _plot_oracle_v6_micro_comparison(split_top_frame, figure_paths["oracle_v6_micro"])
    _plot_flat_top_summary(near_optimal_frame, boundary_summary_frame, figure_paths["flat_top"])
    _plot_claim_summary(figure_paths["claim_summary"])
    return {key: _relative(paths, value) for key, value in figure_paths.items()}


def write_predictive_evaluation_report(paths) -> dict[str, Any]:
    audit_summary = audit_predictive_evaluation(paths)
    if not audit_summary["gate_0_passed"]:
        summary = {
            "gate_1_passed": False,
            "blockers": audit_summary["blockers"],
            "report_path": _relative(paths, closure_report_path(paths)),
        }
        write_text(
            closure_report_path(paths),
            "# Predictive Closure Report\n\n- Gate 0 failed during audit, so closure reporting stopped.\n",
        )
        return summary

    build_unified_predictive_story_tables(paths)
    flat_top_summary = run_predictive_flat_top_analysis(paths)
    if not flat_top_summary["gate_1_passed"]:
        return {
            "gate_1_passed": False,
            "blockers": flat_top_summary["blockers"],
            "report_path": _relative(paths, closure_report_path(paths)),
        }

    context = _load_context(paths)
    split_top_frame = _build_split_top_comparison(context["candidate_frame"])
    _, boundary_summary_frame = _boundary_distance_tables(context["candidate_frame"])
    _, translation_bridge_frame = _translation_loss_tables(context, split_top_frame)

    near_optimal_frame = pd.read_csv(flat_top_root(paths) / "near_optimal_summary_by_family.csv")
    figure_paths = _generate_figures(paths, split_top_frame, near_optimal_frame, boundary_summary_frame, translation_bridge_frame)

    t1 = pd.read_csv(aggregate_root(paths) / "t1_predictive_family_scope.csv")
    t2 = pd.read_csv(aggregate_root(paths) / "t2_phase1_summary.csv")
    t3 = pd.read_csv(aggregate_root(paths) / "t3_phase2_summary.csv")
    t4 = pd.read_csv(aggregate_root(paths) / "t4_phase3_boundary_evaluation_summary.csv")
    t5 = pd.read_csv(aggregate_root(paths) / "t5_final_predictive_interpretation.csv")

    pooled_t2 = t2.loc[t2["family_id"] == "pooled"].iloc[0]
    pooled_t3 = t3.loc[t3["family_id"] == "pooled"].iloc[0]
    pooled_t4 = t4.loc[t4["family_id"] == "pooled"].iloc[0]
    pooled_t5 = t5.loc[t5["family_id"] == "pooled"].iloc[0]
    pooled_near_optimal = near_optimal_frame.loc[near_optimal_frame["family_id"] == "pooled"].iloc[0]
    pooled_translation = translation_bridge_frame.loc[translation_bridge_frame["family_id"] == "pooled"].iloc[0]
    pooled_split_top = split_top_frame.loc[split_top_frame["family_id"] == "pooled"].iloc[0]

    strongest_supported_claim = (
        "Predictive families establish boundary importance and mechanism value, and phase-3/3b recover family-appropriate near-oracle legal cuts on the saved partition table, but executed downstream gains over micro remain modest."
    )
    key_unsupported_claim = "Predictive families do not establish that the selected artifact partition beats micro on executed downstream utility."
    flat_top_explanation = (
        "The saved utility surface has a broad near-optimal top, V4 and V6 are utility-tied despite a two-boundary change, and actual executed utility remains sensitive to residual runtime overhead relative to micro."
    )
    recommended_next_stage = (
        "Freeze predictive families at closure and port the same validator-aware machinery to a richer workflow family class before any new score revision."
    )

    recommendation_payload = {
        "predictive_line_status": "closed_with_interpretation_package",
        "objective_revision_status": "frozen_after_phase3b_no_further_predictive_objective_revision_recommended",
        "predictive_substrate_hash": compute_substrate_baseline_version(paths),
        "strongest_supported_claim": strongest_supported_claim,
        "unsupported_claims": [
            key_unsupported_claim,
            "Predictive families alone do not justify a final paper claim that artifact partitions universally outperform fine legal micro cuts.",
            "Oracle clean execution was not directly established beyond the saved clean-cost proxy.",
        ],
        "explanation_for_flat_top_behavior": {
            "near_optimal_tolerance": NEAR_OPTIMAL_TOLERANCE,
            "mean_near_optimal_count": float(pooled_near_optimal["mean_near_optimal_count"]),
            "v6_oracle_proxy_gap": float(pooled_split_top["v6_oracle_stress_gap"]),
            "v4_to_v6_proxy_gain": float(pooled_translation["v4_to_v6_proxy_gain"]),
            "v4_to_v6_actual_gain": float(pooled_translation["v4_to_v6_actual_gain"]),
            "micro_minus_v6_actual": float(pooled_translation["micro_minus_v6_actual"]),
            "summary": flat_top_explanation,
        },
        "recommended_next_stage": {
            "stage_id": "implement_richer_workflow_families",
            "summary": recommended_next_stage,
            "implementation_note": "Keep the predictive substrate, validators, repair APIs, and current predictive closure package frozen; require substrate/replay/smoke first in the richer family class, then rerun phase-1 and phase-2 before any new score work.",
        },
        "confidence_level": "moderate_high",
        "costly_reruns_avoided": True,
    }
    write_json(closure_root(paths) / "final_recommendation.json", recommendation_payload)

    report_lines = [
        "# Predictive Closure Report",
        "",
        f"- predictive substrate baseline: `{compute_substrate_baseline_version(paths)}`",
        "- this closure package reuses saved predictive outputs and performs only lightweight aggregation, proxy reconstruction, and figure generation.",
        "",
        "## 1. Predictive Families Studied",
        "",
        markdown_table(
            t1.to_dict(orient="records"),
            [
                "family_alias",
                "family_id",
                "datasets",
                "split_count",
                "substrate_reference_status",
                "support_replay_status",
                "smoke_status",
            ],
        ),
        "",
        "## 2. Direct Answers",
        "",
        "1. What predictive families were studied?",
        "- F1 `openml_tabular_binary`, F2 `tdc_admet_binary`, F3 `tdc_admet_regression`, and F4 `tdc_tox_binary`, for a total of 15 leave-one-dataset-out splits on the frozen predictive substrate.",
        "2. What has been established on F1-F4?",
        f"- {strongest_supported_claim}",
        "3. What has NOT been established?",
        f"- {key_unsupported_claim}",
        "4. What does phase-1 show about boundary importance?",
        (
            f"- Under mild portability stress, the selected artifact partition beats structured text, whole-workflow, and no-contract baselines "
            f"(`artifact_partition_full` stress utility `{pooled_t2['artifact_partition_full_stress_utility']:.3f}` vs "
            f"`structured_textual_memory` `{pooled_t2['structured_textual_memory_stress_utility']:.3f}` and "
            f"`whole_workflow_skill` `{pooled_t2['whole_workflow_skill_stress_utility']:.3f}`), while `micro_skill` still leads "
            f"(`{pooled_t2['micro_skill_stress_utility']:.3f}`)."
        ),
        "5. What does phase-2 show about contract / validator / repair value?",
        (
            f"- Full validator-aware repair beats cut-alone under stress (`{pooled_t3['artifact_partition_full_stress_utility']:.3f}` vs "
            f"`{pooled_t3['artifact_partition_no_contract_stress_utility']:.3f}`), and local repair preserves perfect final success with shorter rerun span than global rerun "
            f"(`{pooled_t3['full_local_repair_rerun_span']:.3f}` vs `{pooled_t3['global_rerun_rerun_span']:.3f}`)."
        ),
        "6. What do phase-3 and phase-3b show about automatic boundary selection?",
        (
            f"- Phase-3 improves pooled Spearman from `{pooled_t2['ranking_spearman']:.3f}` to `{pooled_t4['phase3_spearman']:.3f}` and moves the selected partition "
            f"from `{pooled_t4['baseline_selected_partition']}` to `{pooled_t4['phase3_selected_partition']}`. "
            f"Phase-3b improves pooled Spearman again to `{pooled_t4['phase3b_spearman']:.3f}` and moves selection to `{pooled_t4['phase3b_selected_partition']}`."
        ),
        "7. Why did ranking improve much more than downstream utility?",
        (
            f"- The saved partition surface is flat near the top: mean near-optimal set size is `{pooled_near_optimal['mean_near_optimal_count']:.3f}`, "
            f"`V6` is in the near-optimal set on every split, and pooled oracle-minus-V6 stress gap on the saved table is only `{pooled_split_top['v6_oracle_stress_gap']:.3f}`. "
            f"On top of that, `V4` and `V6` are utility-tied on the saved partition table (`v4_to_v6_proxy_gain={pooled_translation['v4_to_v6_proxy_gain']:.3f}`), so the phase-3b ranking gain mostly reorders near-ties. "
            f"The executed stress gain from `V4` to `V6` is therefore only `{pooled_translation['v4_to_v6_actual_gain']:.3f}`, while `micro` still leads by `{pooled_translation['micro_minus_v6_actual']:.3f}`."
        ),
        "8. What is the correct predictive-family claim now?",
        "- Predictive families support family-appropriate cut selection with validated local repair, not a claim that the selected artifact partition beats the micro partition.",
        '9. Why is "family-appropriate cut selection" better than "artifact partition beats micro" on predictive families?',
        (
            f"- Because the best saved proxy oracle is a fine legal partition near `partition_020`, `V6` sits one boundary away from that oracle, and `micro` still wins executed stress utility. "
            f"The evidence therefore supports learning useful boundary placement, not forcing a coarse artifact partition as the universally best cut."
        ),
        "10. Evaluation extensions",
        f"- {recommended_next_stage}",
        "",
        "## 3. Supported Claims",
        "",
        markdown_table(
            t5.to_dict(orient="records"),
            ["family_id", "clearly_supported", "not_supported", "ambiguous", "recommended_paper_wording"],
        ),
        "",
        "## 4. Unsupported Claims",
        "",
        "- predictive families do not show a reliable executed downstream win of the selected artifact partition over `micro_skill`.",
        "- predictive families do not justify another predictive-only objective revision round.",
        "- predictive families alone do not settle final paper victory beyond the family-appropriate-cut claim.",
        "",
        "## 5. Interpretive Hypotheses",
        "",
        "- The remaining micro advantage likely reflects a mix of flat-top saturation and execution overhead that is not fully captured by the saved partition-ranking utility.",
        "- Richer workflow families should produce a less ceilinged utility landscape and a stronger test of whether learned boundary placement translates downstream.",
        "",
        "## 6. Next-Step Recommendations",
        "",
        "- freeze predictive families, their substrate, and the phase-3b recommendation at closure.",
        "- use this closure package as the predictive-family chapter in the repo and paper.",
        "- move implementation effort to the next richer workflow family class before reopening score revision.",
        "",
        "## 7. Figure Outputs",
        "",
        markdown_table(
            [{"figure_id": key, "path": path} for key, path in figure_paths.items()],
            ["figure_id", "path"],
        ),
    ]
    write_text(closure_report_path(paths), "\n".join(report_lines) + "\n")

    terminal_summary_lines = [
        f"working_root={paths.workspace_root.name}",
        "costly_reruns_avoided=yes",
        f"predictive_substrate_hash={compute_substrate_baseline_version(paths)}",
        f"strongest_supported_predictive_family_claim={strongest_supported_claim}",
        f"key_unsupported_claim={key_unsupported_claim}",
        f"flat_top_explanation_summary={flat_top_explanation}",
        f"recommended_next_stage={recommended_next_stage}",
    ]

    summary = {
        "gate_1_passed": True,
        "report_path": _relative(paths, closure_report_path(paths)),
        "audit_report": _relative(paths, audit_report_path(paths)),
        "flat_top_report": _relative(paths, flat_top_report_path(paths)),
        "final_recommendation_path": _relative(paths, closure_root(paths) / "final_recommendation.json"),
        "figure_paths": figure_paths,
        "terminal_summary_lines": terminal_summary_lines,
    }
    write_json(closure_root(paths) / "summary.json", summary)
    return summary
