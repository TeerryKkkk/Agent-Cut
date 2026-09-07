from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runtime.singlecell_workflow_evaluation import load_reference_trace, total_reference_wall_clock
from utils.io_utils import markdown_table, read_json, write_json, write_text
from utils.pathing import detect_project_paths
from validators.singlecell_mapping_roles import VALIDATOR_ROLES, validate_run_directory

FOLLOWUP_ROOT_NAME = "singlecell_robustness"
ANALYSIS_REPORT_NAME = "singlecell_neurips_analysis_followup.md"
VALIDATOR_RESCAN_REPORT_NAME = "singlecell_phase1_validator_rescan.md"

FAMILIES = ["scanpy_pancreas_ingest", "tabula_muris_label_transfer"]
VARIABLE_BOUNDARIES = ["prepared_query", "latent_or_graph", "predicted_labels", "mapping_metrics"]
UTILITY_IDS = ["transfer_first", "containment_first", "balanced"]
MAX_RERUN_SPAN = len(VALIDATOR_ROLES) - 1
RIDGE_LAMBDA = 1.0
BOOTSTRAP_REPS = 500
RNG_SEED = 0
FLOAT_TOLERANCE = 1.0e-12
BOUNDARY_COLORS = {
    "prepared_query": "#9c6644",
    "latent_or_graph": "#386641",
    "predicted_labels": "#0f4c5c",
    "mapping_metrics": "#bc4749",
}
FAMILY_COLORS = {
    "scanpy_pancreas_ingest": "#355070",
    "tabula_muris_label_transfer": "#6d597a",
    "pooled": "#2a9d8f",
}


def followup_root(paths) -> Path:
    return paths.results_dir / FOLLOWUP_ROOT_NAME


def analysis_tables_root(paths) -> Path:
    return followup_root(paths) / "analysis_tables"


def figures_root(paths) -> Path:
    return followup_root(paths) / "figures"


def validator_rescan_root(paths) -> Path:
    return followup_root(paths) / "validator_rescan"


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_ready(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return float(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return float(value)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _rank_relation(delta: float, *, tolerance: float = 1.0e-9) -> str:
    if abs(delta) <= tolerance:
        return "tie"
    return "left_gt_right" if delta > 0.0 else "right_gt_left"


def _metric_fidelity(metric_gap_to_reference: float | None) -> float:
    if metric_gap_to_reference is None:
        return 0.0
    return max(0.0, 1.0 - abs(float(metric_gap_to_reference)))


def _rerun_efficiency(rerun_span: float | None) -> float:
    if rerun_span is None:
        return 0.0
    return max(0.0, 1.0 - (float(rerun_span) / float(MAX_RERUN_SPAN)))


def _runtime_efficiency(wall_clock_seconds: float, reference_wall_clock_seconds: float) -> float:
    return 1.0 / (1.0 + (float(wall_clock_seconds) / max(float(reference_wall_clock_seconds), 1.0e-6)))


def _localization_from_depth(first_failed_boundary_depth: float | None, *, rerun_span: float | None, success: bool) -> float:
    if first_failed_boundary_depth is not None:
        return max(0.0, 1.0 - ((float(first_failed_boundary_depth) - 1.0) / float(MAX_RERUN_SPAN)))
    if success and (rerun_span is None or float(rerun_span) == 0.0):
        return 1.0
    return 0.0


def _localization_score(
    *,
    localization_value: float | bool | None,
    first_failed_boundary_depth: float | None,
    rerun_span: float | None,
    success: bool,
) -> float:
    if localization_value is not None and not (isinstance(localization_value, float) and pd.isna(localization_value)):
        return float(_safe_bool(localization_value))
    return _localization_from_depth(first_failed_boundary_depth, rerun_span=rerun_span, success=success)


def _artifact_validity_from_validation_results(payload: Any) -> float:
    parsed = _parse_payload(payload)
    if not isinstance(parsed, dict) or not parsed:
        return 0.0
    values = [float(_safe_bool(result.get("passed"))) for result in parsed.values() if isinstance(result, dict)]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _utility_formula_rows() -> pd.DataFrame:
    rows = [
        {
            "row_type": "component",
            "utility_id": "",
            "component_id": "success_score",
            "formula_text": "success_score = success or final_success cast to {0,1}",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "For E4, success uses final_success.",
        },
        {
            "row_type": "component",
            "utility_id": "",
            "component_id": "artifact_validity_score",
            "formula_text": "artifact_validity_score = artifact_validity_rate; if absent, mean(passed over validation_results_json roles)",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "E4 reconstructs final artifact validity from saved validator outputs.",
        },
        {
            "row_type": "component",
            "utility_id": "",
            "component_id": "metric_fidelity_score",
            "formula_text": "metric_fidelity_score = max(0, 1 - abs(metric_gap_to_reference))",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "E4 uses final_metric_gap_to_reference.",
        },
        {
            "row_type": "component",
            "utility_id": "",
            "component_id": "rerun_efficiency_score",
            "formula_text": f"rerun_efficiency_score = max(0, 1 - rerun_span / {MAX_RERUN_SPAN})",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "Uses the frozen six-role single-cell template, so the maximum rerun span is five.",
        },
        {
            "row_type": "component",
            "utility_id": "",
            "component_id": "runtime_efficiency_score",
            "formula_text": "runtime_efficiency_score = 1 / (1 + elapsed_seconds / reference_wall_clock_seconds)",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "E4 uses extra_wall_clock_seconds against the same saved reference run wall clock for that split.",
        },
        {
            "row_type": "component",
            "utility_id": "",
            "component_id": "localization_score",
            "formula_text": "localization_score = root_localization_accuracy if available; else 1 - (first_failed_boundary_depth - 1) / 5; if no failure and rerun_span = 0, localization_score = 1; if localization is absent under stress, localization_score = 0",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "The depth fallback only applies when root-localization accuracy is not saved directly.",
        },
        {
            "row_type": "utility",
            "utility_id": "transfer_first",
            "component_id": "",
            "formula_text": "(success_score + artifact_validity_score + metric_fidelity_score) / 3",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "Transfer quality only; no containment components enter the score.",
        },
        {
            "row_type": "utility",
            "utility_id": "containment_first",
            "component_id": "",
            "formula_text": "(rerun_efficiency_score + runtime_efficiency_score + localization_score) / 3",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "Containment quality only; no transfer-fidelity components enter the score.",
        },
        {
            "row_type": "utility",
            "utility_id": "balanced",
            "component_id": "",
            "formula_text": "(success_score + artifact_validity_score + metric_fidelity_score + rerun_efficiency_score + runtime_efficiency_score + localization_score) / 6",
            "applicable_contexts": "E1_executed; E2_partition_ranking; E3_ablation; E4_repair",
            "notes": "Equal-weight composite over transfer and containment components.",
        },
    ]
    return pd.DataFrame(rows)


def _attach_utility_variants(
    frame: pd.DataFrame,
    *,
    success_column: str,
    validity_column: str,
    gap_column: str,
    rerun_span_column: str,
    wall_clock_column: str,
    reference_wall_clock_column: str,
    first_failed_boundary_depth_column: str | None = None,
    localization_column: str | None = None,
) -> pd.DataFrame:
    enriched = frame.copy()
    success_values = enriched[success_column].apply(_safe_bool).astype(float)
    validity_values = enriched[validity_column].apply(_safe_float)
    gap_values = enriched[gap_column].apply(_optional_float)
    rerun_values = enriched[rerun_span_column].apply(_optional_float)
    elapsed_values = enriched[wall_clock_column].apply(_safe_float)
    reference_values = enriched[reference_wall_clock_column].apply(_safe_float)
    if first_failed_boundary_depth_column is not None and first_failed_boundary_depth_column in enriched.columns:
        depth_values = enriched[first_failed_boundary_depth_column].apply(_optional_float)
    else:
        depth_values = pd.Series([None] * len(enriched), index=enriched.index)
    if localization_column is not None and localization_column in enriched.columns:
        localization_values = enriched[localization_column]
    else:
        localization_values = pd.Series([None] * len(enriched), index=enriched.index)

    enriched["component_success_score"] = success_values
    enriched["component_artifact_validity_score"] = validity_values
    enriched["component_metric_fidelity_score"] = gap_values.apply(_metric_fidelity)
    enriched["component_rerun_efficiency_score"] = rerun_values.apply(_rerun_efficiency)
    enriched["component_runtime_efficiency_score"] = [
        _runtime_efficiency(elapsed, reference)
        for elapsed, reference in zip(elapsed_values.tolist(), reference_values.tolist())
    ]
    enriched["component_localization_score"] = [
        _localization_score(
            localization_value=localization,
            first_failed_boundary_depth=depth,
            rerun_span=rerun_span,
            success=_safe_bool(success),
        )
        for localization, depth, rerun_span, success in zip(
            localization_values.tolist(),
            depth_values.tolist(),
            rerun_values.tolist(),
            enriched[success_column].tolist(),
        )
    ]
    enriched["utility__transfer_first"] = (
        enriched["component_success_score"]
        + enriched["component_artifact_validity_score"]
        + enriched["component_metric_fidelity_score"]
    ) / 3.0
    enriched["utility__containment_first"] = (
        enriched["component_rerun_efficiency_score"]
        + enriched["component_runtime_efficiency_score"]
        + enriched["component_localization_score"]
    ) / 3.0
    enriched["utility__balanced"] = (
        enriched["component_success_score"]
        + enriched["component_artifact_validity_score"]
        + enriched["component_metric_fidelity_score"]
        + enriched["component_rerun_efficiency_score"]
        + enriched["component_runtime_efficiency_score"]
        + enriched["component_localization_score"]
    ) / 6.0
    return enriched


def _reference_wall_clock_lookup(paths) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for family_id in FAMILIES:
        per_run_path = paths.results_dir / "singlecell_workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv"
        frame = pd.read_csv(per_run_path)
        family_rows = frame[frame["family_id"] == family_id][["split_id", "heldout_task"]].drop_duplicates()
        for row in family_rows.to_dict(orient="records"):
            lookup[(family_id, row["split_id"])] = total_reference_wall_clock(
                load_reference_trace(paths, family_id, row["heldout_task"])
            )
    return lookup


def _load_partition_metadata(paths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id in FAMILIES:
        partitions = read_json(paths.results_dir / "partitions" / family_id / "legal_partitions.json")
        selected_id = read_json(paths.results_dir / "partitions" / family_id / "selected_partition.json")["partition_id"]
        micro_partition = min(partitions, key=lambda row: (-int(row["segment_count"]), str(row["partition_id"])))
        whole_partition = min(partitions, key=lambda row: (int(row["segment_count"]), str(row["partition_id"])))
        for partition in partitions:
            row = {
                "family_id": family_id,
                "partition_id": partition["partition_id"],
                "segment_count": int(partition["segment_count"]),
                "boundary_roles_json": _json_ready(partition["boundary_roles"]),
                "segment_ids_json": _json_ready(partition["segment_ids"]),
                "selected_partition_id_fixed": selected_id,
                "micro_partition_id_fixed": micro_partition["partition_id"],
                "whole_partition_id_fixed": whole_partition["partition_id"],
            }
            for boundary_role in VARIABLE_BOUNDARIES:
                row[boundary_role] = int(boundary_role in partition["boundary_roles"])
            rows.append(row)
    return pd.DataFrame(rows)


def _load_e2_partition_frame(paths, partition_metadata: pd.DataFrame) -> pd.DataFrame:
    partition_path = paths.results_dir / "singlecell_workflow_evaluation" / "boundary_ranking" / "per_partition_results.csv"
    frame = pd.read_csv(partition_path)
    merged = frame.merge(
        partition_metadata,
        on=["family_id", "partition_id", "segment_count"],
        how="left",
    )
    merged["selected_partition_id"] = merged["selected_partition_id"].fillna(merged["selected_partition_id_fixed"])
    merged["micro_partition_id"] = merged["micro_partition_id"].fillna(merged["micro_partition_id_fixed"])
    return merged


def _family_and_pooled_frames(frame: pd.DataFrame):
    for family_id in FAMILIES:
        yield family_id, frame[frame["family_id"] == family_id].copy()
    yield "pooled", frame.copy()


def _tie_rank(frame: pd.DataFrame, value_column: str, selected_value: float) -> int:
    return 1 + int((frame[value_column] > (float(selected_value) + FLOAT_TOLERANCE)).sum())


def _ordinal_rank(frame: pd.DataFrame, id_column: str, target_id: str) -> int:
    positions = frame.index[frame[id_column] == target_id].tolist()
    if not positions:
        raise KeyError(f"Could not locate {target_id} in ranking frame.")
    return int(positions[0] + 1)


def _same_cardinality_analysis(paths, e2_partition_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    for (family_id, split_id), split_frame in e2_partition_frame.groupby(["family_id", "split_id"], dropna=False):
        ordered = split_frame.sort_values(["utility", "partition_score", "partition_id"], ascending=[False, False, True]).reset_index(drop=True)
        selected_id = split_frame["selected_partition_id"].iloc[0]
        micro_id = split_frame["micro_partition_id"].iloc[0]
        whole_id = split_frame["whole_partition_id_fixed"].iloc[0]
        selected_row = split_frame[split_frame["partition_id"] == selected_id].iloc[0]
        same_bucket = split_frame[split_frame["segment_count"] == int(selected_row["segment_count"])].sort_values(
            ["utility", "partition_score", "partition_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        best_same_utility = float(same_bucket["utility"].max())
        best_same_ids = same_bucket.loc[
            np.isclose(same_bucket["utility"], best_same_utility, atol=1.0e-9),
            "partition_id",
        ].tolist()
        micro_row = split_frame[split_frame["partition_id"] == micro_id].iloc[0]
        whole_row = split_frame[split_frame["partition_id"] == whole_id].iloc[0]
        selected_global_tie_rank = _tie_rank(ordered, "utility", float(selected_row["utility"]))
        micro_global_tie_rank = _tie_rank(ordered, "utility", float(micro_row["utility"]))
        whole_global_tie_rank = _tie_rank(ordered, "utility", float(whole_row["utility"]))
        split_rows.append(
            {
                "family_id": family_id,
                "split_id": split_id,
                "heldout_task": split_frame["heldout_task"].iloc[0],
                "selected_partition_id": selected_id,
                "selected_segment_count": int(selected_row["segment_count"]),
                "same_cardinality_bucket_size": int(len(same_bucket)),
                "selected_same_cardinality_tie_rank": _tie_rank(same_bucket, "utility", float(selected_row["utility"])),
                "selected_same_cardinality_ordinal_rank": _ordinal_rank(same_bucket, "partition_id", selected_id),
                "selected_same_cardinality_regret": round(best_same_utility - float(selected_row["utility"]), 6),
                "best_same_cardinality_partition_ids_json": _json_ready(best_same_ids),
                "best_same_cardinality_utility": round(best_same_utility, 6),
                "selected_utility": round(float(selected_row["utility"]), 6),
                "selected_global_tie_rank": selected_global_tie_rank,
                "selected_global_ordinal_rank": _ordinal_rank(ordered, "partition_id", selected_id),
                "micro_partition_id": micro_id,
                "micro_utility": round(float(micro_row["utility"]), 6),
                "micro_global_tie_rank": micro_global_tie_rank,
                "micro_global_ordinal_rank": _ordinal_rank(ordered, "partition_id", micro_id),
                "whole_partition_id": whole_id,
                "whole_utility": round(float(whole_row["utility"]), 6),
                "whole_global_tie_rank": whole_global_tie_rank,
                "whole_global_ordinal_rank": _ordinal_rank(ordered, "partition_id", whole_id),
                "selected_minus_micro": round(float(selected_row["utility"]) - float(micro_row["utility"]), 6),
                "selected_minus_whole": round(float(selected_row["utility"]) - float(whole_row["utility"]), 6),
                "selected_tied_best_same_cardinality": bool(abs(best_same_utility - float(selected_row["utility"])) <= 1.0e-9),
            }
        )
    split_summary = pd.DataFrame(split_rows).sort_values(["family_id", "split_id"]).reset_index(drop=True)

    family_rows: list[dict[str, Any]] = []
    for family_id, frame in _family_and_pooled_frames(split_summary):
        family_rows.append(
            {
                "family_id": family_id,
                "split_count": int(len(frame)),
                "mean_selected_same_cardinality_tie_rank": round(float(frame["selected_same_cardinality_tie_rank"].mean()), 6),
                "mean_selected_same_cardinality_regret": round(float(frame["selected_same_cardinality_regret"].mean()), 6),
                "share_selected_tied_best_same_cardinality": round(float(frame["selected_tied_best_same_cardinality"].mean()), 6),
                "mean_selected_global_tie_rank": round(float(frame["selected_global_tie_rank"].mean()), 6),
                "mean_micro_global_tie_rank": round(float(frame["micro_global_tie_rank"].mean()), 6),
                "mean_whole_global_tie_rank": round(float(frame["whole_global_tie_rank"].mean()), 6),
                "mean_selected_minus_micro": round(float(frame["selected_minus_micro"].mean()), 6),
                "mean_selected_minus_whole": round(float(frame["selected_minus_whole"].mean()), 6),
                "same_cardinality_bucket_size": int(frame["same_cardinality_bucket_size"].iloc[0]),
            }
        )
    family_summary = pd.DataFrame(family_rows)
    _save_dataframe(split_summary, analysis_tables_root(paths) / "same_cardinality_per_split.csv")
    _save_dataframe(family_summary, analysis_tables_root(paths) / "same_cardinality_family_summary.csv")
    _plot_same_cardinality(split_summary, figures_root(paths) / "same_cardinality_rank_plot.png")
    return split_summary, family_summary


def _plot_same_cardinality(summary_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    family_positions = {family_id: index for index, family_id in enumerate(FAMILIES)}
    for family_id in FAMILIES:
        subset = summary_frame[summary_frame["family_id"] == family_id].reset_index(drop=True)
        x_positions = [family_positions[family_id] + (index - (len(subset) - 1) / 2.0) * 0.05 for index in range(len(subset))]
        axes[0].scatter(
            x_positions,
            subset["selected_same_cardinality_tie_rank"],
            color=FAMILY_COLORS[family_id],
            s=45,
            label=family_id,
        )
        axes[1].scatter(
            x_positions,
            subset["selected_same_cardinality_regret"],
            color=FAMILY_COLORS[family_id],
            s=45,
            label=family_id,
        )
    axes[0].axhline(1.0, color="#6c757d", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(list(family_positions.values()), [family_id.replace("_", "\n") for family_id in FAMILIES])
    axes[0].set_ylabel("selected tie-rank within 4-segment bucket")
    axes[0].set_ylim(0.75, max(2.25, float(summary_frame["selected_same_cardinality_tie_rank"].max()) + 0.25))
    axes[1].axhline(0.0, color="#6c757d", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(list(family_positions.values()), [family_id.replace("_", "\n") for family_id in FAMILIES])
    axes[1].set_ylabel("selected regret to best same-cardinality peer")
    axes[1].set_ylim(
        min(-0.005, float(summary_frame["selected_same_cardinality_regret"].min()) - 0.005),
        max(0.01, float(summary_frame["selected_same_cardinality_regret"].max()) + 0.01),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _ridge_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    design = frame.copy()
    design["segment_count_centered"] = design["segment_count"] - float(design["segment_count"].mean())
    design["family__tabula_muris_label_transfer"] = (design["family_id"] == "tabula_muris_label_transfer").astype(float)
    columns = ["intercept", "segment_count_centered", "family__tabula_muris_label_transfer", *VARIABLE_BOUNDARIES]
    matrix = np.column_stack(
        [
            np.ones(len(design), dtype=float),
            design["segment_count_centered"].to_numpy(dtype=float),
            design["family__tabula_muris_label_transfer"].to_numpy(dtype=float),
            *[design[column].to_numpy(dtype=float) for column in VARIABLE_BOUNDARIES],
        ]
    )
    return matrix, columns


def _fit_ridge(frame: pd.DataFrame, *, outcome_column: str) -> dict[str, float]:
    matrix, columns = _ridge_design(frame)
    outcome = frame[outcome_column].to_numpy(dtype=float)
    penalty = RIDGE_LAMBDA * np.eye(matrix.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ outcome)
    return {column: float(value) for column, value in zip(columns, coefficients)}


def _bootstrap_ridge_coefficients(frame: pd.DataFrame, *, outcome_column: str) -> pd.DataFrame:
    split_groups = {
        split_key: split_frame.copy()
        for split_key, split_frame in frame.groupby(["family_id", "split_id"], dropna=False)
    }
    split_keys = list(split_groups)
    rng = np.random.default_rng(RNG_SEED)
    records: list[dict[str, Any]] = []
    for _ in range(BOOTSTRAP_REPS):
        sampled_indices = rng.integers(0, len(split_keys), size=len(split_keys))
        boot_frame = pd.concat([split_groups[split_keys[index]] for index in sampled_indices], ignore_index=True)
        coefficients = _fit_ridge(boot_frame, outcome_column=outcome_column)
        records.append(coefficients)
    return pd.DataFrame(records)


def _conditional_mean_difference(frame: pd.DataFrame, boundary_role: str, *, outcome_column: str) -> float:
    differences: list[float] = []
    for _, subset in frame.groupby(["family_id", "split_id", "segment_count"], dropna=False):
        present = subset[subset[boundary_role] == 1][outcome_column]
        absent = subset[subset[boundary_role] == 0][outcome_column]
        if present.empty or absent.empty:
            continue
        differences.append(float(present.mean()) - float(absent.mean()))
    if not differences:
        return 0.0
    return float(np.mean(differences))


def _boundary_attribution(paths, e2_partition_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = _fit_ridge(e2_partition_frame, outcome_column="utility")
    bootstrap_frame = _bootstrap_ridge_coefficients(e2_partition_frame, outcome_column="utility")
    rows: list[dict[str, Any]] = []
    for term in ["segment_count_centered", "family__tabula_muris_label_transfer", *VARIABLE_BOUNDARIES]:
        bootstrap_values = bootstrap_frame[term].to_numpy(dtype=float)
        row = {
            "term": term,
            "ridge_coefficient": round(float(coefficients[term]), 6),
            "bootstrap_ci_low": round(float(np.quantile(bootstrap_values, 0.025)), 6),
            "bootstrap_ci_high": round(float(np.quantile(bootstrap_values, 0.975)), 6),
            "bootstrap_positive_share": round(float((bootstrap_values > 0.0).mean()), 6),
            "bootstrap_negative_share": round(float((bootstrap_values < 0.0).mean()), 6),
            "is_boundary_term": bool(term in VARIABLE_BOUNDARIES),
            "segment_count_controlled_mean_difference": round(
                _conditional_mean_difference(e2_partition_frame, term, outcome_column="utility") if term in VARIABLE_BOUNDARIES else 0.0,
                6,
            ),
        }
        rows.append(row)
    coefficient_frame = pd.DataFrame(rows)
    recurrence_frame = _boundary_topk_recurrence(e2_partition_frame)
    _save_dataframe(coefficient_frame, analysis_tables_root(paths) / "boundary_attribution_coefficients.csv")
    _save_dataframe(recurrence_frame, analysis_tables_root(paths) / "boundary_topk_recurrence.csv")
    _plot_boundary_coefficients(coefficient_frame, figures_root(paths) / "boundary_attribution_effects.png")
    _plot_boundary_recurrence(recurrence_frame, figures_root(paths) / "boundary_topk_recurrence.png")
    return coefficient_frame, recurrence_frame


def _boundary_topk_recurrence(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id, subset in _family_and_pooled_frames(frame):
        split_ids = sorted(subset["split_id"].dropna().unique().tolist())
        if not split_ids:
            continue
        baseline_prevalence = {boundary_role: float(subset[boundary_role].mean()) for boundary_role in VARIABLE_BOUNDARIES}
        for k in [1, 3]:
            total_slots = 0
            present_slot_counts = {boundary_role: 0 for boundary_role in VARIABLE_BOUNDARIES}
            split_any_counts = {boundary_role: 0 for boundary_role in VARIABLE_BOUNDARIES}
            for _, split_frame in subset.groupby("split_id", dropna=False):
                topk = split_frame.sort_values(["utility", "partition_score", "partition_id"], ascending=[False, False, True]).head(k)
                total_slots += len(topk)
                for boundary_role in VARIABLE_BOUNDARIES:
                    present_count = int(topk[boundary_role].sum())
                    present_slot_counts[boundary_role] += present_count
                    split_any_counts[boundary_role] += int(present_count > 0)
            for boundary_role in VARIABLE_BOUNDARIES:
                rows.append(
                    {
                        "family_id": family_id,
                        "k": k,
                        "boundary_role": boundary_role,
                        "slot_recurrence_rate": round(float(present_slot_counts[boundary_role] / max(total_slots, 1)), 6),
                        "split_coverage_rate": round(float(split_any_counts[boundary_role] / len(split_ids)), 6),
                        "baseline_prevalence": round(baseline_prevalence[boundary_role], 6),
                        "slot_lift_over_baseline": round(
                            float((present_slot_counts[boundary_role] / max(total_slots, 1)) - baseline_prevalence[boundary_role]),
                            6,
                        ),
                        "num_splits": int(len(split_ids)),
                        "total_slots": int(total_slots),
                    }
                )
    return pd.DataFrame(rows).sort_values(["family_id", "k", "boundary_role"]).reset_index(drop=True)


def _plot_boundary_coefficients(frame: pd.DataFrame, output_path: Path) -> None:
    boundary_frame = frame[frame["is_boundary_term"]].copy()
    boundary_frame = boundary_frame.sort_values("ridge_coefficient", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    y_positions = np.arange(len(boundary_frame))
    colors = [BOUNDARY_COLORS.get(term, "#6c757d") for term in boundary_frame["term"]]
    coefficients = boundary_frame["ridge_coefficient"].to_numpy(dtype=float)
    lower_errors = coefficients - boundary_frame["bootstrap_ci_low"].to_numpy(dtype=float)
    upper_errors = boundary_frame["bootstrap_ci_high"].to_numpy(dtype=float) - coefficients
    ax.barh(y_positions, coefficients, color=colors, alpha=0.85)
    ax.errorbar(
        coefficients,
        y_positions,
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor="#1f2933",
        capsize=3,
        linewidth=1.0,
    )
    ax.axvline(0.0, color="#6c757d", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions, boundary_frame["term"].tolist())
    ax.set_xlabel("ridge utility effect after family + segment-count control")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_boundary_recurrence(frame: pd.DataFrame, output_path: Path) -> None:
    pooled = frame[frame["family_id"] == "pooled"].copy()
    top1 = pooled[pooled["k"] == 1].set_index("boundary_role")
    top3 = pooled[pooled["k"] == 3].set_index("boundary_role")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    roles = VARIABLE_BOUNDARIES
    colors = [BOUNDARY_COLORS[role] for role in roles]
    axes[0].bar(roles, [float(top1.loc[role, "slot_recurrence_rate"]) for role in roles], color=colors)
    axes[0].bar(
        roles,
        [float(top1.loc[role, "baseline_prevalence"]) for role in roles],
        fill=False,
        edgecolor="#1f2933",
        linewidth=1.2,
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("top-1 slot recurrence")
    axes[1].bar(roles, [float(top3.loc[role, "slot_recurrence_rate"]) for role in roles], color=colors)
    axes[1].bar(
        roles,
        [float(top3.loc[role, "baseline_prevalence"]) for role in roles],
        fill=False,
        edgecolor="#1f2933",
        linewidth=1.2,
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("top-3 slot recurrence")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _e1_stress_variant_frame(paths) -> pd.DataFrame:
    frame = pd.read_csv(paths.results_dir / "singlecell_workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv")
    stress = frame[frame["clean_or_stress"] == "stress"].copy()
    return _attach_utility_variants(
        stress,
        success_column="success",
        validity_column="artifact_validity_rate",
        gap_column="metric_gap_to_reference",
        rerun_span_column="rerun_span",
        wall_clock_column="wall_clock_seconds",
        reference_wall_clock_column="reference_wall_clock_seconds",
        first_failed_boundary_depth_column="first_failed_boundary_depth",
    )


def _e2_stress_partition_variants(paths, partition_metadata: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(paths.results_dir / "singlecell_workflow_evaluation" / "boundary_ranking" / "per_run_results.csv")
    stress = frame[frame["clean_or_stress"] == "stress"].copy()
    stress = stress.merge(
        partition_metadata,
        on=["family_id", "partition_id", "segment_count"],
        how="left",
    )
    stress = _attach_utility_variants(
        stress,
        success_column="success",
        validity_column="artifact_validity_rate",
        gap_column="metric_gap_to_reference",
        rerun_span_column="rerun_span",
        wall_clock_column="wall_clock_seconds",
        reference_wall_clock_column="reference_wall_clock_seconds",
        first_failed_boundary_depth_column="first_failed_boundary_depth",
    )
    group_columns = [
        "family_id",
        "split_id",
        "heldout_task",
        "partition_id",
        "partition_score",
        "segment_count",
        "selected_partition_id",
        "micro_partition_id",
        "boundary_roles_json",
        "segment_ids_json",
        "prepared_query",
        "latent_or_graph",
        "predicted_labels",
        "mapping_metrics",
        "whole_partition_id_fixed",
    ]
    aggregate_map = {f"utility__{utility_id}": "mean" for utility_id in UTILITY_IDS}
    grouped = stress.groupby(group_columns, dropna=False).agg(aggregate_map).reset_index()
    return grouped.sort_values(["family_id", "split_id", "partition_id"]).reset_index(drop=True)


def _e3_variant_frame(paths) -> pd.DataFrame:
    frame = pd.read_csv(paths.results_dir / "singlecell_repair_evaluation" / "ablation" / "per_run_results.csv")
    stress = frame[frame["clean_or_stress"] == "stress"].copy()
    reference_lookup = _reference_wall_clock_lookup(paths)
    stress["reference_wall_clock_seconds"] = [
        reference_lookup[(family_id, split_id)]
        for family_id, split_id in zip(stress["family_id"].tolist(), stress["split_id"].tolist())
    ]
    return _attach_utility_variants(
        stress,
        success_column="success",
        validity_column="artifact_validity_rate",
        gap_column="metric_gap_to_reference",
        rerun_span_column="rerun_span",
        wall_clock_column="wall_clock_seconds",
        reference_wall_clock_column="reference_wall_clock_seconds",
        first_failed_boundary_depth_column="first_failed_boundary_depth",
    )


def _e4_variant_frame(paths) -> pd.DataFrame:
    frame = pd.read_csv(paths.results_dir / "singlecell_repair_evaluation" / "induced_fault" / "per_run_results.csv")
    reference_lookup = _reference_wall_clock_lookup(paths)
    enriched = frame.copy()
    enriched["artifact_validity_rate_final"] = enriched["validation_results_json"].apply(_artifact_validity_from_validation_results)
    enriched["reference_wall_clock_seconds"] = [
        reference_lookup[(family_id, split_id)]
        for family_id, split_id in zip(enriched["family_id"].tolist(), enriched["split_id"].tolist())
    ]
    return _attach_utility_variants(
        enriched,
        success_column="final_success",
        validity_column="artifact_validity_rate_final",
        gap_column="final_metric_gap_to_reference",
        rerun_span_column="rerun_span",
        wall_clock_column="extra_wall_clock_seconds",
        reference_wall_clock_column="reference_wall_clock_seconds",
        localization_column="root_localization_accuracy",
    )


def _ranking_summaries(partition_frame: pd.DataFrame, *, utility_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for (family_id, split_id), split_frame in partition_frame.groupby(["family_id", "split_id"], dropna=False):
        scored = split_frame.sort_values(["partition_score", "segment_count", "partition_id"], ascending=[False, True, True]).reset_index(drop=True)
        utility_sorted = split_frame.sort_values([utility_column, "partition_score", "partition_id"], ascending=[False, False, True]).reset_index(drop=True)
        predicted_best = scored.iloc[0]
        oracle_best = utility_sorted.iloc[0]
        top3_ids = set(scored.head(3)["partition_id"].tolist())
        if (
            split_frame["partition_score"].nunique(dropna=False) <= 1
            or split_frame[utility_column].nunique(dropna=False) <= 1
        ):
            spearman_value = None
            kendall_value = None
        else:
            spearman_value = spearmanr(split_frame["partition_score"], split_frame[utility_column]).statistic
            kendall_value = kendalltau(split_frame["partition_score"], split_frame[utility_column]).statistic
        selected_row = split_frame[split_frame["partition_id"] == split_frame["selected_partition_id"].iloc[0]].iloc[0]
        micro_row = split_frame[split_frame["partition_id"] == split_frame["micro_partition_id"].iloc[0]].iloc[0]
        split_rows.append(
            {
                "family_id": family_id,
                "split_id": split_id,
                "heldout_task": split_frame["heldout_task"].iloc[0],
                "predicted_best_partition": predicted_best["partition_id"],
                "oracle_best_partition": oracle_best["partition_id"],
                "predicted_best_score": float(predicted_best["partition_score"]),
                "predicted_best_utility": float(predicted_best[utility_column]),
                "oracle_best_utility": float(oracle_best[utility_column]),
                "top1_regret": float(oracle_best[utility_column]) - float(predicted_best[utility_column]),
                "top3_hit": float(oracle_best["partition_id"] in top3_ids),
                "spearman": None if pd.isna(spearman_value) else float(spearman_value),
                "kendall_tau": None if pd.isna(kendall_value) else float(kendall_value),
            }
        )
        selected_rows.append(
            {
                "family_id": family_id,
                "split_id": split_id,
                "heldout_task": split_frame["heldout_task"].iloc[0],
                "selected_partition_id": selected_row["partition_id"],
                "selected_utility": float(selected_row[utility_column]),
                "micro_partition_id": micro_row["partition_id"],
                "micro_utility": float(micro_row[utility_column]),
                "oracle_partition_id": oracle_best["partition_id"],
                "oracle_utility": float(oracle_best[utility_column]),
                "selected_top1_regret": float(oracle_best[utility_column]) - float(selected_row[utility_column]),
                "micro_top1_regret": float(oracle_best[utility_column]) - float(micro_row[utility_column]),
                "selected_minus_micro": float(selected_row[utility_column]) - float(micro_row[utility_column]),
                "selected_minus_oracle": float(selected_row[utility_column]) - float(oracle_best[utility_column]),
            }
        )
    split_summary = pd.DataFrame(split_rows).sort_values(["family_id", "split_id"]).reset_index(drop=True)
    selected_summary = pd.DataFrame(selected_rows).sort_values(["family_id", "split_id"]).reset_index(drop=True)
    family_summary = (
        split_summary.groupby("family_id", dropna=False)[["spearman", "kendall_tau", "top1_regret", "top3_hit"]]
        .mean(numeric_only=True)
        .reset_index()
        .merge(
            selected_summary.groupby("family_id", dropna=False)[
                ["selected_utility", "micro_utility", "oracle_utility", "selected_top1_regret", "micro_top1_regret", "selected_minus_micro", "selected_minus_oracle"]
            ]
            .mean(numeric_only=True)
            .reset_index(),
            on="family_id",
            how="inner",
        )
    )
    pooled_row = {
        "family_id": "pooled",
        "spearman": float(split_summary["spearman"].mean()),
        "kendall_tau": float(split_summary["kendall_tau"].mean()),
        "top1_regret": float(split_summary["top1_regret"].mean()),
        "top3_hit": float(split_summary["top3_hit"].mean()),
        "selected_utility": float(selected_summary["selected_utility"].mean()),
        "micro_utility": float(selected_summary["micro_utility"].mean()),
        "oracle_utility": float(selected_summary["oracle_utility"].mean()),
        "selected_top1_regret": float(selected_summary["selected_top1_regret"].mean()),
        "micro_top1_regret": float(selected_summary["micro_top1_regret"].mean()),
        "selected_minus_micro": float(selected_summary["selected_minus_micro"].mean()),
        "selected_minus_oracle": float(selected_summary["selected_minus_oracle"].mean()),
    }
    family_summary = pd.concat([family_summary, pd.DataFrame([pooled_row])], ignore_index=True)
    return split_summary, selected_summary, family_summary


def _summarize_condition_frame(
    frame: pd.DataFrame,
    *,
    context: str,
    family_summary_columns: dict[str, str],
    split_group_columns: list[str],
    condition_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for utility_id in UTILITY_IDS:
        utility_column = f"utility__{utility_id}"
        split_condition = (
            frame.groupby(split_group_columns + [condition_column], dropna=False)[utility_column]
            .mean()
            .reset_index()
        )
        for (family_id, split_id), split_frame in split_condition.groupby(["family_id", "split_id"], dropna=False):
            row: dict[str, Any] = {
                "context": context,
                "utility_id": utility_id,
                "family_id": family_id,
                "split_id": split_id,
                "heldout_task": split_frame["heldout_task"].iloc[0],
            }
            means = split_frame.set_index(condition_column)[utility_column]
            for output_column, condition_id in family_summary_columns.items():
                row[output_column] = _optional_float(means.get(condition_id))
            if "selected_utility" in row and "micro_utility" in row:
                row["selected_minus_micro"] = (
                    None
                    if row["selected_utility"] is None or row["micro_utility"] is None
                    else float(row["selected_utility"]) - float(row["micro_utility"])
                )
            split_rows.append(row)
        for family_id, family_frame in _family_and_pooled_frames(split_condition):
            row = {
                "context": context,
                "utility_id": utility_id,
                "family_id": family_id,
            }
            means = family_frame.groupby(condition_column, dropna=False)[utility_column].mean()
            for output_column, condition_id in family_summary_columns.items():
                row[output_column] = _optional_float(means.get(condition_id))
            ranked = means.sort_values(ascending=False)
            row["best_label"] = ranked.index[0] if not ranked.empty else ""
            if "selected_utility" in row and "micro_utility" in row and row["selected_utility"] is not None and row["micro_utility"] is not None:
                row["selected_minus_micro"] = float(row["selected_utility"]) - float(row["micro_utility"])
            if "selected_utility" in row and "artifact_partition_no_contract_utility" in row:
                left = row["selected_utility"]
                right = row["artifact_partition_no_contract_utility"]
                row["full_minus_no_contract"] = None if left is None or right is None else float(left) - float(right)
            if "artifact_partition_full_utility" in row and "artifact_partition_no_contract_utility" in row:
                left = row["artifact_partition_full_utility"]
                right = row["artifact_partition_no_contract_utility"]
                row["full_minus_no_contract"] = None if left is None or right is None else float(left) - float(right)
            if "artifact_partition_full_utility" in row and "artifact_partition_contract_validator_no_repair_utility" in row:
                left = row["artifact_partition_full_utility"]
                right = row["artifact_partition_contract_validator_no_repair_utility"]
                row["full_minus_validator_no_repair"] = None if left is None or right is None else float(left) - float(right)
            if "full_local_repair_utility" in row and "global_end_to_end_rerun_utility" in row:
                left = row["full_local_repair_utility"]
                right = row["global_end_to_end_rerun_utility"]
                row["local_minus_global"] = None if left is None or right is None else float(left) - float(right)
            family_rows.append(row)
    return pd.DataFrame(family_rows), pd.DataFrame(split_rows)


def _utility_robustness(
    paths,
    partition_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    formulas = _utility_formula_rows()
    _save_dataframe(formulas, analysis_tables_root(paths) / "utility_formulas.csv")

    e1_frame = _e1_stress_variant_frame(paths)
    e2_frame = _e2_stress_partition_variants(paths, partition_metadata)
    e3_frame = _e3_variant_frame(paths)
    e4_frame = _e4_variant_frame(paths)

    family_tables: list[pd.DataFrame] = []
    split_tables: list[pd.DataFrame] = []

    e1_family, e1_split = _summarize_condition_frame(
        e1_frame,
        context="E1_executed",
        family_summary_columns={
            "selected_utility": "artifact_partition_full",
            "micro_utility": "micro_skill",
            "artifact_partition_no_contract_utility": "artifact_partition_no_contract",
            "structured_textual_memory_utility": "structured_textual_memory",
            "whole_workflow_skill_utility": "whole_workflow_skill",
        },
        split_group_columns=["family_id", "split_id", "heldout_task"],
        condition_column="condition",
    )
    family_tables.append(e1_family)
    split_tables.append(e1_split)

    for utility_id in UTILITY_IDS:
        utility_column = f"utility__{utility_id}"
        split_summary, _, family_summary = _ranking_summaries(e2_frame, utility_column=utility_column)
        split_summary["context"] = "E2_partition_ranking"
        split_summary["utility_id"] = utility_id
        family_summary["context"] = "E2_partition_ranking"
        family_summary["utility_id"] = utility_id
        split_tables.append(
            split_summary[
                [
                    "context",
                    "utility_id",
                    "family_id",
                    "split_id",
                    "heldout_task",
                    "predicted_best_partition",
                    "oracle_best_partition",
                    "predicted_best_score",
                    "predicted_best_utility",
                    "oracle_best_utility",
                    "top1_regret",
                    "top3_hit",
                    "spearman",
                    "kendall_tau",
                ]
            ]
        )
        family_tables.append(
            family_summary[
                [
                    "context",
                    "utility_id",
                    "family_id",
                    "spearman",
                    "kendall_tau",
                    "top1_regret",
                    "top3_hit",
                    "selected_utility",
                    "micro_utility",
                    "oracle_utility",
                    "selected_top1_regret",
                    "micro_top1_regret",
                    "selected_minus_micro",
                    "selected_minus_oracle",
                ]
            ]
        )

    e3_family, _ = _summarize_condition_frame(
        e3_frame,
        context="E3_ablation",
        family_summary_columns={
            "artifact_partition_full_utility": "artifact_partition_full",
            "artifact_partition_no_contract_utility": "artifact_partition_no_contract",
            "artifact_partition_contract_no_structured_validator_utility": "artifact_partition_contract_no_structured_validator",
            "artifact_partition_contract_validator_no_repair_utility": "artifact_partition_contract_validator_no_repair",
        },
        split_group_columns=["family_id", "split_id", "heldout_task"],
        condition_column="ablation_condition",
    )
    family_tables.append(e3_family)

    e4_family, _ = _summarize_condition_frame(
        e4_frame,
        context="E4_repair",
        family_summary_columns={
            "full_local_repair_utility": "full_local_repair",
            "global_end_to_end_rerun_utility": "global_end_to_end_rerun",
            "validator_detect_no_local_repair_utility": "validator_detect_no_local_repair",
        },
        split_group_columns=["family_id", "split_id", "heldout_task"],
        condition_column="repair_policy",
    )
    family_tables.append(e4_family)

    family_summary = pd.concat(family_tables, ignore_index=True, sort=False)
    split_summary = pd.concat(
        [table.dropna(axis=1, how="all") for table in split_tables if not table.empty],
        ignore_index=True,
        sort=False,
    )
    _save_dataframe(family_summary, analysis_tables_root(paths) / "utility_robustness_family_summary.csv")
    _save_dataframe(split_summary, analysis_tables_root(paths) / "utility_robustness_split_summary.csv")
    _plot_utility_selected_vs_micro(family_summary, figures_root(paths) / "utility_robustness_selected_vs_micro.png")
    _plot_utility_alignment(family_summary, figures_root(paths) / "utility_alignment_across_definitions.png")
    return formulas, family_summary, split_summary


def _format_report_float(value: Any, *, precision: int = 3, undefined_label: str = "undefined") -> str:
    if value is None:
        return undefined_label
    if isinstance(value, float) and pd.isna(value):
        return undefined_label
    return f"{float(value):.{precision}f}"


def _plot_utility_selected_vs_micro(family_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    e1 = family_summary[(family_summary["context"] == "E1_executed") & (family_summary["family_id"] == "pooled")].set_index("utility_id")
    e2 = family_summary[(family_summary["context"] == "E2_partition_ranking") & (family_summary["family_id"] == "pooled")].set_index("utility_id")
    x_positions = np.arange(len(UTILITY_IDS))
    width = 0.28
    axes[0].bar(x_positions - width / 2.0, [float(e1.loc[utility_id, "selected_utility"]) for utility_id in UTILITY_IDS], width=width, color="#355070", label="selected/full")
    axes[0].bar(x_positions + width / 2.0, [float(e1.loc[utility_id, "micro_utility"]) for utility_id in UTILITY_IDS], width=width, color="#2a9d8f", label="micro")
    axes[0].set_xticks(x_positions, UTILITY_IDS, rotation=15)
    axes[0].set_ylabel("mean stress utility")
    axes[0].set_title("E1 executed")
    axes[1].bar(x_positions - width, [float(e2.loc[utility_id, "selected_utility"]) for utility_id in UTILITY_IDS], width=width, color="#355070", label="selected")
    axes[1].bar(x_positions, [float(e2.loc[utility_id, "micro_utility"]) for utility_id in UTILITY_IDS], width=width, color="#2a9d8f", label="micro")
    axes[1].bar(x_positions + width, [float(e2.loc[utility_id, "oracle_utility"]) for utility_id in UTILITY_IDS], width=width, color="#e76f51", label="oracle")
    axes[1].set_xticks(x_positions, UTILITY_IDS, rotation=15)
    axes[1].set_ylabel("mean stress utility")
    axes[1].set_title("E2 proxy ranking")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_utility_alignment(family_summary: pd.DataFrame, output_path: Path) -> None:
    pooled = family_summary[(family_summary["context"] == "E2_partition_ranking") & (family_summary["family_id"] == "pooled")].set_index("utility_id")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#355070", "#bc4749", "#2a9d8f"]
    axes[0].bar(UTILITY_IDS, [float(pooled.loc[utility_id, "spearman"]) for utility_id in UTILITY_IDS], color=colors)
    axes[0].set_ylabel("pooled Spearman")
    axes[1].bar(UTILITY_IDS, [float(pooled.loc[utility_id, "top1_regret"]) for utility_id in UTILITY_IDS], color=colors)
    axes[1].set_ylabel("mean top-1 regret")
    axes[2].bar(UTILITY_IDS, [float(pooled.loc[utility_id, "top3_hit"]) for utility_id in UTILITY_IDS], color=colors)
    axes[2].set_ylabel("mean top-3 hit")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _fault_level_repair_breakdown(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(paths.results_dir / "singlecell_repair_evaluation" / "induced_fault" / "per_run_results.csv")
    summary = (
        frame.groupby(["family_id", "fault_type", "repair_policy"], dropna=False)[
            ["root_localization_accuracy", "repair_success", "final_success", "rerun_span", "extra_wall_clock_seconds"]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"root_localization_accuracy": "localization_accuracy"})
    )
    pooled = (
        frame.groupby(["fault_type", "repair_policy"], dropna=False)[
            ["root_localization_accuracy", "repair_success", "final_success", "rerun_span", "extra_wall_clock_seconds"]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"root_localization_accuracy": "localization_accuracy"})
    )
    pooled.insert(0, "family_id", "pooled")
    summary = pd.concat([summary, pooled], ignore_index=True)
    latent_focus = summary[summary["fault_type"] == "latent_or_graph_cell_order_mismatch"].copy()
    _save_dataframe(summary, analysis_tables_root(paths) / "fault_level_policy_summary.csv")
    _save_dataframe(latent_focus, analysis_tables_root(paths) / "latent_mismatch_focus.csv")
    _plot_fault_heatmap(summary, figures_root(paths) / "fault_level_repair_heatmap.png")
    _plot_latent_focus(latent_focus, figures_root(paths) / "latent_mismatch_policy_comparison.png")
    return summary, latent_focus


def _plot_fault_heatmap(summary_frame: pd.DataFrame, output_path: Path) -> None:
    pooled = summary_frame[summary_frame["family_id"] == "pooled"].copy()
    faults = sorted(pooled["fault_type"].unique().tolist())
    policies = ["full_local_repair", "validator_detect_no_local_repair", "global_end_to_end_rerun"]
    final_success = np.array(
        [
            [
                float(
                    pooled[
                        (pooled["fault_type"] == fault_type) & (pooled["repair_policy"] == repair_policy)
                    ]["final_success"].iloc[0]
                )
                for repair_policy in policies
            ]
            for fault_type in faults
        ]
    )
    localization = np.array(
        [
            [
                float(
                    pooled[
                        (pooled["fault_type"] == fault_type) & (pooled["repair_policy"] == repair_policy)
                    ]["localization_accuracy"].iloc[0]
                )
                for repair_policy in policies
            ]
            for fault_type in faults
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, matrix, title in zip(
        axes,
        [final_success, localization],
        ["final success", "localization accuracy"],
    ):
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
        axis.set_xticks(np.arange(len(policies)), policies, rotation=20)
        axis.set_yticks(np.arange(len(faults)), faults)
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", color="#1f2933")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_focus(latent_focus: pd.DataFrame, output_path: Path) -> None:
    pooled = latent_focus[latent_focus["family_id"] == "pooled"].set_index("repair_policy")
    policies = ["full_local_repair", "validator_detect_no_local_repair", "global_end_to_end_rerun"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(policies, [float(pooled.loc[policy, "final_success"]) for policy in policies], color=["#355070", "#9c6644", "#2a9d8f"])
    axes[0].bar(
        policies,
        [float(pooled.loc[policy, "localization_accuracy"]) for policy in policies],
        fill=False,
        edgecolor="#1f2933",
        linewidth=1.2,
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("rate")
    axes[1].bar(policies, [float(pooled.loc[policy, "rerun_span"]) for policy in policies], color=["#355070", "#9c6644", "#2a9d8f"])
    axes[1].set_ylabel("mean rerun span")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_utility(
    *,
    success: bool,
    artifact_validity_rate: float,
    metric_gap_to_reference: float | None,
    rerun_span: int | None,
    wall_clock_seconds: float,
    reference_wall_clock_seconds: float,
) -> float:
    metric_fidelity = _metric_fidelity(metric_gap_to_reference)
    repair_efficiency = _rerun_efficiency(rerun_span)
    cost_efficiency = _runtime_efficiency(wall_clock_seconds, reference_wall_clock_seconds)
    utility = (
        0.35 * float(_safe_bool(success))
        + 0.20 * float(artifact_validity_rate)
        + 0.20 * metric_fidelity
        + 0.15 * repair_efficiency
        + 0.10 * cost_efficiency
    )
    return round(float(utility), 6)


def _sanitize_message(value: str, workspace_root: Path) -> str:
    message = str(value)
    message = message.replace(str(workspace_root), ".")
    message = message.replace(str(workspace_root).replace("\\", "/"), ".")
    return message


def run_singlecell_phase1_validator_rescan(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    source_path = resolved_paths.results_dir / "singlecell_workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv"
    frame = pd.read_csv(source_path)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        relative_run_dir = row["run_dir"]
        run_dir = resolved_paths.workspace_root / relative_run_dir
        record: dict[str, Any] = {
            "family_id": row["family_id"],
            "split_id": row["split_id"],
            "heldout_task": row["heldout_task"],
            "condition": row["condition"],
            "clean_or_stress": row["clean_or_stress"],
            "stress_protocol_id": row["stress_protocol_id"],
            "run_dir": relative_run_dir,
            "run_dir_exists": bool(run_dir.exists()),
            "rescanned": False,
            "skip_reason": "",
            "original_artifact_validity_rate": round(float(row["artifact_validity_rate"]), 6),
            "original_all_segment_validations_passed": bool(_safe_bool(row["all_segment_validations_passed"])),
            "original_utility": round(float(row["utility"]), 6),
            "rescanned_artifact_validity_rate": None,
            "rescanned_all_roles_passed": None,
            "rescanned_utility": None,
            "rescanned_first_failed_role": "",
            "rescanned_failed_role_count": None,
            "rescanned_failed_roles_json": "[]",
            "rescanned_failure_codes_json": "{}",
        }
        if not run_dir.exists():
            record["skip_reason"] = "run_dir_missing"
            rows.append(record)
            continue
        try:
            validation_results = validate_run_directory(run_dir, workspace_root=resolved_paths.workspace_root)
            failed_roles = [role for role in VALIDATOR_ROLES if not validation_results[role].passed]
            failure_codes = {
                role: _sanitize_message(validation_results[role].error_code, resolved_paths.workspace_root)
                for role in failed_roles
            }
            rescanned_artifact_validity_rate = float(
                np.mean([float(validation_results[role].passed) for role in VALIDATOR_ROLES])
            )
            rescanned_utility = _run_utility(
                success=_safe_bool(row["success"]),
                artifact_validity_rate=rescanned_artifact_validity_rate,
                metric_gap_to_reference=_optional_float(row["metric_gap_to_reference"]),
                rerun_span=None if pd.isna(row["rerun_span"]) else int(row["rerun_span"]),
                wall_clock_seconds=float(row["wall_clock_seconds"]),
                reference_wall_clock_seconds=float(row["reference_wall_clock_seconds"]),
            )
            record.update(
                {
                    "rescanned": True,
                    "rescanned_artifact_validity_rate": round(rescanned_artifact_validity_rate, 6),
                    "rescanned_all_roles_passed": bool(len(failed_roles) == 0),
                    "rescanned_utility": round(rescanned_utility, 6),
                    "rescanned_first_failed_role": failed_roles[0] if failed_roles else "",
                    "rescanned_failed_role_count": int(len(failed_roles)),
                    "rescanned_failed_roles_json": _json_ready(failed_roles),
                    "rescanned_failure_codes_json": _json_ready(failure_codes),
                }
            )
        except Exception as exc:
            record["skip_reason"] = _sanitize_message(str(exc), resolved_paths.workspace_root)
        rows.append(record)

    per_run = pd.DataFrame(rows)
    _save_dataframe(per_run, validator_rescan_root(resolved_paths) / "per_run_rescan.csv")

    stress = per_run[(per_run["clean_or_stress"] == "stress") & (per_run["rescanned"])].copy()
    condition_summary = (
        stress.groupby(["family_id", "condition"], dropna=False)[
            ["rescanned_artifact_validity_rate", "rescanned_utility", "original_utility", "rescanned_all_roles_passed"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    ranking_rows: list[dict[str, Any]] = []
    for family_id, subset in _family_and_pooled_frames(stress):
        mean_by_condition = subset.groupby("condition", dropna=False)[["original_utility", "rescanned_utility"]].mean()
        original_ranking = mean_by_condition["original_utility"].sort_values(ascending=False).index.tolist()
        rescanned_ranking = mean_by_condition["rescanned_utility"].sort_values(ascending=False).index.tolist()
        full_original = float(mean_by_condition.loc["artifact_partition_full", "original_utility"])
        micro_original = float(mean_by_condition.loc["micro_skill", "original_utility"])
        no_contract_original = float(mean_by_condition.loc["artifact_partition_no_contract", "original_utility"])
        full_rescanned = float(mean_by_condition.loc["artifact_partition_full", "rescanned_utility"])
        micro_rescanned = float(mean_by_condition.loc["micro_skill", "rescanned_utility"])
        no_contract_rescanned = float(mean_by_condition.loc["artifact_partition_no_contract", "rescanned_utility"])
        ranking_rows.append(
            {
                "family_id": family_id,
                "original_condition_ranking_json": _json_ready(original_ranking),
                "rescanned_condition_ranking_json": _json_ready(rescanned_ranking),
                "ranking_changed": bool(original_ranking != rescanned_ranking),
                "full_vs_micro_original_relation": _rank_relation(full_original - micro_original),
                "full_vs_micro_rescanned_relation": _rank_relation(full_rescanned - micro_rescanned),
                "full_vs_micro_relation_changed": bool(
                    _rank_relation(full_original - micro_original) != _rank_relation(full_rescanned - micro_rescanned)
                ),
                "full_vs_no_contract_original_relation": _rank_relation(full_original - no_contract_original),
                "full_vs_no_contract_rescanned_relation": _rank_relation(full_rescanned - no_contract_rescanned),
                "full_vs_no_contract_relation_changed": bool(
                    _rank_relation(full_original - no_contract_original)
                    != _rank_relation(full_rescanned - no_contract_rescanned)
                ),
            }
        )
    ranking_summary = pd.DataFrame(ranking_rows)
    e2_path = resolved_paths.results_dir / "singlecell_workflow_evaluation" / "boundary_ranking" / "per_run_results.csv"
    e2_frame = pd.read_csv(e2_path)
    summary = {
        "source_csv": resolved_paths.relative_to_workspace(source_path),
        "runs_total": int(len(per_run)),
        "runs_rescanned": int(per_run["rescanned"].sum()),
        "runs_skipped": int((~per_run["rescanned"]).sum()),
        "stress_runs_rescanned": int(len(stress)),
        "clean_runs_rescanned": int(len(per_run[(per_run["clean_or_stress"] == "clean") & (per_run["rescanned"])])),
        "pass_fail_by_condition": condition_summary.to_dict(orient="records"),
        "ranking_changes": ranking_summary.to_dict(orient="records"),
        "skipped_runs": per_run.loc[~per_run["rescanned"], ["family_id", "split_id", "condition", "skip_reason"]].to_dict(orient="records"),
        "e2_rescan_note": {
            "rescanned": False,
            "reason": "E2 boundary_ranking/per_run_results.csv does not carry run_dir pointers, so the saved proxy table cannot be rescanned in the same turnkey way as E1.",
            "rows_checked": int(len(e2_frame)),
            "run_dir_column_present": bool("run_dir" in e2_frame.columns),
        },
    }
    write_json(validator_rescan_root(resolved_paths) / "summary.json", summary)

    condition_rows = condition_summary.copy()
    condition_rows["rescanned_all_roles_passed"] = condition_rows["rescanned_all_roles_passed"].round(6)
    ranking_table = ranking_summary[
        [
            "family_id",
            "ranking_changed",
            "full_vs_micro_original_relation",
            "full_vs_micro_rescanned_relation",
            "full_vs_no_contract_original_relation",
            "full_vs_no_contract_rescanned_relation",
        ]
    ].to_dict(orient="records")
    skipped_rows = summary["skipped_runs"]
    report_lines = [
        "# Single-Cell Phase-1 Validator Rescan",
        "",
        f"- source CSV: `{resolved_paths.relative_to_workspace(source_path)}`",
        f"- runs rescanned: `{summary['runs_rescanned']}` / `{summary['runs_total']}`",
        f"- stress rows rescanned: `{summary['stress_runs_rescanned']}`",
        f"- clean rows rescanned: `{summary['clean_runs_rescanned']}`",
        f"- skipped rows: `{summary['runs_skipped']}`",
        "",
        "## Stress condition summary",
        "",
        markdown_table(
            condition_rows.to_dict(orient="records"),
            ["family_id", "condition", "rescanned_artifact_validity_rate", "rescanned_utility", "original_utility", "rescanned_all_roles_passed"],
        ),
        "",
        "## Executed conclusion check",
        "",
        markdown_table(
            ranking_table,
            [
                "family_id",
                "ranking_changed",
                "full_vs_micro_original_relation",
                "full_vs_micro_rescanned_relation",
                "full_vs_no_contract_original_relation",
                "full_vs_no_contract_rescanned_relation",
            ],
        ),
        "",
        "## E2 limitation",
        "",
        "- E2 was not rescanned here. The saved `boundary_ranking/per_run_results.csv` table has no `run_dir` column, so it is not a direct rerun ledger in the same turnkey sense as E1.",
        "",
    ]
    if skipped_rows:
        report_lines.extend(
            [
                "## Skipped rows",
                "",
                markdown_table(skipped_rows, ["family_id", "split_id", "condition", "skip_reason"]),
                "",
            ]
        )
    write_text(resolved_paths.reports_dir / VALIDATOR_RESCAN_REPORT_NAME, "\n".join(report_lines) + "\n")
    return {
        "per_run_rescan_csv": resolved_paths.relative_to_workspace(validator_rescan_root(resolved_paths) / "per_run_rescan.csv"),
        "summary_json": resolved_paths.relative_to_workspace(validator_rescan_root(resolved_paths) / "summary.json"),
        "report_md": resolved_paths.relative_to_workspace(resolved_paths.reports_dir / VALIDATOR_RESCAN_REPORT_NAME),
        "summary": summary,
    }


def _paper_synthesis_table(
    paths,
    same_cardinality_family: pd.DataFrame,
    utility_family_summary: pd.DataFrame,
    latent_focus: pd.DataFrame,
    validator_rescan_summary: dict[str, Any],
) -> pd.DataFrame:
    e1_balanced = utility_family_summary[
        (utility_family_summary["context"] == "E1_executed")
        & (utility_family_summary["utility_id"] == "balanced")
    ].set_index("family_id")
    e2_balanced = utility_family_summary[
        (utility_family_summary["context"] == "E2_partition_ranking")
        & (utility_family_summary["utility_id"] == "balanced")
    ].set_index("family_id")
    latent_pooled = latent_focus[latent_focus["family_id"] == "pooled"].set_index("repair_policy")
    rescan_lookup = {row["family_id"]: row for row in validator_rescan_summary["ranking_changes"]}
    rows: list[dict[str, Any]] = []
    for family_id in [*FAMILIES, "pooled"]:
        rows.append(
            {
                "family_id": family_id,
                "e1_full_minus_micro_balanced": round(float(e1_balanced.loc[family_id, "selected_minus_micro"]), 6),
                "e2_selected_minus_micro_balanced": round(float(e2_balanced.loc[family_id, "selected_minus_micro"]), 6),
                "e2_selected_minus_oracle_balanced": round(float(e2_balanced.loc[family_id, "selected_minus_oracle"]), 6),
                "same_cardinality_share_tied_best": round(
                    float(same_cardinality_family.set_index("family_id").loc[family_id, "share_selected_tied_best_same_cardinality"]),
                    6,
                ),
                "same_cardinality_mean_regret": round(
                    float(same_cardinality_family.set_index("family_id").loc[family_id, "mean_selected_same_cardinality_regret"]),
                    6,
                ),
                "validator_rescan_ranking_changed": bool(rescan_lookup[family_id]["ranking_changed"]),
                "validator_rescan_full_vs_micro_relation": rescan_lookup[family_id]["full_vs_micro_rescanned_relation"],
                "validator_rescan_full_vs_no_contract_relation": rescan_lookup[family_id]["full_vs_no_contract_rescanned_relation"],
                "repair_weak_spot": "latent_or_graph_cell_order_mismatch",
                "latent_mismatch_local_final_success": round(float(latent_pooled.loc["full_local_repair", "final_success"]), 6),
                "latent_mismatch_global_final_success": round(float(latent_pooled.loc["global_end_to_end_rerun", "final_success"]), 6),
            }
        )
    synthesis = pd.DataFrame(rows)
    _save_dataframe(synthesis, analysis_tables_root(paths) / "paper_synthesis_table.csv")
    return synthesis


def _analysis_report(
    paths,
    same_cardinality_split: pd.DataFrame,
    same_cardinality_family: pd.DataFrame,
    boundary_coefficients: pd.DataFrame,
    boundary_recurrence: pd.DataFrame,
    utility_family_summary: pd.DataFrame,
    validator_rescan_summary: dict[str, Any],
    fault_summary: pd.DataFrame,
    latent_focus: pd.DataFrame,
) -> Path:
    same_cardinality_pooled = same_cardinality_family.set_index("family_id").loc["pooled"]
    e2_balanced = utility_family_summary[
        (utility_family_summary["context"] == "E2_partition_ranking") & (utility_family_summary["utility_id"] == "balanced")
    ].set_index("family_id")
    e1_by_utility = utility_family_summary[
        (utility_family_summary["context"] == "E1_executed") & (utility_family_summary["family_id"] == "pooled")
    ].set_index("utility_id")
    e2_by_utility = utility_family_summary[
        (utility_family_summary["context"] == "E2_partition_ranking") & (utility_family_summary["family_id"] == "pooled")
    ].set_index("utility_id")
    boundary_term_rows = boundary_coefficients[boundary_coefficients["is_boundary_term"]].copy()
    stable_positive = boundary_term_rows[
        (boundary_term_rows["bootstrap_ci_low"] > 0.0)
        | (
            (boundary_term_rows["bootstrap_positive_share"] >= 0.8)
            & (boundary_term_rows["segment_count_controlled_mean_difference"] > 0.0)
        )
    ].sort_values("ridge_coefficient", ascending=False)
    latent_pooled = latent_focus[latent_focus["family_id"] == "pooled"].set_index("repair_policy")
    rescan_rows = pd.DataFrame(validator_rescan_summary["ranking_changes"]).set_index("family_id")
    weak_spot_row = latent_pooled.loc["full_local_repair"]
    robust_selected_over_micro = bool(
        (
            utility_family_summary[
                (utility_family_summary["context"] == "E2_partition_ranking") & (utility_family_summary["family_id"] == "pooled")
            ]["selected_minus_micro"]
            >= -1.0e-9
        ).all()
    )
    e1_report_rows = utility_family_summary[
        (utility_family_summary["context"] == "E1_executed")
        & (utility_family_summary["family_id"].isin(FAMILIES + ["pooled"]))
    ][["utility_id", "family_id", "selected_utility", "micro_utility", "selected_minus_micro", "artifact_partition_no_contract_utility", "full_minus_no_contract"]].copy()
    e2_report_rows = utility_family_summary[
        (utility_family_summary["context"] == "E2_partition_ranking")
        & (utility_family_summary["family_id"].isin(FAMILIES + ["pooled"]))
    ][["utility_id", "family_id", "selected_utility", "micro_utility", "oracle_utility", "selected_minus_micro", "selected_minus_oracle", "spearman", "top1_regret", "top3_hit"]].copy()
    e2_report_rows["spearman"] = e2_report_rows["spearman"].apply(lambda value: _format_report_float(value, precision=3, undefined_label="undefined"))

    report_lines = [
        "# Single-Cell NeurIPS Analysis Follow-Up",
        "",
        "Saved inputs reused:",
        "",
        "- `results/partitions/scanpy_pancreas_ingest/legal_partitions.json`",
        "- `results/partitions/tabula_muris_label_transfer/legal_partitions.json`",
        "- `results/partitions/scanpy_pancreas_ingest/selected_partition.json`",
        "- `results/partitions/tabula_muris_label_transfer/selected_partition.json`",
        "- `results/singlecell_workflow_evaluation/boundary_ranking/per_partition_results.csv`",
        "- `results/singlecell_workflow_evaluation/boundary_ranking/per_run_results.csv`",
        "- `results/singlecell_workflow_evaluation/same_info_diff_cut/per_run_results.csv`",
        "- `results/singlecell_repair_evaluation/ablation/per_run_results.csv`",
        "- `results/singlecell_repair_evaluation/induced_fault/per_run_results.csv`",
        "- `runs/singlecell_phase1_same_info_diff_cut/...` saved run directories for the validator rescan",
        "",
        "## 1. What can now be claimed more strongly from saved F5/F6 evidence?",
        "",
        f"- `partition_009` is not only benefiting from a convenient segment count. Across all `10` saved splits it stayed tied-best inside the fixed 4-segment bucket, with pooled same-cardinality regret `{same_cardinality_pooled['mean_selected_same_cardinality_regret']:.6f}` and tie-best rate `{same_cardinality_pooled['share_selected_tied_best_same_cardinality']:.3f}`.",
        "- The frozen partition score still aligns positively with downstream proxy utility once the utility definition actually separates partitions. "
        f"Pooled E2 Spearman values were `{_format_report_float(e2_by_utility.loc['transfer_first', 'spearman'])}` for `transfer_first` "
        "(undefined because the transfer-only utility saturates on the saved E2 table), "
        f"`{_format_report_float(e2_by_utility.loc['containment_first', 'spearman'])}` for `containment_first`, and "
        f"`{_format_report_float(e2_by_utility.loc['balanced', 'spearman'])}` for `balanced`.",
        f"- The repair story is fault-conditional rather than universally pro-local-repair. For the isolated weak spot `latent_or_graph_cell_order_mismatch`, pooled local final success stayed `{float(latent_pooled.loc['full_local_repair', 'final_success']):.3f}` while global rerun stayed `{float(latent_pooled.loc['global_end_to_end_rerun', 'final_success']):.3f}`.",
        "",
        "## 2. Does selected `partition_009` remain strong within its same-cardinality bucket?",
        "",
        markdown_table(
            same_cardinality_family[
                [
                    "family_id",
                    "mean_selected_same_cardinality_tie_rank",
                    "mean_selected_same_cardinality_regret",
                    "share_selected_tied_best_same_cardinality",
                    "mean_selected_minus_micro",
                    "mean_selected_minus_whole",
                ]
            ].to_dict(orient="records"),
            [
                "family_id",
                "mean_selected_same_cardinality_tie_rank",
                "mean_selected_same_cardinality_regret",
                "share_selected_tied_best_same_cardinality",
                "mean_selected_minus_micro",
                "mean_selected_minus_whole",
            ],
        ),
        "",
        "- Interpretation: the saved table does not support the weak claim that `partition_009` wins only because it is medium-sized. Inside the 4-segment bucket it is never below the tied-best utility level, and it remains well above the whole-workflow partition.",
        "",
        "## 3. Which specific boundaries appear to carry positive value?",
        "",
        markdown_table(
            boundary_term_rows[
                [
                    "term",
                    "ridge_coefficient",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "bootstrap_positive_share",
                    "segment_count_controlled_mean_difference",
                ]
            ].to_dict(orient="records"),
            [
                "term",
                "ridge_coefficient",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "bootstrap_positive_share",
                "segment_count_controlled_mean_difference",
            ],
        ),
        "",
        markdown_table(
            boundary_recurrence[
                (boundary_recurrence["family_id"] == "pooled") & (boundary_recurrence["k"] == 1)
            ][["boundary_role", "slot_recurrence_rate", "baseline_prevalence", "slot_lift_over_baseline"]].to_dict(orient="records"),
            ["boundary_role", "slot_recurrence_rate", "baseline_prevalence", "slot_lift_over_baseline"],
        ),
        "",
        (
            "- Stable positive boundaries: "
            + (", ".join(f"`{term}`" for term in stable_positive["term"].tolist()) if not stable_positive.empty else "none stood out cleanly enough to call stable.")
        ),
        "- The top-1 and top-3 recurrence table is descriptive rather than causal, but it agrees with the coefficient table on which boundaries recur in the best saved partitions.",
        "",
        "## 4. Does the selected-vs-micro story change under alternative utility definitions?",
        "",
        markdown_table(
            e1_report_rows.to_dict(orient="records"),
            [
                "utility_id",
                "family_id",
                "selected_utility",
                "micro_utility",
                "selected_minus_micro",
                "artifact_partition_no_contract_utility",
                "full_minus_no_contract",
            ],
        ),
        "",
        markdown_table(
            e2_report_rows.to_dict(orient="records"),
            [
                "utility_id",
                "family_id",
                "selected_utility",
                "micro_utility",
                "oracle_utility",
                "selected_minus_micro",
                "selected_minus_oracle",
                "spearman",
                "top1_regret",
                "top3_hit",
            ],
        ),
        "",
        "- Executed E1 does not stay a tie under all alternative utility definitions. "
        f"`transfer_first` remains tied at `{float(e1_by_utility.loc['transfer_first', 'selected_minus_micro']):.6f}`, "
        f"but `containment_first` and `balanced` favor full over micro by "
        f"`{float(e1_by_utility.loc['containment_first', 'selected_minus_micro']):.6f}` and "
        f"`{float(e1_by_utility.loc['balanced', 'selected_minus_micro']):.6f}` respectively.",
        f"- E2 proxy rankings remain near-oracle, and the selected partition stays non-worse than micro across all three utility definitions: `{robust_selected_over_micro}`.",
        "",
        "## 5. Do tightened validators materially change the executed phase-1 conclusion?",
        "",
        markdown_table(
            rescan_rows[
                [
                    "ranking_changed",
                    "full_vs_micro_original_relation",
                    "full_vs_micro_rescanned_relation",
                    "full_vs_no_contract_original_relation",
                    "full_vs_no_contract_rescanned_relation",
                ]
            ]
            .reset_index()
            .to_dict(orient="records"),
            [
                "family_id",
                "ranking_changed",
                "full_vs_micro_original_relation",
                "full_vs_micro_rescanned_relation",
                "full_vs_no_contract_original_relation",
                "full_vs_no_contract_rescanned_relation",
            ],
        ),
        "",
        f"- Rescanned E1 rows: `{validator_rescan_summary['runs_rescanned']}` / `{validator_rescan_summary['runs_total']}`. Skipped rows: `{validator_rescan_summary['runs_skipped']}`.",
        "- No turnkey E2 rescan was attempted because the saved E2 proxy table has no `run_dir` pointers.",
        "",
        "## 6. Which repair failures remain unsolved, and how should they be described?",
        "",
        markdown_table(
            latent_focus[
                [
                    "family_id",
                    "repair_policy",
                    "localization_accuracy",
                    "repair_success",
                    "final_success",
                    "rerun_span",
                    "extra_wall_clock_seconds",
                ]
            ].to_dict(orient="records"),
            [
                "family_id",
                "repair_policy",
                "localization_accuracy",
                "repair_success",
                "final_success",
                "rerun_span",
                "extra_wall_clock_seconds",
            ],
        ),
        "",
        f"- The exact remaining weak spot is `latent_or_graph_cell_order_mismatch`.",
        f"- Local repair still localizes better than global rerun overall, but on this fault its pooled final success is only `{float(weak_spot_row['final_success']):.3f}` while global rerun stays at `{float(latent_pooled.loc['global_end_to_end_rerun', 'final_success']):.3f}`.",
        "- The most honest description is an upstream integrity fault: the corruption is injected in `latent_or_graph`, but the observed downstream failure surfaces later, so the current local repair trigger restarts too late to rebuild the broken latent state.",
        "",
        "## 7. What is the single most justified next experiment after this analysis round?",
        "",
        "- Run one targeted repair-follow-up on `latent_or_graph_cell_order_mismatch` that allows local repair to restart from `latent_or_graph` (or one step earlier) once the downstream inconsistency is detected. This is the narrowest next experiment that directly tests the only clearly isolated failure mode left in the saved single-cell package.",
        "",
        "## Bottom line",
        "",
        "- Stronger saved-data claim: the selected cut is a boundary-specific near-top choice, not just a generic 4-segment cut.",
        "- Mixed claim that should stay mixed: selected beats micro on the E2 proxy table, while the executed E1 head-to-head flips from an original tie to a full-favoring result once containment is weighted explicitly.",
        "- Stable mechanism claim: repair helps on localization and rerun scope, but not universally on final success.",
    ]
    report_path = paths.reports_dir / ANALYSIS_REPORT_NAME
    write_text(report_path, "\n".join(report_lines) + "\n")
    return report_path


def _terminal_summary(
    same_cardinality_family: pd.DataFrame,
    boundary_coefficients: pd.DataFrame,
    utility_family_summary: pd.DataFrame,
    validator_rescan_summary: dict[str, Any],
    latent_focus: pd.DataFrame,
) -> list[str]:
    same_cardinality_pooled = same_cardinality_family.set_index("family_id").loc["pooled"]
    boundary_term_rows = boundary_coefficients[boundary_coefficients["is_boundary_term"]].copy()
    stable_positive = boundary_term_rows[
        (boundary_term_rows["bootstrap_ci_low"] > 0.0)
        | (
            (boundary_term_rows["bootstrap_positive_share"] >= 0.8)
            & (boundary_term_rows["segment_count_controlled_mean_difference"] > 0.0)
        )
    ]["term"].tolist()
    e2_pooled = utility_family_summary[
        (utility_family_summary["context"] == "E2_partition_ranking") & (utility_family_summary["family_id"] == "pooled")
    ]
    e1_pooled = utility_family_summary[
        (utility_family_summary["context"] == "E1_executed") & (utility_family_summary["family_id"] == "pooled")
    ].set_index("utility_id")
    robust_selected_over_micro = bool((e2_pooled["selected_minus_micro"] >= -1.0e-9).all())
    rescan_rows = pd.DataFrame(validator_rescan_summary["ranking_changes"]).set_index("family_id")
    latent_pooled = latent_focus[latent_focus["family_id"] == "pooled"].set_index("repair_policy")
    return [
        "saved_files_reused="
        "results/partitions/*/legal_partitions.json; "
        "results/partitions/*/selected_partition.json; "
        "results/singlecell_workflow_evaluation/boundary_ranking/per_partition_results.csv; "
        "results/singlecell_workflow_evaluation/boundary_ranking/per_run_results.csv; "
        "results/singlecell_workflow_evaluation/same_info_diff_cut/per_run_results.csv; "
        "results/singlecell_repair_evaluation/ablation/per_run_results.csv; "
        "results/singlecell_repair_evaluation/induced_fault/per_run_results.csv; "
        "runs/singlecell_phase1_same_info_diff_cut/...",
        "same_cardinality_conclusion="
        f"selected stays tied-best within the 4-segment bucket on the saved splits "
        f"(mean_regret={float(same_cardinality_pooled['mean_selected_same_cardinality_regret']):.6f}, "
        f"tie_best_rate={float(same_cardinality_pooled['share_selected_tied_best_same_cardinality']):.3f})",
        "boundary_attribution_conclusion="
        + (
            "stable useful boundaries=" + ",".join(stable_positive)
            if stable_positive
            else "no boundary cleared the stability threshold cleanly enough to call it stable"
        ),
        "utility_robustness_conclusion="
        + (
            "selected remains non-worse than micro across all saved E2 utility definitions; "
            f"E1 tie breaks toward full under containment_first ({float(e1_pooled.loc['containment_first', 'selected_minus_micro']):.6f}) "
            f"and balanced ({float(e1_pooled.loc['balanced', 'selected_minus_micro']):.6f})"
            if robust_selected_over_micro
            else "selected-vs-micro is sensitive to the utility definition"
        ),
        "validator_rescan_conclusion="
        f"ranking_changed={bool(rescan_rows.loc['pooled', 'ranking_changed'])} "
        f"full_vs_micro={rescan_rows.loc['pooled', 'full_vs_micro_rescanned_relation']} "
        f"full_vs_no_contract={rescan_rows.loc['pooled', 'full_vs_no_contract_rescanned_relation']}",
        "repair_weak_spot="
        f"latent_or_graph_cell_order_mismatch remains unsolved for local repair "
        f"(local_final_success={float(latent_pooled.loc['full_local_repair', 'final_success']):.3f}, "
        f"global_final_success={float(latent_pooled.loc['global_end_to_end_rerun', 'final_success']):.3f})",
        "best_next_step=allow a targeted local repair restart from latent_or_graph for latent_or_graph_cell_order_mismatch",
    ]


def run_singlecell_robustness(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    partition_metadata = _load_partition_metadata(resolved_paths)
    e2_partition_frame = _load_e2_partition_frame(resolved_paths, partition_metadata)
    same_cardinality_split, same_cardinality_family = _same_cardinality_analysis(resolved_paths, e2_partition_frame)
    boundary_coefficients, boundary_recurrence = _boundary_attribution(resolved_paths, e2_partition_frame)
    _, utility_family_summary, utility_split_summary = _utility_robustness(resolved_paths, partition_metadata)
    validator_rescan = run_singlecell_phase1_validator_rescan(resolved_paths)
    fault_summary, latent_focus = _fault_level_repair_breakdown(resolved_paths)
    synthesis = _paper_synthesis_table(
        resolved_paths,
        same_cardinality_family,
        utility_family_summary,
        latent_focus,
        validator_rescan["summary"],
    )
    report_path = _analysis_report(
        resolved_paths,
        same_cardinality_split,
        same_cardinality_family,
        boundary_coefficients,
        boundary_recurrence,
        utility_family_summary,
        validator_rescan["summary"],
        fault_summary,
        latent_focus,
    )
    terminal_summary = _terminal_summary(
        same_cardinality_family,
        boundary_coefficients,
        utility_family_summary,
        validator_rescan["summary"],
        latent_focus,
    )
    return {
        "same_cardinality_per_split_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "same_cardinality_per_split.csv"),
        "same_cardinality_family_summary_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "same_cardinality_family_summary.csv"),
        "boundary_attribution_coefficients_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "boundary_attribution_coefficients.csv"),
        "boundary_topk_recurrence_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "boundary_topk_recurrence.csv"),
        "utility_robustness_family_summary_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "utility_robustness_family_summary.csv"),
        "utility_robustness_split_summary_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "utility_robustness_split_summary.csv"),
        "fault_level_policy_summary_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "fault_level_policy_summary.csv"),
        "latent_mismatch_focus_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "latent_mismatch_focus.csv"),
        "paper_synthesis_table_csv": resolved_paths.relative_to_workspace(analysis_tables_root(resolved_paths) / "paper_synthesis_table.csv"),
        "validator_rescan_summary_json": validator_rescan["summary_json"],
        "analysis_report_md": resolved_paths.relative_to_workspace(report_path),
        "validator_rescan_report_md": validator_rescan["report_md"],
        "terminal_summary": terminal_summary,
        "summary": {
            "same_cardinality_family": same_cardinality_family.to_dict(orient="records"),
            "utility_family_summary": utility_family_summary.to_dict(orient="records"),
            "fault_level_policy_summary": fault_summary.to_dict(orient="records"),
            "paper_synthesis_table": synthesis.to_dict(orient="records"),
        },
    }
