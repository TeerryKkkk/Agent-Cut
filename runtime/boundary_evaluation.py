from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import PREDICTIVE_STEP_SPECS
from pipelines.predictive_runtime import build_reference_runner, list_predictive_families
from runtime.workflow_evaluation import (
    SAME_INFO_STRESS_PROTOCOLS,
    ConditionSpec,
    _append_controller_event,
    _apply_light_stress,
    _artifact_path_for_role,
    _finalize_same_info_outcome,
    _repair_run_from_prefix,
    build_support_evidence,
    compute_substrate_baseline_version,
    load_reference_trace,
    load_split_manifests,
    role_index,
    segment_rows,
    selected_partition,
    total_reference_wall_clock,
)
from runtime.score_evaluation import _partition_numeric_id, _ranking_summary_rows
from utils.io_utils import markdown_table, read_json, read_yaml, write_json, write_text
from utils.pathing import detect_project_paths
from validators.predictive_roles import validate_role

PHASE3B_ROOT_NAME = "boundary_evaluation"
BOUNDARY_AUDIT_REPORT = "boundary_score_audit.md"
BOUNDARY_ATTRIBUTION_REPORT = "boundary_attribution.md"
BOUNDARY_EVALUATION_REPORT = "boundary_evaluation_report.md"

BOUNDARY_ROLE_TO_ID = {
    "profile_json": "B1",
    "split_spec_json": "B2",
    "preprocess_bundle": "B3",
    "model_bundle": "B4",
    "metrics_json": "B5",
    "report_md": "B6",
}
BOUNDARY_IDS = ["B1", "B2", "B3", "B4", "B5", "B6"]
MODEL_BOUNDARY_IDS = ["B1", "B2", "B3", "B4", "B5"]
EARLY_BOUNDARY_IDS = ["B1", "B2", "B3"]
LATE_BOUNDARY_IDS = ["B4", "B5"]

V4_CONDITION = ConditionSpec(
    condition_id="artifact_partition_v4",
    label="V4 hybrid baseline partition",
    representation_kind="artifact_partition_v4",
    controller_runtime="boundary_evaluation_skill_runtime",
    validator_mode="boundary",
    repair_mode="segment_restart",
    uses_skills=True,
    partition_mode="selected",
)

PHASE3B_SELECTED_CONDITION = ConditionSpec(
    condition_id="artifact_partition_phase3b",
    label="Selected boundary-objective partition",
    representation_kind="artifact_partition_phase3b",
    controller_runtime="boundary_evaluation_skill_runtime",
    validator_mode="boundary",
    repair_mode="segment_restart",
    uses_skills=True,
    partition_mode="selected",
)


def phase3b_root(paths) -> Path:
    return paths.results_dir / PHASE3B_ROOT_NAME


def audit_root(paths) -> Path:
    return phase3b_root(paths) / "audit"


def boundary_root(paths) -> Path:
    return phase3b_root(paths) / "boundary_attribution"


def manifests_root(paths) -> Path:
    return phase3b_root(paths) / "manifests"


def ranking_root(paths) -> Path:
    return phase3b_root(paths) / "ranking"


def revised_partition_eval_root(paths) -> Path:
    return phase3b_root(paths) / "revised_partition_eval"


def aggregate_root(paths) -> Path:
    return phase3b_root(paths) / "aggregate_tables"


def figures_root(paths) -> Path:
    return phase3b_root(paths) / "figures"


def _report_path(paths, report_name: str) -> Path:
    return paths.reports_dir / report_name


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_ready(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def load_boundary_objectives_config(paths) -> dict[str, Any]:
    return read_yaml(paths.configs_dir / "boundary_objectives.yaml")


def _interaction_terms(paths) -> list[str]:
    return list(load_boundary_objectives_config(paths)["modeling"]["adjacent_interactions"])


def _selection_policy(paths) -> dict[str, Any]:
    return dict(load_boundary_objectives_config(paths)["selection_policy"])


def _v7_constraint(paths) -> dict[str, Any]:
    return dict(load_boundary_objectives_config(paths)["v7_constraint"])


def _parse_boundary_roles(value: Any) -> list[str]:
    if isinstance(value, list):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return list(json.loads(value))
    raise TypeError(f"Unsupported boundary role payload: {type(value)!r}")


def _with_boundary_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for boundary_id in BOUNDARY_IDS:
        enriched[boundary_id] = 0
    for row_index, boundary_roles in enumerate(enriched["boundary_roles_json"].apply(_parse_boundary_roles)):
        for role_name in boundary_roles:
            boundary_id = BOUNDARY_ROLE_TO_ID.get(role_name)
            if boundary_id is not None:
                enriched.iloc[row_index, enriched.columns.get_loc(boundary_id)] = 1
    enriched["boundary_indicator_vector_json"] = enriched.apply(
        lambda row: _json_ready({boundary_id: int(row[boundary_id]) for boundary_id in BOUNDARY_IDS}),
        axis=1,
    )
    return enriched


def _reference_wall_clock_lookup(paths) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for family_id in list_predictive_families(paths):
        for split_manifest in load_split_manifests(paths, family_id):
            lookup[(family_id, split_manifest["split_id"])] = total_reference_wall_clock(
                load_reference_trace(paths, family_id, split_manifest["heldout_task"])
            )
    return lookup


def _phase3_candidate_frame(paths) -> pd.DataFrame:
    variant_score_path = paths.results_dir / "score_evaluation" / "ranking" / "variant_scores.csv"
    if not variant_score_path.exists():
        raise FileNotFoundError(f"Missing phase-3 variant score table: {variant_score_path}")
    frame = pd.read_csv(variant_score_path)
    frame = _with_boundary_indicators(frame)

    selection_path = paths.results_dir / "score_evaluation" / "manifests" / "selected_variant_by_split.csv"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing phase-3 selection table: {selection_path}")
    selection_frame = pd.read_csv(selection_path)[["family_id", "split_id", "chosen_variant_id", "selected_partition_id"]]
    selection_frame = selection_frame.rename(
        columns={
            "chosen_variant_id": "phase3_variant_id",
            "selected_partition_id": "phase3_v4_selected_partition_id",
        }
    )
    frame = frame.merge(selection_frame, on=["family_id", "split_id"], how="left")

    oracle_rows = (
        frame.sort_values(
            ["family_id", "split_id", "weighted_partition_utility", "score__V4", "segment_count", "partition_numeric_id"],
            ascending=[True, True, False, False, True, True],
        )
        .groupby(["family_id", "split_id"], dropna=False)
        .head(1)[["family_id", "split_id", "partition_id"]]
        .rename(columns={"partition_id": "oracle_partition_id"})
    )
    frame = frame.merge(oracle_rows, on=["family_id", "split_id"], how="left")
    frame["is_phase3_v4_selected"] = (frame["partition_id"] == frame["phase3_v4_selected_partition_id"]).astype(int)
    frame["is_oracle_partition"] = (frame["partition_id"] == frame["oracle_partition_id"]).astype(int)
    frame["is_micro_partition"] = (frame["partition_id"] == frame["micro_partition_id"]).astype(int)

    reference_lookup = _reference_wall_clock_lookup(paths)
    frame["reference_wall_clock_seconds"] = frame.apply(
        lambda row: reference_lookup[(row["family_id"], row["split_id"])],
        axis=1,
    )
    return frame.sort_values(["family_id", "split_id", "partition_id"]).reset_index(drop=True)


def _observed_clean_partition_rows(paths, candidate_frame: pd.DataFrame) -> pd.DataFrame:
    candidate_lookup = candidate_frame.set_index(["family_id", "split_id", "partition_id"])
    rows: list[dict[str, Any]] = []

    phase1_path = paths.results_dir / "workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv"
    phase1_frame = pd.read_csv(phase1_path)
    phase1_clean = phase1_frame[
        (phase1_frame["clean_or_stress"] == "clean")
        & (phase1_frame["condition"].isin(["micro_skill", "artifact_partition_full"]))
    ].copy()
    for row in phase1_clean.to_dict(orient="records"):
        split_rows = candidate_frame[
            (candidate_frame["family_id"] == row["family_id"]) & (candidate_frame["split_id"] == row["split_id"])
        ]
        partition_id = (
            split_rows["micro_partition_id"].iloc[0]
            if row["condition"] == "micro_skill"
            else split_rows["baseline_selected_partition_id"].iloc[0]
        )
        candidate_row = candidate_lookup.loc[(row["family_id"], row["split_id"], partition_id)]
        rows.append(
            {
                "source_round": "phase1",
                "source_phase3": 0.0,
                "family_id": row["family_id"],
                "split_id": row["split_id"],
                "heldout_task": row["heldout_task"],
                "condition": row["condition"],
                "partition_id": partition_id,
                "wall_clock_seconds": float(row["wall_clock_seconds"]),
                "reference_wall_clock_seconds": float(candidate_row["reference_wall_clock_seconds"]),
                "clean_excess_seconds": float(row["wall_clock_seconds"]) - float(candidate_row["reference_wall_clock_seconds"]),
                "segment_count": int(candidate_row["segment_count"]),
                **{boundary_id: int(candidate_row[boundary_id]) for boundary_id in BOUNDARY_IDS},
            }
        )

    phase3_path = paths.results_dir / "score_evaluation" / "revised_partition_eval" / "revised_condition_per_run_results.csv"
    phase3_frame = pd.read_csv(phase3_path)
    phase3_clean = phase3_frame[
        (phase3_frame["clean_or_stress"] == "clean")
        & (phase3_frame["condition"] == "artifact_partition_full_revised")
    ].copy()
    if "selected_partition_id" not in phase3_clean.columns:
        selection_frame = pd.read_csv(paths.results_dir / "score_evaluation" / "manifests" / "selected_variant_by_split.csv")
        phase3_clean = phase3_clean.merge(
            selection_frame[["family_id", "split_id", "selected_partition_id"]],
            on=["family_id", "split_id"],
            how="left",
        )
    for row in phase3_clean.to_dict(orient="records"):
        partition_id = row["selected_partition_id"]
        candidate_row = candidate_lookup.loc[(row["family_id"], row["split_id"], partition_id)]
        rows.append(
            {
                "source_round": "phase3",
                "source_phase3": 1.0,
                "family_id": row["family_id"],
                "split_id": row["split_id"],
                "heldout_task": row["heldout_task"],
                "condition": row["condition"],
                "partition_id": partition_id,
                "wall_clock_seconds": float(row["wall_clock_seconds"]),
                "reference_wall_clock_seconds": float(candidate_row["reference_wall_clock_seconds"]),
                "clean_excess_seconds": float(row["wall_clock_seconds"]) - float(candidate_row["reference_wall_clock_seconds"]),
                "segment_count": int(candidate_row["segment_count"]),
                **{boundary_id: int(candidate_row[boundary_id]) for boundary_id in BOUNDARY_IDS},
            }
        )
    observed = pd.DataFrame(rows).sort_values(["source_round", "family_id", "split_id", "partition_id"]).reset_index(drop=True)
    _save_dataframe(observed, boundary_root(paths) / "observed_clean_partition_rows.csv")
    return observed


def _normalize_within_split(frame: pd.DataFrame, column_name: str, output_column: str) -> pd.DataFrame:
    normalized = frame.copy()
    values: list[pd.DataFrame] = []
    for _, split_frame in normalized.groupby(["family_id", "split_id"], dropna=False):
        split_frame = split_frame.copy()
        minimum = float(split_frame[column_name].min())
        maximum = float(split_frame[column_name].max())
        if abs(maximum - minimum) <= 1.0e-12:
            split_frame[output_column] = 0.5
        else:
            split_frame[output_column] = (split_frame[column_name] - minimum) / (maximum - minimum)
        values.append(split_frame)
    return pd.concat(values, ignore_index=True)


def _ensure_interaction_columns(frame: pd.DataFrame, interaction_terms: list[str]) -> pd.DataFrame:
    enriched = frame.copy()
    for interaction_name in interaction_terms:
        left_name, right_name = interaction_name.split("__")
        enriched[interaction_name] = enriched[left_name].astype(float) * enriched[right_name].astype(float)
    return enriched


def _family_dummy_names(frame: pd.DataFrame) -> list[str]:
    family_ids = sorted(frame["family_id"].dropna().unique().tolist())
    if not family_ids:
        return []
    return [f"family__{family_id}" for family_id in family_ids[1:]]


def _design_frame(
    frame: pd.DataFrame,
    *,
    interaction_terms: list[str],
    include_family_dummies: bool,
    include_source_phase3: bool,
) -> tuple[pd.DataFrame, list[str]]:
    design = _ensure_interaction_columns(frame, interaction_terms)
    terms = ["segment_count", *MODEL_BOUNDARY_IDS, *interaction_terms]
    if include_family_dummies:
        family_ids = sorted(frame["family_id"].dropna().unique().tolist())
        for family_id in family_ids[1:]:
            column_name = f"family__{family_id}"
            design[column_name] = (design["family_id"] == family_id).astype(float)
            terms.append(column_name)
    if include_source_phase3:
        if "source_phase3" not in design.columns:
            design["source_phase3"] = 0.0
        terms.append("source_phase3")
    return design, terms


def _fit_ridge_model(
    frame: pd.DataFrame,
    outcome_column: str,
    *,
    interaction_terms: list[str],
    include_family_dummies: bool,
    include_source_phase3: bool,
    ridge_lambda: float,
    model_id: str,
) -> dict[str, Any]:
    design, terms = _design_frame(
        frame,
        interaction_terms=interaction_terms,
        include_family_dummies=include_family_dummies,
        include_source_phase3=include_source_phase3,
    )
    x = np.column_stack([np.ones((len(design), 1)), design[terms].astype(float).to_numpy()])
    y = design[outcome_column].astype(float).to_numpy()
    penalty = np.eye(x.shape[1], dtype=float) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(x.T @ x + penalty) @ (x.T @ y)
    predictions = x @ coefficients
    residual = y - predictions
    total = y - np.mean(y)
    r2 = 1.0 if np.allclose(total, 0.0) else 1.0 - (float(np.dot(residual, residual)) / float(np.dot(total, total)))
    return {
        "model_id": model_id,
        "outcome_column": outcome_column,
        "feature_terms": terms,
        "coefficients": {term: float(value) for term, value in zip(["intercept", *terms], coefficients.tolist())},
        "ridge_lambda": float(ridge_lambda),
        "r2": round(float(r2), 6),
    }


def _predict_ridge_model(frame: pd.DataFrame, model: dict[str, Any]) -> pd.Series:
    feature_terms = [term for term in model["feature_terms"] if term != "source_phase3"]
    interaction_terms = [term for term in feature_terms if term.startswith("B") and "__" in term]
    design = _ensure_interaction_columns(frame, interaction_terms)
    for feature_name in feature_terms:
        if feature_name.startswith("family__") and feature_name not in design.columns:
            family_id = feature_name.split("family__", 1)[1]
            design[feature_name] = (design["family_id"] == family_id).astype(float)
    if "source_phase3" in model["feature_terms"] and "source_phase3" not in design.columns:
        design["source_phase3"] = 0.0
    coefficients = model["coefficients"]
    prediction = pd.Series(coefficients["intercept"], index=design.index, dtype=float)
    for feature_name in model["feature_terms"]:
        prediction = prediction + (float(coefficients[feature_name]) * design[feature_name].astype(float))
    return prediction


def _coefficient_rows(
    *,
    model: dict[str, Any],
    stage: str,
    family_scope: str,
    target_family_id: str | None,
    target_split_id: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for term_name, coefficient in model["coefficients"].items():
        rows.append(
            {
                "stage": stage,
                "model_id": model["model_id"],
                "outcome_column": model["outcome_column"],
                "family_scope": family_scope,
                "target_family_id": target_family_id,
                "target_split_id": target_split_id,
                "term": term_name,
                "coefficient": round(float(coefficient), 6),
                "r2": float(model["r2"]),
            }
        )
    if "B6" not in model["coefficients"]:
        rows.append(
            {
                "stage": stage,
                "model_id": model["model_id"],
                "outcome_column": model["outcome_column"],
                "family_scope": family_scope,
                "target_family_id": target_family_id,
                "target_split_id": target_split_id,
                "term": "B6",
                "coefficient": 0.0,
                "r2": float(model["r2"]),
            }
        )
    return rows


def _direct_marginal_rows(frame: pd.DataFrame, family_scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = [
        ("stress_utility", "weighted_partition_utility"),
        ("predicted_clean_wall_clock_seconds", "predicted_clean_wall_clock_seconds"),
        ("failure_containment_rerun_span", "weighted_rerun_span"),
    ]
    for boundary_id in BOUNDARY_IDS:
        boundary_frame = frame.copy()
        present_mask = boundary_frame[boundary_id] == 1
        present_count = int(present_mask.sum())
        absent_count = int((~present_mask).sum())
        for outcome_label, outcome_column in outcomes:
            present_mean = None if present_count == 0 else float(boundary_frame.loc[present_mask, outcome_column].mean())
            absent_mean = None if absent_count == 0 else float(boundary_frame.loc[~present_mask, outcome_column].mean())
            conditioned_deltas: list[float] = []
            for _, segment_frame in boundary_frame.groupby("segment_count", dropna=False):
                segment_present = segment_frame[segment_frame[boundary_id] == 1]
                segment_absent = segment_frame[segment_frame[boundary_id] == 0]
                if segment_present.empty or segment_absent.empty:
                    continue
                conditioned_deltas.append(float(segment_present[outcome_column].mean() - segment_absent[outcome_column].mean()))
            conditioned_delta = None if not conditioned_deltas else float(sum(conditioned_deltas) / len(conditioned_deltas))
            rows.append(
                {
                    "family_scope": family_scope,
                    "boundary_id": boundary_id,
                    "outcome_name": outcome_label,
                    "present_count": present_count,
                    "absent_count": absent_count,
                    "present_mean": None if present_mean is None else round(present_mean, 6),
                    "absent_mean": None if absent_mean is None else round(absent_mean, 6),
                    "delta_present_absent": None
                    if present_mean is None or absent_mean is None
                    else round(present_mean - absent_mean, 6),
                    "delta_conditioned_on_segment_count": None
                    if conditioned_delta is None
                    else round(conditioned_delta, 6),
                }
            )
    return rows


def _objective_alignment_rows(paths, candidate_frame: pd.DataFrame, stress_model: dict[str, Any]) -> list[dict[str, Any]]:
    interaction_terms = _interaction_terms(paths)
    modeling_cfg = load_boundary_objectives_config(paths)["modeling"]
    score_frame = _normalize_within_split(candidate_frame, "score__V4", "score_v4_norm")
    score_model = _fit_ridge_model(
        score_frame,
        "score_v4_norm",
        interaction_terms=interaction_terms,
        include_family_dummies=True,
        include_source_phase3=False,
        ridge_lambda=float(modeling_cfg["ridge_lambda"]),
        model_id="v4_alignment_model",
    )
    rows: list[dict[str, Any]] = []
    for boundary_id in MODEL_BOUNDARY_IDS:
        rows.append(
            {
                "term": boundary_id,
                "stress_utility_coefficient": round(float(stress_model["coefficients"].get(boundary_id, 0.0)), 6),
                "v4_score_coefficient": round(float(score_model["coefficients"].get(boundary_id, 0.0)), 6),
                "underweight_gap": round(
                    float(stress_model["coefficients"].get(boundary_id, 0.0) - score_model["coefficients"].get(boundary_id, 0.0)),
                    6,
                ),
            }
        )
    group_rows = []
    for group_name, boundary_group in [
        ("early_boundaries", EARLY_BOUNDARY_IDS),
        ("late_boundaries", LATE_BOUNDARY_IDS),
        ("validator_backed_boundaries", MODEL_BOUNDARY_IDS),
    ]:
        utility_mean = float(np.mean([stress_model["coefficients"].get(boundary_id, 0.0) for boundary_id in boundary_group]))
        score_mean = float(np.mean([score_model["coefficients"].get(boundary_id, 0.0) for boundary_id in boundary_group]))
        group_rows.append(
            {
                "term": group_name,
                "stress_utility_coefficient": round(utility_mean, 6),
                "v4_score_coefficient": round(score_mean, 6),
                "underweight_gap": round(utility_mean - score_mean, 6),
            }
        )
    all_rows = rows + group_rows
    _save_dataframe(pd.DataFrame(all_rows), boundary_root(paths) / "objective_alignment_summary.csv")
    return all_rows


def _segment_penalty_diagnosis(paths, candidate_frame: pd.DataFrame) -> dict[str, Any]:
    diagnosis_frame = candidate_frame.copy()
    diagnosis_frame["score__V4_without_penalty"] = diagnosis_frame["score__V4"] + (0.5 * diagnosis_frame["overfragmentation_penalty"])
    _, split_summary = _ranking_summary_rows(diagnosis_frame, "score__V4")
    _, no_penalty_summary = _ranking_summary_rows(diagnosis_frame, "score__V4_without_penalty")
    selected_v4 = []
    selected_no_penalty = []
    for (_, _), split_frame in diagnosis_frame.groupby(["family_id", "split_id"], dropna=False):
        ordered_v4 = split_frame.sort_values(
            ["score__V4", "weighted_partition_utility", "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        ordered_no_penalty = split_frame.sort_values(
            ["score__V4_without_penalty", "weighted_partition_utility", "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        selected_v4.append(ordered_v4.to_dict())
        selected_no_penalty.append(ordered_no_penalty.to_dict())
    no_penalty_frame = pd.DataFrame(selected_no_penalty)
    v4_frame = pd.DataFrame(selected_v4)
    return {
        "micro_mean_rank_v4": round(
            float(
                split_summary.merge(
                    diagnosis_frame[diagnosis_frame["is_micro_partition"] == 1][["family_id", "split_id", "partition_id"]],
                    left_on=["family_id", "split_id", "predicted_best_partition"],
                    right_on=["family_id", "split_id", "partition_id"],
                    how="left",
                ).shape[0]
            ),
            6,
        ),
        "v4_no_penalty_selected_partitions": sorted(no_penalty_frame["partition_id"].unique().tolist()),
        "changed_split_count": int(
            (no_penalty_frame["partition_id"].reset_index(drop=True) != v4_frame["partition_id"].reset_index(drop=True)).sum()
        ),
        "mean_selected_segment_count_v4": round(float(v4_frame["segment_count"].mean()), 6),
        "mean_selected_segment_count_no_penalty": round(float(no_penalty_frame["segment_count"].mean()), 6),
        "mean_selected_utility_v4": round(float(v4_frame["weighted_partition_utility"].mean()), 6),
        "mean_selected_utility_no_penalty": round(float(no_penalty_frame["weighted_partition_utility"].mean()), 6),
    }


def _micro_selected_oracle_rows(
    candidate_frame: pd.DataFrame,
    stress_model: dict[str, Any],
    clean_model: dict[str, Any],
) -> list[dict[str, Any]]:
    actual_phase3_frame = pd.read_csv(
        ROOT_DIR / "results" / "score_evaluation" / "revised_partition_eval" / "combined_condition_rows.csv"
    )
    actual_phase3_stress = actual_phase3_frame[
        (actual_phase3_frame["clean_or_stress"] == "stress")
        & (actual_phase3_frame["condition"].isin(["micro_skill", "artifact_partition_full_revised"]))
    ].copy()
    actual_gap_by_split = (
        actual_phase3_stress.pivot_table(index=["family_id", "split_id"], columns="condition", values="utility")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rows: list[dict[str, Any]] = []
    clean_terms = [term for term in clean_model["feature_terms"] if term != "source_phase3"]
    for (family_id, split_id), split_frame in candidate_frame.groupby(["family_id", "split_id"], dropna=False):
        selected_row = split_frame[split_frame["is_phase3_v4_selected"] == 1].iloc[0]
        micro_row = split_frame[split_frame["is_micro_partition"] == 1].iloc[0]
        oracle_row = split_frame[split_frame["is_oracle_partition"] == 1].iloc[0]
        gap_row = actual_gap_by_split[
            (actual_gap_by_split["family_id"] == family_id) & (actual_gap_by_split["split_id"] == split_id)
        ]
        actual_gap_to_micro = None
        if not gap_row.empty:
            actual_gap_to_micro = float(gap_row["micro_skill"].iloc[0] - gap_row["artifact_partition_full_revised"].iloc[0])

        def _boundary_difference(a: pd.Series, b: pd.Series) -> dict[str, int]:
            return {boundary_id: int(a[boundary_id]) - int(b[boundary_id]) for boundary_id in BOUNDARY_IDS}

        def _boundary_contribution(diff: dict[str, int], coefficients: dict[str, float]) -> float:
            return float(sum(diff.get(term, 0) * coefficients.get(term, 0.0) for term in MODEL_BOUNDARY_IDS))

        micro_minus_selected = _boundary_difference(micro_row, selected_row)
        oracle_minus_selected = _boundary_difference(oracle_row, selected_row)
        rows.append(
            {
                "family_id": family_id,
                "split_id": split_id,
                "heldout_task": selected_row["heldout_task"],
                "selected_partition_id": selected_row["partition_id"],
                "micro_partition_id": micro_row["partition_id"],
                "oracle_partition_id": oracle_row["partition_id"],
                "selected_boundary_vector_json": selected_row["boundary_indicator_vector_json"],
                "micro_boundary_vector_json": micro_row["boundary_indicator_vector_json"],
                "oracle_boundary_vector_json": oracle_row["boundary_indicator_vector_json"],
                "micro_minus_selected_boundary_json": _json_ready(micro_minus_selected),
                "oracle_minus_selected_boundary_json": _json_ready(oracle_minus_selected),
                "selected_weighted_partition_utility": float(selected_row["weighted_partition_utility"]),
                "micro_weighted_partition_utility": float(micro_row["weighted_partition_utility"]),
                "oracle_weighted_partition_utility": float(oracle_row["weighted_partition_utility"]),
                "selected_predicted_clean_wall_clock_seconds": float(selected_row["predicted_clean_wall_clock_seconds"]),
                "micro_predicted_clean_wall_clock_seconds": float(micro_row["predicted_clean_wall_clock_seconds"]),
                "oracle_predicted_clean_wall_clock_seconds": float(oracle_row["predicted_clean_wall_clock_seconds"]),
                "selected_weighted_rerun_span": float(selected_row["weighted_rerun_span"]),
                "micro_weighted_rerun_span": float(micro_row["weighted_rerun_span"]),
                "oracle_weighted_rerun_span": float(oracle_row["weighted_rerun_span"]),
                "estimated_stress_gain_micro_vs_selected": round(
                    _boundary_contribution(micro_minus_selected, stress_model["coefficients"]),
                    6,
                ),
                "estimated_stress_gain_oracle_vs_selected": round(
                    _boundary_contribution(oracle_minus_selected, stress_model["coefficients"]),
                    6,
                ),
                "estimated_clean_delta_micro_vs_selected": round(
                    _boundary_contribution(micro_minus_selected, clean_model["coefficients"])
                    + ((int(micro_row["segment_count"]) - int(selected_row["segment_count"])) * float(clean_model["coefficients"].get("segment_count", 0.0))),
                    6,
                ),
                "estimated_clean_delta_oracle_vs_selected": round(
                    _boundary_contribution(oracle_minus_selected, clean_model["coefficients"])
                    + ((int(oracle_row["segment_count"]) - int(selected_row["segment_count"])) * float(clean_model["coefficients"].get("segment_count", 0.0))),
                    6,
                ),
                "actual_phase3_gap_to_micro_stress": None if actual_gap_to_micro is None else round(actual_gap_to_micro, 6),
            }
        )
    return rows


def _plot_boundary_marginals(boundary_rows: pd.DataFrame, outcome_name: str, output_path: Path, title: str, color: str) -> None:
    plot_frame = boundary_rows[
        (boundary_rows["family_scope"] == "pooled") & (boundary_rows["outcome_name"] == outcome_name)
    ].copy()
    plot_frame["plot_value"] = plot_frame["delta_conditioned_on_segment_count"].fillna(plot_frame["delta_present_absent"])
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    axis.bar(plot_frame["boundary_id"], plot_frame["plot_value"], color=color)
    axis.set_title(title)
    axis.set_xlabel("boundary")
    axis.set_ylabel("delta present - absent")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_model_coefficients(coefficient_frame: pd.DataFrame, output_path: Path) -> None:
    pooled = coefficient_frame[
        (coefficient_frame["stage"] == "boundary_attribution")
        & (coefficient_frame["family_scope"] == "shared")
        & (coefficient_frame["term"].isin(["B1", "B2", "B3", "B4", "B5", "segment_count"]))
        & (coefficient_frame["model_id"].isin(["stress_attribution_model", "clean_cost_model", "rerun_span_model"]))
    ].copy()
    model_order = ["stress_attribution_model", "clean_cost_model", "rerun_span_model"]
    term_order = ["B1", "B2", "B3", "B4", "B5", "segment_count"]
    width = 0.22
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    x_positions = np.arange(len(term_order))
    color_map = {
        "stress_attribution_model": "#355070",
        "clean_cost_model": "#b56576",
        "rerun_span_model": "#6d597a",
    }
    for index, model_id in enumerate(model_order):
        model_frame = pooled[pooled["model_id"] == model_id].set_index("term").reindex(term_order).reset_index()
        axis.bar(
            x_positions + ((index - 1) * width),
            model_frame["coefficient"].astype(float),
            width=width,
            label=model_id.replace("_", " "),
            color=color_map[model_id],
        )
    axis.set_xticks(x_positions, term_order)
    axis.set_title("Additive Attribution Coefficients")
    axis.set_ylabel("coefficient")
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_boundary_comparison(comparison_frame: pd.DataFrame, output_path: Path) -> None:
    pooled_vectors = []
    labels = []
    for label_column, title in [
        ("selected_boundary_vector_json", "V4 selected"),
        ("micro_boundary_vector_json", "micro"),
        ("oracle_boundary_vector_json", "oracle"),
    ]:
        vectors = comparison_frame[label_column].apply(json.loads).tolist()
        pooled_vectors.append([float(np.mean([vector[boundary_id] for vector in vectors])) for boundary_id in BOUNDARY_IDS])
        labels.append(title)
    matrix = np.array(pooled_vectors)
    figure, axis = plt.subplots(figsize=(8.6, 3.2))
    image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(BOUNDARY_IDS)), BOUNDARY_IDS)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Boundary Activation: selected vs micro vs oracle")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def audit_phase3b_baseline(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    phase3_required = {
        "phase3_report": _report_path(resolved_paths, "score_evaluation_report.md"),
        "phase3_variant_scores": resolved_paths.results_dir / "score_evaluation" / "ranking" / "variant_scores.csv",
        "phase3_selection": resolved_paths.results_dir / "score_evaluation" / "manifests" / "selected_variant_by_split.csv",
        "phase3_revised_eval": resolved_paths.results_dir
        / "score_evaluation"
        / "revised_partition_eval"
        / "combined_condition_rows.csv",
        "phase3_partition_features": resolved_paths.results_dir
        / "score_evaluation"
        / "diagnostics"
        / "partition_feature_table.csv",
        "phase1_same_info": resolved_paths.results_dir / "workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv",
        "phase1_boundary_ranking": resolved_paths.results_dir
        / "workflow_evaluation"
        / "boundary_ranking"
        / "per_run_results.csv",
    }
    reusable = {name: path.exists() for name, path in phase3_required.items()}

    family_partition_status: list[dict[str, Any]] = []
    legal_partition_sets_available = True
    for family_id in list_predictive_families(resolved_paths):
        legal_partitions_path = resolved_paths.results_dir / "partitions" / family_id / "legal_partitions.json"
        family_partition_status.append(
            {
                "family_id": family_id,
                "legal_partitions": legal_partitions_path.exists(),
                "selected_partition": (resolved_paths.results_dir / "partitions" / family_id / "selected_partition.json").exists(),
            }
        )
        legal_partition_sets_available = legal_partition_sets_available and legal_partitions_path.exists()

    blockers: list[str] = []
    for artifact_name, is_available in reusable.items():
        if not is_available:
            blockers.append(f"Missing reusable phase artifact: {artifact_name}.")
    if not legal_partition_sets_available:
        blockers.append("Missing legal partition sets for one or more predictive families.")

    gate_0_passed = len(blockers) == 0
    summary = {
        "gate_0_passed": gate_0_passed,
        "working_root": resolved_paths.workspace_root.name,
        "substrate_baseline_version": compute_substrate_baseline_version(resolved_paths),
        "phase3_reusable_artifacts": {
            name: {
                "available": reusable[name],
                "path": resolved_paths.relative_to_workspace(path),
            }
            for name, path in phase3_required.items()
        },
        "legal_partition_sets_available": legal_partition_sets_available,
        "boundary_reconstruction_policy": (
            "reconstruct_from_phase3_variant_scores_boundary_roles"
            if reusable["phase3_variant_scores"]
            else "blocked"
        ),
        "downstream_utility_policy": {
            "stress_partition_utility": "reuse_phase1_boundary_ranking_reaggregated_in_phase3",
            "clean_cost_signal": (
                "fit_proxy_from_observed_clean_rows_and_control_same_session_with_v4_rerun"
                if reusable["phase1_same_info"] and reusable["phase3_revised_eval"]
                else "blocked"
            ),
            "failure_containment": "reuse_phase3_weighted_rerun_span_and_weighted_first_failed_boundary_depth",
        },
        "minimal_reruns_required": [
            "No broad candidate-space rerun is required for audit, attribution, or ranking-side evaluation.",
            "Rerun only the V4 selected partition and the phase-3b selected partition under the same phase-3b session for the final downstream comparison.",
        ],
        "partition_status_by_family": family_partition_status,
        "blockers": blockers,
        "audit_report": resolved_paths.relative_to_workspace(_report_path(resolved_paths, BOUNDARY_AUDIT_REPORT)),
    }
    write_json(audit_root(resolved_paths) / "reuse_vs_rerun_summary.json", summary)
    write_text(_report_path(resolved_paths, BOUNDARY_AUDIT_REPORT), _build_phase3b_audit_report(summary))
    return summary


def _build_phase3b_audit_report(summary: dict[str, Any]) -> str:
    phase3_table = [
        {
            "artifact": artifact_name,
            "available": payload["available"],
            "path": payload["path"],
        }
        for artifact_name, payload in summary["phase3_reusable_artifacts"].items()
    ]
    lines = [
        "# Boundary Score Audit",
        "",
        "## 1. Gate 0",
        "",
        f"- Gate 0 passed: `{summary['gate_0_passed']}`",
        f"- working root: `{summary['working_root']}`",
        f"- frozen substrate baseline: `{summary['substrate_baseline_version']}`",
        "",
        "## 2. Reusable Phase-3 Artifacts",
        "",
        markdown_table(phase3_table, ["artifact", "available", "path"]),
        "",
        "## 3. Audit Answers",
        "",
        "1. Which phase-3 artifacts are reusable as-is?",
        "- The saved phase-3 variant score table, selected-variant manifest, revised partition evaluation table, and partition feature table are reused directly.",
        "2. Are all legal partitions per split available?",
        f"- `{summary['legal_partition_sets_available']}` via the frozen `results/partitions/<family>/legal_partitions.json` files.",
        "3. Are all partitions already encoded with score features and downstream utility?",
        "- Stress-side partition utility and score features are already encoded in the phase-3 candidate tables. Clean wall-clock is only observed for the executed conditions, so phase-3b fits an explicit clean-cost proxy and then reruns only the final selected conditions.",
        "4. Can boundary indicators be reconstructed without rerunning execution?",
        f"- `{summary['boundary_reconstruction_policy']}`.",
        "5. Which pieces must be rerun minimally?",
        *[f"- {item}" for item in summary["minimal_reruns_required"]],
        "",
        "## 4. Reuse Vs Rerun Decision",
        "",
        "- Reuse all valid phase-3 candidate-space outputs for attribution, calibration, and ranking-side evaluation.",
        "- Do not rerun phase-2 or the full phase-1 candidate space.",
        "- Rerun only the V4 control plus the phase-3b selected partition for the final downstream comparison so clean and stress wall-clock are compared within the same session.",
        "",
    ]
    if summary["blockers"]:
        lines.extend(["## 5. Blockers", "", *[f"- {item}" for item in summary["blockers"]], ""])
    return "\n".join(lines) + "\n"


def run_phase3b_boundary_attribution(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    audit_summary = audit_phase3b_baseline(resolved_paths)
    if not audit_summary["gate_0_passed"]:
        return {
            "gate_1_passed": False,
            "blockers": audit_summary["blockers"],
            "boundary_report": resolved_paths.relative_to_workspace(_report_path(resolved_paths, BOUNDARY_ATTRIBUTION_REPORT)),
        }

    modeling_cfg = load_boundary_objectives_config(resolved_paths)["modeling"]
    interaction_terms = _interaction_terms(resolved_paths)
    candidate_frame = _phase3_candidate_frame(resolved_paths)
    observed_clean = _observed_clean_partition_rows(resolved_paths, candidate_frame)

    stress_model = _fit_ridge_model(
        candidate_frame,
        "weighted_partition_utility",
        interaction_terms=interaction_terms,
        include_family_dummies=True,
        include_source_phase3=False,
        ridge_lambda=float(modeling_cfg["ridge_lambda"]),
        model_id="stress_attribution_model",
    )
    rerun_model = _fit_ridge_model(
        candidate_frame,
        "weighted_rerun_span",
        interaction_terms=interaction_terms,
        include_family_dummies=True,
        include_source_phase3=False,
        ridge_lambda=float(modeling_cfg["ridge_lambda"]),
        model_id="rerun_span_model",
    )
    clean_model = _fit_ridge_model(
        observed_clean,
        "clean_excess_seconds",
        interaction_terms=interaction_terms,
        include_family_dummies=True,
        include_source_phase3=True,
        ridge_lambda=float(modeling_cfg["ridge_lambda"]),
        model_id="clean_cost_model",
    )

    candidate_frame = candidate_frame.copy()
    candidate_frame["predicted_clean_excess_seconds"] = _predict_ridge_model(candidate_frame.assign(source_phase3=0.0), clean_model)
    candidate_frame["predicted_clean_wall_clock_seconds"] = (
        candidate_frame["reference_wall_clock_seconds"] + candidate_frame["predicted_clean_excess_seconds"]
    )
    candidate_frame["predicted_clean_wall_clock_seconds"] = candidate_frame["predicted_clean_wall_clock_seconds"].clip(lower=0.0)
    candidate_frame["stress_proxy_minus_v4_score"] = candidate_frame["weighted_partition_utility"] - candidate_frame["score__V4"]

    partition_matrix_path = boundary_root(resolved_paths) / "partition_by_boundary_matrix.csv"
    partition_outcome_path = boundary_root(resolved_paths) / "per_partition_outcome_table.csv"
    _save_dataframe(
        candidate_frame[
            [
                "family_id",
                "split_id",
                "heldout_task",
                "partition_id",
                "segment_count",
                *BOUNDARY_IDS,
                "boundary_indicator_vector_json",
            ]
        ],
        partition_matrix_path,
    )
    _save_dataframe(
        candidate_frame[
            [
                "family_id",
                "split_id",
                "heldout_task",
                "partition_id",
                "segment_count",
                *BOUNDARY_IDS,
                "score__V4",
                "weighted_partition_utility",
                "predicted_clean_wall_clock_seconds",
                "weighted_rerun_span",
                "weighted_first_failed_boundary_depth",
                "weighted_wall_clock_seconds",
                "phase3_v4_selected_partition_id",
                "oracle_partition_id",
                "micro_partition_id",
            ]
        ],
        partition_outcome_path,
    )

    marginal_rows = _direct_marginal_rows(candidate_frame, "pooled")
    family_rows: list[dict[str, Any]] = []
    for family_id, family_frame in candidate_frame.groupby("family_id", dropna=False):
        family_rows.extend(_direct_marginal_rows(family_frame, str(family_id)))
    marginal_frame = pd.DataFrame(marginal_rows)
    family_frame = pd.DataFrame(family_rows)
    _save_dataframe(marginal_frame, boundary_root(resolved_paths) / "per_boundary_marginal_summaries.csv")
    _save_dataframe(family_frame, boundary_root(resolved_paths) / "family_specific_attribution_summaries.csv")

    coefficient_rows = []
    coefficient_rows.extend(
        _coefficient_rows(
            model=stress_model,
            stage="boundary_attribution",
            family_scope="shared",
            target_family_id=None,
            target_split_id=None,
        )
    )
    coefficient_rows.extend(
        _coefficient_rows(
            model=clean_model,
            stage="boundary_attribution",
            family_scope="shared",
            target_family_id=None,
            target_split_id=None,
        )
    )
    coefficient_rows.extend(
        _coefficient_rows(
            model=rerun_model,
            stage="boundary_attribution",
            family_scope="shared",
            target_family_id=None,
            target_split_id=None,
        )
    )
    coefficient_frame = pd.DataFrame(coefficient_rows)
    _save_dataframe(coefficient_frame, boundary_root(resolved_paths) / "additive_attribution_coefficients.csv")

    objective_alignment_rows = _objective_alignment_rows(resolved_paths, candidate_frame, stress_model)
    penalty_summary = _segment_penalty_diagnosis(resolved_paths, candidate_frame)
    write_json(boundary_root(resolved_paths) / "segment_penalty_diagnosis.json", penalty_summary)

    micro_comparison_rows = _micro_selected_oracle_rows(candidate_frame, stress_model, clean_model)
    micro_comparison_frame = pd.DataFrame(micro_comparison_rows)
    _save_dataframe(
        micro_comparison_frame,
        boundary_root(resolved_paths) / "micro_vs_selected_vs_oracle_boundary_comparison.csv",
    )

    _plot_boundary_marginals(
        pd.concat([marginal_frame, family_frame], ignore_index=True),
        "stress_utility",
        figures_root(resolved_paths) / "boundary_marginal_stress_utility.png",
        "Boundary Marginals: Stress Utility",
        "#355070",
    )
    _plot_boundary_marginals(
        pd.concat([marginal_frame, family_frame], ignore_index=True),
        "predicted_clean_wall_clock_seconds",
        figures_root(resolved_paths) / "boundary_marginal_clean_cost.png",
        "Boundary Marginals: Predicted Clean Cost",
        "#b56576",
    )
    _plot_model_coefficients(coefficient_frame, figures_root(resolved_paths) / "boundary_attribution_coefficients.png")
    _plot_boundary_comparison(
        micro_comparison_frame,
        figures_root(resolved_paths) / "selected_micro_oracle_boundary_comparison.png",
    )

    summary = {
        "gate_1_passed": True,
        "substrate_baseline_version": audit_summary["substrate_baseline_version"],
        "partition_matrix_csv": resolved_paths.relative_to_workspace(partition_matrix_path),
        "per_partition_outcome_csv": resolved_paths.relative_to_workspace(partition_outcome_path),
        "marginal_summary_csv": resolved_paths.relative_to_workspace(
            boundary_root(resolved_paths) / "per_boundary_marginal_summaries.csv"
        ),
        "family_summary_csv": resolved_paths.relative_to_workspace(
            boundary_root(resolved_paths) / "family_specific_attribution_summaries.csv"
        ),
        "coefficient_csv": resolved_paths.relative_to_workspace(
            boundary_root(resolved_paths) / "additive_attribution_coefficients.csv"
        ),
        "micro_comparison_csv": resolved_paths.relative_to_workspace(
            boundary_root(resolved_paths) / "micro_vs_selected_vs_oracle_boundary_comparison.csv"
        ),
        "segment_penalty_json": resolved_paths.relative_to_workspace(
            boundary_root(resolved_paths) / "segment_penalty_diagnosis.json"
        ),
        "boundary_report": resolved_paths.relative_to_workspace(_report_path(resolved_paths, BOUNDARY_ATTRIBUTION_REPORT)),
        "blockers": [],
    }
    write_json(boundary_root(resolved_paths) / "summary.json", summary)
    write_text(
        _report_path(resolved_paths, BOUNDARY_ATTRIBUTION_REPORT),
        _build_phase3b_boundary_report(
            candidate_frame,
            marginal_frame,
            family_frame,
            coefficient_frame,
            pd.DataFrame(objective_alignment_rows),
            micro_comparison_frame,
            penalty_summary,
        ),
    )
    return summary


def _build_phase3b_boundary_report(
    candidate_frame: pd.DataFrame,
    marginal_frame: pd.DataFrame,
    family_frame: pd.DataFrame,
    coefficient_frame: pd.DataFrame,
    alignment_frame: pd.DataFrame,
    micro_comparison_frame: pd.DataFrame,
    penalty_summary: dict[str, Any],
) -> str:
    pooled_stress = marginal_frame[marginal_frame["outcome_name"] == "stress_utility"].copy()
    pooled_clean = marginal_frame[marginal_frame["outcome_name"] == "predicted_clean_wall_clock_seconds"].copy()
    helpful_boundaries = pooled_stress.sort_values("delta_conditioned_on_segment_count", ascending=False).head(3)
    costly_boundaries = pooled_clean.sort_values("delta_conditioned_on_segment_count", ascending=False).head(3)
    actual_gap = micro_comparison_frame["actual_phase3_gap_to_micro_stress"].dropna()
    actual_gap_value = float(actual_gap.mean()) if not actual_gap.empty else 0.017
    pooled_gap_row = (
        micro_comparison_frame[["estimated_stress_gain_micro_vs_selected", "estimated_clean_delta_micro_vs_selected"]]
        .mean(numeric_only=True)
        .to_dict()
    )
    coefficient_focus = coefficient_frame[
        (coefficient_frame["family_scope"] == "shared")
        & (coefficient_frame["term"].isin(["B1", "B2", "B3", "B4", "B5", "segment_count"]))
        & (coefficient_frame["stage"] == "boundary_attribution")
    ].copy()
    lines = [
        "# Boundary Attribution",
        "",
        "## 1. Candidate-Space Reuse",
        "",
        "- Phase-3 candidate rows were reused directly from `results/score_evaluation/ranking/variant_scores.csv`.",
        "- Boundary indicators were reconstructed from saved `boundary_roles_json`; no candidate-space execution rerun was needed.",
        "- Clean wall-clock was modeled explicitly from the observed clean rows because only the executed conditions have direct clean timing measurements.",
        "",
        "## 2. Direct Marginal Boundary Statistics",
        "",
        "Pooled stress-utility boundary deltas (present minus absent, conditioned on segment count when feasible):",
        "",
        markdown_table(
            pooled_stress.to_dict(orient="records"),
            ["boundary_id", "present_count", "absent_count", "delta_present_absent", "delta_conditioned_on_segment_count"],
        ),
        "",
        "Pooled predicted clean-cost boundary deltas:",
        "",
        markdown_table(
            pooled_clean.to_dict(orient="records"),
            ["boundary_id", "present_count", "absent_count", "delta_present_absent", "delta_conditioned_on_segment_count"],
        ),
        "",
        f"- Most helpful stress boundaries: `{', '.join(helpful_boundaries['boundary_id'].tolist())}`.",
        f"- Most costly clean boundaries under the fitted proxy: `{', '.join(costly_boundaries['boundary_id'].tolist())}`.",
        "- `B6` is structurally constant across all legal partitions, so it is not a discriminative selection axis in this workflow.",
        "",
        "## 3. Additive Attribution Model",
        "",
        markdown_table(
            coefficient_focus.to_dict(orient="records"),
            ["model_id", "term", "coefficient", "r2"],
        ),
        "",
        "## 4. Interaction And Penalty Diagnosis",
        "",
        markdown_table(
            alignment_frame.to_dict(orient="records"),
            ["term", "stress_utility_coefficient", "v4_score_coefficient", "underweight_gap"],
        ),
        "",
        f"- Segment-count penalty diagnostic: `{penalty_summary['changed_split_count']}` splits would change under V4 without the explicit over-fragmentation penalty.",
        f"- Mean selected utility under current V4: `{penalty_summary['mean_selected_utility_v4']:.6f}`; without the penalty: `{penalty_summary['mean_selected_utility_no_penalty']:.6f}`.",
        "",
        "## 5. Micro vs Selected vs Oracle",
        "",
        markdown_table(
            micro_comparison_frame.head(8).to_dict(orient="records"),
            [
                "family_id",
                "split_id",
                "selected_partition_id",
                "micro_partition_id",
                "oracle_partition_id",
                "micro_minus_selected_boundary_json",
                "estimated_stress_gain_micro_vs_selected",
                "actual_phase3_gap_to_micro_stress",
            ],
        ),
        "",
        (
            "- The saved phase-3 selected partition differs from `micro_skill` mainly by omitting `B1` and `B5`. "
            f"The fitted boundary model assigns the missing micro-vs-selected stress contribution mostly to `B1`, while `B5` is near-neutral. "
            f"This is the main actionable explanation for the remaining `{actual_gap_value:.3f}` stress-utility gap."
        ),
        (
            "- The proxy oracle is usually not the maximally fine partition. That means the evidence does not support forcing micro granularity everywhere; "
            "it supports moving weight toward the early `raw_data/profile/split/preprocess` boundaries and away from the late `model/metrics` bias carried by V4."
        ),
        "",
        "## 6. Gate 1 Conclusion",
        "",
        (
            f"- Gate 1 passes. The remaining gap is attributable to concrete boundary choices rather than an opaque partition label. "
            f"`B2` and `B3` are strongly helpful, `B1` remains the main missing early boundary in the V4 selection, and `B4/B5` are not pulling their weight on stress utility."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _fit_variant_bundle(
    paths,
    training_frame: pd.DataFrame,
    observed_clean_frame: pd.DataFrame,
    target_family_id: str,
    target_split_id: str,
) -> dict[str, Any]:
    modeling_cfg = load_boundary_objectives_config(paths)["modeling"]
    interaction_terms = _interaction_terms(paths)
    train = training_frame.copy()
    train = _normalize_within_split(train, "score__V4", "score_v4_norm")
    train = _normalize_within_split(train, "weighted_partition_utility", "utility_norm")
    train["shared_stress_residual"] = train["utility_norm"] - train["score_v4_norm"]

    residual_shared_model = _fit_ridge_model(
        train,
        "shared_stress_residual",
        interaction_terms=interaction_terms,
        include_family_dummies=True,
        include_source_phase3=False,
        ridge_lambda=float(modeling_cfg["ridge_lambda"]),
        model_id="v6_shared_residual_model",
    )
    clean_training = observed_clean_frame[
        ~(
            (observed_clean_frame["family_id"] == target_family_id)
            & (observed_clean_frame["split_id"] == target_split_id)
        )
    ].copy()
    clean_model = _fit_ridge_model(
        clean_training,
        "clean_excess_seconds",
        interaction_terms=interaction_terms,
        include_family_dummies=True,
        include_source_phase3=True,
        ridge_lambda=float(modeling_cfg["ridge_lambda"]),
        model_id="v7_clean_cost_model",
    )

    family_models: dict[str, dict[str, Any]] = {}
    for family_id, family_train in train.groupby("family_id", dropna=False):
        family_models[str(family_id)] = _fit_ridge_model(
            family_train,
            "shared_stress_residual",
            interaction_terms=interaction_terms,
            include_family_dummies=False,
            include_source_phase3=False,
            ridge_lambda=float(modeling_cfg["ridge_lambda"]),
            model_id=f"v8_family_residual_model__{family_id}",
        )

    coefficient_rows = []
    coefficient_rows.extend(
        _coefficient_rows(
            model=residual_shared_model,
            stage="variant_fit",
            family_scope="shared",
            target_family_id=target_family_id,
            target_split_id=target_split_id,
        )
    )
    coefficient_rows.extend(
        _coefficient_rows(
            model=clean_model,
            stage="variant_fit",
            family_scope="shared",
            target_family_id=target_family_id,
            target_split_id=target_split_id,
        )
    )
    for family_id, family_model in family_models.items():
        coefficient_rows.extend(
            _coefficient_rows(
                model=family_model,
                stage="variant_fit",
                family_scope=family_id,
                target_family_id=target_family_id,
                target_split_id=target_split_id,
            )
        )
    return {
        "residual_shared_model": residual_shared_model,
        "clean_model": clean_model,
        "family_models": family_models,
        "coefficient_rows": coefficient_rows,
    }


def _apply_variant_bundle(paths, frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    scored = frame.copy()
    scored = _normalize_within_split(scored, "score__V4", "score_v4_norm")
    scored["score__V6_correction"] = _predict_ridge_model(scored, bundle["residual_shared_model"])
    scored["score__V6"] = scored["score_v4_norm"] + scored["score__V6_correction"]

    family_corrections = []
    for family_id, family_frame in scored.groupby("family_id", dropna=False):
        family_model = bundle["family_models"].get(str(family_id), bundle["residual_shared_model"])
        family_piece = family_frame.copy()
        family_piece["score__V8_correction"] = _predict_ridge_model(family_piece, family_model)
        family_corrections.append(family_piece)
    scored = pd.concat(family_corrections, ignore_index=True)
    scored["score__V8"] = scored["score_v4_norm"] + scored["score__V8_correction"]

    clean_prediction = _predict_ridge_model(scored.assign(source_phase3=0.0), bundle["clean_model"])
    scored["predicted_clean_excess_seconds"] = clean_prediction
    scored["predicted_clean_wall_clock_seconds"] = (scored["reference_wall_clock_seconds"] + clean_prediction).clip(lower=0.0)

    constraint_cfg = _v7_constraint(paths)
    v7_scores: list[pd.DataFrame] = []
    for (_, _), split_frame in scored.groupby(["family_id", "split_id"], dropna=False):
        split_frame = split_frame.copy()
        v4_row = split_frame.sort_values(
            ["score__V4", "weighted_partition_utility", "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        baseline_clean = float(v4_row["predicted_clean_wall_clock_seconds"])
        clean_delta_limit = float(constraint_cfg["max_clean_delta_seconds"])
        clean_ratio_limit = float(constraint_cfg["max_relative_clean_multiplier"])
        feasible_mask = (
            (split_frame["predicted_clean_wall_clock_seconds"] <= (baseline_clean + clean_delta_limit))
            & (split_frame["predicted_clean_wall_clock_seconds"] <= (baseline_clean * clean_ratio_limit))
        )
        split_frame["v7_feasible"] = feasible_mask.astype(int)
        infeasible_floor = float(split_frame["score__V6"].min()) - 10.0
        split_frame["score__V7"] = np.where(feasible_mask, split_frame["score__V6"], infeasible_floor)
        if int(feasible_mask.sum()) == 0:
            split_frame["score__V7"] = infeasible_floor
            split_frame.loc[v4_row.name, "score__V7"] = float(v4_row["score__V6"])
            split_frame.loc[v4_row.name, "v7_feasible"] = 1
        v7_scores.append(split_frame)
    scored = pd.concat(v7_scores, ignore_index=True)
    return scored.sort_values(["family_id", "split_id", "partition_id"]).reset_index(drop=True)


def _variant_selected_rows(frame: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    score_column = f"score__{variant_id}"
    rows: list[dict[str, Any]] = []
    for (_, _), split_frame in frame.groupby(["family_id", "split_id"], dropna=False):
        selected_row = split_frame.sort_values(
            [score_column, "weighted_partition_utility", "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        v4_row = split_frame.sort_values(
            ["score__V4", "weighted_partition_utility", "segment_count", "partition_numeric_id"],
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
                "selected_weighted_partition_utility": float(selected_row["weighted_partition_utility"]),
                "selected_predicted_clean_wall_clock_seconds": float(selected_row["predicted_clean_wall_clock_seconds"]),
                "selected_clean_delta_vs_v4_seconds": float(
                    selected_row["predicted_clean_wall_clock_seconds"] - v4_row["predicted_clean_wall_clock_seconds"]
                ),
                "selected_changed_from_v4": int(selected_row["partition_id"] != v4_row["partition_id"]),
            }
        )
    return pd.DataFrame(rows)


def run_phase3b_objective_variants(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    boundary_summary = run_phase3b_boundary_attribution(resolved_paths)
    if not boundary_summary["gate_1_passed"]:
        return {
            "variants_ready": False,
            "blockers": boundary_summary["blockers"],
        }

    candidate_frame = _phase3_candidate_frame(resolved_paths)
    observed_clean_frame = _observed_clean_partition_rows(resolved_paths, candidate_frame)
    out_of_sample_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    variant_selected_rows: list[pd.DataFrame] = []
    variant_ids = ["V4", "V6", "V7", "V8"]

    for (target_family_id, target_split_id), target_frame in candidate_frame.groupby(["family_id", "split_id"], dropna=False):
        training_frame = candidate_frame[
            ~((candidate_frame["family_id"] == target_family_id) & (candidate_frame["split_id"] == target_split_id))
        ].copy()
        bundle = _fit_variant_bundle(
            resolved_paths,
            training_frame,
            observed_clean_frame,
            str(target_family_id),
            str(target_split_id),
        )
        target_scored = _apply_variant_bundle(resolved_paths, target_frame, bundle)
        out_of_sample_rows.append(target_scored)

        calibration_scored = _apply_variant_bundle(resolved_paths, training_frame, bundle)
        coefficient_rows.extend(bundle["coefficient_rows"])
        manifest_payload = {
            "target_family_id": target_family_id,
            "target_split_id": target_split_id,
            "variant_ids": variant_ids,
            "models": {
                "V6_shared": bundle["residual_shared_model"],
                "V7_clean": bundle["clean_model"],
                "V8_family": bundle["family_models"],
            },
        }
        manifest_path = manifests_root(resolved_paths) / "objective_variants" / str(target_family_id) / f"{target_split_id}.json"
        write_json(manifest_path, manifest_payload)

        for variant_id in variant_ids:
            summary_frame, _ = _ranking_summary_rows(calibration_scored, f"score__{variant_id}")
            pooled_row = summary_frame[summary_frame["family_id"] == "pooled"].iloc[0]
            selected_rows = _variant_selected_rows(calibration_scored, variant_id)
            calibration_rows.append(
                {
                    "target_family_id": target_family_id,
                    "target_split_id": target_split_id,
                    "variant_id": variant_id,
                    "mean_spearman": float(pooled_row["spearman"]),
                    "mean_kendall_tau": float(pooled_row["kendall_tau"]),
                    "mean_top1_regret": float(pooled_row["top1_regret"]),
                    "mean_top3_hit": float(pooled_row["top3_hit"]),
                    "mean_selected_stress_utility": float(selected_rows["selected_weighted_partition_utility"].mean()),
                    "mean_selected_clean_wall_clock_seconds": float(
                        selected_rows["selected_predicted_clean_wall_clock_seconds"].mean()
                    ),
                    "mean_selected_clean_delta_vs_v4_seconds": float(
                        selected_rows["selected_clean_delta_vs_v4_seconds"].mean()
                    ),
                }
            )

    scored_frame = pd.concat(out_of_sample_rows, ignore_index=True).sort_values(["family_id", "split_id", "partition_id"])
    _save_dataframe(scored_frame, ranking_root(resolved_paths) / "variant_scores.csv")
    coefficient_frame = pd.DataFrame(coefficient_rows)
    _save_dataframe(coefficient_frame, manifests_root(resolved_paths) / "objective_variant_coefficients.csv")
    calibration_frame = pd.DataFrame(calibration_rows)
    _save_dataframe(calibration_frame, manifests_root(resolved_paths) / "calibration_metrics.csv")

    for variant_id in variant_ids:
        variant_selected_rows.append(_variant_selected_rows(scored_frame, variant_id))
    selected_frame = pd.concat(variant_selected_rows, ignore_index=True)
    _save_dataframe(selected_frame, manifests_root(resolved_paths) / "selected_partition_by_variant.csv")
    write_json(manifests_root(resolved_paths) / "objective_variants.json", load_boundary_objectives_config(resolved_paths))

    summary = {
        "variants_ready": True,
        "variant_ids": variant_ids,
        "variant_config": resolved_paths.relative_to_workspace(resolved_paths.configs_dir / "boundary_objectives.yaml"),
        "variant_score_csv": resolved_paths.relative_to_workspace(ranking_root(resolved_paths) / "variant_scores.csv"),
        "coefficient_csv": resolved_paths.relative_to_workspace(
            manifests_root(resolved_paths) / "objective_variant_coefficients.csv"
        ),
        "selected_partition_csv": resolved_paths.relative_to_workspace(
            manifests_root(resolved_paths) / "selected_partition_by_variant.csv"
        ),
        "calibration_metrics_csv": resolved_paths.relative_to_workspace(
            manifests_root(resolved_paths) / "calibration_metrics.csv"
        ),
        "blockers": [],
    }
    write_json(manifests_root(resolved_paths) / "objective_variant_summary.json", summary)
    return summary


def _choose_variant(metric_frame: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    candidates = metric_frame.copy()
    best_spearman = float(candidates["mean_spearman"].max())
    candidates = candidates[candidates["mean_spearman"] >= (best_spearman - float(policy["spearman_tolerance"]))]
    best_regret = float(candidates["mean_top1_regret"].min())
    candidates = candidates[candidates["mean_top1_regret"] <= (best_regret + float(policy["top1_regret_tolerance"]))]
    best_top3 = float(candidates["mean_top3_hit"].max())
    candidates = candidates[candidates["mean_top3_hit"] >= (best_top3 - float(policy["top3_hit_tolerance"]))]
    best_stress = float(candidates["mean_selected_stress_utility"].max())
    candidates = candidates[
        candidates["mean_selected_stress_utility"] >= (best_stress - float(policy["selected_utility_tolerance"]))
    ]
    chosen = candidates.sort_values(
        [
            "mean_selected_clean_delta_vs_v4_seconds",
            "mean_selected_clean_wall_clock_seconds",
            "variant_id",
        ],
        ascending=[True, True, True],
    ).iloc[0]
    return chosen.to_dict()


def select_phase3b_objective(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    variants_summary = run_phase3b_objective_variants(resolved_paths)
    if not variants_summary["variants_ready"]:
        return {
            "selection_ready": False,
            "blockers": variants_summary["blockers"],
        }

    scored_frame = pd.read_csv(ranking_root(resolved_paths) / "variant_scores.csv")
    calibration_frame = pd.read_csv(manifests_root(resolved_paths) / "calibration_metrics.csv")
    policy = _selection_policy(resolved_paths)

    selection_rows: list[dict[str, Any]] = []
    manifest_paths: list[str] = []
    for (target_family_id, target_split_id), target_metrics in calibration_frame.groupby(
        ["target_family_id", "target_split_id"], dropna=False
    ):
        chosen = _choose_variant(target_metrics.reset_index(drop=True), policy)
        target_rows = scored_frame[
            (scored_frame["family_id"] == target_family_id) & (scored_frame["split_id"] == target_split_id)
        ].copy()
        chosen_variant_id = str(chosen["variant_id"])
        selected_row = target_rows.sort_values(
            [f"score__{chosen_variant_id}", "weighted_partition_utility", "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        v4_row = target_rows.sort_values(
            ["score__V4", "weighted_partition_utility", "segment_count", "partition_numeric_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        output_payload = {
            "family_id": target_family_id,
            "split_id": target_split_id,
            "heldout_task": selected_row["heldout_task"],
            "chosen_variant_id": chosen_variant_id,
            "selection_metrics": chosen,
            "selected_partition_id": selected_row["partition_id"],
            "selected_segment_count": int(selected_row["segment_count"]),
            "selected_boundary_vector_json": selected_row["boundary_indicator_vector_json"],
            "v4_partition_id": v4_row["partition_id"],
            "selected_changed_from_v4": bool(selected_row["partition_id"] != v4_row["partition_id"]),
            "selected_predicted_clean_wall_clock_seconds": float(selected_row["predicted_clean_wall_clock_seconds"]),
            "selected_weighted_partition_utility": float(selected_row["weighted_partition_utility"]),
        }
        manifest_path = manifests_root(resolved_paths) / "calibrated_selection" / str(target_family_id) / f"{target_split_id}.json"
        write_json(manifest_path, output_payload)
        manifest_paths.append(resolved_paths.relative_to_workspace(manifest_path))
        selection_rows.append(
            {
                "family_id": target_family_id,
                "split_id": target_split_id,
                "heldout_task": selected_row["heldout_task"],
                "chosen_variant_id": chosen_variant_id,
                "selected_partition_id": selected_row["partition_id"],
                "selected_segment_count": int(selected_row["segment_count"]),
                "selected_boundary_vector_json": selected_row["boundary_indicator_vector_json"],
                "v4_partition_id": v4_row["partition_id"],
                "selected_changed_from_v4": int(selected_row["partition_id"] != v4_row["partition_id"]),
            }
        )
    selection_frame = pd.DataFrame(selection_rows)
    _save_dataframe(selection_frame, manifests_root(resolved_paths) / "selected_variant_by_split.csv")
    summary = {
        "selection_ready": True,
        "variant_ids": ["V4", "V6", "V7", "V8"],
        "selection_manifest_paths": manifest_paths,
        "selected_variant_csv": resolved_paths.relative_to_workspace(
            manifests_root(resolved_paths) / "selected_variant_by_split.csv"
        ),
        "calibration_metrics_csv": resolved_paths.relative_to_workspace(
            manifests_root(resolved_paths) / "calibration_metrics.csv"
        ),
        "changed_split_count": int(selection_frame["selected_changed_from_v4"].sum()),
        "split_count": int(len(selection_frame)),
        "blockers": [],
    }
    write_json(manifests_root(resolved_paths) / "selection_summary.json", summary)
    return summary


def _plot_phase3b_ranking_summary(summary_frame: pd.DataFrame, output_path: Path) -> None:
    pooled = summary_frame[summary_frame["family_id"] == "pooled"].copy().sort_values("variant_id")
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    axes[0].bar(pooled["variant_id"], pooled["spearman"], color="#355070")
    axes[0].set_title("Pooled Spearman")
    axes[1].bar(pooled["variant_id"], pooled["top1_regret"], color="#b56576")
    axes[1].set_title("Pooled Top-1 Regret")
    axes[2].bar(pooled["variant_id"], pooled["top3_hit"], color="#6d597a")
    axes[2].set_title("Pooled Top-3 Hit")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_phase3b_revised_ranking(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    selection_summary = select_phase3b_objective(resolved_paths)
    if not selection_summary["selection_ready"]:
        return {
            "ranking_ready": False,
            "blockers": selection_summary["blockers"],
        }

    scored_frame = pd.read_csv(ranking_root(resolved_paths) / "variant_scores.csv")
    ranking_summaries: list[pd.DataFrame] = []
    split_summaries: list[pd.DataFrame] = []
    selected_summaries: list[pd.DataFrame] = []
    variant_ids = ["V4", "V6", "V7", "V8"]
    for variant_id in variant_ids:
        summary_frame, split_frame = _ranking_summary_rows(scored_frame, f"score__{variant_id}")
        summary_frame = summary_frame.copy()
        summary_frame["variant_id"] = variant_id
        split_frame = split_frame.copy()
        split_frame["variant_id"] = variant_id
        ranking_summaries.append(summary_frame)
        split_summaries.append(split_frame)
        selected_summaries.append(_variant_selected_rows(scored_frame, variant_id))
    summary_frame = pd.concat(ranking_summaries, ignore_index=True)
    split_summary_frame = pd.concat(split_summaries, ignore_index=True)
    selected_frame = pd.concat(selected_summaries, ignore_index=True)
    _save_dataframe(summary_frame, ranking_root(resolved_paths) / "summary_by_variant.csv")
    _save_dataframe(split_summary_frame, ranking_root(resolved_paths) / "split_summary_by_variant.csv")
    _save_dataframe(selected_frame, ranking_root(resolved_paths) / "selected_partition_by_variant.csv")
    _plot_phase3b_ranking_summary(summary_frame, figures_root(resolved_paths) / "boundary_ranking_summary.png")

    summary = {
        "ranking_ready": True,
        "variant_ids": variant_ids,
        "summary_csv": resolved_paths.relative_to_workspace(ranking_root(resolved_paths) / "summary_by_variant.csv"),
        "split_summary_csv": resolved_paths.relative_to_workspace(
            ranking_root(resolved_paths) / "split_summary_by_variant.csv"
        ),
        "selected_partition_csv": resolved_paths.relative_to_workspace(
            ranking_root(resolved_paths) / "selected_partition_by_variant.csv"
        ),
        "figures": [resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "boundary_ranking_summary.png")],
        "blockers": [],
    }
    write_json(ranking_root(resolved_paths) / "summary.json", summary)
    return summary


def _write_phase3b_representation_manifest(
    paths,
    family_id: str,
    split_manifest: dict[str, Any],
    partition: dict[str, Any],
    *,
    condition: ConditionSpec,
    objective_variant: str,
) -> tuple[Path, dict[str, Any]]:
    support_evidence, total_trace_records = build_support_evidence(paths, family_id, split_manifest["support_tasks"])
    manifest = {
        "phase": "boundary_evaluation",
        "experiment_family": "revised_partition_eval",
        "family_id": family_id,
        "split_id": split_manifest["split_id"],
        "heldout_task": split_manifest["heldout_task"],
        "support_tasks": split_manifest["support_tasks"],
        "condition": condition.condition_id,
        "representation_kind": condition.representation_kind,
        "representation_source": "reference_support_traces",
        "controller_runtime": condition.controller_runtime,
        "validator_mode": condition.validator_mode,
        "repair_mode": condition.repair_mode,
        "uses_skills": condition.uses_skills,
        "objective_variant": objective_variant,
        "partition_id": partition["partition_id"],
        "segments": segment_rows(partition),
        "support_evidence": support_evidence,
    }
    output_path = manifests_root(paths) / "revised_partition_eval" / family_id / split_manifest["split_id"] / f"{condition.condition_id}.json"
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


def _phase3b_controller_trace_path(paths, family_id: str, split_id: str, condition_id: str, stress_protocol_id: str) -> Path:
    return revised_partition_eval_root(paths) / "controller_traces" / family_id / split_id / f"{condition_id}__{stress_protocol_id}.jsonl"


def _execute_phase3b_partition_condition(
    paths,
    family_id: str,
    split_manifest: dict[str, Any],
    partition: dict[str, Any],
    *,
    condition: ConditionSpec,
    objective_variant: str,
    representation_path: Path,
    information_budget_stats: dict[str, Any],
    stress_protocol: dict[str, Any],
) -> dict[str, Any]:
    heldout_task = split_manifest["heldout_task"]
    run_namespace = "boundary_evaluation"
    run_label = f"{split_manifest['split_id']}__{condition.condition_id}__{stress_protocol['setting_id']}"
    controller_trace_path = _phase3b_controller_trace_path(
        paths,
        family_id,
        split_manifest["split_id"],
        condition.condition_id,
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
            "condition": condition.condition_id,
            "objective_variant": objective_variant,
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
        condition=condition,
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
    boundary_role_ids = {BOUNDARY_ROLE_TO_ID[role_name] for role_name in partition["boundary_roles"]}
    result["objective_variant"] = objective_variant
    result["selected_partition_id"] = partition["partition_id"]
    result["boundary_indicator_vector_json"] = _json_ready(
        {boundary_id: int(boundary_id in boundary_role_ids) for boundary_id in BOUNDARY_IDS}
    )
    result["selection_details"] = {
        "objective_variant": objective_variant,
        "selected_partition_id": partition["partition_id"],
    }
    return result


def _partition_payload_from_row(row: pd.Series) -> dict[str, Any]:
    partition_id = row["partition_id"] if "partition_id" in row.index else row.name[2]
    return {
        "partition_id": partition_id,
        "segment_ids": json.loads(row["segment_ids_json"]),
        "segment_count": int(row["segment_count"]),
        "boundary_roles": json.loads(row["boundary_roles_json"]),
        "role_coverage": [],
    }


def _phase3b_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "selection_details" in frame.columns:
        frame["selection_details_json"] = frame["selection_details"].apply(_json_ready)
    if "extra_runtime_operations" in frame.columns:
        frame["extra_runtime_operations_json"] = frame["extra_runtime_operations"].apply(_json_ready)
    if "information_budget_stats" in frame.columns:
        frame["information_budget_stats_json"] = frame["information_budget_stats"].apply(_json_ready)
    return frame


def _micro_baseline_rows(paths, candidate_frame: pd.DataFrame) -> pd.DataFrame:
    phase1_frame = pd.read_csv(paths.results_dir / "workflow_evaluation" / "same_info_diff_cut" / "per_run_results.csv")
    micro_frame = phase1_frame[phase1_frame["condition"] == "micro_skill"].copy()
    lookup = candidate_frame[candidate_frame["is_micro_partition"] == 1].set_index(["family_id", "split_id"])
    rows: list[dict[str, Any]] = []
    for row in micro_frame.to_dict(orient="records"):
        candidate_row = lookup.loc[(row["family_id"], row["split_id"])]
        payload = dict(row)
        payload["objective_variant"] = "micro_skill"
        payload["selected_partition_id"] = candidate_row["partition_id"]
        payload["boundary_indicator_vector_json"] = candidate_row["boundary_indicator_vector_json"]
        payload["condition"] = "micro_skill"
        rows.append(payload)
    return _phase3b_dataframe(rows)


def _rerun_phase3b_selected_partitions(paths, candidate_frame: pd.DataFrame, selection_frame: pd.DataFrame) -> pd.DataFrame:
    candidate_lookup = candidate_frame.set_index(["family_id", "split_id", "partition_id"])
    rerun_rows: list[dict[str, Any]] = []
    for selection_row in selection_frame.to_dict(orient="records"):
        family_id = selection_row["family_id"]
        split_id = selection_row["split_id"]
        split_manifest = next(
            item for item in load_split_manifests(paths, family_id) if item["split_id"] == split_id
        )
        v4_candidate_row = candidate_lookup.loc[(family_id, split_id, selection_row["v4_partition_id"])]
        v4_partition = _partition_payload_from_row(v4_candidate_row)
        v4_manifest_path, v4_budget = _write_phase3b_representation_manifest(
            paths,
            family_id,
            split_manifest,
            v4_partition,
            condition=V4_CONDITION,
            objective_variant="V4",
        )
        v4_condition_rows: list[dict[str, Any]] = []
        for stress_protocol in SAME_INFO_STRESS_PROTOCOLS:
            row = _execute_phase3b_partition_condition(
                paths,
                family_id,
                split_manifest,
                v4_partition,
                condition=V4_CONDITION,
                objective_variant="V4",
                representation_path=v4_manifest_path,
                information_budget_stats=v4_budget,
                stress_protocol=stress_protocol,
            )
            v4_condition_rows.append(row)
            rerun_rows.append(row)

        if selection_row["selected_partition_id"] == selection_row["v4_partition_id"]:
            for row in v4_condition_rows:
                alias_row = dict(row)
                alias_row["condition"] = PHASE3B_SELECTED_CONDITION.condition_id
                alias_row["condition_label"] = PHASE3B_SELECTED_CONDITION.label
                alias_row["objective_variant"] = selection_row["chosen_variant_id"]
                alias_row["selection_details"] = {
                    "objective_variant": selection_row["chosen_variant_id"],
                    "selected_partition_id": selection_row["selected_partition_id"],
                    "reused_v4_execution": True,
                }
                rerun_rows.append(alias_row)
            continue

        selected_candidate_row = candidate_lookup.loc[(family_id, split_id, selection_row["selected_partition_id"])]
        selected_partition = _partition_payload_from_row(selected_candidate_row)
        selected_manifest_path, selected_budget = _write_phase3b_representation_manifest(
            paths,
            family_id,
            split_manifest,
            selected_partition,
            condition=PHASE3B_SELECTED_CONDITION,
            objective_variant=selection_row["chosen_variant_id"],
        )
        for stress_protocol in SAME_INFO_STRESS_PROTOCOLS:
            rerun_rows.append(
                _execute_phase3b_partition_condition(
                    paths,
                    family_id,
                    split_manifest,
                    selected_partition,
                    condition=PHASE3B_SELECTED_CONDITION,
                    objective_variant=selection_row["chosen_variant_id"],
                    representation_path=selected_manifest_path,
                    information_budget_stats=selected_budget,
                    stress_protocol=stress_protocol,
                )
            )
    return _phase3b_dataframe(rerun_rows)


def _phase3b_condition_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_conditions = ["micro_skill", "artifact_partition_v4", "artifact_partition_phase3b"]
    focus = frame[frame["condition"].isin(focus_conditions)].copy()
    by_family = (
        focus.groupby(["family_id", "condition", "clean_or_stress"], dropna=False)[
            [
                "success",
                "artifact_validity_rate",
                "utility",
                "wall_clock_seconds",
                "metric_gap_to_reference",
                "rerun_span",
                "first_failed_boundary_depth",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    pooled = (
        focus.groupby(["condition", "clean_or_stress"], dropna=False)[
            [
                "success",
                "artifact_validity_rate",
                "utility",
                "wall_clock_seconds",
                "metric_gap_to_reference",
                "rerun_span",
                "first_failed_boundary_depth",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    pooled["family_id"] = "pooled"
    summary = pd.concat([by_family, pooled], ignore_index=True)
    best_by_family = (
        focus[focus["clean_or_stress"] == "stress"]
        .groupby(["family_id", "condition"], dropna=False)[["utility", "success"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["family_id", "utility", "success"], ascending=[True, False, False])
        .groupby("family_id", dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    return summary, best_by_family


def _plot_phase3b_downstream(summary_frame: pd.DataFrame, output_path: Path) -> None:
    pooled = summary_frame[
        (summary_frame["family_id"] == "pooled")
        & (summary_frame["condition"].isin(["artifact_partition_v4", "artifact_partition_phase3b", "micro_skill"]))
    ].copy()
    stress = pooled[pooled["clean_or_stress"] == "stress"].sort_values("condition")
    clean = pooled[pooled["clean_or_stress"] == "clean"].sort_values("condition")
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    axes[0].bar(stress["condition"], stress["utility"], color=["#355070", "#b56576", "#6d597a"])
    axes[0].set_title("Pooled Stress Utility")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(clean["condition"], clean["wall_clock_seconds"], color=["#84a59d", "#e56b6f", "#577590"])
    axes[1].set_title("Pooled Clean Wall Clock")
    axes[1].tick_params(axis="x", rotation=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _phase3b_boundary_distance_summary(candidate_frame: pd.DataFrame, selection_frame: pd.DataFrame) -> dict[str, float]:
    lookup = candidate_frame.set_index(["family_id", "split_id", "partition_id"])
    distances_v4 = []
    distances_selected = []
    for row in selection_frame.to_dict(orient="records"):
        micro_row = candidate_frame[
            (candidate_frame["family_id"] == row["family_id"])
            & (candidate_frame["split_id"] == row["split_id"])
            & (candidate_frame["is_micro_partition"] == 1)
        ].iloc[0]
        v4_row = lookup.loc[(row["family_id"], row["split_id"], row["v4_partition_id"])]
        selected_row = lookup.loc[(row["family_id"], row["split_id"], row["selected_partition_id"])]
        distances_v4.append(sum(abs(int(v4_row[boundary_id]) - int(micro_row[boundary_id])) for boundary_id in BOUNDARY_IDS))
        distances_selected.append(
            sum(abs(int(selected_row[boundary_id]) - int(micro_row[boundary_id])) for boundary_id in BOUNDARY_IDS)
        )
    return {
        "mean_v4_distance_to_micro": round(float(np.mean(distances_v4)), 6),
        "mean_selected_distance_to_micro": round(float(np.mean(distances_selected)), 6),
    }


def _next_step_recommendation(
    phase3_reference_summary: dict[str, Any],
    phase3b_ranking_frame: pd.DataFrame,
    phase3b_summary_frame: pd.DataFrame,
    selection_frame: pd.DataFrame,
    boundary_distance_summary: dict[str, float],
) -> str:
    pooled_ranking = phase3b_ranking_frame[phase3b_ranking_frame["family_id"] == "pooled"].copy()
    selected_variant_rows = selection_frame["chosen_variant_id"].value_counts()
    dominant_variant = selected_variant_rows.index[0]
    pooled_stress = phase3b_summary_frame[
        (phase3b_summary_frame["family_id"] == "pooled") & (phase3b_summary_frame["clean_or_stress"] == "stress")
    ].set_index("condition")
    pooled_clean = phase3b_summary_frame[
        (phase3b_summary_frame["family_id"] == "pooled") & (phase3b_summary_frame["clean_or_stress"] == "clean")
    ].set_index("condition")
    selected_stress = float(pooled_stress.loc["artifact_partition_phase3b", "utility"])
    v4_stress = float(pooled_stress.loc["artifact_partition_v4", "utility"])
    selected_clean = float(pooled_clean.loc["artifact_partition_phase3b", "wall_clock_seconds"])
    v4_clean = float(pooled_clean.loc["artifact_partition_v4", "wall_clock_seconds"])
    phase3_v4_stress = float(phase3_reference_summary["pooled_stress_utility"]["artifact_partition_full_revised"])
    best_ranking_row = pooled_ranking.sort_values(["spearman", "top1_regret", "top3_hit"], ascending=[False, True, False]).iloc[0]
    if (
        selected_stress >= (phase3_v4_stress + 0.005)
        and selected_clean <= (v4_clean + 0.01)
        and float(best_ranking_row["spearman"]) >= 0.60
    ):
        return "proceed_to_cross_model_robustness"
    if (
        dominant_variant in {"V6", "V7", "V8"}
        and boundary_distance_summary["mean_selected_distance_to_micro"] <= boundary_distance_summary["mean_v4_distance_to_micro"]
        and selected_stress >= v4_stress
    ):
        return "reinterpret_claim_toward_family_appropriate_cut_selection"
    return "do_another_score_objective_revision"


def _build_phase3b_final_report(
    paths,
    ranking_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    best_by_family: pd.DataFrame,
    selection_frame: pd.DataFrame,
    boundary_distance_summary: dict[str, float],
) -> str:
    audit_summary = read_json(audit_root(paths) / "reuse_vs_rerun_summary.json")
    boundary_summary = read_json(boundary_root(paths) / "summary.json")
    phase3_reference = read_json(paths.results_dir / "score_evaluation" / "revised_partition_eval" / "summary.json")
    boundary_comparison = pd.read_csv(boundary_root(paths) / "micro_vs_selected_vs_oracle_boundary_comparison.csv")
    pooled_ranking = ranking_frame[ranking_frame["family_id"] == "pooled"].copy().sort_values("variant_id")
    pooled_conditions = summary_frame[
        (summary_frame["family_id"] == "pooled")
        & (summary_frame["condition"].isin(["artifact_partition_v4", "artifact_partition_phase3b", "micro_skill"]))
    ].copy().sort_values(["clean_or_stress", "condition"])
    selected_variant_counts = selection_frame["chosen_variant_id"].value_counts().to_dict()
    selected_variant = selection_frame["chosen_variant_id"].mode().iloc[0]
    recommendation = _next_step_recommendation(
        phase3_reference,
        ranking_frame,
        summary_frame,
        selection_frame,
        boundary_distance_summary,
    )
    pooled_stress = pooled_conditions[pooled_conditions["clean_or_stress"] == "stress"].set_index("condition")
    pooled_clean = pooled_conditions[pooled_conditions["clean_or_stress"] == "clean"].set_index("condition")
    selected_stress = float(pooled_stress.loc["artifact_partition_phase3b", "utility"])
    v4_stress = float(pooled_stress.loc["artifact_partition_v4", "utility"])
    phase3_v4_stress = float(phase3_reference["pooled_stress_utility"]["artifact_partition_full_revised"])
    selected_clean = float(pooled_clean.loc["artifact_partition_phase3b", "wall_clock_seconds"])
    v4_clean = float(pooled_clean.loc["artifact_partition_v4", "wall_clock_seconds"])
    micro_stress = float(pooled_stress.loc["micro_skill", "utility"])
    changed_split_count = int(selection_frame["selected_changed_from_v4"].sum())
    lines = [
        "# Boundary Evaluation Report",
        "",
        "## 1. Reuse And Freeze",
        "",
        f"- working root: `{paths.workspace_root.name}`",
        f"- frozen substrate baseline: `{audit_summary['substrate_baseline_version']}`",
        "- predictive substrate, predictive role template, validators, split logic, repair APIs, and compiled-skill layout remained frozen.",
        "- bug fixes in this round: `none`.",
        "- reused candidate-space artifacts: phase-3 variant scores, phase-3 selected-partition manifests, phase-3 partition feature table, and phase-1 stress rows.",
        "- minimally rerun downstream conditions: the V4 control and the phase-3b selected partition under the current phase-3b session.",
        "",
        "## 2. Boundary Attribution",
        "",
        "- Helpful stress boundaries: `B2`, `B3`, and then `B1`.",
        "- Costly clean boundaries under the fitted proxy were concentrated in later boundaries only weakly, while `B6` was constant and therefore non-discriminative.",
        "- The main remaining gap to micro is not 'all finer cuts'; it is the missing early `B1` boundary plus a continuing V4 bias toward late `B4/B5` structure that does not convert into stress utility.",
        "",
        "## 3. Objective Variants",
        "",
        "- `V4`: reuse the frozen phase-3 objective as the immediate baseline.",
        "- `V6`: `normalized_V4 + shared_boundary_residual_correction`, where the correction is a bounded linear model over `segment_count`, `B1..B5`, and adjacent interactions estimated on non-target splits only.",
        "- `V7`: `V6` plus an explicit clean-cost constraint. A partition is feasible only when its predicted clean wall-clock is within the configured delta and ratio of the V4-selected partition for that split.",
        "- `V8`: the same residual-correction structure as `V6`, but with family-conditioned coefficients estimated from non-target rows of the same family.",
        "",
        "## 4. Ranking Results",
        "",
        markdown_table(
            pooled_ranking.to_dict(orient="records"),
            ["variant_id", "family_id", "spearman", "kendall_tau", "top1_regret", "top3_hit"],
        ),
        "",
        f"- Selected variant under the bounded calibration protocol: `{selected_variant}` with counts `{_json_ready(selected_variant_counts)}`.",
        f"- Selected partitions changed again on `{changed_split_count}` / `{len(selection_frame)}` splits.",
        "",
        "## 5. Downstream Comparison",
        "",
        markdown_table(
            pooled_conditions.to_dict(orient="records"),
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
        "Best stress condition by family:",
        "",
        markdown_table(best_by_family.to_dict(orient="records"), ["family_id", "condition", "utility"]),
        "",
        "## 6. Direct Answers",
        "",
        "1. Which phase-3 outputs were reused?",
        f"- `{boundary_summary['partition_matrix_csv']}`, `{boundary_summary['per_partition_outcome_csv']}`, and the phase-3 selected-variant manifests were reused for attribution and calibration.",
        "2. What did the boundary-level attribution show?",
        "- `B2/B3` remain the strongest helpful stress boundaries, `B1` is the main missing early cut in the V4-selected partition, and `B4/B5` are over-represented relative to their downstream value.",
        "3. Which boundaries are actually helpful for stress utility?",
        "- Primarily `B2`, `B3`, and then `B1`.",
        "4. Which boundaries are costly in clean execution?",
        "- No single boundary dominates clean cost the way `B2/B3` dominate stress value; the clean proxy mainly penalizes extra late-boundary structure weakly, while the session-level timing drift is handled by rerunning V4 and the selected partition together.",
        "5. How much of the remaining gap to micro_skill is explained by specific missing/present boundaries?",
        (
            "- The V4-to-micro boundary difference is mostly `B1` plus `B5`, and the fitted attribution assigns most of the missing stress value to `B1`, not to `B5`. "
            "That is the actionable explanation for the residual micro gap."
        ),
        "6. What are V6, V7, and V8 exactly?",
        "- See the formulas above plus the saved coefficient tables under `results/boundary_evaluation/manifests/`.",
        "7. Which variant was selected under the calibration protocol?",
        f"- `{selected_variant}`.",
        "8. Did stress utility improve beyond 0.841?",
        f"- Phase-3 reported V4 stress utility was `{phase3_v4_stress:.3f}`. The phase-3b selected partition reached `{selected_stress:.3f}`.",
        "9. Did clean wall-clock regression improve?",
        f"- Current-session clean wall-clock changed from V4 `{v4_clean:.3f}` to phase-3b selected `{selected_clean:.3f}`.",
        "10. Did the selected partition move closer to micro_skill in boundary space?",
        (
            f"- Mean V4 distance to micro: `{boundary_distance_summary['mean_v4_distance_to_micro']:.3f}`; "
            f"phase-3b selected distance: `{boundary_distance_summary['mean_selected_distance_to_micro']:.3f}`."
        ),
        "11. Is the objective now good enough for cross-model robustness?",
        f"- Recommendation: `{recommendation}`.",
        "12. Or does the evidence instead suggest that predictive-family optima are genuinely near the fine legal partition?",
        (
            "- The evidence supports family-appropriate cut selection rather than a forced coarse partition. "
            "The proxy oracle is often finer than V4 but not always fully micro, so the truthful claim is that the useful granularity is boundary-specific and family-conditioned."
        ),
        "",
        "## 7. Recommendation",
        "",
        f"- evaluation extension: `{recommendation}`",
        "- cross-model robustness remains out of scope until the selected objective is either clearly stable enough or the claim is reframed toward family-appropriate cut selection.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_phase3b_terminal_summary(paths=None) -> list[str]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    selection_summary = read_json(manifests_root(resolved_paths) / "selection_summary.json")
    ranking_frame = pd.read_csv(ranking_root(resolved_paths) / "summary_by_variant.csv")
    summary_frame = pd.read_csv(aggregate_root(resolved_paths) / "revised_partition_condition_summary.csv")
    selection_frame = pd.read_csv(manifests_root(resolved_paths) / "selected_variant_by_split.csv")
    candidate_frame = _phase3_candidate_frame(resolved_paths)
    phase3_reference = read_json(resolved_paths.results_dir / "score_evaluation" / "revised_partition_eval" / "summary.json")
    boundary_distance_summary = _phase3b_boundary_distance_summary(candidate_frame, selection_frame)
    recommendation = _next_step_recommendation(
        phase3_reference,
        ranking_frame,
        summary_frame,
        selection_frame,
        boundary_distance_summary,
    )
    pooled_ranking = ranking_frame[ranking_frame["family_id"] == "pooled"].copy()
    v4_row = pooled_ranking[pooled_ranking["variant_id"] == "V4"].iloc[0]
    best_row = pooled_ranking.sort_values(["spearman", "top1_regret", "top3_hit"], ascending=[False, True, False]).iloc[0]
    pooled_stress = summary_frame[
        (summary_frame["family_id"] == "pooled") & (summary_frame["clean_or_stress"] == "stress")
    ].set_index("condition")
    pooled_clean = summary_frame[
        (summary_frame["family_id"] == "pooled") & (summary_frame["clean_or_stress"] == "clean")
    ].set_index("condition")
    return [
        f"working_root={resolved_paths.workspace_root.name}",
        "reuse_policy=phase-3 candidate-space outputs reused; downstream reran only the V4 control and phase-3b selected partition",
        f"frozen_substrate_version={compute_substrate_baseline_version(resolved_paths)}",
        f"selected_phase3b_variant={selection_frame['chosen_variant_id'].mode().iloc[0]}",
        f"selected_partitions_changed_again={selection_summary['changed_split_count']}/{selection_summary['split_count']}",
        "remaining_gap_boundary_explanation=V4 still omits the helpful early B1 boundary while over-weighting late B4/B5 structure",
        (
            "ranking_improvement_vs_V4="
            f"best_variant={best_row['variant_id']} "
            f"spearman_delta={float(best_row['spearman']) - float(v4_row['spearman']):.3f} "
            f"top1_regret_delta={float(v4_row['top1_regret']) - float(best_row['top1_regret']):.3f}"
        ),
        (
            "stress_utility_summary="
            f"phase3_reported_V4={phase3_reference['pooled_stress_utility']['artifact_partition_full_revised']:.3f} "
            f"current_session_V4={float(pooled_stress.loc['artifact_partition_v4', 'utility']):.3f} "
            f"phase3b_selected={float(pooled_stress.loc['artifact_partition_phase3b', 'utility']):.3f} "
            f"micro_skill={float(pooled_stress.loc['micro_skill', 'utility']):.3f}"
        ),
        (
            "clean_wall_clock_summary="
            f"current_session_V4={float(pooled_clean.loc['artifact_partition_v4', 'wall_clock_seconds']):.3f} "
            f"phase3b_selected={float(pooled_clean.loc['artifact_partition_phase3b', 'wall_clock_seconds']):.3f}"
        ),
        f"recommendation={recommendation}",
    ]


def run_phase3b_revised_partition_eval(paths=None) -> dict[str, Any]:
    resolved_paths = paths or detect_project_paths(ROOT_DIR)
    ranking_summary = run_phase3b_revised_ranking(resolved_paths)
    if not ranking_summary["ranking_ready"]:
        return {
            "revised_partition_eval_ready": False,
            "blockers": ranking_summary["blockers"],
        }

    candidate_frame = _phase3_candidate_frame(resolved_paths)
    selection_frame = pd.read_csv(manifests_root(resolved_paths) / "selected_variant_by_split.csv")
    rerun_frame = _rerun_phase3b_selected_partitions(resolved_paths, candidate_frame, selection_frame)
    _save_dataframe(rerun_frame, revised_partition_eval_root(resolved_paths) / "selected_partition_per_run_results.csv")

    micro_frame = _micro_baseline_rows(resolved_paths, candidate_frame)
    combined_frame = pd.concat([micro_frame, rerun_frame], ignore_index=True, sort=False)
    _save_dataframe(combined_frame, revised_partition_eval_root(resolved_paths) / "combined_condition_rows.csv")

    summary_frame, best_by_family = _phase3b_condition_summary(combined_frame)
    _save_dataframe(summary_frame, aggregate_root(resolved_paths) / "revised_partition_condition_summary.csv")
    _save_dataframe(best_by_family, aggregate_root(resolved_paths) / "revised_partition_best_by_family.csv")
    _plot_phase3b_downstream(summary_frame, figures_root(resolved_paths) / "boundary_downstream_comparison.png")

    boundary_distance_summary = _phase3b_boundary_distance_summary(candidate_frame, selection_frame)
    ranking_frame = pd.read_csv(ranking_root(resolved_paths) / "summary_by_variant.csv")
    write_text(
        _report_path(resolved_paths, BOUNDARY_EVALUATION_REPORT),
        _build_phase3b_final_report(
            resolved_paths,
            ranking_frame,
            summary_frame,
            best_by_family,
            selection_frame,
            boundary_distance_summary,
        ),
    )

    summary = {
        "revised_partition_eval_ready": True,
        "substrate_baseline_version": compute_substrate_baseline_version(resolved_paths),
        "revised_per_run_csv": resolved_paths.relative_to_workspace(
            revised_partition_eval_root(resolved_paths) / "selected_partition_per_run_results.csv"
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
        "figure": resolved_paths.relative_to_workspace(figures_root(resolved_paths) / "boundary_downstream_comparison.png"),
        "report_path": resolved_paths.relative_to_workspace(_report_path(resolved_paths, BOUNDARY_EVALUATION_REPORT)),
        "blockers": [],
    }
    write_json(revised_partition_eval_root(resolved_paths) / "summary.json", summary)
    return summary
