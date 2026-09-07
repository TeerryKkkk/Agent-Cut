from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.family_registry import family_result_namespace, list_families, load_family_config, load_family_manifest_summary
from utils.io_utils import read_json, write_json
from utils.pathing import detect_project_paths

PRIOR_TABLES = {
    "predictive": {
        "validator_strength": {
            "profile_json": 0.72,
            "split_spec_json": 0.86,
            "preprocess_bundle": 0.9,
            "model_bundle": 0.93,
            "metrics_json": 0.95,
            "report_md": 0.65,
        },
        "semantic_closure": {
            "profile_json": 0.56,
            "split_spec_json": 0.76,
            "preprocess_bundle": 0.9,
            "model_bundle": 0.93,
            "metrics_json": 0.84,
            "report_md": 0.6,
        },
        "external_coupling": {
            "raw_data": 0.62,
            "profile_json": 0.74,
            "split_spec_json": 0.84,
            "preprocess_bundle": 0.88,
            "model_bundle": 0.9,
            "metrics_json": 0.94,
        },
    },
    "singlecell_mapping": {
        "validator_strength": {
            "reference_query_raw": 0.78,
            "prepared_query": 0.86,
            "latent_or_graph": 0.93,
            "predicted_labels": 0.95,
            "mapping_metrics": 0.97,
            "report_md": 0.68,
        },
        "semantic_closure": {
            "reference_query_raw": 0.58,
            "prepared_query": 0.75,
            "latent_or_graph": 0.9,
            "predicted_labels": 0.94,
            "mapping_metrics": 0.98,
            "report_md": 0.62,
        },
        "external_coupling": {
            "reference_query_raw": 0.66,
            "prepared_query": 0.8,
            "latent_or_graph": 0.9,
            "predicted_labels": 0.94,
            "mapping_metrics": 0.97,
        },
    },
}


def _load_trace_transitions(trace_path: Path) -> list[tuple[str, str]]:
    transitions: list[tuple[str, str]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("status") != "pass":
                continue
            transitions.append((payload["role_in"], payload["role_out"]))
    return transitions


def _load_validator_qa_summary(paths) -> dict[str, Any]:
    raise RuntimeError("_load_validator_qa_summary now requires family_id.")


def _qa_rows_for_family(qa_summary: dict[str, Any], family_id: str, row_key: str) -> list[dict[str, Any]]:
    if "by_family" in qa_summary and family_id in qa_summary["by_family"]:
        return qa_summary["by_family"][family_id][row_key]
    return [row for row in qa_summary[row_key] if row.get("family_id", family_id) == family_id]


def _qa_evidence(qa_summary: dict[str, Any], family_id: str, role: str) -> float:
    valid_rows = [row for row in _qa_rows_for_family(qa_summary, family_id, "valid_results") if row["role"] == role]
    bad_rows = [row for row in _qa_rows_for_family(qa_summary, family_id, "bad_results") if row["role"] == role]
    valid_pass = sum(1 for row in valid_rows if row["status"] == "pass")
    bad_detect = sum(1 for row in bad_rows if row["status"] == "detected")
    valid_rate = valid_pass / len(valid_rows) if valid_rows else 1.0
    bad_rate = bad_detect / len(bad_rows) if bad_rows else 1.0
    return max(0.0, min(1.0, valid_rate * bad_rate))


def _resolve_trace_path(paths, family_id: str, task_name: str) -> Path:
    base_dir = paths.runs_dir / "reference" / family_id / task_name
    direct_trace = base_dir / "trace.jsonl"
    if direct_trace.exists():
        return direct_trace
    candidates = sorted(base_dir.glob("**/trace.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No trace.jsonl found for family={family_id} task={task_name} under {base_dir}")
    return candidates[0]


def _default_support_tasks(paths, family_id: str, family_spec: dict[str, Any]) -> list[str]:
    if "dev_split" in family_spec:
        return list(family_spec["dev_split"]["support_tasks"])
    manifest_summary = load_family_manifest_summary(paths, family_id)
    if manifest_summary["split_paths"]:
        first_split = read_json(paths.workspace_root / manifest_summary["split_paths"][0])
        return list(first_split["support_tasks"])
    return []


def _load_validator_qa_summary_for_family(paths, family_id: str) -> dict[str, Any]:
    namespace = family_result_namespace(paths, family_id)
    return read_json(paths.results_dir / namespace / "validator_qa" / "summary.json")


def score_family_segments(paths, family_id: str, support_tasks: list[str] | None = None) -> list[dict[str, Any]]:
    family_spec = load_family_config(paths, family_id)
    partitions_dir = paths.results_dir / "partitions" / family_id
    qa_summary = _load_validator_qa_summary_for_family(paths, family_id)
    legal_segments = read_json(partitions_dir / "legal_segments.json")
    prior_tables = PRIOR_TABLES[family_spec["family_type"]]

    active_support_tasks = support_tasks or _default_support_tasks(paths, family_id, family_spec)
    support_transition_sets = []
    for task_name in active_support_tasks:
        trace_path = _resolve_trace_path(paths, family_id, task_name)
        support_transition_sets.append(set(_load_trace_transitions(trace_path)))

    max_span = len(family_spec["role_template"]) - 1
    scored_segments = []
    for segment in legal_segments:
        required_transitions = {(item["role_in"], item["role_out"]) for item in segment["transitions"]}
        trace_coverage = sum(1 for transitions in support_transition_sets if required_transitions.issubset(transitions))
        recurrence = trace_coverage / len(support_transition_sets) if support_transition_sets else 0.0

        span = segment["transition_count"]
        qa_factor = _qa_evidence(qa_summary, family_id, segment["end_role"])
        validator_strength = prior_tables["validator_strength"][segment["end_role"]] * qa_factor
        semantic_closure = min(1.0, prior_tables["semantic_closure"][segment["end_role"]] + 0.04 * min(2, span - 1))
        external_coupling = max(0.0, prior_tables["external_coupling"][segment["start_role"]] - 0.03 * max(0, span - 2))
        repair_span = 1.0 - ((span - 1) / max(1, max_span - 1))

        total_score = (
            0.2 * recurrence
            + 0.25 * validator_strength
            + 0.25 * semantic_closure
            + 0.15 * external_coupling
            + 0.15 * repair_span
        )
        scored_segments.append(
            {
                **segment,
                "score_components": {
                    "recurrence": round(recurrence, 6),
                    "validator_strength": round(validator_strength, 6),
                    "semantic_closure": round(semantic_closure, 6),
                    "external_coupling": round(external_coupling, 6),
                    "repair_span": round(repair_span, 6),
                },
                "support_tasks_used": active_support_tasks,
                "total_score": round(total_score, 6),
            }
        )

    write_json(partitions_dir / "segment_scores.json", scored_segments)
    return scored_segments


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    for family_id in list_families(paths):
        scored_segments = score_family_segments(paths, family_id)
        print(f"{family_id}: scored_segments={len(scored_segments)}")


if __name__ == "__main__":
    main()
