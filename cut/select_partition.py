from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.family_registry import list_families, load_family_config
from utils.io_utils import read_json, write_json
from utils.pathing import detect_project_paths

BOUNDARY_UTILITY = {
    "predictive": {
        "profile_json": 0.03,
        "split_spec_json": 0.09,
        "preprocess_bundle": 0.11,
        "model_bundle": 0.11,
        "metrics_json": 0.08,
        "report_md": 0.0,
    },
    "singlecell_mapping": {
        "reference_query_raw": 0.03,
        "prepared_query": 0.08,
        "latent_or_graph": 0.14,
        "predicted_labels": 0.16,
        "mapping_metrics": 0.11,
        "report_md": 0.0,
    },
}


def select_family_partition(paths, family_id: str) -> dict[str, Any]:
    partitions_dir = paths.results_dir / "partitions" / family_id
    scored_segments = read_json(partitions_dir / "segment_scores.json")
    partitions = read_json(partitions_dir / "legal_partitions.json")
    segment_index = {segment["segment_id"]: segment for segment in scored_segments}
    family_spec = load_family_config(paths, family_id)
    boundary_prior = BOUNDARY_UTILITY[family_spec["family_type"]]

    scored_partitions: list[dict[str, Any]] = []
    for partition in partitions:
        segments = [segment_index[segment_id] for segment_id in partition["segment_ids"]]
        segment_scores = [segment["total_score"] for segment in segments]
        mean_segment_score = sum(segment_scores) / len(segment_scores)
        internal_boundaries = [segment["end_role"] for segment in segments[:-1]]
        boundary_bonus = (
            sum(boundary_prior[role] for role in internal_boundaries) / max(1, len(internal_boundaries))
            if internal_boundaries
            else 0.0
        )
        overfragmentation_penalty = 0.06 * max(0, len(segments) - 4)
        partition_score = mean_segment_score + boundary_bonus - overfragmentation_penalty
        scored_partitions.append(
            {
                **partition,
                "segment_scores": segment_scores,
                "mean_segment_score": round(mean_segment_score, 6),
                "boundary_bonus": round(boundary_bonus, 6),
                "overfragmentation_penalty": round(overfragmentation_penalty, 6),
                "partition_score": round(partition_score, 6),
            }
        )

    scored_partitions.sort(key=lambda item: (-item["partition_score"], item["segment_count"]))
    selected = scored_partitions[0]
    selected["selection_rationale"] = (
        "Chosen for the best tradeoff between validator-backed semantic closure and manageable "
        "repair span without over-fragmenting the predictive workflow."
    )

    write_json(partitions_dir / "partition_scores.json", scored_partitions)
    write_json(partitions_dir / "selected_partition.json", selected)
    return selected


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    for family_id in list_families(paths):
        selected = select_family_partition(paths, family_id)
        print(f"{family_id}: selected_partition={selected['partition_id']}")


if __name__ == "__main__":
    main()
