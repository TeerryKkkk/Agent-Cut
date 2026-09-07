from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.family_registry import list_families, load_family_config
from utils.io_utils import write_json
from utils.pathing import detect_project_paths


def _family_spec(paths, family_id: str) -> dict[str, Any]:
    return load_family_config(paths, family_id)


def enumerate_legal_segments(role_template: list[str], validator_roles: set[str]) -> list[dict[str, Any]]:
    legal_segments: list[dict[str, Any]] = []
    for start_index in range(len(role_template) - 1):
        for end_index in range(start_index + 1, len(role_template)):
            end_role = role_template[end_index]
            if end_role not in validator_roles:
                continue

            role_sequence = role_template[start_index : end_index + 1]
            transitions = [
                {"role_in": role_template[index], "role_out": role_template[index + 1]}
                for index in range(start_index, end_index)
            ]
            legal_segments.append(
                {
                    "segment_id": f"{role_sequence[0]}__to__{role_sequence[-1]}",
                    "start_role": role_sequence[0],
                    "end_role": role_sequence[-1],
                    "start_index": start_index,
                    "end_index": end_index,
                    "role_sequence": role_sequence,
                    "transition_count": len(transitions),
                    "transitions": transitions,
                    "validator_bearing_output": True,
                    "parameterizable_io": True,
                    "hidden_state_dependency": "low",
                    "scientifically_meaningful_output": True,
                    "legal": True,
                }
            )
    return legal_segments


def enumerate_legal_partitions(role_template: list[str], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments_by_start: dict[int, list[dict[str, Any]]] = {}
    for segment in segments:
        segments_by_start.setdefault(segment["start_index"], []).append(segment)

    partitions: list[dict[str, Any]] = []

    def _walk(cursor: int, chosen: list[dict[str, Any]]) -> None:
        if cursor == len(role_template) - 1:
            partitions.append(
                {
                    "partition_id": "",
                    "segment_ids": [segment["segment_id"] for segment in chosen],
                    "segment_count": len(chosen),
                    "boundary_roles": [segment["end_role"] for segment in chosen],
                    "role_coverage": role_template,
                }
            )
            return

        for segment in segments_by_start.get(cursor, []):
            _walk(segment["end_index"], [*chosen, segment])

    _walk(0, [])

    for index, partition in enumerate(partitions, start=1):
        partition["partition_id"] = f"partition_{index:03d}"
    return partitions


def enumerate_family_partitions(paths, family_id: str) -> dict[str, Any]:
    spec = _family_spec(paths, family_id)
    role_template = list(spec["role_template"])
    validator_roles = set(spec["validator_roles"])
    output_dir = paths.results_dir / "partitions" / family_id
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = enumerate_legal_segments(role_template, validator_roles)
    partitions = enumerate_legal_partitions(role_template, segments)
    summary = {
        "family_id": family_id,
        "role_template": role_template,
        "legal_segment_count": len(segments),
        "legal_partition_count": len(partitions),
    }

    write_json(output_dir / "legal_segments.json", segments)
    write_json(output_dir / "legal_partitions.json", partitions)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    for family_id in list_families(paths):
        summary = enumerate_family_partitions(paths, family_id)
        print(
            f"{family_id}: legal_segments={summary['legal_segment_count']} "
            f"legal_partitions={summary['legal_partition_count']}"
        )


if __name__ == "__main__":
    main()
