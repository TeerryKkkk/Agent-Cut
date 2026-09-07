from __future__ import annotations

import traceback
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compiler.compile_skill_library import compile_split_skill_library
from cut.enumerate_partitions import enumerate_family_partitions
from cut.score_segments import score_family_segments
from cut.select_partition import select_family_partition
from pipelines.predictive_runtime import list_predictive_families
from runtime.support_replay import replay_library
from utils.io_utils import read_json, write_json
from utils.pathing import detect_project_paths
from utils.predictive_splits import load_family_split_summary, write_family_split_manifests


def _aggregate_family_replay(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_replays = sum(int(row["total_replays"]) for row in split_rows if row["status"] == "pass")
    passed_replays = sum(int(row["passed_replays"]) for row in split_rows if row["status"] == "pass")
    pass_rate = passed_replays / total_replays if total_replays else 0.0
    return {
        "total_replays": total_replays,
        "passed_replays": passed_replays,
        "pass_rate": pass_rate,
        "passes_target": pass_rate >= 0.95 if total_replays else False,
    }


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    substrate_root = paths.results_dir / "predictive_substrate"
    partition_root = substrate_root / "partitions"
    replay_root = substrate_root / "support_replay"
    partition_root.mkdir(parents=True, exist_ok=True)
    replay_root.mkdir(parents=True, exist_ok=True)

    family_results: dict[str, Any] = {}
    for family_id in list_predictive_families(paths):
        write_family_split_manifests(paths, family_id)

        partition_summary = enumerate_family_partitions(paths, family_id)
        score_family_segments(paths, family_id)
        selected_partition = select_family_partition(paths, family_id)

        family_partition_summary = {
            **partition_summary,
            "selected_partition_id": selected_partition["partition_id"],
            "selected_segment_ids": selected_partition["segment_ids"],
            "selection_rationale": selected_partition["selection_rationale"],
        }
        family_partition_dir = partition_root / family_id
        family_partition_dir.mkdir(parents=True, exist_ok=True)
        write_json(family_partition_dir / "summary.json", family_partition_summary)

        split_summary = load_family_split_summary(paths, family_id)
        split_rows = []
        for split_path in split_summary["split_paths"]:
            split_manifest = read_json(paths.workspace_root / split_path)
            try:
                library_manifest = compile_split_skill_library(paths, family_id, split_manifest)
                replay_summary = replay_library(paths, family_id, split_manifest["split_id"])
                row = {
                    "split_id": split_manifest["split_id"],
                    "heldout_task": split_manifest["heldout_task"],
                    "support_tasks": split_manifest["support_tasks"],
                    "status": "pass",
                    "compiled_skill_count": len(library_manifest["skills"]),
                    "total_replays": replay_summary["total_replays"],
                    "passed_replays": replay_summary["passed_replays"],
                    "pass_rate": replay_summary["pass_rate"],
                    "passes_target": replay_summary["passes_target"],
                    "library_manifest": paths.relative_to_workspace(
                        paths.compiled_skills_dir / family_id / split_manifest["split_id"] / "library_manifest.json"
                    ),
                    "support_replay_summary": paths.relative_to_workspace(
                        paths.results_dir / "support_replay" / family_id / split_manifest["split_id"] / "summary.json"
                    ),
                }
            except Exception as exc:
                row = {
                    "split_id": split_manifest["split_id"],
                    "heldout_task": split_manifest["heldout_task"],
                    "support_tasks": split_manifest["support_tasks"],
                    "status": "fail",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            split_rows.append(row)

        family_replay = _aggregate_family_replay(split_rows)
        family_summary = {
            "family_id": family_id,
            "partition_summary": family_partition_summary,
            "split_results": split_rows,
            **family_replay,
        }
        family_results[family_id] = family_summary
        family_root = replay_root / family_id
        family_root.mkdir(parents=True, exist_ok=True)
        write_json(family_root / "summary.json", family_summary)

    aggregate = {
        "families": family_results,
        "all_families_passed": all(summary["passes_target"] for summary in family_results.values()),
    }
    write_json(replay_root / "summary.json", aggregate)

    for family_id, summary in family_results.items():
        print(
            f"{family_id}: selected_partition={summary['partition_summary']['selected_partition_id']} "
            f"support_replay_pass_rate={summary['pass_rate']:.3f}"
        )
    print(f"Predictive support replay summary: {paths.relative_to_workspace(replay_root / 'summary.json')}")


if __name__ == "__main__":
    main()
