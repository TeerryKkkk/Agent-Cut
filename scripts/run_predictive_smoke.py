from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import load_family_metric_policy, resolve_primary_metric, threshold_passes
from pipelines.predictive_runtime import build_reference_runner, list_predictive_families
from utils.io_utils import read_json, read_yaml, write_json
from utils.pathing import detect_project_paths
from utils.predictive_splits import load_family_split_summary
from validators.predictive_roles import artifact_paths_for_run, validate_role


def _select_dev_split(paths, family_id: str) -> dict[str, Any]:
    family_spec = read_yaml(paths.configs_dir / "families.yaml")["families"][family_id]
    split_summary = load_family_split_summary(paths, family_id)
    for split_path in split_summary["split_paths"]:
        split_manifest = read_json(paths.workspace_root / split_path)
        if split_manifest["heldout_task"] == family_spec["dev_split"]["heldout_task"]:
            return split_manifest
    raise RuntimeError(f"Could not find dev split manifest for {family_id}")


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    substrate_root = paths.results_dir / "predictive_substrate"
    reference_summary = read_json(substrate_root / "reference" / "summary.json")
    validator_summary = read_json(substrate_root / "validator_qa" / "summary.json")
    support_replay_summary = read_json(substrate_root / "support_replay" / "summary.json")

    smoke_root = substrate_root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    family_rows: dict[str, Any] = {}

    for family_id in list_predictive_families(paths):
        split_manifest = _select_dev_split(paths, family_id)
        family_reference = reference_summary["families"][family_id]
        family_validator = validator_summary["by_family"][family_id]
        family_replay = support_replay_summary["families"][family_id]
        gated = family_reference["passed_all"] and family_validator["passes_target"] and family_replay["passes_target"]

        if not gated:
            summary = {
                "family_id": family_id,
                "split_id": split_manifest["split_id"],
                "heldout_task": split_manifest["heldout_task"],
                "status": "skipped",
                "reason": "Reference, validator QA, or support replay gate did not pass.",
                "reference_passed": family_reference["passed_all"],
                "validator_passed": family_validator["passes_target"],
                "support_replay_passed": family_replay["passes_target"],
            }
            family_rows[family_id] = summary
            family_dir = smoke_root / family_id
            family_dir.mkdir(parents=True, exist_ok=True)
            write_json(family_dir / "summary.json", summary)
            continue

        library_manifest = read_json(
            paths.compiled_skills_dir / family_id / split_manifest["split_id"] / "library_manifest.json"
        )
        runner = build_reference_runner(
            paths,
            family_id,
            split_manifest["heldout_task"],
            run_namespace="smoke",
            run_label=split_manifest["split_id"],
        )

        rows = []
        for skill_entry in library_manifest["skills"]:
            skill_dir = paths.workspace_root / skill_entry["skill_dir"]
            skill_payload = read_yaml(skill_dir / "skill.yaml")
            runner.run_segment(skill_payload["role_start"], skill_payload["role_end"])
            artifact_path = artifact_paths_for_run(runner.run_dir)[skill_payload["role_end"]]
            validation = validate_role(
                skill_payload["role_end"],
                artifact_path,
                workspace_root=paths.workspace_root,
                run_dir=runner.run_dir,
            )
            rows.append(
                {
                    "skill_id": skill_payload["skill_id"],
                    "role_end": skill_payload["role_end"],
                    "status": "pass" if validation.passed else "fail",
                    "error_code": validation.error_code or "none",
                }
            )

        smoke_metrics = read_json(runner.run_dir / "artifacts" / "metrics.json")
        reference_metrics = read_json(
            paths.runs_dir / "reference" / family_id / split_manifest["heldout_task"] / "artifacts" / "metrics.json"
        )
        metric_policy = load_family_metric_policy(paths, family_id)
        primary_metric = resolve_primary_metric(metric_policy, split_manifest["heldout_task"])
        meets_threshold, metric_gap = threshold_passes(
            float(reference_metrics[primary_metric]),
            float(smoke_metrics[primary_metric]),
            primary_metric,
            metric_policy["success_threshold_rule"],
        )
        all_valid = all(row["status"] == "pass" for row in rows)
        summary = {
            "family_id": family_id,
            "split_id": split_manifest["split_id"],
            "heldout_task": split_manifest["heldout_task"],
            "selected_partition_id": library_manifest["selected_partition_id"],
            "all_segment_validations_passed": all_valid,
            "primary_metric": primary_metric,
            "reference_primary_metric": reference_metrics[primary_metric],
            "smoke_primary_metric": smoke_metrics[primary_metric],
            "metric_gap": metric_gap,
            "meets_reference_threshold": meets_threshold,
            "results": rows,
            "smoke_run_dir": paths.relative_to_workspace(runner.run_dir),
            "status": "pass" if all_valid and meets_threshold else "fail",
        }
        family_rows[family_id] = summary
        family_dir = smoke_root / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        write_json(family_dir / "summary.json", summary)

    aggregate = {
        "families": family_rows,
        "all_families_ready": all(summary["status"] == "pass" for summary in family_rows.values()),
    }
    write_json(smoke_root / "summary.json", aggregate)

    for family_id, summary in family_rows.items():
        print(f"{family_id}: smoke_status={summary['status']}")
    print(f"Predictive smoke summary: {paths.relative_to_workspace(smoke_root / 'summary.json')}")


if __name__ == "__main__":
    main()
