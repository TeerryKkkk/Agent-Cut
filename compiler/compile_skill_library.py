from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.family_registry import list_families, load_family_config, load_runner_specs
from utils.io_utils import read_json, write_json, write_yaml
from utils.pathing import detect_project_paths
from utils.predictive_splits import load_family_split_summary


def _read_trace(trace_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def _resolve_trace_path(paths, family_id: str, task_name: str) -> Path:
    base_dir = paths.runs_dir / "reference" / family_id / task_name
    direct_trace = base_dir / "trace.jsonl"
    if direct_trace.exists():
        return direct_trace
    candidates = sorted(base_dir.glob("**/trace.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No trace.jsonl found for family={family_id} task={task_name} under {base_dir}")
    return candidates[0]


def _segment_step_specs(role_template: list[str], step_specs: list[tuple[str, str, str, str]], start_role: str, end_role: str) -> list[dict[str, Any]]:
    start_index = role_template.index(start_role)
    end_index = role_template.index(end_role)
    selected = []
    for step_id, role_in, role_out, function_name in step_specs:
        role_out_index = role_template.index(role_out)
        if role_out_index <= start_index or role_out_index > end_index:
            continue
        selected.append(
            {
                "step_id": step_id,
                "role_in": role_in,
                "role_out": role_out,
                "function_name": function_name,
            }
        )
    return selected


def compile_split_skill_library(paths, family_id: str, split_manifest: dict[str, Any]) -> dict[str, Any]:
    family_spec = load_family_config(paths, family_id)
    role_template, step_specs = load_runner_specs(paths, family_id)
    selected_partition = read_json(paths.results_dir / "partitions" / family_id / "selected_partition.json")
    support_tasks = list(split_manifest["support_tasks"])
    compiled_root = paths.compiled_skills_dir / family_id / split_manifest["split_id"]
    compiled_root.mkdir(parents=True, exist_ok=True)

    skills_manifest: list[dict[str, Any]] = []
    for index, segment_id in enumerate(selected_partition["segment_ids"], start=1):
        start_role, end_role = segment_id.split("__to__")
        role_start_index = role_template.index(start_role)
        role_end_index = role_template.index(end_role)
        role_sequence = role_template[role_start_index : role_end_index + 1]
        segment_step_specs = _segment_step_specs(role_template, step_specs, start_role, end_role)

        skill_id = f"skill_{index:02d}_{start_role}_to_{end_role}"
        skill_dir = compiled_root / skill_id
        tests_dir = skill_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        trace_examples: dict[str, Any] = {}
        for task_name in support_tasks:
            trace_path = _resolve_trace_path(paths, family_id, task_name)
            trace_records = _read_trace(trace_path)
            allowed_steps = {item["step_id"] for item in segment_step_specs}
            trace_examples[task_name] = [record for record in trace_records if record["step_id"] in allowed_steps]

        skill_payload = {
            "family_id": family_id,
            "split_id": split_manifest["split_id"],
            "skill_id": skill_id,
            "name": skill_id.replace("_", "-"),
            "description": f"Compiled predictive skill for {start_role} to {end_role}.",
            "selected_partition_id": selected_partition["partition_id"],
            "role_start": start_role,
            "role_end": end_role,
            "role_sequence": role_sequence,
            "step_specs": segment_step_specs,
            "validator_role": end_role,
            "support_tasks": support_tasks,
            "heldout_task": split_manifest["heldout_task"],
            "input_contract": {
                "primary_input_role": start_role,
                "artifacts_are_explicit": True,
                "portable_paths_only": True,
            },
            "output_contract": {
                "output_role": end_role,
                "validator_required": True,
                "scientifically_meaningful_artifact": True,
            },
            "replay_entrypoint": {
                "module": "runtime.support_replay",
                "callable": "replay_skill_on_task",
            },
        }
        validator_binding = {
            "role": end_role,
            "module": family_spec["validator_module"],
            "callable": "validate_role",
        }
        repair_binding = {
            "strategy": "rerun_segment_from_input_role",
            "module": family_spec["reference_pipeline"]["module"],
            "runner_class": family_spec["reference_pipeline"]["runner_class"],
        }
        provenance_template = {
            "family_id": family_id,
            "split_id": split_manifest["split_id"],
            "skill_id": skill_id,
            "task_name": "{{task_name}}",
            "run_namespace": "support_replay",
            "start_role": start_role,
            "end_role": end_role,
        }
        replay_manifest = {
            "family_id": family_id,
            "split_id": split_manifest["split_id"],
            "skill_id": skill_id,
            "support_tasks": support_tasks,
            "expected_validator_role": end_role,
            "reference_runs": [
                f"runs/reference/{family_id}/{task_name}"
                for task_name in support_tasks
            ],
        }

        write_yaml(skill_dir / "skill.yaml", skill_payload)
        write_yaml(skill_dir / "validator_binding.yaml", validator_binding)
        write_yaml(skill_dir / "repair_binding.yaml", repair_binding)
        write_json(skill_dir / "trace_examples.json", trace_examples)
        write_json(skill_dir / "provenance_template.json", provenance_template)
        write_yaml(tests_dir / "replay_manifest.yaml", replay_manifest)

        skills_manifest.append(
            {
                "skill_id": skill_id,
                "role_start": start_role,
                "role_end": end_role,
                "skill_dir": paths.relative_to_workspace(skill_dir),
            }
        )

    library_manifest = {
        "family_id": family_id,
        "split_id": split_manifest["split_id"],
        "selected_partition_id": selected_partition["partition_id"],
        "heldout_task": split_manifest["heldout_task"],
        "support_tasks": support_tasks,
        "skills": skills_manifest,
    }
    write_json(compiled_root / "library_manifest.json", library_manifest)
    return library_manifest


def compile_family_skill_libraries(paths, family_id: str) -> list[dict[str, Any]]:
    split_summary = load_family_split_summary(paths, family_id)
    manifests = []
    for split_path in split_summary["split_paths"]:
        split_manifest = read_json(paths.workspace_root / split_path)
        manifests.append(compile_split_skill_library(paths, family_id, split_manifest))
    return manifests


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    for family_id in list_families(paths, family_type="predictive"):
        split_summary = load_family_split_summary(paths, family_id)
        for split_path in split_summary["split_paths"]:
            split_manifest = read_json(paths.workspace_root / split_path)
            library_manifest = compile_split_skill_library(paths, family_id, split_manifest)
            print(
                f"{family_id}/{split_manifest['split_id']}: compiled_skills="
                f"{len(library_manifest['skills'])}"
            )


if __name__ == "__main__":
    main()
