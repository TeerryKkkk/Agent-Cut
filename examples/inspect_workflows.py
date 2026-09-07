"""Inspect legal workflow decompositions without downloading data or calling a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cut.enumerate_partitions import enumerate_legal_partitions, enumerate_legal_segments
from utils.family_registry import list_families, load_family_config
from utils.io_utils import read_yaml
from utils.pathing import detect_project_paths


def inspect_workflows(root: Path = ROOT) -> list[dict]:
    paths = detect_project_paths(root)
    result = []
    for family_id in list_families(paths):
        spec = load_family_config(paths, family_id)
        roles = spec["role_template"]
        segments = enumerate_legal_segments(roles, set(spec["validator_roles"]))
        partitions = enumerate_legal_partitions(roles, segments)
        skills = []
        for definition in sorted((paths.compiled_skills_dir / family_id).rglob("skill.yaml")):
            payload = read_yaml(definition)
            if payload["role_start"] not in roles or payload["role_end"] not in roles:
                raise ValueError(f"Skill boundary outside its workflow: {definition.relative_to(root)}")
            if payload["validator_role"] not in spec["validator_roles"]:
                raise ValueError(f"Skill output has no validator: {definition.relative_to(root)}")
            skills.append(payload["skill_id"])
        result.append({"family": family_id, "roles": roles, "legal_segments": len(segments),
                       "legal_partitions": len(partitions), "compiled_skill_definitions": len(skills)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the complete workflow summary as JSON.")
    args = parser.parse_args()
    rows = inspect_workflows()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['family']}: {row['legal_partitions']} partitions, "
                  f"{row['compiled_skill_definitions']} compiled skill definitions")


if __name__ == "__main__":
    main()
