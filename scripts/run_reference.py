"""Run one reference task from a configured scientific workflow family."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.family_registry import list_families, list_family_tasks
from utils.pathing import detect_project_paths, ensure_workspace_dirs


def main() -> None:
    paths = detect_project_paths(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=list_families(paths), required=True)
    parser.add_argument("--task", required=True, help="Dataset or task name in the selected family.")
    args = parser.parse_args()
    available = list_family_tasks(paths, args.family)
    if args.task not in available:
        parser.error(f"Task must be one of: {', '.join(available)}")
    from pipelines.family_runtime import run_reference_task

    ensure_workspace_dirs(paths)
    runner = run_reference_task(paths, args.family, args.task)
    print(f"Artifacts: {runner.run_dir.relative_to(paths.workspace_root)}")


if __name__ == "__main__":
    main()
