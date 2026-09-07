from __future__ import annotations

import traceback
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_common import family_dataset_names
from pipelines.predictive_runtime import list_predictive_families, run_reference_task
from utils.io_utils import write_json
from utils.pathing import detect_project_paths


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    reference_root = paths.results_dir / "predictive_substrate" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)

    family_summaries = {}
    for family_id in list_predictive_families(paths):
        rows = []
        for task_name in family_dataset_names(paths, family_id):
            try:
                runner = run_reference_task(paths, family_id, task_name, run_namespace="reference")
                rows.append(
                    {
                        "family_id": family_id,
                        "task_name": task_name,
                        "status": "pass",
                        "run_dir": paths.relative_to_workspace(runner.run_dir),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "family_id": family_id,
                        "task_name": task_name,
                        "status": "fail",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        passed = sum(1 for row in rows if row["status"] == "pass")
        summary = {
            "family_id": family_id,
            "task_count": len(rows),
            "passed_task_count": passed,
            "failed_task_count": len(rows) - passed,
            "passed_all": passed == len(rows),
            "results": rows,
        }
        family_summaries[family_id] = summary
        family_root = reference_root / family_id
        family_root.mkdir(parents=True, exist_ok=True)
        write_json(family_root / "summary.json", summary)

    aggregate = {
        "families": family_summaries,
        "all_families_passed": all(summary["passed_all"] for summary in family_summaries.values()),
    }
    write_json(reference_root / "summary.json", aggregate)

    for family_id, summary in family_summaries.items():
        print(
            f"{family_id}: passed_tasks={summary['passed_task_count']}/"
            f"{summary['task_count']}"
        )
    print(f"Reference summary: {paths.relative_to_workspace(reference_root / 'summary.json')}")


if __name__ == "__main__":
    main()
