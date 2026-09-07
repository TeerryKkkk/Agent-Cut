from __future__ import annotations

import traceback
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.family_runtime import run_reference_task
from utils.family_registry import (
    ensure_family_manifests,
    list_families,
    list_family_tasks,
    load_family_config,
    load_validator_module,
)
from utils.io_utils import read_json, write_json
from utils.pathing import detect_project_paths


def _seed_list(family_spec: dict[str, Any]) -> list[int]:
    recipe = family_spec["reference_recipe"]
    if "random_seeds" in recipe:
        return [int(seed) for seed in recipe["random_seeds"]]
    if "random_seed" in recipe:
        return [int(recipe["random_seed"])]
    return [0]


def _metrics_for_run(validator_module, run_dir: Path) -> dict[str, Any]:
    metrics_path = validator_module.artifact_paths_for_run(run_dir)["mapping_metrics"]
    return read_json(metrics_path)


def _metric_band(metric_rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    band = {}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in metric_rows if metric_name in row]
        if not values:
            continue
        band[metric_name] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "values": values,
        }
    return band


def _run_label_for_seed(seed: int, multi_seed: bool) -> str | None:
    if not multi_seed:
        return None
    return f"seed_{seed:03d}"


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    reference_root = paths.results_dir / "singlecell_substrate" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)

    family_summaries = {}
    for family_id in list_families(paths, family_type="singlecell_mapping"):
        ensure_family_manifests(paths, family_id)
        family_spec = load_family_config(paths, family_id)
        validator_module = load_validator_module(paths, family_id)
        task_names = list_family_tasks(paths, family_id)
        seeds = _seed_list(family_spec)
        multi_seed = len(seeds) > 1

        task_rows = {}
        family_results = []
        for task_name in task_names:
            seed_rows = []
            metrics_rows = []
            for seed in seeds:
                run_label = _run_label_for_seed(seed, multi_seed)
                try:
                    runner = run_reference_task(
                        paths,
                        family_id,
                        task_name,
                        run_namespace="reference",
                        run_label=run_label,
                        seed=seed,
                    )
                    validation_results = validator_module.validate_run_directory(
                        runner.run_dir,
                        workspace_root=paths.workspace_root,
                    )
                    failures = {
                        role: result.as_dict()
                        for role, result in validation_results.items()
                        if not result.passed
                    }
                    metrics = _metrics_for_run(validator_module, runner.run_dir)
                    metrics_rows.append(metrics)
                    seed_rows.append(
                        {
                            "seed": seed,
                            "run_dir": paths.relative_to_workspace(runner.run_dir),
                            "status": "pass" if not failures else "validator_fail",
                            "validator_failures": failures,
                            "metrics": metrics,
                        }
                    )
                    family_results.append(
                        {
                            "task_name": task_name,
                            "seed": seed,
                            "status": "pass" if not failures else "validator_fail",
                            "run_dir": paths.relative_to_workspace(runner.run_dir),
                        }
                    )
                except Exception as exc:
                    row = {
                        "seed": seed,
                        "status": "fail",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    seed_rows.append(row)
                    family_results.append({"task_name": task_name, "seed": seed, "status": "fail", "error": str(exc)})

            metric_names = [
                family_spec["primary_metric"],
                *load_family_config(paths, family_id).get("reference_recipe", {}).get("band_metrics", []),
                *load_family_config(paths, family_id).get("reference_recipe", {}).get("report_metrics", []),
            ]
            if family_id == "scanpy_pancreas_ingest":
                metric_names = ["acc_all", "acc_conserved", "macro_f1", "ref_type_coverage"]
            if family_id == "tabula_muris_label_transfer":
                metric_names = ["accuracy", "macro_f1", "unknown_rate"]

            canonical_row = next((row for row in seed_rows if row["status"] in {"pass", "validator_fail"}), None)
            task_rows[task_name] = {
                "task_name": task_name,
                "seed_runs": seed_rows,
                "metric_band": _metric_band(metrics_rows, metric_names),
                "canonical_run_dir": canonical_row["run_dir"] if canonical_row else None,
                "passed_all_seed_runs": all(row["status"] == "pass" for row in seed_rows),
            }

        passed = sum(1 for row in family_results if row["status"] == "pass")
        summary = {
            "family_id": family_id,
            "task_count": len(task_names),
            "seed_count": len(seeds),
            "total_runs": len(family_results),
            "passed_runs": passed,
            "failed_runs": len(family_results) - passed,
            "passed_all": all(task_row["passed_all_seed_runs"] for task_row in task_rows.values()),
            "tasks": task_rows,
            "results": family_results,
        }
        family_summaries[family_id] = summary
        family_dir = reference_root / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        write_json(family_dir / "summary.json", summary)
        print(f"{family_id}: passed_runs={passed}/{len(family_results)}")

    aggregate = {
        "families": family_summaries,
        "all_families_passed": all(summary["passed_all"] for summary in family_summaries.values()),
    }
    write_json(reference_root / "summary.json", aggregate)
    print(f"Single-cell reference summary: {paths.relative_to_workspace(reference_root / 'summary.json')}")


if __name__ == "__main__":
    main()
