from __future__ import annotations

import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.family_registry import ensure_family_manifests, list_families
from utils.io_utils import write_json
from utils.pathing import detect_project_paths


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    substrate_root = paths.results_dir / "singlecell_substrate"
    aggregate_root = substrate_root / "aggregate_tables"
    aggregate_root.mkdir(parents=True, exist_ok=True)

    family_rows = []
    for family_id in list_families(paths, family_type="singlecell_mapping"):
        started_at = time.perf_counter()
        print(f"[singlecell-build] start family={family_id}", flush=True)
        summary = ensure_family_manifests(paths, family_id)
        family_rows.append(summary)
        elapsed = time.perf_counter() - started_at
        print(
            f"[singlecell-build] done family={family_id} "
            f"task_count={summary['task_count']} split_count={summary['split_count']} elapsed_s={elapsed:.2f}"
            ,
            flush=True,
        )

    write_json(aggregate_root / "family_table.json", {"families": family_rows})
    print(f"Single-cell manifests: {paths.relative_to_workspace(substrate_root / 'manifests')}", flush=True)


if __name__ == "__main__":
    main()
