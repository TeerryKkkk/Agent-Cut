from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_runtime import list_predictive_families
from tasks.tdc_tasks import build_authoritative_tdc_registry
from utils.io_utils import write_json
from utils.pathing import detect_project_paths
from utils.predictive_splits import write_family_split_manifests


def main() -> None:
    paths = detect_project_paths(ROOT_DIR)
    substrate_root = paths.results_dir / "predictive_substrate"
    substrate_root.mkdir(parents=True, exist_ok=True)

    family_rows = []
    for family_id in list_predictive_families(paths):
        family_rows.append(write_family_split_manifests(paths, family_id))

    write_json(
        substrate_root / "family_table.json",
        {
            "families": family_rows,
        },
    )
    write_json(substrate_root / "tdc_authority.json", build_authoritative_tdc_registry(paths))

    for row in family_rows:
        print(f"{row['family_id']}: split_count={row['split_count']}")
    print(f"Saved split manifests under: {paths.relative_to_workspace(substrate_root / 'splits')}")


if __name__ == "__main__":
    main()
