from pathlib import Path
import shutil

from cut.score_segments import score_family_segments
from cut.select_partition import select_family_partition
from runtime.workflow_evaluation import score_partitions_for_split
from utils.io_utils import append_jsonl, read_json, read_yaml, write_json
from utils.pathing import ProjectPaths


def test_runtime_and_compiler_agree_on_predictive_partition_scores(tmp_path):
    root = Path(__file__).resolve().parents[2]
    family = "openml_tabular_binary"
    paths = ProjectPaths(launch_root=tmp_path, workspace_root=tmp_path)
    paths.configs_dir.mkdir()
    shutil.copyfile(root / "configs/families.yaml", paths.configs_dir / "families.yaml")
    partitions_dir = paths.results_dir / "partitions" / family
    partitions_dir.mkdir(parents=True)
    for name in ("legal_segments.json", "legal_partitions.json"):
        shutil.copyfile(root / "benchmarks/partitions" / family / name, partitions_dir / name)

    write_json(
        paths.results_dir / "predictive_substrate/validator_qa/summary.json",
        {"valid_results": [], "bad_results": []},
    )
    roles = read_yaml(paths.configs_dir / "families.yaml")["families"][family]["role_template"]
    for role_in, role_out in zip(roles, roles[1:]):
        append_jsonl(
            paths.runs_dir / "reference" / family / "synthetic/trace.jsonl",
            {"status": "pass", "role_in": role_in, "role_out": role_out},
        )

    score_family_segments(paths, family, support_tasks=["synthetic"])
    select_family_partition(paths, family)
    expected = read_json(partitions_dir / "partition_scores.json")
    actual = score_partitions_for_split(paths, family, support_tasks=["synthetic"])
    assert len(actual) == 32
    assert [row["partition_id"] for row in actual] == [row["partition_id"] for row in expected]
    for observed, saved in zip(actual, expected):
        for key in ("segment_scores", "boundary_bonus", "partition_score"):
            assert observed[key] == saved[key]
