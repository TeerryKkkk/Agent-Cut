import pytest

from cut.enumerate_partitions import enumerate_legal_partitions, enumerate_legal_segments
from examples.inspect_workflows import inspect_workflows
from utils.pathing import detect_project_paths


def test_workflow_partitions_cover_every_transition_once():
    roles = ["raw", "prepared", "model", "metrics"]
    segments = enumerate_legal_segments(roles, set(roles[1:]))
    partitions = enumerate_legal_partitions(roles, segments)
    assert len(partitions) == 4
    index = {segment["segment_id"]: segment for segment in segments}
    for partition in partitions:
        cursor = 0
        for segment_id in partition["segment_ids"]:
            segment = index[segment_id]
            assert segment["start_index"] == cursor
            cursor = segment["end_index"]
        assert cursor == len(roles) - 1


def test_only_validated_roles_can_be_boundaries():
    roles = ["raw", "unvalidated", "model", "metrics"]
    segments = enumerate_legal_segments(roles, {"model", "metrics"})
    partitions = enumerate_legal_partitions(roles, segments)
    assert len(partitions) == 2
    assert all("unvalidated" not in p["boundary_roles"] for p in partitions)


def test_repository_discovery_needs_no_secret_file(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/families.yaml").write_text("families: {}", encoding="utf-8")
    (tmp_path / "cut").mkdir()
    nested = tmp_path / "examples/nested"
    nested.mkdir(parents=True)
    assert detect_project_paths(nested).workspace_root == tmp_path.resolve()


def test_unrelated_directory_does_not_become_a_repository(tmp_path):
    with pytest.raises(RuntimeError, match="Agent-Cut"):
        detect_project_paths(tmp_path)


def test_all_saved_skill_boundaries_have_validators():
    rows = inspect_workflows()
    assert len(rows) == 6
    assert all(row["compiled_skill_definitions"] > 0 for row in rows)
    assert {row["legal_partitions"] for row in rows} == {16, 32}
