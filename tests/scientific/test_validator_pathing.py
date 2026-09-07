from importlib import import_module

import pytest


@pytest.fixture(params=["predictive_roles", "singlecell_mapping_roles"])
def validator(request):
    return import_module(f"validators.{request.param}")


def test_validator_finds_repository_without_credentials(validator, tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/families.yaml").write_text("families: {}", encoding="utf-8")
    (tmp_path / "cut").mkdir()
    artifact = tmp_path / "runs/reference/demo/artifacts/metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    assert validator.infer_workspace_root(artifact) == tmp_path.resolve()


def test_validator_does_not_infer_unrelated_directory(validator, tmp_path):
    assert validator.infer_workspace_root(tmp_path / "metrics.json") is None


def test_mapping_metrics_infers_required_metric_policy(tmp_path):
    from validators.singlecell_mapping_roles import validate_mapping_metrics

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/families.yaml").write_text("families: {}", encoding="utf-8")
    (tmp_path / "configs/metrics.yaml").write_text(
        "families:\n  scanpy_pancreas_ingest:\n    required_metrics: [acc_all]\n",
        encoding="utf-8",
    )
    (tmp_path / "cut").mkdir()
    artifact = tmp_path / "runs/reference/demo/artifacts/mapping_metrics/metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    result = validate_mapping_metrics(artifact, "scanpy_pancreas_ingest")
    assert not result.passed
    assert result.error_code == "metrics_missing_fields"
    assert result.details["missing_metrics"] == ["acc_all"]
