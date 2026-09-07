from __future__ import annotations

from pipelines.family_runtime import build_reference_runner, run_reference_family, run_reference_task
from utils.family_registry import list_families


def list_predictive_families(paths) -> list[str]:
    return list_families(paths, family_type="predictive")
