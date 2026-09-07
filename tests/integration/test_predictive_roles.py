from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.predictive_runtime import run_reference_task
from utils.io_utils import read_json, write_json
from utils.pathing import detect_project_paths
from validators.predictive_roles import validate_metrics_json, validate_run_directory


class PredictiveRoleValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = detect_project_paths(ROOT_DIR)
        cls.openml_runner = run_reference_task(cls.paths, "openml_tabular_binary", "adult")
        cls.tdc_binary_runner = run_reference_task(cls.paths, "tdc_admet_binary", "HIA_Hou")
        cls.tdc_regression_runner = run_reference_task(cls.paths, "tdc_admet_regression", "Caco2_Wang")

    def test_openml_reference_run_passes_all_validators(self) -> None:
        results = validate_run_directory(self.openml_runner.run_dir, workspace_root=self.paths.workspace_root)
        failures = {role: result.as_dict() for role, result in results.items() if not result.passed}
        self.assertFalse(failures, failures)

    def test_tdc_binary_reference_run_passes_all_validators(self) -> None:
        results = validate_run_directory(self.tdc_binary_runner.run_dir, workspace_root=self.paths.workspace_root)
        failures = {role: result.as_dict() for role, result in results.items() if not result.passed}
        self.assertFalse(failures, failures)

    def test_tdc_regression_reference_run_passes_all_validators(self) -> None:
        results = validate_run_directory(self.tdc_regression_runner.run_dir, workspace_root=self.paths.workspace_root)
        failures = {role: result.as_dict() for role, result in results.items() if not result.passed}
        self.assertFalse(failures, failures)

    def test_metrics_validator_detects_corruption(self) -> None:
        tmp_root = self.paths.results_dir / "test_validator_tmp" / "metrics_corruption"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_root / "metrics.json"
        payload = read_json(self.openml_runner.run_dir / "artifacts" / "metrics.json")
        payload["balanced_accuracy"] = float(payload["balanced_accuracy"]) + 0.25
        write_json(tmp_path, payload)
        result = validate_metrics_json(tmp_path, workspace_root=self.paths.workspace_root)
        self.assertFalse(result.passed, result.as_dict())
        self.assertEqual(result.error_code, "metrics_inconsistent")


if __name__ == "__main__":
    unittest.main()
