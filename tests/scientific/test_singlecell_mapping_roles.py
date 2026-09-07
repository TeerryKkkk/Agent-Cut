from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from utils.io_utils import write_json, write_text
from validators.singlecell_mapping_roles import (
    validate_mapping_metrics,
    validate_predicted_labels,
    validate_report_md,
)


class SingleCellMappingValidatorTests(unittest.TestCase):
    def test_predicted_labels_detects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            (run_dir / "artifacts" / "predicted_labels").mkdir(parents=True, exist_ok=True)
            (run_dir / "artifacts" / "latent_or_graph").mkdir(parents=True, exist_ok=True)

            adata = ad.AnnData(X=np.eye(2))
            adata.obs["tech"] = pd.Categorical(["10x", "SS2"])
            adata.obs["cell_ontology_class"] = ["A", "B"]
            adata.write_h5ad(run_dir / "artifacts" / "latent_or_graph" / "combined_latent.h5ad")
            write_json(run_dir / "summary.json", {"family_id": "tabula_muris_label_transfer"})
            pd.DataFrame(columns=["cell_id", "true_label", "predicted_label"]).to_csv(
                run_dir / "artifacts" / "predicted_labels" / "predictions.csv",
                index=False,
            )

            result = validate_predicted_labels(
                run_dir / "artifacts" / "predicted_labels",
                "tabula_muris_label_transfer",
                run_dir=run_dir,
            )
            self.assertFalse(result.passed, result.as_dict())
            self.assertEqual(result.error_code, "predictions_empty")

    def test_predicted_labels_detects_cell_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            (run_dir / "artifacts" / "predicted_labels").mkdir(parents=True, exist_ok=True)
            (run_dir / "artifacts" / "latent_or_graph").mkdir(parents=True, exist_ok=True)

            adata = ad.AnnData(X=np.eye(3))
            adata.obs_names = ["cell_a", "cell_b", "cell_c"]
            adata.obs["tech"] = pd.Categorical(["10x", "10x", "SS2"])
            adata.obs["cell_ontology_class"] = ["A", "B", "A"]
            adata.write_h5ad(run_dir / "artifacts" / "latent_or_graph" / "combined_latent.h5ad")
            write_json(run_dir / "summary.json", {"family_id": "tabula_muris_label_transfer"})
            pd.DataFrame(
                {
                    "cell_id": ["cell_b", "cell_a"],
                    "true_label": ["A", "B"],
                    "predicted_label": ["A", "B"],
                }
            ).to_csv(run_dir / "artifacts" / "predicted_labels" / "predictions.csv", index=False)

            result = validate_predicted_labels(
                run_dir / "artifacts" / "predicted_labels",
                "tabula_muris_label_transfer",
                run_dir=run_dir,
            )
            self.assertFalse(result.passed, result.as_dict())
            self.assertEqual(result.error_code, "predictions_cell_order_mismatch")

    def test_mapping_metrics_requires_confusion_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs_dir = root / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)
            write_text(
                configs_dir / "metrics.yaml",
                "\n".join(
                    [
                        "families:",
                        "  scanpy_pancreas_ingest:",
                        "    required_metrics:",
                        "      - acc_all",
                        "      - acc_conserved",
                        "      - macro_f1",
                        "      - ref_type_coverage",
                        "      - confusion_matrix_path",
                    ]
                )
                + "\n",
            )
            metrics_path = root / "metrics.json"
            write_json(
                metrics_path,
                {
                    "family_id": "scanpy_pancreas_ingest",
                    "task_name": "x",
                    "primary_metric": "acc_all",
                    "acc_all": 1.0,
                    "acc_conserved": 1.0,
                    "macro_f1": 1.0,
                    "ref_type_coverage": 1.0,
                    "confusion_matrix_path": "missing.csv",
                },
            )
            workspace_root = root
            result = validate_mapping_metrics(metrics_path, "scanpy_pancreas_ingest", workspace_root=workspace_root)
            self.assertFalse(result.passed, result.as_dict())
            self.assertEqual(result.error_code, "metrics_confusion_missing_file")

    def test_mapping_metrics_detects_prediction_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs_dir = root / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)
            write_text(
                configs_dir / "metrics.yaml",
                "\n".join(
                    [
                        "families:",
                        "  tabula_muris_label_transfer:",
                        "    required_metrics:",
                        "      - accuracy",
                        "      - macro_f1",
                        "      - confusion_matrix_path",
                        "      - unknown_rate",
                    ]
                )
                + "\n",
            )
            run_dir = root / "run"
            (run_dir / "artifacts" / "predicted_labels").mkdir(parents=True, exist_ok=True)
            (run_dir / "artifacts" / "mapping_metrics" / "confusion").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "cell_id": ["c1", "c2"],
                    "true_label": ["A", "B"],
                    "predicted_label": ["A", "B"],
                }
            ).to_csv(run_dir / "artifacts" / "predicted_labels" / "predictions.csv", index=False)
            confusion_path = run_dir / "artifacts" / "mapping_metrics" / "confusion" / "confusion_matrix.csv"
            confusion_path.write_text("truth,pred\nA,A\n", encoding="utf-8")
            metrics_path = run_dir / "artifacts" / "mapping_metrics" / "metrics.json"
            write_json(
                metrics_path,
                {
                    "family_id": "tabula_muris_label_transfer",
                    "task_name": "demo",
                    "primary_metric": "accuracy",
                    "accuracy": 0.5,
                    "macro_f1": 1.0,
                    "unknown_rate": 0.0,
                    "confusion_matrix_path": "run/artifacts/mapping_metrics/confusion/confusion_matrix.csv",
                },
            )

            result = validate_mapping_metrics(metrics_path, "tabula_muris_label_transfer", workspace_root=root)
            self.assertFalse(result.passed, result.as_dict())
            self.assertEqual(result.error_code, "metrics_prediction_mismatch")

    def test_report_validator_detects_missing_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.md"
            write_text(report_path, "# Mapping Report: demo\n\n## Task\n\n## Metrics\n")
            result = validate_report_md(report_path, "scanpy_pancreas_ingest")
            self.assertFalse(result.passed, result.as_dict())
            self.assertEqual(result.error_code, "report_missing_sections")


if __name__ == "__main__":
    unittest.main()
