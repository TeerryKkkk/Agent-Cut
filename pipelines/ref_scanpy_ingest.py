from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from pipelines.singlecell_common import (
    RoleArtifact,
    SINGLECELL_ROLE_TEMPLATE,
    SINGLECELL_STEP_SPECS,
    SingleCellReferenceRunnerBase,
    classification_metrics,
    save_confusion_outputs,
    seed_everything,
)
from utils.io_utils import read_json, write_json

from tasks.scanpy_ingest_tasks import load_clean_pancreas

FAMILY_ID = "scanpy_pancreas_ingest"


@contextmanager
def _serial_ingest_index():
    from scanpy.tools import _ingest
    from pynndescent import NNDescent

    original = _ingest.Ingest._init_pynndescent

    def _init_pynndescent_serial(self, distances):
        first_col = np.arange(distances.shape[0])[:, None]
        init_indices = np.hstack((first_col, np.stack(distances.tolil().rows)))
        self._nnd_idx = NNDescent(
            data=self._rep,
            metric=self._metric,
            metric_kwds=self._metric_kwds,
            n_neighbors=self._n_neighbors,
            init_graph=init_indices,
            random_state=self._neigh_random_state,
            n_jobs=1,
            tree_init=False,
        )

    _ingest.Ingest._init_pynndescent = _init_pynndescent_serial
    try:
        yield
    finally:
        _ingest.Ingest._init_pynndescent = original


class ScanpyIngestReferenceRunner(SingleCellReferenceRunnerBase):
    FAMILY_ID = FAMILY_ID
    ROLE_TEMPLATE = SINGLECELL_ROLE_TEMPLATE
    STEP_SPECS = SINGLECELL_STEP_SPECS

    def __init__(
        self,
        paths,
        task_name: str,
        *,
        run_namespace: str = "reference",
        run_label: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            paths,
            task_name,
            FAMILY_ID,
            run_namespace=run_namespace,
            run_label=run_label,
            seed=seed,
        )

    def build_reference_query_raw(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("reference_query_raw")
        role_dir.mkdir(parents=True, exist_ok=True)

        adata, _ = load_clean_pancreas(self.paths)
        reference_batch = str(self.task_manifest["reference_batch"])
        query_batch = str(self.task_manifest["query_batch"])
        reference = adata[adata.obs["batch"].astype(str) == reference_batch].copy()
        query = adata[adata.obs["batch"].astype(str) == query_batch].copy()
        reference.obs["batch"] = reference.obs["batch"].astype(str).astype("category")
        reference.obs["celltype"] = reference.obs["celltype"].astype(str).astype("category")
        query.obs["batch"] = query.obs["batch"].astype(str).astype("category")
        query.obs["celltype"] = query.obs["celltype"].astype(str).astype("category")

        reference_path = role_dir / "reference_raw.h5ad"
        query_path = role_dir / "query_raw.h5ad"
        manifest_path = role_dir / "task_manifest.json"
        metadata_path = role_dir / "metadata.json"
        reference.write_h5ad(reference_path)
        query.write_h5ad(query_path)

        metadata = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "task_id": self.task_manifest["task_id"],
            "reference_batch": reference_batch,
            "query_batch": query_batch,
            "reference_n_cells": int(reference.n_obs),
            "query_n_cells": int(query.n_obs),
            "reference_label_count": int(reference.obs["celltype"].nunique()),
            "query_label_count": int(query.obs["celltype"].nunique()),
        }
        write_json(manifest_path, self.task_manifest)
        write_json(metadata_path, metadata)

        artifact = RoleArtifact(
            "reference_query_raw",
            [reference_path, query_path, manifest_path, metadata_path],
            metadata,
        )
        self.role_artifacts["reference_query_raw"] = artifact
        self._record_success(
            step_id="step_01_materialize_reference_query_raw",
            role_in="source_dataset",
            role_out="reference_query_raw",
            function_name="build_reference_query_raw",
            started_at=started_at,
            input_paths=[],
            output_paths=artifact.paths,
            details=metadata,
        )
        return artifact

    def build_prepared_query(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("prepared_query")
        role_dir.mkdir(parents=True, exist_ok=True)

        raw_role = self.role_artifacts["reference_query_raw"]
        reference = sc.read_h5ad(raw_role.paths[0])
        query = sc.read_h5ad(raw_role.paths[1])
        gene_intersection = reference.var_names.intersection(query.var_names)
        reference = reference[:, gene_intersection].copy()
        query = query[:, gene_intersection].copy()
        query.obs["celltype_true"] = query.obs["celltype"].astype(str)

        reference_path = role_dir / "reference_prepared.h5ad"
        query_path = role_dir / "query_prepared.h5ad"
        summary_path = role_dir / "summary.json"
        reference.write_h5ad(reference_path)
        query.write_h5ad(query_path)
        summary = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "gene_count": int(len(gene_intersection)),
            "reference_n_cells": int(reference.n_obs),
            "query_n_cells": int(query.n_obs),
            "query_empty_after_alignment": bool(query.n_obs == 0 or len(gene_intersection) == 0),
        }
        write_json(summary_path, summary)

        artifact = RoleArtifact(
            "prepared_query",
            [reference_path, query_path, summary_path],
            summary,
        )
        self.role_artifacts["prepared_query"] = artifact
        self._record_success(
            step_id="step_02_prepare_query",
            role_in="reference_query_raw",
            role_out="prepared_query",
            function_name="build_prepared_query",
            started_at=started_at,
            input_paths=raw_role.paths,
            output_paths=artifact.paths,
            details=summary,
        )
        return artifact

    def build_latent_or_graph(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("latent_or_graph")
        role_dir.mkdir(parents=True, exist_ok=True)

        prepared_role = self.role_artifacts["prepared_query"]
        reference = sc.read_h5ad(prepared_role.paths[0])
        query = sc.read_h5ad(prepared_role.paths[1])
        recipe = self.family_config["reference_recipe"]

        seed_everything(self.seed)
        sc.settings.seed = self.seed
        sc.pp.pca(reference, n_comps=min(int(recipe["pca_n_comps"]), max(1, reference.n_vars - 1)))
        sc.pp.neighbors(
            reference,
            n_neighbors=min(int(recipe["n_neighbors"]), max(2, reference.n_obs - 1)),
            random_state=self.seed,
            transformer="sklearn",
        )
        sc.tl.umap(reference, min_dist=float(recipe["umap_min_dist"]), random_state=self.seed)

        # Force ingest onto a serial NNDescent path; the default tree build hits
        # Windows shared-memory restrictions in this workspace.
        with _serial_ingest_index():
            sc.tl.ingest(
                query,
                reference,
                obs=recipe["labeling_obs"],
                embedding_method=tuple(recipe["embedding_method"]),
            )
        query.obs["predicted_celltype"] = query.obs["celltype"].astype(str)

        reference_path = role_dir / "reference_graph.h5ad"
        query_path = role_dir / "query_ingested.h5ad"
        summary_path = role_dir / "summary.json"
        reference.write_h5ad(reference_path)
        query.write_h5ad(query_path)
        summary = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "seed": self.seed,
            "reference_has_pca": "X_pca" in reference.obsm,
            "reference_has_umap": "X_umap" in reference.obsm,
            "reference_has_neighbors": "neighbors" in reference.uns,
            "query_has_pca": "X_pca" in query.obsm,
            "query_has_umap": "X_umap" in query.obsm,
            "query_prediction_count": int(query.n_obs),
        }
        write_json(summary_path, summary)

        artifact = RoleArtifact(
            "latent_or_graph",
            [reference_path, query_path, summary_path],
            summary,
        )
        self.role_artifacts["latent_or_graph"] = artifact
        self._record_success(
            step_id="step_03_build_latent_or_graph",
            role_in="prepared_query",
            role_out="latent_or_graph",
            function_name="build_latent_or_graph",
            started_at=started_at,
            input_paths=prepared_role.paths,
            output_paths=artifact.paths,
            details=summary,
        )
        return artifact

    def build_predicted_labels(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("predicted_labels")
        role_dir.mkdir(parents=True, exist_ok=True)

        latent_role = self.role_artifacts["latent_or_graph"]
        query = sc.read_h5ad(latent_role.paths[1])
        prediction_frame = pd.DataFrame(
            {
                "cell_id": query.obs_names.astype(str),
                "batch": query.obs["batch"].astype(str).tolist(),
                "true_label": query.obs["celltype_true"].astype(str).tolist(),
                "predicted_label": query.obs["predicted_celltype"].astype(str).tolist(),
            }
        )
        predictions_path = role_dir / "predictions.csv"
        summary_path = role_dir / "summary.json"
        prediction_frame.to_csv(predictions_path, index=False)
        summary = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "row_count": int(len(prediction_frame)),
            "predicted_label_count": int(prediction_frame["predicted_label"].nunique()),
        }
        write_json(summary_path, summary)

        artifact = RoleArtifact("predicted_labels", [predictions_path, summary_path], summary)
        self.role_artifacts["predicted_labels"] = artifact
        self._record_success(
            step_id="step_04_predict_labels",
            role_in="latent_or_graph",
            role_out="predicted_labels",
            function_name="build_predicted_labels",
            started_at=started_at,
            input_paths=latent_role.paths,
            output_paths=artifact.paths,
            details=summary,
        )
        return artifact

    def build_mapping_metrics(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("mapping_metrics")
        role_dir.mkdir(parents=True, exist_ok=True)

        prediction_frame = pd.read_csv(self.role_artifacts["predicted_labels"].paths[0])
        reference = sc.read_h5ad(self.role_artifacts["latent_or_graph"].paths[0])
        reference_vocab = set(reference.obs["celltype"].astype(str).tolist())
        query_vocab = set(prediction_frame["true_label"].astype(str).tolist())
        conserved_mask = prediction_frame["true_label"].astype(str).isin(reference_vocab)

        metrics = classification_metrics(
            prediction_frame["true_label"].astype(str),
            prediction_frame["predicted_label"].astype(str),
        )
        acc_conserved = float(
            (
                prediction_frame.loc[conserved_mask, "true_label"].astype(str)
                == prediction_frame.loc[conserved_mask, "predicted_label"].astype(str)
            ).mean()
        ) if conserved_mask.any() else 0.0
        confusion = save_confusion_outputs(
            prediction_frame,
            role_dir / "confusion",
            truth_key="true_label",
            prediction_key="predicted_label",
            normalize_rows=False,
        )
        metrics_payload = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "task_id": self.task_manifest["task_id"],
            "seed": self.seed,
            "primary_metric": self.metric_policy["primary_metric"],
            "required_metrics": self.metric_policy["required_metrics"],
            "acc_all": metrics["accuracy"],
            "acc_conserved": acc_conserved,
            "macro_f1": metrics["macro_f1"],
            "ref_type_coverage": len(reference_vocab & query_vocab) / max(1, len(query_vocab)),
            "confusion_matrix_path": self._rel(confusion["csv_path"]),
            "confusion_matrix_png_path": self._rel(confusion["png_path"]),
            "query_rows": int(len(prediction_frame)),
        }
        metrics_path = role_dir / "metrics.json"
        write_json(metrics_path, metrics_payload)

        artifact = RoleArtifact(
            "mapping_metrics",
            [metrics_path, confusion["csv_path"], confusion["png_path"]],
            metrics_payload,
        )
        self.role_artifacts["mapping_metrics"] = artifact
        self._record_success(
            step_id="step_05_compute_mapping_metrics",
            role_in="predicted_labels",
            role_out="mapping_metrics",
            function_name="build_mapping_metrics",
            started_at=started_at,
            input_paths=self.role_artifacts["predicted_labels"].paths,
            output_paths=artifact.paths,
            details={"acc_all": metrics_payload["acc_all"], "acc_conserved": metrics_payload["acc_conserved"]},
        )
        return artifact

    def build_report(self) -> RoleArtifact:
        started_at = time.perf_counter()
        metrics_payload = read_json(self.role_artifacts["mapping_metrics"].paths[0])
        report_path = self.write_role_report(
            [
                f"# Mapping Report: {self.task_name}",
                "",
                "## Task",
                "",
                f"- family_id: `{FAMILY_ID}`",
                f"- task_id: `{self.task_manifest['task_id']}`",
                f"- reference_batch: `{self.task_manifest['reference_batch']}`",
                f"- query_batch: `{self.task_manifest['query_batch']}`",
                "",
                "## Metrics",
                "",
                f"- acc_all: `{metrics_payload['acc_all']:.6f}`",
                f"- acc_conserved: `{metrics_payload['acc_conserved']:.6f}`",
                f"- macro_f1: `{metrics_payload['macro_f1']:.6f}`",
                f"- ref_type_coverage: `{metrics_payload['ref_type_coverage']:.6f}`",
                "",
                "## Determinism",
                "",
                f"- seed: `{self.seed}`",
                "- deterministic_note: `tutorial-consistent Scanpy ingest over a fixed processed pancreas matrix`",
                "",
                "## Warnings",
                "",
                "- none",
            ]
        )
        artifact = RoleArtifact("report_md", [report_path], {"task_name": self.task_name})
        self.role_artifacts["report_md"] = artifact
        self._record_success(
            step_id="step_06_write_report",
            role_in="mapping_metrics",
            role_out="report_md",
            function_name="build_report",
            started_at=started_at,
            input_paths=self.role_artifacts["mapping_metrics"].paths,
            output_paths=artifact.paths,
            details={"seed": self.seed},
        )
        return artifact
