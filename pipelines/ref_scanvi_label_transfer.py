from __future__ import annotations

import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy import sparse

from pipelines.singlecell_common import (
    RoleArtifact,
    SINGLECELL_ROLE_TEMPLATE,
    SINGLECELL_STEP_SPECS,
    SingleCellReferenceRunnerBase,
    classification_metrics,
    resolve_training_device,
    save_confusion_outputs,
    seed_everything,
)
from utils.io_utils import read_json, write_json

from tasks.tabula_muris_tasks import ensure_tabula_muris_files

FAMILY_ID = "tabula_muris_label_transfer"


def _round_sparse_in_place(matrix) -> None:
    if sparse.issparse(matrix):
        matrix.data = np.rint(matrix.data)


def _normalize_facs_by_gene_length(adata, gene_length: pd.DataFrame):
    gene_length = gene_length.reindex(adata.var_names).dropna()
    adata = adata[:, gene_length.index].copy()
    scale = np.median(gene_length[1].values) / gene_length[1].values
    if sparse.issparse(adata.X):
        adata.X = adata.X.multiply(scale).tocsr()
        _round_sparse_in_place(adata.X)
    else:
        adata.X = np.rint(np.asarray(adata.X) * scale)
    return adata


class ScanviLabelTransferReferenceRunner(SingleCellReferenceRunnerBase):
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

        files = ensure_tabula_muris_files(self.paths)
        reference = sc.read_h5ad(files["facs_path"])
        query = sc.read_h5ad(files["droplet_path"])

        tissue = self.task_manifest["tissue"]
        reference = reference[
            (reference.obs["tissue"].astype(str) == tissue)
            & (reference.obs["sex"].astype(str) == "female")
            & (~reference.obs["cell_ontology_class"].isna())
        ].copy()
        query = query[
            (query.obs["tissue"].astype(str) == tissue)
            & (query.obs["sex"].astype(str) == "female")
            & (~query.obs["cell_ontology_class"].isna())
        ].copy()
        reference.obs["tech"] = "SS2"
        query.obs["tech"] = "10x"
        reference.layers["counts"] = reference.X.copy()
        query.layers["counts"] = query.X.copy()

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
            "tissue": tissue,
            "reference_n_cells": int(reference.n_obs),
            "query_n_cells": int(query.n_obs),
            "reference_label_count": int(reference.obs["cell_ontology_class"].nunique()),
            "query_label_count": int(query.obs["cell_ontology_class"].nunique()),
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
        files = ensure_tabula_muris_files(self.paths)
        reference = sc.read_h5ad(raw_role.paths[0])
        query = sc.read_h5ad(raw_role.paths[1])
        gene_len = pd.read_csv(files["gene_length_path"], delimiter=" ", header=None, index_col=0)
        reference = _normalize_facs_by_gene_length(reference, gene_len)

        combined = ad.concat([query, reference], join="inner", label="dataset_role", keys=["query", "reference"])
        combined.layers["counts"] = combined.X.copy()
        combined.obs["tech"] = combined.obs["tech"].astype(str).astype("category")
        combined.obs["cell_ontology_class"] = combined.obs["cell_ontology_class"].astype(str)
        combined.obs["celltype_scanvi"] = "Unknown"
        reference_mask = combined.obs["tech"].astype(str) == "SS2"
        combined.obs.loc[reference_mask, "celltype_scanvi"] = combined.obs.loc[reference_mask, "cell_ontology_class"].astype(str)
        sc.pp.normalize_total(combined, target_sum=1e4)
        sc.pp.log1p(combined)
        combined.raw = combined.copy()
        sc.pp.highly_variable_genes(
            combined,
            flavor="seurat_v3",
            n_top_genes=int(self.family_config["reference_recipe"]["highly_variable_n_top_genes"]),
            layer="counts",
            batch_key="tech",
            subset=True,
        )

        query_prepared = combined[combined.obs["tech"].astype(str) == "10x"].copy()
        reference_prepared = combined[combined.obs["tech"].astype(str) == "SS2"].copy()
        reference_path = role_dir / "reference_prepared.h5ad"
        query_path = role_dir / "query_prepared.h5ad"
        combined_path = role_dir / "combined_prepared.h5ad"
        summary_path = role_dir / "summary.json"
        reference_prepared.write_h5ad(reference_path)
        query_prepared.write_h5ad(query_path)
        combined.write_h5ad(combined_path)

        summary = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "gene_count": int(combined.n_vars),
            "reference_n_cells": int(reference_prepared.n_obs),
            "query_n_cells": int(query_prepared.n_obs),
            "combined_n_cells": int(combined.n_obs),
            "counts_layer_present": "counts" in combined.layers,
        }
        write_json(summary_path, summary)

        artifact = RoleArtifact(
            "prepared_query",
            [reference_path, query_path, combined_path, summary_path],
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
        adata = sc.read_h5ad(prepared_role.paths[2])
        recipe = self.family_config["reference_recipe"]

        seed_everything(self.seed)
        accelerator, devices = resolve_training_device(
            recipe.get("accelerator", "auto"),
            int(recipe.get("devices", 1)),
        )
        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key=recipe["batch_key"],
        )
        scvi_model = scvi.model.SCVI(adata, n_latent=int(recipe["n_latent"]))
        scvi_model.train(
            max_epochs=int(recipe["scvi_max_epochs"]),
            accelerator=accelerator,
            devices=devices,
            batch_size=int(recipe["batch_size"]),
            early_stopping=False,
            datasplitter_kwargs={"drop_last": True},
        )
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model,
            adata=adata,
            unlabeled_category=recipe["unlabeled_category"],
            labels_key=recipe["labels_key"],
        )
        scanvi_model.train(
            max_epochs=int(recipe["scanvi_max_epochs"]),
            n_samples_per_label=int(recipe["n_samples_per_label"]),
            accelerator=accelerator,
            devices=devices,
            batch_size=int(recipe["batch_size"]),
            datasplitter_kwargs={"drop_last": True},
        )

        adata.obsm["X_scVI"] = scvi_model.get_latent_representation(adata)
        adata.obsm["X_scANVI"] = scanvi_model.get_latent_representation(adata)
        adata.obs["predicted_celltype"] = scanvi_model.predict(adata).astype(str)

        latent_path = role_dir / "combined_latent.h5ad"
        scvi_dir = role_dir / "scvi_model"
        scanvi_dir = role_dir / "scanvi_model"
        summary_path = role_dir / "summary.json"
        scvi_model.save(scvi_dir, overwrite=True, save_anndata=False)
        scanvi_model.save(scanvi_dir, overwrite=True, save_anndata=False)
        adata.write_h5ad(latent_path)
        summary = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "seed": self.seed,
            "accelerator": accelerator,
            "devices": devices,
            "latent_key_scvi": "X_scVI",
            "latent_key_scanvi": "X_scANVI",
            "prediction_count": int(adata.n_obs),
        }
        write_json(summary_path, summary)

        artifact = RoleArtifact(
            "latent_or_graph",
            [latent_path, scvi_dir, scanvi_dir, summary_path],
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

        latent_adata = sc.read_h5ad(self.role_artifacts["latent_or_graph"].paths[0])
        query_mask = latent_adata.obs["tech"].astype(str) == "10x"
        query_adata = latent_adata[query_mask].copy()
        prediction_frame = pd.DataFrame(
            {
                "cell_id": query_adata.obs_names.astype(str),
                "tissue": query_adata.obs["tissue"].astype(str).tolist(),
                "true_label": query_adata.obs["cell_ontology_class"].astype(str).tolist(),
                "predicted_label": query_adata.obs["predicted_celltype"].astype(str).tolist(),
                "label_training_key": query_adata.obs["celltype_scanvi"].astype(str).tolist(),
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
            input_paths=self.role_artifacts["latent_or_graph"].paths,
            output_paths=artifact.paths,
            details=summary,
        )
        return artifact

    def build_mapping_metrics(self) -> RoleArtifact:
        started_at = time.perf_counter()
        role_dir = self._role_dir("mapping_metrics")
        role_dir.mkdir(parents=True, exist_ok=True)

        prediction_frame = pd.read_csv(self.role_artifacts["predicted_labels"].paths[0])
        metrics = classification_metrics(
            prediction_frame["true_label"].astype(str),
            prediction_frame["predicted_label"].astype(str),
        )
        confusion = save_confusion_outputs(
            prediction_frame,
            role_dir / "confusion",
            truth_key="true_label",
            prediction_key="predicted_label",
            normalize_rows=False,
        )
        unknown_rate = float((prediction_frame["predicted_label"].astype(str) == "Unknown").mean())
        metrics_payload = {
            "family_id": FAMILY_ID,
            "task_name": self.task_name,
            "task_id": self.task_manifest["task_id"],
            "seed": self.seed,
            "primary_metric": self.metric_policy["primary_metric"],
            "required_metrics": self.metric_policy["required_metrics"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "unknown_rate": unknown_rate,
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
            details={"accuracy": metrics_payload["accuracy"], "macro_f1": metrics_payload["macro_f1"]},
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
                f"- tissue: `{self.task_manifest['tissue']}`",
                f"- reference_technology: `{self.task_manifest['reference_technology']}`",
                f"- query_technology: `{self.task_manifest['query_technology']}`",
                "",
                "## Metrics",
                "",
                f"- accuracy: `{metrics_payload['accuracy']:.6f}`",
                f"- macro_f1: `{metrics_payload['macro_f1']:.6f}`",
                f"- unknown_rate: `{metrics_payload['unknown_rate']:.6f}`",
                "",
                "## Seeds",
                "",
                f"- seed: `{self.seed}`",
                "- warning_note: `scVI/scANVI training is stochastic; family-level thresholds use a seed-aware reference band`",
                "",
                "## Reference Query Summary",
                "",
                f"- reference_n_cells: `{self.task_manifest['reference_n_cells']}`",
                f"- query_n_cells: `{self.task_manifest['query_n_cells']}`",
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
