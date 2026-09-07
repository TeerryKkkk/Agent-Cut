from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from provenance.trace_utils import TraceRecorder
from utils.family_registry import ensure_family_manifests, load_family_config, load_family_metric_policy, load_task_manifest
from utils.io_utils import write_json, write_text
from utils.pathing import ProjectPaths, ensure_workspace_dirs

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
SINGLECELL_ROLE_TEMPLATE = [
    "reference_query_raw",
    "prepared_query",
    "latent_or_graph",
    "predicted_labels",
    "mapping_metrics",
    "report_md",
]
SINGLECELL_STEP_SPECS = [
    ("step_01_materialize_reference_query_raw", "source_dataset", "reference_query_raw", "build_reference_query_raw"),
    ("step_02_prepare_query", "reference_query_raw", "prepared_query", "build_prepared_query"),
    ("step_03_build_latent_or_graph", "prepared_query", "latent_or_graph", "build_latent_or_graph"),
    ("step_04_predict_labels", "latent_or_graph", "predicted_labels", "build_predicted_labels"),
    ("step_05_compute_mapping_metrics", "predicted_labels", "mapping_metrics", "build_mapping_metrics"),
    ("step_06_write_report", "mapping_metrics", "report_md", "build_report"),
]


@dataclass
class RoleArtifact:
    role: str
    paths: list[Path]
    metadata: dict[str, Any]


def _validate_hdf5_header(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < len(HDF5_SIGNATURE):
        return False
    with path.open("rb") as handle:
        return handle.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE


def ensure_h5ad_download(url: str, destination: Path) -> Path:
    if _validate_hdf5_header(destination):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    if temporary_path.exists():
        try:
            temporary_path.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                f"Temporary download path is locked by another process: {temporary_path}. "
                "An earlier download attempt is likely still running."
            ) from exc
    if destination.exists():
        destination.unlink()

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    started_at = time.perf_counter()
    print(f"[download] opening url={url} destination={destination.as_posix()}", flush=True)
    with urlopen(request, timeout=120) as response:
        total_bytes = int(response.headers.get("Content-Length", "0") or 0)
        downloaded_bytes = 0
        next_report_bytes = 128 * 1024 * 1024
        next_report_time = started_at + 10.0
        print(
            f"[download] start url={url} destination={destination.as_posix()} total_bytes={total_bytes}",
            flush=True,
        )
        with temporary_path.open("wb") as handle:
            print(f"[download] waiting_for_first_chunk destination={destination.name}", flush=True)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded_bytes += len(chunk)
                if downloaded_bytes == len(chunk):
                    print(
                        f"[download] first_chunk destination={destination.name} bytes={downloaded_bytes}",
                        flush=True,
                    )
                if downloaded_bytes >= next_report_bytes or time.perf_counter() >= next_report_time:
                    handle.flush()
                    elapsed = max(1.0e-6, time.perf_counter() - started_at)
                    rate = downloaded_bytes / elapsed / (1024 * 1024)
                    print(
                        "[download] progress "
                        f"destination={destination.name} downloaded_bytes={downloaded_bytes} "
                        f"total_bytes={total_bytes} rate_mb_s={rate:.2f}"
                    ,
                        flush=True,
                    )
                    next_report_bytes = downloaded_bytes + 128 * 1024 * 1024
                    next_report_time = time.perf_counter() + 10.0
            handle.flush()

    if not _validate_hdf5_header(temporary_path):
        temporary_size = temporary_path.stat().st_size if temporary_path.exists() else 0
        raise RuntimeError(f"Downloaded file at {temporary_path} is not a valid HDF5 payload (size={temporary_size}).")

    temporary_path.replace(destination)
    elapsed = max(1.0e-6, time.perf_counter() - started_at)
    print(
        "[download] complete "
        f"destination={destination.as_posix()} bytes={destination.stat().st_size} elapsed_s={elapsed:.2f}"
        ,
        flush=True,
    )
    return destination


def ensure_text_download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    if temporary_path.exists():
        temporary_path.unlink()

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        content = response.read()
    if not content:
        raise RuntimeError(f"Downloaded text payload from {url} is empty.")
    temporary_path.write_bytes(content)
    temporary_path.replace(destination)
    return destination


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        import scvi

        scvi.settings.seed = seed
    except Exception:
        pass


def resolve_training_device(preference: str = "auto", devices: int = 1) -> tuple[str, int]:
    import torch

    if preference != "auto":
        return preference, devices
    if torch.cuda.is_available():
        return "gpu", devices
    return "cpu", 1


def save_confusion_outputs(
    prediction_frame: pd.DataFrame,
    output_dir: Path,
    *,
    truth_key: str,
    prediction_key: str,
    normalize_rows: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion = pd.crosstab(
        prediction_frame[truth_key].astype(str),
        prediction_frame[prediction_key].astype(str),
        dropna=False,
    )
    if normalize_rows:
        confusion = confusion.div(confusion.sum(axis=1).replace(0, 1), axis=0)

    csv_path = output_dir / "confusion_matrix.csv"
    png_path = output_dir / "confusion_matrix.png"
    confusion.to_csv(csv_path)

    figure_width = max(6, 0.4 * max(1, confusion.shape[1]))
    figure_height = max(6, 0.4 * max(1, confusion.shape[0]))
    plt.figure(figsize=(figure_width, figure_height))
    plt.imshow(confusion.to_numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(confusion.columns)), confusion.columns, rotation=90)
    plt.yticks(range(len(confusion.index)), confusion.index)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return {
        "csv_path": csv_path,
        "png_path": png_path,
        "row_count": int(confusion.shape[0]),
        "column_count": int(confusion.shape[1]),
    }


def classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
    }


class SingleCellReferenceRunnerBase:
    ROLE_TEMPLATE = SINGLECELL_ROLE_TEMPLATE
    STEP_SPECS = SINGLECELL_STEP_SPECS

    def __init__(
        self,
        paths: ProjectPaths,
        task_name: str,
        family_id: str,
        *,
        run_namespace: str = "reference",
        run_label: str | None = None,
        seed: int | None = None,
    ) -> None:
        ensure_workspace_dirs(paths)
        ensure_family_manifests(paths, family_id)

        self.paths = paths
        self.task_name = task_name
        self.family_id = family_id
        self.family_config = load_family_config(paths, family_id)
        self.metric_policy = load_family_metric_policy(paths, family_id)
        self.task_manifest = load_task_manifest(paths, family_id, task_name)
        self.seed = int(seed if seed is not None else self._default_seed())
        self.run_namespace = run_namespace
        self.run_label = run_label

        run_root = paths.runs_dir / run_namespace / family_id / task_name
        self.run_dir = run_root if run_label is None else run_root / run_label
        self.artifacts_dir = self.run_dir / "artifacts"
        self.trace_path = self.run_dir / "trace.jsonl"
        self.prov_path = self.run_dir / "prov.json"
        self.summary_path = self.run_dir / "summary.json"
        self.trace = TraceRecorder(self.trace_path)
        self.role_artifacts: dict[str, RoleArtifact] = {}

        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _default_seed(self) -> int:
        recipe = self.family_config["reference_recipe"]
        if "random_seed" in recipe:
            return int(recipe["random_seed"])
        if "random_seeds" in recipe:
            return int(list(recipe["random_seeds"])[0])
        return 0

    def _rel(self, path: Path) -> str:
        return self.paths.relative_to_workspace(path)

    def _role_dir(self, role: str) -> Path:
        if role == "report_md":
            return self.artifacts_dir
        return self.artifacts_dir / role

    def _record_success(
        self,
        *,
        step_id: str,
        role_in: str,
        role_out: str,
        function_name: str,
        started_at: float,
        input_paths: list[Path],
        output_paths: list[Path],
        details: dict[str, Any] | None = None,
    ) -> None:
        self.trace.record(
            step_id=step_id,
            role_in=role_in,
            role_out=role_out,
            input_artifacts=[self._rel(path) for path in input_paths],
            output_artifacts=[self._rel(path) for path in output_paths],
            function=function_name,
            status="pass",
            wall_clock_s=time.perf_counter() - started_at,
            details=details or {},
        )

    def run_until(self, end_role: str = "report_md") -> dict[str, RoleArtifact]:
        for _, _, role_out, function_name in self.STEP_SPECS:
            if role_out in self.role_artifacts:
                if role_out == end_role:
                    break
                continue
            getattr(self, function_name)()
            if role_out == end_role:
                break

        self._write_run_summary()
        return self.role_artifacts

    def run_segment(self, start_role: str, end_role: str) -> dict[str, RoleArtifact]:
        start_index = self.ROLE_TEMPLATE.index(start_role)
        end_index = self.ROLE_TEMPLATE.index(end_role)
        if start_index > end_index:
            raise ValueError(f"Invalid segment order: {start_role} -> {end_role}")

        if start_role == self.ROLE_TEMPLATE[0] and self.ROLE_TEMPLATE[0] not in self.role_artifacts:
            getattr(self, self.STEP_SPECS[0][3])()
        if start_role != self.ROLE_TEMPLATE[0]:
            self.run_until(start_role)

        for _, _, role_out, function_name in self.STEP_SPECS:
            if self.ROLE_TEMPLATE.index(role_out) <= start_index:
                continue
            if role_out in self.role_artifacts:
                continue
            getattr(self, function_name)()
            if role_out == end_role:
                break

        self._write_run_summary()
        return self.role_artifacts

    def _write_run_summary(self) -> None:
        artifact_registry = {
            role: [self._rel(path) for path in artifact.paths]
            for role, artifact in self.role_artifacts.items()
        }
        summary = {
            "family_id": self.family_id,
            "task_name": self.task_name,
            "task_id": self.task_manifest["task_id"],
            "run_namespace": self.run_namespace,
            "run_label": self.run_label,
            "role_template": self.ROLE_TEMPLATE,
            "artifact_registry": artifact_registry,
            "trace_path": self._rel(self.trace_path),
            "provenance_path": self._rel(self.prov_path),
            "workspace_layout_mode": self.paths.layout_mode,
            "seed": self.seed,
        }
        write_json(self.summary_path, summary)
        self.trace.write_summary(
            self.prov_path,
            {
                "family_id": self.family_id,
                "task_name": self.task_name,
                "task_manifest": self.task_manifest,
                "artifact_registry": artifact_registry,
                "reference_recipe": self.family_config["reference_recipe"],
                "seed": self.seed,
            },
        )

    def write_role_report(self, lines: list[str]) -> Path:
        report_path = self._role_dir("report_md") / "report.md"
        write_text(report_path, "\n".join(lines) + "\n")
        return report_path
