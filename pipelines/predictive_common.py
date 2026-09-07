from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.io_utils import read_yaml

PREDICTIVE_ROLE_TEMPLATE = [
    "raw_data",
    "profile_json",
    "split_spec_json",
    "preprocess_bundle",
    "model_bundle",
    "metrics_json",
    "report_md",
]

PREDICTIVE_STEP_SPECS = [
    ("step_01_materialize_raw_data", "source_dataset", "raw_data", "materialize_raw_data"),
    ("step_02_profile_dataset", "raw_data", "profile_json", "build_profile"),
    ("step_03_define_split", "profile_json", "split_spec_json", "build_split_spec"),
    ("step_04_fit_preprocess", "split_spec_json", "preprocess_bundle", "build_preprocess_bundle"),
    ("step_05_fit_model", "preprocess_bundle", "model_bundle", "build_model_bundle"),
    ("step_06_compute_metrics", "model_bundle", "metrics_json", "build_metrics"),
    ("step_07_write_report", "metrics_json", "report_md", "build_report"),
]

INTERNAL_METRIC_ALIASES = {
    "roc-auc": "roc_auc",
    "auroc": "roc_auc",
    "roc_auc": "roc_auc",
    "pr-auc": "average_precision",
    "auprc": "average_precision",
    "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
    "macro_f1": "macro_f1",
    "mae": "mae",
    "rmse": "rmse",
    "r2": "r2",
    "spearman": "spearman",
}

HIGHER_IS_BETTER = {
    "roc_auc": True,
    "average_precision": True,
    "balanced_accuracy": True,
    "accuracy": True,
    "macro_f1": True,
    "r2": True,
    "spearman": True,
    "mae": False,
    "rmse": False,
}

_RDKIT_CACHE: tuple[Any, Any, Any] | None = None


def normalize_metric_name(metric_name: str) -> str:
    if metric_name not in INTERNAL_METRIC_ALIASES:
        raise KeyError(f"Unsupported metric name: {metric_name}")
    return INTERNAL_METRIC_ALIASES[metric_name]


def metric_higher_is_better(metric_name: str) -> bool:
    normalized = normalize_metric_name(metric_name)
    if normalized not in HIGHER_IS_BETTER:
        raise KeyError(f"No direction registered for metric: {metric_name}")
    return HIGHER_IS_BETTER[normalized]


def compute_binary_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    probability_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, float], str | None]:
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
    }

    positive_label = None
    if probability_frame is not None:
        probability_columns = [column for column in probability_frame.columns if column.startswith("proba_")]
        if len(probability_columns) == 2:
            positive_column = probability_columns[-1]
            positive_label = positive_column.replace("proba_", "", 1)
            actual_binary = (actual.astype(str) == positive_label).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(actual_binary, probability_frame[positive_column]))
            metrics["average_precision"] = float(average_precision_score(actual_binary, probability_frame[positive_column]))

    return metrics, positive_label


def compute_regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    spearman_value = spearmanr(actual, predicted).statistic
    if np.isnan(spearman_value):
        spearman_value = 0.0
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(rmse),
        "r2": float(r2_score(actual, predicted)),
        "spearman": float(spearman_value),
    }


def reference_preprocessor_identity() -> Pipeline:
    return Pipeline(steps=[("identity", FunctionTransformer(validate=False))])


def suppress_rdkit_stderr() -> contextlib.AbstractContextManager[None]:
    return contextlib.redirect_stderr(io.StringIO())


def load_rdkit_modules() -> tuple[Any, Any, Any]:
    global _RDKIT_CACHE
    if _RDKIT_CACHE is None:
        with suppress_rdkit_stderr():
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem

        _RDKIT_CACHE = (Chem, DataStructs, AllChem)
    return _RDKIT_CACHE


def canonicalize_smiles(smiles: str) -> tuple[str, bool]:
    Chem, _, _ = load_rdkit_modules()
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return "", False
    return Chem.MolToSmiles(molecule, canonical=True), True


def featurize_smiles(
    smiles_series: pd.Series,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[str], list[bool]]:
    Chem, DataStructs, AllChem = load_rdkit_modules()
    features: list[np.ndarray] = []
    canonical_smiles: list[str] = []
    valid_mask: list[bool] = []

    with suppress_rdkit_stderr():
        for smiles in smiles_series.astype(str):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                features.append(np.zeros(n_bits, dtype=np.float32))
                canonical_smiles.append("")
                valid_mask.append(False)
                continue

            canonical = Chem.MolToSmiles(molecule, canonical=True)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
            array = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            features.append(array.astype(np.float32))
            canonical_smiles.append(canonical)
            valid_mask.append(True)

    return np.vstack(features), canonical_smiles, valid_mask


def featurize_smiles_hashing(
    smiles_series: pd.Series,
    *,
    n_features: int = 2048,
    ngram_range: tuple[int, int] = (1, 3),
) -> tuple[np.ndarray, list[str], list[bool]]:
    normalized_smiles = smiles_series.fillna("").astype(str).str.strip().tolist()
    valid_mask = [bool(item) for item in normalized_smiles]
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        lowercase=False,
    )
    matrix = vectorizer.transform(normalized_smiles).astype(np.float32).toarray()
    return matrix, normalized_smiles, valid_mask


def load_family_config(paths, family_id: str) -> dict[str, Any]:
    return read_yaml(paths.configs_dir / "families.yaml")["families"][family_id]


def load_dataset_catalog(paths) -> dict[str, dict[str, Any]]:
    return read_yaml(paths.configs_dir / "datasets.yaml")["datasets"]


def load_family_metric_policy(paths, family_id: str) -> dict[str, Any]:
    return read_yaml(paths.configs_dir / "metrics.yaml")["families"][family_id]


def family_dataset_names(paths, family_id: str) -> list[str]:
    catalog = load_dataset_catalog(paths)
    return [name for name, spec in catalog.items() if spec["family_id"] == family_id]


def resolve_primary_metric(metric_policy: dict[str, Any], task_name: str) -> str:
    dataset_overrides = metric_policy.get("dataset_metric_overrides", {})
    metric_name = dataset_overrides.get(task_name, metric_policy["primary_metric"])
    return normalize_metric_name(metric_name)


def threshold_passes(reference_value: float, observed_value: float, primary_metric: str, threshold_rule: dict[str, Any]) -> tuple[bool, float]:
    normalized_metric = normalize_metric_name(primary_metric)
    higher_is_better = metric_higher_is_better(normalized_metric)

    if higher_is_better:
        drop = float(reference_value) - float(observed_value)
        allowed_drop = float(threshold_rule.get("max_primary_metric_drop", 0.0))
        return drop <= allowed_drop, drop

    increase = float(observed_value) - float(reference_value)
    allowed_increase = float(threshold_rule.get("max_primary_metric_increase", threshold_rule.get("max_primary_metric_drop", 0.0)))
    return increase <= allowed_increase, increase
