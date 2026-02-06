#!/usr/bin/env python3
"""
Train + Evaluate + Log to MLflow + Register model.

Usage (local example):
  python scripts/train_and_evaluate.py

Config via env vars:
  DATA_PATH               Path to CSV containing features + label column.
  LABEL_COL               Label column name (default: "label")
  DROP_COLS               Comma-separated columns to drop (e.g. "timestamp,subsystem")
  TEST_SIZE               Test split fraction (default: 0.2)
  RANDOM_SEED             Random seed (default: 42)
  MLFLOW_TRACKING_URI     e.g. "http://localhost:5001" or inside docker "http://mlflow:5000"
  MLFLOW_EXPERIMENT       Experiment name (default: "hvac_fdd_training")
  REGISTERED_MODEL_NAME   Model registry name (default: "hvac_fdd_rtu")
  REGISTER_ALIAS          Alias to set (default: "Production")
  ARTIFACT_MODEL_PATH     MLflow artifact path name (default: "model")
  OUTPUT_MODEL_PATH       Local model path (default: "models/hvac_fdd_model.joblib")
  OUTPUT_LABEL_MAP_PATH   Local label map path (default: "models/label_map.yaml")

Assumptions:
  - CSV is already "feature matrix" format: one row = one window, columns = engineered features.
  - LABEL_COL contains class names (e.g. "normal", "refrigerant_overcharge", ...)

If you instead have raw telemetry windows, you should run your feature pipeline first
to create the training feature CSV (this script is intentionally focused).
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import mlflow
from mlflow.tracking import MlflowClient


# --------------------------
# Utilities
# --------------------------

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def _safe_mkdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _try_import_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore
        return lgb
    except Exception:
        return None

def _make_label_map(y: pd.Series) -> Dict[str, int]:
    classes = sorted(map(str, y.unique()))
    return {c: i for i, c in enumerate(classes)}

def _encode_labels(y: pd.Series, label_map: Dict[str, int]) -> np.ndarray:
    return np.array([label_map[str(v)] for v in y], dtype=int)

def _decode_labels(y_ids: np.ndarray, inv_map: Dict[int, str]) -> np.ndarray:
    return np.array([inv_map[int(i)] for i in y_ids], dtype=object)


# --------------------------
# Main training routine
# --------------------------

def load_dataset(
    data_path: Path,
    label_col: str,
    drop_cols: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    if label_col not in df.columns:
        raise RuntimeError(f"LABEL_COL='{label_col}' not found. Columns: {list(df.columns)[:30]} ...")

    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])

    if drop_cols:
        drops = [c.strip() for c in drop_cols.split(",") if c.strip()]
        keep_drops = [c for c in drops if c in X.columns]
        if keep_drops:
            X = X.drop(columns=keep_drops)

    # Basic cleanup
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        # simple imputation (median). For production, you may want a pipeline object.
        X = X.fillna(X.median(numeric_only=True))

    # Ensure numeric
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise RuntimeError(f"Non-numeric feature columns found: {non_numeric[:20]} (make features numeric)")

    return X, y


def train_model(X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 42):
    lgb = _try_import_lightgbm()
    if lgb is not None:
        # LightGBM is usually best for tabular classification
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        # Fallback if lightgbm is not installed
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray, id_to_name: Dict[int, str]) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))

    # human-readable report using names
    y_test_names = _decode_labels(y_test, id_to_name)
    y_pred_names = _decode_labels(y_pred, id_to_name)

    report_txt = classification_report(y_test_names, y_pred_names, digits=4)
    cm = confusion_matrix(y_test_names, y_pred_names, labels=[id_to_name[i] for i in sorted(id_to_name)])

    return {
        "metrics": {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        },
        "classification_report_text": report_txt,
        "confusion_matrix": cm,
        "labels_order": [id_to_name[i] for i in sorted(id_to_name)],
    }


def register_and_set_alias(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
    registered_model_name: str,
    alias: str,
) -> str:
    """
    Register model from run artifact and set alias to newest version.
    Returns latest version string.
    """
    # Create a new model version in registry
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, registered_model_name)
    # Wait until READY
    for _ in range(60):
        v = client.get_model_version(registered_model_name, mv.version)
        if v.status == "READY":
            break
        time.sleep(1)

    # Set alias to this version
    client.set_registered_model_alias(registered_model_name, alias, mv.version)
    return str(mv.version)


def main():
    # ---- configuration ----
    data_path = Path(_env("DATA_PATH", "data/processed/train_features.csv"))
    label_col = _env("LABEL_COL", "label")
    drop_cols = os.getenv("DROP_COLS", "")

    test_size = float(_env("TEST_SIZE", "0.2"))
    seed = int(_env("RANDOM_SEED", "42"))

    tracking_uri = _env("MLFLOW_TRACKING_URI", "http://localhost:5001")
    experiment_name = _env("MLFLOW_EXPERIMENT", "hvac_fdd_training")

    registered_model_name = _env("REGISTERED_MODEL_NAME", "hvac_fdd_rtu")
    register_alias = _env("REGISTER_ALIAS", "Production")
    artifact_model_path = _env("ARTIFACT_MODEL_PATH", "model")

    output_model_path = Path(_env("OUTPUT_MODEL_PATH", "models/hvac_fdd_model.joblib"))
    output_label_map_path = Path(_env("OUTPUT_LABEL_MAP_PATH", "models/label_map.yaml"))

    # ---- load data ----
    X, y = load_dataset(data_path, label_col=label_col, drop_cols=drop_cols)
    label_map = _make_label_map(y)
    id_to_name = {v: k for k, v in label_map.items()}
    y_ids = _encode_labels(y, label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ids,
        test_size=test_size,
        random_state=seed,
        stratify=y_ids if len(np.unique(y_ids)) > 1 else None,
    )

    # ---- mlflow setup ----
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    with mlflow.start_run(run_name="train_and_evaluate") as run:
        run_id = run.info.run_id

        # Log dataset + shapes
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("label_col", label_col)
        mlflow.log_param("drop_cols", drop_cols)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("n_rows", int(len(X)))
        mlflow.log_param("n_features", int(X.shape[1]))
        mlflow.log_param("n_classes", int(len(label_map)))
        mlflow.log_text(json.dumps(label_map, indent=2), "label_map.json")

        # Train
        model = train_model(X_train, y_train, seed=seed)

        # Make sure API can read feature order
        # (Your API uses model.feature_name_ — LightGBM has feature_name_ or feature_name())
        # We'll enforce a consistent attribute for both.
        try:
            model.feature_name_ = list(X.columns)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Evaluate
        ev = evaluate_model(model, X_test, y_test, id_to_name=id_to_name)
        for k, v in ev["metrics"].items():
            mlflow.log_metric(k, v)

        # Artifacts: report + confusion matrix csv
        mlflow.log_text(ev["classification_report_text"], "classification_report.txt")
        cm_df = pd.DataFrame(ev["confusion_matrix"], index=ev["labels_order"], columns=ev["labels_order"])
        cm_path = Path("confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=True)
        mlflow.log_artifact(str(cm_path))

        # Log model to MLflow
        # If lightgbm model exists, mlflow.sklearn still works for LGBMClassifier because it is sklearn API.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_model_path,
            registered_model_name=registered_model_name,  # creates/updates registry too
        )

        # Also save local artifacts for API fallback
        _safe_mkdir(output_model_path)
        joblib.dump(model, output_model_path)

        _safe_mkdir(output_label_map_path)
        output_label_map_path.write_text(yaml.safe_dump(label_map, sort_keys=True))

        mlflow.log_artifact(str(output_model_path))
        mlflow.log_artifact(str(output_label_map_path))

        # Ensure alias points at latest version
        # The log_model call above already registered; we now set alias to newest version.
        # Pick latest by numeric max.
        vers = client.search_model_versions(f"name='{registered_model_name}'")
        latest = str(max(int(v.version) for v in vers))
        client.set_registered_model_alias(registered_model_name, register_alias, latest)

        print("\n✅ Training complete")
        print("Run ID:", run_id)
        print("Metrics:", ev["metrics"])
        print(f"Registered model: {registered_model_name} (alias {register_alias} -> {latest})")
        print("Local model saved to:", output_model_path)
        print("Local label map saved to:", output_label_map_path)


if __name__ == "__main__":
    main()