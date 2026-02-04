"""
FastAPI service for HVAC FDD Platform
"""
from fastapi import FastAPI, File, UploadFile, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import mlflow
import joblib
import yaml
import io
import uuid
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime, timezone

from src.common.schemas import (
    FaultTicket,
    PredictionRequest,
    BatchPredictionResponse,
    ExplanationRequest,
    ExplanationResponse,
    HealthResponse,
)
from src.features.window_features import build_rtu_features
from src.common.config import load_config
from src.common.logging import setup_logging, get_logger
from src.features.windowing import create_windows, pivot_window_data
from src.diagnostics.playbook import get_playbook_actions

# Setup logging
setup_logging(log_level="INFO", log_file="api.log")
logger = get_logger(__name__)

app = FastAPI(
    title="HVAC FDD Platform API",
    description="API for HVAC Fault Detection & Diagnostics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model = None
label_encoder: Dict[str, int] | None = None  # name -> id
config = None
mlflow_connected = False


def _load_local_artifacts() -> None:
    """Load local model + label_map.yaml"""
    global model, label_encoder

    local_model_path = Path("models/hvac_fdd_model.joblib")
    local_label_map_path = Path("models/label_map.yaml")

    if not local_model_path.exists():
        raise RuntimeError(f"Local model not found: {local_model_path.resolve()}")
    if not local_label_map_path.exists():
        raise RuntimeError(f"Local label map not found: {local_label_map_path.resolve()}")

    model = joblib.load(local_model_path)
    label_encoder = yaml.safe_load(local_label_map_path.read_text())  # name -> id

    logger.info(f"Loaded local model: {local_model_path.resolve()}")
    logger.info(f"Loaded local label map: {local_label_map_path.resolve()}")


def _invert_label_map(name_to_id: Dict[str, int]) -> Dict[int, str]:
    """name->id to id->name"""
    return {int(v): str(k) for k, v in name_to_id.items()}


def _predict_from_raw_rtu_window(raw_signals: Dict[str, List[float]]) -> Dict[str, Any]:
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model/labels not loaded")
    if config is None:
        raise HTTPException(status_code=500, detail="Config not loaded")

    # 0) validate required signals exist
    required = [
        "RTU_SA_TEMP",
        "RTU_RA_TEMP",
        "RTU_OA_TEMP",
        "RTU_SA_FAN_WATT",
        "RTU_REFG_COND_PRES",
        "RTU_REFG_SUCT_PRES",
    ]
    missing_signals = [s for s in required if s not in raw_signals]
    if missing_signals:
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_signals", "missing": missing_signals},
        )

    # 1) Build engineered features (convert any ValueError to clean 422)
    try:
        feats = build_rtu_features(raw_signals)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_window",
                "message": str(e),
            },
        )

    # 2) Build X in exact feature order expected by model
    cols = list(model.feature_name_)
    missing_feats = [c for c in cols if c not in feats]
    if missing_feats:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_engineered_features",
                "missing": missing_feats,
                "expected": cols,
            },
        )

    X = pd.DataFrame([[float(feats[c]) for c in cols]], columns=cols)

    # 3) Predict
    pred_id = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    conf = float(np.max(proba))

    # 4) id->name mapping
    id_to_name = _invert_label_map(label_encoder)
    pred_name = id_to_name.get(pred_id, str(pred_id))

    # 5) proba mapping must follow model.classes_ order
    class_ids = [int(c) for c in getattr(model, "classes_", range(len(proba)))]
    proba_map = {id_to_name.get(cid, str(cid)): float(p) for cid, p in zip(class_ids, proba)}

    # 6) Threshold logic (single source of truth)
    fault_present_threshold = float(config.get("thresholds.fault_present_threshold", 0.5))
    fault_type_threshold = float(config.get("thresholds.fault_type_threshold", 0.6))

    # Decide whether we declare a fault at all
    fault_present = (pred_name != "normal") and (conf >= fault_present_threshold)

    # If we don't declare a fault, force outputs to normal
    fault_type = pred_name if fault_present else "normal"

    # Only show fault_type_confidence when we actually declare fault and it's strong enough
    fault_type_confidence = conf if (fault_present and conf >= fault_type_threshold) else None

    return {
        "pred_id": pred_id,
        "pred_name": pred_name,
        "fault_type": fault_type,
        "fault_present": fault_present,
        "confidence": conf,
        "fault_type_confidence": fault_type_confidence,
        "proba_map": proba_map,
        "feature_names": cols,
        "engineered_features": {c: float(feats[c]) for c in cols},
    }


@app.post("/debug_features")
async def debug_features(request: PredictionRequest):
    """Returns engineered features and probability breakdown."""
    out = _predict_from_raw_rtu_window(request.data)
    return {
        "fault_type": out["fault_type"],
        "fault_present": out["fault_present"],
        "confidence": out["confidence"],
        "fault_type_confidence": out["fault_type_confidence"],
        "proba": out["proba_map"],
        "feature_names": out["feature_names"],
        "engineered_features": out["engineered_features"],
    }


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.on_event("startup")
async def startup_event():
    global model, label_encoder, config, mlflow_connected

    logger.info("Starting HVAC FDD API...")

    try:
        config = load_config("configs/inference.yaml")

        tracking_uri = config.get("model.mlflow_tracking_uri", "http://localhost:5000")
        model_name = config.get("model.model_name", "hvac_fdd_rtu")
        model_alias = config.get("model.model_alias", "Production")

        mlflow.set_tracking_uri(tracking_uri)

        try:
            model_uri = f"models:/{model_name}@{model_alias}"
            model = mlflow.sklearn.load_model(model_uri)
            mlflow_connected = True
            logger.info(f"Loaded model from MLflow: {model_uri}")

            # still load label map locally
            local_label_map_path = Path("models/label_map.yaml")
            if not local_label_map_path.exists():
                raise RuntimeError(f"Local label map not found: {local_label_map_path.resolve()}")
            label_encoder = yaml.safe_load(local_label_map_path.read_text())
            logger.info(f"Loaded local label map: {local_label_map_path.resolve()}")

        except Exception as e:
            mlflow_connected = False
            logger.warning(f"Could not load model from MLflow: {e}")
            _load_local_artifacts()

        logger.info("API startup complete")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=(model is not None and label_encoder is not None),
        mlflow_connected=bool(mlflow_connected),
    )


@app.post("/predict_window", response_model=FaultTicket)
async def predict_window(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if label_encoder is None:
        raise HTTPException(status_code=500, detail="Label map not loaded")
    if config is None:
        raise HTTPException(status_code=500, detail="Config not loaded")

    try:
        out = _predict_from_raw_rtu_window(request.data)
    except HTTPException:
        # keep clean API errors as-is
        raise
    except ValueError as e:
        # BAD USER INPUT -> 422 (not 500)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_window",
                "message": str(e),
                "hint": "Your window is too short or missing a required RTU signal. Provide >= MIN_SAMPLES per signal.",
            },
        )
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

    fault_type = out["fault_type"]
    fault_present = bool(out["fault_present"])
    conf = float(out["confidence"])
    ft_conf = out.get("fault_type_confidence")

    severity_level = "high" if conf >= 0.8 else "medium" if conf >= 0.6 else "low"

    playbook = get_playbook_actions(fault_type)
    recommended_checks = (playbook.get("actions") or [])[:5]

    evidence = {
        "top_features": out["feature_names"][:5],
        "proba": out["proba_map"],
        "playbook_severity": playbook.get("severity"),
        "playbook_description": playbook.get("description"),
    }

    return FaultTicket(
        ticket_id=str(uuid.uuid4()),
        timestamp=request.timestamp,
        fault_present=fault_present,
        fault_confidence=conf,
        fault_type=fault_type,
        fault_type_confidence=ft_conf,
        severity_score=conf,
        severity_level=severity_level,
        evidence=evidence,
        recommended_checks=recommended_checks,
        model_version=config.get("versioning.min_model_version", "1.0.0"),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if config is None:
        raise HTTPException(status_code=500, detail="Config not loaded")

    required = {
        "RTU_SA_TEMP",
        "RTU_RA_TEMP",
        "RTU_OA_TEMP",
        "RTU_SA_FAN_WATT",
        "RTU_REFG_COND_PRES",
        "RTU_REFG_SUCT_PRES",
    }

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if df.empty:
            raise HTTPException(status_code=422, detail={"error": "empty_csv", "message": "Uploaded CSV has no rows"})

        tickets: List[FaultTicket] = []
        total_windows = 0

        # ---- Case A: WIDE CSV (preferred): timestamp + 6 signal columns ----
        if required.issubset(df.columns):
            # find a timestamp column
            time_col = None
            for c in ("timestamp", "time", "datetime", "date"):
                if c in df.columns:
                    time_col = c
                    break

            if time_col is None:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "missing_timestamp_column",
                        "message": "Wide CSV detected (has RTU_* columns) but no timestamp column found.",
                        "hint": "Add a 'timestamp' column (ISO8601) or rename your time column to 'timestamp'.",
                        "columns": list(df.columns),
                    },
                )

            # window parameters (use your config if present; otherwise safe defaults)
            window_size = int(config.get("windowing.window_size", 20))
            stride = int(config.get("windowing.stride", window_size))

            if len(df) < window_size:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "not_enough_rows",
                        "message": f"CSV has {len(df)} rows but window_size is {window_size}.",
                        "hint": f"Provide at least {window_size} samples per signal.",
                    },
                )

            for start in range(0, len(df) - window_size + 1, stride):
                end = start + window_size
                total_windows += 1

                raw = {sig: df.iloc[start:end][sig].astype(float).tolist() for sig in required}

                try:
                    out = _predict_from_raw_rtu_window(raw)
                except HTTPException as he:
                    # invalid window (e.g., too short) -> skip
                    logger.warning(f"Skipping window {start}:{end} due to HTTPException: {he.detail}")
                    continue

                if not out.get("fault_present", False):
                    continue

                fault_type = out["fault_type"]
                conf = float(out["confidence"])
                ft_conf = out.get("fault_type_confidence")

                severity_level = "high" if conf >= 0.8 else "medium" if conf >= 0.6 else "low"

                playbook = get_playbook_actions(fault_type)
                recommended_checks = (playbook.get("actions") or [])[:5]

                evidence = {
                    "top_features": out["feature_names"][:5],
                    "proba": out["proba_map"],
                    "playbook_severity": playbook.get("severity"),
                    "playbook_description": playbook.get("description"),
                }

                # window timestamp: take first row timestamp in that window
                ts = str(df.iloc[start][time_col])

                tickets.append(
                    FaultTicket(
                        ticket_id=str(uuid.uuid4()),
                        timestamp=ts,
                        fault_present=True,
                        fault_confidence=conf,
                        fault_type=fault_type,
                        fault_type_confidence=ft_conf,
                        severity_score=conf,
                        severity_level=severity_level,
                        evidence=evidence,
                        recommended_checks=recommended_checks,
                        model_version=str(config.get("versioning.min_model_version", "1.0.0")),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )

            return BatchPredictionResponse(
                tickets=tickets,
                total_windows=total_windows,
                faults_detected=len(tickets),
                processing_time_seconds=0.0,
            )

        # ---- Case B: LONG CSV: fallback to your existing pipeline ----
        # Expected columns are not guaranteed; handle errors cleanly.
        windows = create_windows(df)
        total_windows = len(windows)

        cols = list(model.feature_name_)
        for w in windows:
            try:
                raw = pivot_window_data(w["data"])

                missing_signals = sorted([s for s in required if s not in raw])
                if missing_signals:
                    logger.warning(f"Skipping window {w.get('window_id','?')}: missing {missing_signals}")
                    continue

                out = _predict_from_raw_rtu_window(raw)
                if not out.get("fault_present", False):
                    continue

                fault_type = out["fault_type"]
                conf = float(out["confidence"])
                ft_conf = out.get("fault_type_confidence")

                severity_level = "high" if conf >= 0.8 else "medium" if conf >= 0.6 else "low"

                playbook = get_playbook_actions(fault_type)
                recommended_checks = (playbook.get("actions") or [])[:5]

                evidence = {
                    "top_features": cols[:5],
                    "proba": out["proba_map"],
                    "playbook_severity": playbook.get("severity"),
                    "playbook_description": playbook.get("description"),
                }

                tickets.append(
                    FaultTicket(
                        ticket_id=w.get("window_id", str(uuid.uuid4())),
                        timestamp=w.get("window_start"),
                        fault_present=True,
                        fault_confidence=conf,
                        fault_type=fault_type,
                        fault_type_confidence=ft_conf,
                        severity_score=conf,
                        severity_level=severity_level,
                        evidence=evidence,
                        recommended_checks=recommended_checks,
                        model_version=str(config.get("versioning.min_model_version", "1.0.0")),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )

            except HTTPException as he:
                logger.warning(f"Skipping window {w.get('window_id','?')} due to HTTPException: {he.detail}")
                continue
            except Exception as e:
                logger.warning(f"Error processing window {w.get('window_id','?')}: {e}")
                continue

        return BatchPredictionResponse(
            tickets=tickets,
            total_windows=total_windows,
            faults_detected=len(tickets),
            processing_time_seconds=0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch prediction error")
        # IMPORTANT: return JSON error, not plain text "Internal Server Error"
        raise HTTPException(status_code=500, detail={"error": "batch_predict_failed", "message": str(e)})


@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    return ExplanationResponse(
        ticket_id=request.ticket_id,
        top_features=["RTU_REFG_COND_PRES_oscillation", "RTU_REFG_SUCT_PRES_oscillation"],
        feature_contributions=[0.35, 0.28],
        anomalous_signals=["RTU_REFG_COND_PRES", "RTU_REFG_SUCT_PRES"],
        evidence_plots=[],
        playbook_actions=["Contact HVAC technician for inspection"],
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"detail": {"error": "invalid_window", "message": str(exc)}},
    )
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, workers=1)