"""
MLflow model registry management
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def register_model(
    model: Any,
    preprocessor: Any,
    model_name: str = "hvac_fdd_rtu",
    run_id: Optional[str] = None,
    tags: Dict[str, str] = None,
    metrics: Dict[str, float] = None
) -> str:
    """
    Register model in MLflow Model Registry
    
    Args:
        model: Trained model
        preprocessor: Preprocessing pipeline
        model_name: Name for registered model
        run_id: MLflow run ID
        tags: Tags for the model
        metrics: Metrics to log
        
    Returns:
        Model URI
    """
    if tags is None:
        tags = {}
    if metrics is None:
        metrics = {}
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log tags
    for tag_name, tag_value in tags.items():
        mlflow.set_tag(tag_name, tag_value)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )
    
    # Get the model URI
    model_uri = f"models:/{model_name}/latest"
    
    logger.info(f"Model registered: {model_uri}")
    
    return model_uri


def load_model_from_registry(
    model_name: str = "hvac_fdd_rtu",
    alias: str = "Production",
    tracking_uri: str = "http://localhost:5000"
) -> Any:
    """
    Load model from MLflow Model Registry
    
    Args:
        model_name: Name of registered model
        alias: Model alias (e.g., 'Production', 'Staging')
        tracking_uri: MLflow tracking server URI
        
    Returns:
        Loaded model
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model: {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}@{alias}: {e}")
        raise


def set_model_stage(
    model_name: str,
    version: int,
    stage: str = "Production",
    tracking_uri: str = "http://localhost:5000"
) -> None:
    """
    Set model stage in registry
    
    Args:
        model_name: Name of registered model
        version: Model version
        stage: Target stage (Staging, Production, Archived)
        tracking_uri: MLflow tracking server URI
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    logger.info(f"Model {model_name} v{version} transitioned to {stage}")


def list_model_versions(
    model_name: str,
    tracking_uri: str = "http://localhost:5000"
) -> list:
    """
    List all versions of a registered model
    
    Args:
        model_name: Name of registered model
        tracking_uri: MLflow tracking server URI
        
    Returns:
        List of model versions
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    
    logger.info(f"Found {len(versions)} versions of {model_name}")
    
    return versions


def get_best_model_version(
    model_name: str,
    metric: str = "f1_score",
    tracking_uri: str = "http://localhost:5000"
) -> Optional[Dict[str, Any]]:
    """
    Get best model version by metric
    
    Args:
        model_name: Name of registered model
        metric: Metric to optimize
        tracking_uri: MLflow tracking server URI
        
    Returns:
        Best model version info
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    
    best_version = None
    best_metric = -float('inf')
    
    for version in versions:
        run = client.get_run(version.run_id)
        if metric in run.data.metrics:
            metric_value = run.data.metrics[metric]
            if metric_value > best_metric:
                best_metric = metric_value
                best_version = version
    
    if best_version:
        logger.info(f"Best version of {model_name}: v{best_version.version} ({metric}={best_metric:.4f})")
    
    return best_version
