"""
Probability calibration for model predictions
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)


def calibrate_model(
    model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = 'platt'
) -> Any:
    """
    Calibrate model probabilities
    
    Args:
        model: Trained model
        X_calib: Calibration feature matrix
        y_calib: Calibration labels
        method: Calibration method ('platt' or 'isotonic')
        
    Returns:
        Calibrated model
    """
    calibrated_model = CalibratedClassifierCV(
        model,
        method=method,
        cv='prefit'
    )
    
    calibrated_model.fit(X_calib, y_calib)
    logger.info(f"Model calibrated using {method} method")
    
    return calibrated_model


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration metrics
    """
    from sklearn.metrics import brier_score_loss
    
    # Brier score
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Expected calibration error (ECE)
    ece = compute_ece(y_true, y_pred_proba, n_bins)
    
    metrics = {
        'brier_score': brier,
        'ece': ece,
    }
    
    logger.info(f"Calibration metrics: Brier={brier:.4f}, ECE={ece:.4f}")
    
    return metrics


def compute_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ece = 0
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        
        if mask.sum() > 0:
            acc = (y_true[mask] == 1).mean()
            conf = y_pred_proba[mask].mean()
            ece += mask.sum() / len(y_true) * np.abs(acc - conf)
    
    return ece


from typing import Any
