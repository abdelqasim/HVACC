"""
Evaluation metrics computation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for binary classification)
        average: Averaging method for multi-class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Binary classification metrics
    if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
    
    logger.info(f"Computed metrics: {metrics}")
    
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        
    Returns:
        Dictionary of per-class metrics
    """
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    
    logger.info(f"Computed per-class metrics for {len(report) - 3} classes")
    
    return report


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Computed confusion matrix: shape={cm.shape}")
    return cm


def compute_false_alarm_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normal_label: int = 0
) -> float:
    """
    Compute false alarm rate (false positives on normal samples)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normal_label: Label for normal (non-faulted) class
        
    Returns:
        False alarm rate
    """
    normal_mask = y_true == normal_label
    if normal_mask.sum() == 0:
        return 0.0
    
    false_alarms = ((y_pred != normal_label) & normal_mask).sum()
    far = false_alarms / normal_mask.sum()
    
    logger.info(f"False alarm rate: {far:.4f}")
    
    return far
