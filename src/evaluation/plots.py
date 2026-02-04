"""
Plotting utilities for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None,
    output_path: str = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        output_path: Path to save plot
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    tick_marks = np.arange(len(np.unique(y_true)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    if labels:
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {output_path}")
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot
        figsize: Figure size
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved ROC curve plot to {output_path}")
    
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    output_path: str = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot calibration curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        output_path: Path to save plot
        figsize: Figure size
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    accuracies = []
    confidences = []
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        
        if mask.sum() > 0:
            acc = (y_true[mask] == 1).mean()
            conf = y_pred_proba[mask].mean()
            accuracies.append(acc)
            confidences.append(conf)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(confidences, accuracies, 'o-', label='Model')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved calibration curve plot to {output_path}")
    
    plt.close()


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_k: int = 20,
    output_path: str = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importances: Importance scores
        top_k: Number of top features to show
        output_path: Path to save plot
        figsize: Figure size
    """
    # Sort by importance
    idx = np.argsort(importances)[-top_k:]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_k} Feature Importance')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path}")
    
    plt.close()
