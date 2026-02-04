"""
Evaluation module for HVAC FDD Platform
"""

from .metrics import compute_metrics, compute_per_class_metrics
from .plots import plot_confusion_matrix, plot_roc_curve, plot_calibration_curve
from .reporting import generate_report

__all__ = [
    "compute_metrics",
    "compute_per_class_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_calibration_curve",
    "generate_report",
]
