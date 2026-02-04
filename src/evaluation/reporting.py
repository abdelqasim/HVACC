"""
Report generation for model evaluation
"""

import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_report(
    metrics: Dict[str, Any],
    model_info: Dict[str, Any],
    output_dir: str = "reports",
    report_name: str = "evaluation_report"
) -> str:
    """
    Generate evaluation report
    
    Args:
        metrics: Dictionary of computed metrics
        model_info: Dictionary with model information
        output_dir: Directory to save report
        report_name: Name of report file
        
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown report
    report_content = f"""# HVAC FDD Model Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information

- **Model Type**: {model_info.get('model_type', 'Unknown')}
- **Subsystem**: {model_info.get('subsystem', 'Unknown')}
- **Training Samples**: {model_info.get('n_train_samples', 'Unknown')}
- **Test Samples**: {model_info.get('n_test_samples', 'Unknown')}

## Performance Metrics

### Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 'N/A'):.4f} |
| Precision | {metrics.get('precision', 'N/A'):.4f} |
| Recall | {metrics.get('recall', 'N/A'):.4f} |
| F1 Score | {metrics.get('f1', 'N/A'):.4f} |

### Binary Classification Metrics (if applicable)

| Metric | Value |
|--------|-------|
| AUC-ROC | {metrics.get('auc', 'N/A'):.4f} |
| AUC-PRC | {metrics.get('auprc', 'N/A'):.4f} |

### Calibration Metrics

| Metric | Value |
|--------|-------|
| Brier Score | {metrics.get('brier_score', 'N/A'):.4f} |
| ECE | {metrics.get('ece', 'N/A'):.4f} |

## Recommendations

1. Review feature importance to understand model decisions
2. Check calibration curve for probability reliability
3. Analyze confusion matrix for error patterns
4. Consider threshold tuning for production deployment

## Artifacts Generated

- Confusion matrix plot: `confusion_matrix.png`
- ROC curve: `roc_curve.png`
- Calibration curve: `calibration_curve.png`
- Feature importance: `feature_importance.png`

---

*For detailed analysis, refer to the generated plots and metrics.*
"""
    
    # Save markdown report
    report_path = output_dir / f"{report_name}.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Generated report: {report_path}")
    
    # Save metrics as JSON
    metrics_path = output_dir / f"{report_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Saved metrics: {metrics_path}")
    
    return str(report_path)
