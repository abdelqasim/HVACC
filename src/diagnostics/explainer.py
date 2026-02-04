"""
Model explainability and interpretation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def get_feature_contributions(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    method: str = 'permutation',
    top_k: int = 10
) -> Tuple[List[str], List[float]]:
    """
    Get feature contributions for a prediction
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        method: Method for computing contributions ('permutation' or 'gain')
        top_k: Number of top features to return
        
    Returns:
        Tuple of (top_feature_names, top_contributions)
    """
    if method == 'gain' and hasattr(model, 'feature_importances_'):
        # Use model's built-in feature importance
        importances = model.feature_importances_
    else:
        # Use permutation importance
        importances = compute_permutation_importance(model, X)
    
    # Get top features
    top_idx = np.argsort(importances)[-top_k:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_contributions = importances[top_idx]
    
    logger.info(f"Top {top_k} features: {top_features}")
    
    return top_features, top_contributions


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray = None,
    n_repeats: int = 10
) -> np.ndarray:
    """
    Compute permutation importance
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Labels (optional)
        n_repeats: Number of repeats
        
    Returns:
        Importance scores
    """
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42
    )
    
    return result.importances_mean


def explain_prediction(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
    method: str = 'permutation'
) -> Dict[str, Any]:
    """
    Explain a single prediction
    
    Args:
        model: Trained model
        X_sample: Single sample feature vector
        feature_names: List of feature names
        top_k: Number of top features to show
        method: Explanation method
        
    Returns:
        Dictionary with explanation
    """
    # Get prediction
    prediction = model.predict(X_sample.reshape(1, -1))[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_sample.reshape(1, -1))[0]
    else:
        probabilities = None
    
    # Get feature contributions
    top_features, top_contributions = get_feature_contributions(
        model, X_sample.reshape(1, -1), feature_names, method, top_k
    )
    
    explanation = {
        'prediction': prediction,
        'probabilities': probabilities.tolist() if probabilities is not None else None,
        'top_features': top_features,
        'feature_contributions': top_contributions.tolist(),
        'feature_values': {
            feature: float(X_sample[i])
            for i, feature in enumerate(feature_names)
            if feature in top_features
        }
    }
    
    logger.info(f"Generated explanation for prediction: {prediction}")
    
    return explanation


def identify_anomalous_signals(
    window_data: pd.DataFrame,
    baseline_stats: Dict[str, Dict[str, float]],
    threshold_std: float = 2.0
) -> List[str]:
    """
    Identify signals that deviate from baseline
    
    Args:
        window_data: DataFrame with window data
        baseline_stats: Dictionary with baseline statistics
        threshold_std: Number of standard deviations for anomaly
        
    Returns:
        List of anomalous signal names
    """
    anomalous = []
    
    for point_name in window_data.columns:
        if point_name not in baseline_stats:
            continue
        
        stats = baseline_stats[point_name]
        values = window_data[point_name].dropna()
        
        if len(values) == 0:
            continue
        
        mean = values.mean()
        baseline_mean = stats.get('mean', mean)
        baseline_std = stats.get('std', 0)
        
        if baseline_std > 0:
            z_score = abs((mean - baseline_mean) / baseline_std)
            if z_score > threshold_std:
                anomalous.append(point_name)
    
    logger.info(f"Identified {len(anomalous)} anomalous signals")
    
    return anomalous


from typing import Any
