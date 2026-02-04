"""
Model training pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def create_model(
    model_type: str = "lightgbm",
    hyperparams: Dict[str, Any] = None,
    random_seed: int = 42
) -> Any:
    """
    Create a model instance
    
    Args:
        model_type: Type of model ('lightgbm', 'xgboost', 'random_forest')
        hyperparams: Model hyperparameters
        random_seed: Random seed for reproducibility
        
    Returns:
        Model instance
    """
    if hyperparams is None:
        hyperparams = {}
    
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(
            random_state=random_seed,
            verbose=-1,
            **hyperparams
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced data
    
    Args:
        y: Label array
        
    Returns:
        Dictionary of class weights
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
    logger.info(f"Computed class weights: {class_weight_dict}")
    
    return class_weight_dict


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "lightgbm",
    hyperparams: Dict[str, Any] = None,
    cv_folds: int = 5,
    random_seed: int = 42,
    use_class_weights: bool = True,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a model with cross-validation
    
    Args:
        X: Feature matrix
        y: Label array
        model_type: Type of model
        hyperparams: Model hyperparameters
        cv_folds: Number of CV folds
        random_seed: Random seed
        use_class_weights: Whether to use class weights
        verbose: Verbose output
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    if hyperparams is None:
        hyperparams = {}
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Compute class weights
    if use_class_weights:
        class_weights = compute_class_weights(y_encoded)
        hyperparams['class_weight'] = class_weights
    
    # Create model
    model = create_model(model_type, hyperparams, random_seed)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(model, X, y_encoded, cv=cv, scoring=scoring, return_train_score=True)
    
    # Train on full data
    model.fit(X, y_encoded)
    
    # Prepare metrics
    metrics = {
        'model': model,
        'label_encoder': le,
        'cv_results': cv_results,
        'mean_cv_score': cv_results['test_accuracy'].mean(),
        'std_cv_score': cv_results['test_accuracy'].std(),
    }
    
    if verbose:
        logger.info(f"Model trained: CV accuracy = {metrics['mean_cv_score']:.4f} ± {metrics['std_cv_score']:.4f}")
    
    return model, metrics


def train_two_stage_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "lightgbm",
    hyperparams: Dict[str, Any] = None,
    cv_folds: int = 5,
    random_seed: int = 42
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Train two-stage model: fault detection + fault classification
    
    Args:
        X: Feature matrix
        y: Label array (with 'normal' and fault types)
        model_type: Type of model
        hyperparams: Model hyperparameters
        cv_folds: Number of CV folds
        random_seed: Random seed
        
    Returns:
        Tuple of (stage1_model, stage2_model, metrics)
    """
    # Stage 1: Binary classification (fault vs normal)
    y_binary = (y != 'normal').astype(int)
    
    stage1_model, stage1_metrics = train_model(
        X, y_binary, model_type, hyperparams, cv_folds, random_seed
    )
    
    logger.info("Stage 1 (fault detection) trained")
    
    # Stage 2: Multi-class classification (fault types, only on faulted samples)
    faulted_mask = y != 'normal'
    X_faulted = X[faulted_mask]
    y_faulted = y[faulted_mask]
    
    if len(np.unique(y_faulted)) > 1:
        stage2_model, stage2_metrics = train_model(
            X_faulted, y_faulted, model_type, hyperparams, cv_folds, random_seed
        )
        logger.info("Stage 2 (fault classification) trained")
    else:
        stage2_model = None
        stage2_metrics = {'warning': 'Only one fault type in training data'}
    
    metrics = {
        'stage1': stage1_metrics,
        'stage2': stage2_metrics,
    }
    
    return stage1_model, stage2_model, metrics


def get_feature_importance(
    model: Any,
    feature_names: list,
    top_k: int = 20
) -> pd.DataFrame:
    """
    Get feature importance from trained model
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_k: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    if top_k:
        importance_df = importance_df.head(top_k)
    
    return importance_df
