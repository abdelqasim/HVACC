"""
Feature storage and retrieval
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_features(
    X: np.ndarray,
    feature_names: list,
    y: np.ndarray,
    output_path: str,
    metadata: dict = None
) -> None:
    """
    Save feature matrix to Parquet format
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        y: Labels array
        output_path: Path to save features
        metadata: Optional metadata dictionary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    # Add metadata columns
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) == len(df):
                    df[key] = value
            else:
                df[key] = value
    
    # Save to Parquet
    df.to_parquet(output_path, compression='snappy', index=False)
    logger.info(f"Saved {len(df)} feature vectors to {output_path}")


def load_features(
    input_path: str
) -> tuple:
    """
    Load feature matrix from Parquet format
    
    Args:
        input_path: Path to feature file
        
    Returns:
        Tuple of (X, feature_names, y, metadata)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")
    
    # Load from Parquet
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} feature vectors from {input_path}")
    
    # Extract label
    if 'label' in df.columns:
        y = df['label'].values
        X_df = df.drop('label', axis=1)
    else:
        y = None
        X_df = df
    
    # Extract metadata
    metadata_cols = ['window_id', 'scenario_id', 'timestamp', 'fault_severity']
    metadata = {}
    for col in metadata_cols:
        if col in X_df.columns:
            metadata[col] = X_df[col].values
            X_df = X_df.drop(col, axis=1)
    
    # Extract feature names and matrix
    feature_names = list(X_df.columns)
    X = X_df.values
    
    return X, feature_names, y, metadata


def save_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    output_path: str
) -> None:
    """
    Save feature importance scores
    
    Args:
        feature_names: List of feature names
        importances: Importance scores
        output_path: Path to save importance
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved feature importance to {output_path}")


def load_feature_importance(
    input_path: str
) -> pd.DataFrame:
    """
    Load feature importance scores
    
    Args:
        input_path: Path to importance file
        
    Returns:
        DataFrame with feature importance
    """
    return pd.read_csv(input_path)
