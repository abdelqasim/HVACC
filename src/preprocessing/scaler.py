"""
Feature scaling utilities
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class StandardScaler:
    """Standardization scaler (z-score normalization)"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Fit scaler on data"""
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self.fitted = True
        logger.info(f"StandardScaler fitted: mean={self.mean_}, std={self.std_}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return X * self.std_ + self.mean_


class RobustScaler:
    """Robust scaler using median and interquartile range"""
    
    def __init__(self, quantile_range: tuple = (25.0, 75.0)):
        self.quantile_range = quantile_range
        self.median_ = None
        self.iqr_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """Fit scaler on data"""
        self.median_ = np.nanpercentile(X, 50, axis=0)
        q1 = np.nanpercentile(X, self.quantile_range[0], axis=0)
        q3 = np.nanpercentile(X, self.quantile_range[1], axis=0)
        self.iqr_ = q3 - q1
        # Avoid division by zero
        self.iqr_[self.iqr_ == 0] = 1.0
        self.fitted = True
        logger.info(f"RobustScaler fitted: median={self.median_}, iqr={self.iqr_}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (X - self.median_) / self.iqr_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return X * self.iqr_ + self.median_


class MinMaxScaler:
    """Min-max normalization scaler"""
    
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Fit scaler on data"""
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        # Avoid division by zero
        range_mask = self.max_ == self.min_
        self.max_[range_mask] = self.min_[range_mask] + 1.0
        self.fitted = True
        logger.info(f"MinMaxScaler fitted: min={self.min_}, max={self.max_}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        return X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X_unscaled = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return X_unscaled * (self.max_ - self.min_) + self.min_


def scale_dataframe(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler_type: str = 'standard',
    fit_on_data: Optional[np.ndarray] = None
) -> tuple:
    """
    Scale DataFrame features
    
    Args:
        df: Input DataFrame
        feature_columns: Columns to scale
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        fit_on_data: Optional data to fit scaler on
        
    Returns:
        Tuple of (scaled_df, scaler)
    """
    # Select scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Extract features
    X = df[feature_columns].values
    
    # Fit scaler
    if fit_on_data is not None:
        scaler.fit(fit_on_data)
    else:
        scaler.fit(X)
    
    # Transform
    X_scaled = scaler.transform(X)
    
    # Create scaled dataframe
    df_scaled = df.copy()
    df_scaled[feature_columns] = X_scaled
    
    logger.info(f"Scaled {len(feature_columns)} features using {scaler_type} scaler")
    
    return df_scaled, scaler
