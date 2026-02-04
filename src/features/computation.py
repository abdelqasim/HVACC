"""
Feature computation for HVAC time series
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def compute_per_point_features(
    series: pd.Series,
    feature_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute per-point features from a time series
    
    Args:
        series: Pandas Series with values
        feature_names: List of features to compute (None = all)
        
    Returns:
        Dictionary of feature values
    """
    if feature_names is None:
        feature_names = ['mean', 'std', 'min', 'max', 'median', 'quantile_25', 
                        'quantile_75', 'range', 'slope', 'delta', 'autocorr_lag1']
    
    features = {}
    
    # Remove NaN values
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {f: np.nan for f in feature_names}
    
    # Basic statistics
    if 'mean' in feature_names:
        features['mean'] = clean_series.mean()
    if 'std' in feature_names:
        features['std'] = clean_series.std()
    if 'min' in feature_names:
        features['min'] = clean_series.min()
    if 'max' in feature_names:
        features['max'] = clean_series.max()
    if 'median' in feature_names:
        features['median'] = clean_series.median()
    if 'quantile_25' in feature_names:
        features['quantile_25'] = clean_series.quantile(0.25)
    if 'quantile_75' in feature_names:
        features['quantile_75'] = clean_series.quantile(0.75)
    if 'range' in feature_names:
        features['range'] = clean_series.max() - clean_series.min()
    
    # Trend features
    if 'slope' in feature_names:
        x = np.arange(len(clean_series))
        slope, _ = np.polyfit(x, clean_series.values, 1)
        features['slope'] = slope
    
    if 'delta' in feature_names:
        features['delta'] = clean_series.iloc[-1] - clean_series.iloc[0]
    
    # Autocorrelation
    if 'autocorr_lag1' in feature_names:
        if len(clean_series) > 1:
            features['autocorr_lag1'] = clean_series.autocorr(lag=1)
        else:
            features['autocorr_lag1'] = np.nan
    
    if 'autocorr_lag5' in feature_names:
        if len(clean_series) > 5:
            features['autocorr_lag5'] = clean_series.autocorr(lag=5)
        else:
            features['autocorr_lag5'] = np.nan
    
    # Distribution features
    if 'variance' in feature_names:
        features['variance'] = clean_series.var()
    
    if 'skewness' in feature_names:
        features['skewness'] = stats.skew(clean_series)
    
    if 'kurtosis' in feature_names:
        features['kurtosis'] = stats.kurtosis(clean_series)
    
    return features


def compute_cross_point_features(
    data: pd.DataFrame,
    point_pairs: List[tuple],
    feature_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute cross-point features between related signals
    
    Args:
        data: DataFrame with point data (columns are point names)
        point_pairs: List of (point1, point2) tuples
        feature_names: List of features to compute
        
    Returns:
        Dictionary of feature values
    """
    if feature_names is None:
        feature_names = ['correlation', 'differential']
    
    features = {}
    
    for point1, point2 in point_pairs:
        if point1 not in data.columns or point2 not in data.columns:
            continue
        
        series1 = data[point1].dropna()
        series2 = data[point2].dropna()
        
        # Align series
        common_idx = series1.index.intersection(series2.index)
        if len(common_idx) < 2:
            continue
        
        s1 = series1[common_idx]
        s2 = series2[common_idx]
        
        # Correlation
        if 'correlation' in feature_names:
            corr = s1.corr(s2)
            features[f'corr_{point1}_{point2}'] = corr
        
        # Differential
        if 'differential' in feature_names:
            diff = (s2 - s1).mean()
            features[f'diff_{point1}_{point2}'] = diff
    
    return features


def compute_stability_features(
    series: pd.Series,
    feature_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute stability indicators
    
    Args:
        series: Pandas Series with values
        feature_names: List of features to compute
        
    Returns:
        Dictionary of feature values
    """
    if feature_names is None:
        feature_names = ['variance_band', 'oscillation_energy', 'zero_crossing_rate']
    
    features = {}
    clean_series = series.dropna()
    
    if len(clean_series) < 2:
        return {f: np.nan for f in feature_names}
    
    # Variance band (variance of rolling variance)
    if 'variance_band' in feature_names:
        rolling_var = clean_series.rolling(window=5).var()
        features['variance_band'] = rolling_var.std()
    
    # Oscillation energy (high-frequency components)
    if 'oscillation_energy' in feature_names:
        # Simple proxy: variance of differences
        diff = clean_series.diff().dropna()
        features['oscillation_energy'] = diff.std()
    
    # Zero crossing rate
    if 'zero_crossing_rate' in feature_names:
        detrended = clean_series - clean_series.mean()
        zero_crossings = np.sum(np.diff(np.sign(detrended)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(detrended)
    
    # Mean absolute change
    if 'mean_absolute_change' in feature_names:
        diff = clean_series.diff().dropna()
        features['mean_absolute_change'] = diff.abs().mean()
    
    return features


def compute_window_features(
    window_data: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    point_name_col: str = 'point_name',
    value_col: str = 'value',
    feature_config: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Compute all features for a window
    
    Args:
        window_data: DataFrame with window data
        timestamp_col: Name of timestamp column
        point_name_col: Name of point name column
        value_col: Name of value column
        feature_config: Feature configuration dictionary
        
    Returns:
        Dictionary of all computed features
    """
    all_features = {}
    
    # Pivot data so each point is a column
    pivoted = window_data.pivot_table(
        index=timestamp_col,
        columns=point_name_col,
        values=value_col,
        aggfunc='first'
    )
    
    # Per-point features
    for point_name in pivoted.columns:
        point_features = compute_per_point_features(pivoted[point_name])
        for feat_name, feat_value in point_features.items():
            all_features[f'{point_name}_{feat_name}'] = feat_value
    
    # Stability features
    for point_name in pivoted.columns:
        stability_features = compute_stability_features(pivoted[point_name])
        for feat_name, feat_value in stability_features.items():
            all_features[f'{point_name}_{feat_name}'] = feat_value
    
    # Cross-point features (if configured)
    if feature_config and 'cross_point_pairs' in feature_config:
        cross_features = compute_cross_point_features(
            pivoted,
            feature_config['cross_point_pairs']
        )
        all_features.update(cross_features)
    
    return all_features


def create_feature_matrix(
    windows: List[Dict],
    feature_config: Dict[str, Any] = None
) -> tuple:
    """
    Create feature matrix from windows
    
    Args:
        windows: List of window dictionaries
        feature_config: Feature configuration
        
    Returns:
        Tuple of (feature_matrix, feature_names, labels)
    """
    X = []
    y = []
    feature_names = None
    
    for window in windows:
        window_features = compute_window_features(
            window['data'],
            feature_config=feature_config
        )
        
        # Initialize feature names from first window
        if feature_names is None:
            feature_names = list(window_features.keys())
        
        # Ensure consistent feature order
        feature_vector = [window_features.get(name, np.nan) for name in feature_names]
        X.append(feature_vector)
        
        # Get label
        fault_id = window.get('fault_id', 'normal')
        y.append(fault_id)
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created feature matrix: {X.shape}")
    
    return X, feature_names, y
