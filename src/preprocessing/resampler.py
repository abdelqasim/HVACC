"""
Data resampling and missing data handling
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def resample_data(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    target_frequency: str = '1min',
    method: str = 'mean',
    point_name_col: str = 'point_name',
    value_col: str = 'value'
) -> pd.DataFrame:
    """
    Resample time series data to consistent frequency
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        target_frequency: Target frequency (e.g., '1min', '5min', '1H')
        method: Resampling method ('mean', 'forward_fill', 'interpolate')
        point_name_col: Name of point name column
        value_col: Name of value column
        
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    
    # Set timestamp as index
    df = df.set_index(timestamp_col)
    
    # Resample by point
    resampled_data = []
    
    for point_name in df[point_name_col].unique():
        point_data = df[df[point_name_col] == point_name].copy()
        
        # Resample based on method
        if method == 'mean':
            resampled = point_data[[value_col]].resample(target_frequency).mean()
        elif method == 'forward_fill':
            resampled = point_data[[value_col]].resample(target_frequency).ffill()
        elif method == 'interpolate':
            resampled = point_data[[value_col]].resample(target_frequency).interpolate()
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        # Add back point name and other columns
        resampled[point_name_col] = point_name
        for col in df.columns:
            if col not in [value_col, point_name_col]:
                resampled[col] = point_data[col].resample(target_frequency).first()
        
        resampled_data.append(resampled)
    
    # Combine all points
    result = pd.concat(resampled_data)
    result = result.reset_index()
    result = result.sort_values([timestamp_col, point_name_col]).reset_index(drop=True)
    
    logger.info(f"Resampled data to {target_frequency}: {len(result)} records")
    
    return result


def handle_missing_data(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    value_col: str = 'value',
    point_name_col: str = 'point_name',
    strategy: str = 'forward_fill',
    forward_fill_limit: int = 5,
    interpolation_method: str = 'linear'
) -> pd.DataFrame:
    """
    Handle missing data in time series
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        value_col: Name of value column
        point_name_col: Name of point name column
        strategy: Strategy for handling missing data ('forward_fill', 'interpolate', 'drop')
        forward_fill_limit: Maximum consecutive forward fills
        interpolation_method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        DataFrame with missing data handled
    """
    df = df.copy()
    
    # Count missing values before
    missing_before = df[value_col].isna().sum()
    
    # Handle by point
    for point_name in df[point_name_col].unique():
        mask = df[point_name_col] == point_name
        
        if strategy == 'forward_fill':
            df.loc[mask, value_col] = df.loc[mask, value_col].fillna(method='ffill', limit=forward_fill_limit)
        elif strategy == 'interpolate':
            df.loc[mask, value_col] = df.loc[mask, value_col].interpolate(method=interpolation_method)
        elif strategy == 'drop':
            df = df[~(mask & df[value_col].isna())]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    missing_after = df[value_col].isna().sum()
    logger.info(f"Handled missing data: {missing_before} -> {missing_after} missing values")
    
    return df.reset_index(drop=True)


def detect_missing_data_patterns(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    value_col: str = 'value',
    point_name_col: str = 'point_name'
) -> Dict[str, Any]:
    """
    Detect patterns in missing data
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        value_col: Name of value column
        point_name_col: Name of point name column
        
    Returns:
        Dictionary with missing data patterns
    """
    report = {
        'total_missing': df[value_col].isna().sum(),
        'missing_percentage': (df[value_col].isna().sum() / len(df) * 100),
        'by_point': {},
    }
    
    for point_name in df[point_name_col].unique():
        mask = df[point_name_col] == point_name
        point_data = df[mask]
        
        missing_count = point_data[value_col].isna().sum()
        report['by_point'][point_name] = {
            'missing_count': missing_count,
            'missing_percentage': (missing_count / len(point_data) * 100),
        }
    
    logger.info(f"Missing data patterns: {report}")
    
    return report
