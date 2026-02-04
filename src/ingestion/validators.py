"""
Data validation utilities
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: List[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame schema
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        optional_columns: List of optional columns
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for empty dataframe
    if len(df) == 0:
        errors.append("DataFrame is empty")
    
    return len(errors) == 0, errors


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    timezone: str = 'UTC'
) -> Tuple[bool, List[str]]:
    """
    Validate timestamp column
    
    Args:
        df: DataFrame to validate
        timestamp_col: Name of timestamp column
        timezone: Expected timezone
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if timestamp_col not in df.columns:
        return False, [f"Timestamp column '{timestamp_col}' not found"]
    
    # Check if timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        errors.append(f"Column '{timestamp_col}' is not datetime type")
        return False, errors
    
    # Check for NaT values
    nat_count = df[timestamp_col].isna().sum()
    if nat_count > 0:
        errors.append(f"Found {nat_count} NaT (missing) timestamps")
    
    # Check if sorted
    if not df[timestamp_col].is_monotonic_increasing:
        errors.append("Timestamps are not sorted in ascending order")
    
    # Check timezone
    if df[timestamp_col].dt.tz is None:
        errors.append(f"Timestamps are not timezone-aware (expected {timezone})")
    elif str(df[timestamp_col].dt.tz) != timezone:
        errors.append(f"Timezone mismatch: expected {timezone}, got {df[timestamp_col].dt.tz}")
    
    return len(errors) == 0, errors


def validate_values(
    df: pd.DataFrame,
    value_col: str = 'value',
    allow_nan: bool = False,
    allow_inf: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate value column
    
    Args:
        df: DataFrame to validate
        value_col: Name of value column
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if value_col not in df.columns:
        return False, [f"Value column '{value_col}' not found"]
    
    # Check for NaN
    nan_count = df[value_col].isna().sum()
    if nan_count > 0 and not allow_nan:
        errors.append(f"Found {nan_count} NaN values in '{value_col}'")
    
    # Check for infinite values
    if pd.api.types.is_numeric_dtype(df[value_col]):
        inf_count = pd.isnull(df[value_col]).sum() + (df[value_col] == np.inf).sum() + (df[value_col] == -np.inf).sum()
        if inf_count > 0 and not allow_inf:
            errors.append(f"Found {inf_count} infinite values in '{value_col}'")
    
    return len(errors) == 0, errors


def validate_data_quality(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    value_col: str = 'value',
    max_missing_pct: float = 0.3
) -> Dict[str, Any]:
    """
    Generate data quality report
    
    Args:
        df: DataFrame to validate
        timestamp_col: Name of timestamp column
        value_col: Name of value column
        max_missing_pct: Maximum allowed missing percentage
        
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isna().sum().to_dict(),
        'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    
    # Timestamp quality
    if timestamp_col in df.columns:
        report['timestamp_quality'] = {
            'min': df[timestamp_col].min(),
            'max': df[timestamp_col].max(),
            'range_days': (df[timestamp_col].max() - df[timestamp_col].min()).days,
            'is_sorted': df[timestamp_col].is_monotonic_increasing,
            'has_timezone': df[timestamp_col].dt.tz is not None,
        }
    
    # Value quality
    if value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
        report['value_quality'] = {
            'min': df[value_col].min(),
            'max': df[value_col].max(),
            'mean': df[value_col].mean(),
            'std': df[value_col].std(),
            'nan_count': df[value_col].isna().sum(),
            'inf_count': ((df[value_col] == np.inf) | (df[value_col] == -np.inf)).sum(),
        }
    
    # Check if quality is acceptable
    missing_pct = (df.isna().sum().max() / len(df) * 100)
    report['acceptable_quality'] = missing_pct <= (max_missing_pct * 100)
    
    logger.info(f"Data quality report: {report}")
    
    return report


import numpy as np
