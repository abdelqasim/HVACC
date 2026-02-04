"""
Sliding window creation for time series data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_windows(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    point_name_col: str = 'point_name',
    value_col: str = 'value',
    window_size_minutes: int = 30,
    stride_minutes: int = 5,
    min_samples_per_window: int = 20
) -> List[Dict]:
    """
    Create sliding windows from time series data
    
    Args:
        df: Input DataFrame with time series data
        timestamp_col: Name of timestamp column
        point_name_col: Name of point name column
        value_col: Name of value column
        window_size_minutes: Window size in minutes
        stride_minutes: Stride in minutes
        min_samples_per_window: Minimum samples required per window
        
    Returns:
        List of window dictionaries
    """
    windows = []
    
    # Convert to timedelta
    window_size = pd.Timedelta(minutes=window_size_minutes)
    stride = pd.Timedelta(minutes=stride_minutes)
    
    # Get unique scenarios/fault_ids
    if 'scenario_id' in df.columns:
        scenarios = df['scenario_id'].unique()
    else:
        scenarios = [None]
    
    for scenario in scenarios:
        # Filter data for this scenario
        if scenario is not None:
            scenario_data = df[df['scenario_id'] == scenario].copy()
        else:
            scenario_data = df.copy()
        
        if len(scenario_data) == 0:
            continue
        
        # Get time range
        min_time = scenario_data[timestamp_col].min()
        max_time = scenario_data[timestamp_col].max()
        
        # Create windows
        window_start = min_time
        window_idx = 0
        
        while window_start + window_size <= max_time:
            window_end = window_start + window_size
            
            # Extract data for this window
            window_mask = (scenario_data[timestamp_col] >= window_start) & \
                         (scenario_data[timestamp_col] < window_end)
            window_data = scenario_data[window_mask]
            
            # Check if we have enough samples
            if len(window_data) >= min_samples_per_window:
                # Get labels (assume same for all samples in window)
                fault_id = window_data['fault_id'].iloc[0] if 'fault_id' in window_data.columns else None
                fault_severity = window_data['fault_severity'].iloc[0] if 'fault_severity' in window_data.columns else None
                
                windows.append({
                    'window_id': f"{scenario}_{window_idx}" if scenario else f"window_{window_idx}",
                    'scenario_id': scenario,
                    'window_start': window_start,
                    'window_end': window_end,
                    'data': window_data,
                    'fault_id': fault_id,
                    'fault_severity': fault_severity,
                    'sample_count': len(window_data),
                })
                
                window_idx += 1
            
            # Move to next window
            window_start += stride
    
    logger.info(f"Created {len(windows)} windows from {len(df)} records")
    
    return windows


def pivot_window_data(
    window_data: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    point_name_col: str = 'point_name',
    value_col: str = 'value'
) -> pd.DataFrame:
    """
    Pivot window data so each point becomes a column
    
    Args:
        window_data: DataFrame with window data
        timestamp_col: Name of timestamp column
        point_name_col: Name of point name column
        value_col: Name of value column
        
    Returns:
        Pivoted DataFrame with points as columns
    """
    pivoted = window_data.pivot_table(
        index=timestamp_col,
        columns=point_name_col,
        values=value_col,
        aggfunc='first'
    )
    
    return pivoted


def get_window_statistics(
    windows: List[Dict]
) -> Dict:
    """
    Get statistics about created windows
    
    Args:
        windows: List of windows
        
    Returns:
        Dictionary with window statistics
    """
    stats = {
        'total_windows': len(windows),
        'samples_per_window': {
            'min': min([w['sample_count'] for w in windows]),
            'max': max([w['sample_count'] for w in windows]),
            'mean': np.mean([w['sample_count'] for w in windows]),
        },
    }
    
    # Count by fault type
    if windows and 'fault_id' in windows[0]:
        fault_counts = {}
        for w in windows:
            fault_id = w['fault_id']
            fault_counts[fault_id] = fault_counts.get(fault_id, 0) + 1
        stats['windows_by_fault'] = fault_counts
    
    logger.info(f"Window statistics: {stats}")
    
    return stats
