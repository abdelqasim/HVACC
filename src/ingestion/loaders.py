"""
Data loaders for HVAC telemetry data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_csv(
    file_path: str,
    timestamp_col: str = "timestamp",
    timestamp_format: Optional[str] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load CSV file with proper timestamp handling
    
    Args:
        file_path: Path to CSV file
        timestamp_col: Name of timestamp column
        timestamp_format: Format string for timestamp parsing
        parse_dates: Whether to parse dates
        
    Returns:
        DataFrame with loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV: {file_path} ({len(df)} rows)")
        
        # Parse timestamps
        if parse_dates and timestamp_col in df.columns:
            if timestamp_format:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=timestamp_format)
            else:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Ensure UTC timezone
            if df[timestamp_col].dt.tz is None:
                df[timestamp_col] = df[timestamp_col].dt.tz_localize('UTC')
            else:
                df[timestamp_col] = df[timestamp_col].dt.tz_convert('UTC')
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise


def load_hvac_data(
    data_path: str,
    subsystem: str = "rtu",
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    point_name_col: str = "point_name"
) -> pd.DataFrame:
    """
    Load HVAC data with standardization
    
    Args:
        data_path: Path to data directory or file
        subsystem: HVAC subsystem type (rtu, ahu, vav, etc.)
        timestamp_col: Name of timestamp column
        value_col: Name of value column
        point_name_col: Name of point name column
        
    Returns:
        Standardized DataFrame
    """
    data_path = Path(data_path)
    
    # Load data
    if data_path.is_file():
        df = load_csv(str(data_path), timestamp_col=timestamp_col)
    elif data_path.is_dir():
        # Load all CSV files in directory
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        dfs = [load_csv(str(f), timestamp_col=timestamp_col) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(csv_files)} CSV files from {data_path}")
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")
    
    # Standardize schema
    required_cols = {timestamp_col, value_col, point_name_col}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create standardized dataframe
    standardized = pd.DataFrame({
        'timestamp': df[timestamp_col],
        'point_name': df[point_name_col],
        'value': df[value_col],
    })
    
    # Add optional columns if present
    optional_cols = ['unit', 'equipment_id', 'system_id', 'scenario_id', 'fault_id', 'fault_severity']
    for col in optional_cols:
        if col in df.columns:
            standardized[col] = df[col]
    
    # Sort by timestamp
    standardized = standardized.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Standardized data: {len(standardized)} records, {standardized['point_name'].nunique()} unique points")
    
    return standardized


def load_sample_data() -> pd.DataFrame:
    """
    Load sample HVAC data for testing
    
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    # Generate synthetic sample data
    timestamps = pd.date_range('2023-01-01', periods=1000, freq='1min', tz='UTC')
    
    data = []
    for ts in timestamps:
        # Normal operation baseline
        supply_temp = 68 + np.random.normal(0, 0.5)
        return_temp = 72 + np.random.normal(0, 0.5)
        supply_fan_speed = 0.85 + np.random.normal(0, 0.02)
        
        # Add some faults in second half
        if ts > timestamps[500]:
            supply_fan_speed *= 0.5  # Simulate fan failure
        
        data.append({
            'timestamp': ts,
            'point_name': 'supply_temp',
            'value': supply_temp,
            'unit': '°C',
            'equipment_id': 'rtu_001',
            'system_id': 'hvac_system_1',
            'scenario_id': 'scenario_001',
            'fault_id': 'normal' if ts <= timestamps[500] else 'supply_fan_failure',
            'fault_severity': 0.0 if ts <= timestamps[500] else 0.8
        })
        
        data.append({
            'timestamp': ts,
            'point_name': 'return_temp',
            'value': return_temp,
            'unit': '°C',
            'equipment_id': 'rtu_001',
            'system_id': 'hvac_system_1',
            'scenario_id': 'scenario_001',
            'fault_id': 'normal' if ts <= timestamps[500] else 'supply_fan_failure',
            'fault_severity': 0.0 if ts <= timestamps[500] else 0.8
        })
        
        data.append({
            'timestamp': ts,
            'point_name': 'supply_fan_speed',
            'value': supply_fan_speed,
            'unit': 'fraction',
            'equipment_id': 'rtu_001',
            'system_id': 'hvac_system_1',
            'scenario_id': 'scenario_001',
            'fault_id': 'normal' if ts <= timestamps[500] else 'supply_fan_failure',
            'fault_severity': 0.0 if ts <= timestamps[500] else 0.8
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated sample data: {len(df)} records")
    
    return df
