"""
End-to-end training script for HVAC FDD Platform (Optimized)
"""

import os
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import joblib

from datetime import datetime

# Correct imports based on existing files
from src.ingestion.loaders import load_csv
from src.preprocessing.resampler import resample_data
from src.features.computation import compute_per_point_features, compute_stability_features
from src.common.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_window_features_fast(window_df):
    """Faster feature computation using only key points"""
    all_features = {}
    # Focus on key RTU points to speed up
    key_points = [
        'RTU_SA_TEMP', 'RTU_RA_TEMP', 'RTU_OA_TEMP', 
        'RTU_SA_FAN_WATT', 'RTU_REFG_COND_PRES', 'RTU_REFG_SUCT_PRES'
    ]
    
    available_points = [p for p in key_points if p in window_df.columns]
    
    for col in available_points:
        series = window_df[col].dropna()
        if len(series) < 2:
            continue
            
        # Basic stats
        all_features[f'{col}_mean'] = series.mean()
        all_features[f'{col}_std'] = series.std()
        all_features[f'{col}_delta'] = series.iloc[-1] - series.iloc[0]
        
        # Stability
        diff = series.diff().dropna()
        all_features[f'{col}_oscillation'] = diff.std()
        
    return all_features

def run_pipeline():
    # 1. Setup paths
    base_dir = Path("/Users/abood/PycharmProjects/HVACC")
    raw_data_dir = base_dir / "data" / "raw" / "Simulated RTU"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Define files to process
    data_files = {
        "normal": "RTU_sim_baseline.csv",
        "condenser_fouling": "RTU_sim_condfouling20.csv",
        "evaporator_fouling": "RTU_sim_evapfouling20.csv",
        "liquid_line_restriction": "RTU_sim_liquidpipe04bar.csv",
        "refrigerant_overcharge": "RTU_sim_overcharge10.csv",
        "refrigerant_undercharge": "RTU_sim_undercharge10.csv"
    }
    
    all_features = []
    all_labels = []
    
    label_map = {name: i for i, name in enumerate(data_files.keys())}
    logger.info(f"Label mapping: {label_map}")
    
    # 3. Process each file
    for label_name, file_name in data_files.items():
        file_path = raw_data_dir / file_name
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Processing {label_name}...")
        
        # Load only first 5000 rows to save time
        df = pd.read_csv(file_path, nrows=5000)
        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to 10 min
        df = df.resample('10min').mean().ffill()
        
        # Create windows (60 min windows, 30 min stride)
        window_size_min = 60
        stride_min = 30
        
        start_time = df.index.min()
        end_time = df.index.max()
        
        current_time = start_time
        while current_time + pd.Timedelta(minutes=window_size_min) <= end_time:
            window_df = df.loc[current_time : current_time + pd.Timedelta(minutes=window_size_min)]
            
            if len(window_df) >= 3:
                features = compute_window_features_fast(window_df)
                if features:
                    all_features.append(list(features.values()))
                    all_labels.append(label_map[label_name])
                
            current_time += pd.Timedelta(minutes=stride_min)
            
    # 4. Convert to DataFrame
    if not all_features:
        logger.error("No features extracted.")
        return

    feature_names = list(features.keys())
    X = pd.DataFrame(all_features, columns=feature_names)
    y = np.array(all_labels)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_df = pd.DataFrame(X, columns=feature_names) if not hasattr(X, "columns") else X.copy()
    y_s = pd.Series(y, name="label")

    processed = pd.concat([X_df, y_s], axis=1)
    processed.to_csv("data/processed/features_labels.csv", index=False)
    print("Saved processed dataset -> data/processed/features_labels.csv")
    
    # 5. Train Model
    logger.info("Starting model training...")
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier
    from sklearn.metrics import classification_report
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        "models/test_split.joblib"
    )
    model = LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=list(data_files.keys()))
    logger.info(f"Training complete. Evaluation Report:\n{report}")
    
    # 7. Save Model

    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "hvac_fdd_model.joblib")
    
    with open(model_dir / "label_map.yaml", 'w') as f:
        yaml.dump(label_map, f)
    logger.info("Model and label map saved.")

if __name__ == "__main__":
    run_pipeline()
