# HVAC Fault Detection & Diagnostics (FDD) Platform

A production-ready machine learning system for detecting and diagnosing faults in HVAC equipment using building telemetry time series data. This platform provides automated fault detection, classification, severity estimation, and actionable diagnostics with a complete MLOps pipeline.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r infra/requirements.txt
# Additional dependencies for training
pip install scikit-learn lightgbm joblib pyyaml scipy
```

### 2. Dataset
Download Simulated RTU data sets.zip from the LBNL RTU dataset page and extract it into data/raw/Simulated RTU/
### 3. Train the Model
Run the end-to-end training pipeline to process the raw data and train the diagnostic model:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 run_training.py
```
This will:
- Load raw CSV data from `data/raw/Simulated RTU/`
- Perform window-based feature extraction
- Train a LightGBM classifier
- Save the model to `models/hvac_fdd_model.joblib`

### 4. Run the Platform
Start the API and Dashboard:
```bash
# Terminal 1: API
python3 -m uvicorn services.api.main:app --reload

# Terminal 2: UI
streamlit run services/ui/app.py
```

## 🏗️ Project Structure

```
hvac_fdd_platform/
├── README.md                          # This file
├── run_training.py                    # Main entry point for training
├── configs/                           # Configuration files
├── data/                              # Data directory
│   ├── raw/                           # Raw dataset files (LBNL RTU Data)
│   └── processed/                     # Processed features
├── src/                               # Source code
│   ├── ingestion/                     # Data loading and validation
│   ├── preprocessing/                 # Data cleaning and preparation
│   ├── features/                      # Feature engineering
│   ├── modeling/                      # Model training
│   ├── evaluation/                    # Model evaluation
│   ├── diagnostics/                   # Explainability
│   └── common/                        # Utilities
├── services/                          # Microservices
│   ├── api/                           # FastAPI service
│   └── ui/                            # Streamlit dashboard
├── infra/                             # Infrastructure (Docker, Requirements)
├── tests/                             # Test suite
└── Makefile                           # Build automation
```

## How It Works

### 1. Data Ingestion
The platform ingests HVAC telemetry data from the LBNL Fault Detection & Diagnostics dataset. Data is loaded, validated against a canonical schema, and stored with proper timestamp handling.

### 2. Feature Engineering
Sliding windows are applied over time series data to create features for ML models.
- **Window-based Features**: Per-point mean, std, min, max, median, quantiles, slope, delta, autocorrelation.
- **Stability indicators**: Variance bands, oscillation energy.

### 3. Model Training
The platform trains LightGBM models with proper handling of class imbalance.
- **Training Pipeline**: Fault detection → fault classification.
- **Experiment tracking**: Logged via MLflow (if configured).

### 4. Inference
FastAPI service provides inference endpoints:
- `POST /predict_window`: Single window prediction.
- `POST /batch_predict`: Batch CSV processing.
- `POST /explain`: Prediction explanations.

### 5. Dashboard UI
Streamlit-based dashboard for:
- Uploading CSV data or selecting sample datasets.
- Running inference and viewing fault timeline.
- Clicking faults to see detailed tickets and recommended actions.

## Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose -f infra/docker-compose.yml logs
```

### Model training fails
```bash
# Check data exists
ls -la data/raw/Simulated\ RTU/
```

---

**Last Updated**: February 2026
**Version**: 1.0.0
