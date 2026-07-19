# HVAC Fault Detection & Diagnostics (FDD) Platform

A production-ready machine learning system for detecting and diagnosing faults in HVAC equipment using building telemetry time series data. This platform provides automated fault detection, classification, severity estimation, and actionable diagnostics with a complete MLOps pipeline.

## Results

Evaluated on the processed feature set (990 sliding-window samples, 6 balanced fault classes, 165 each), LightGBM classifier, 80/20 stratified split:

| Metric | Score |
|---|---|
| Test Accuracy | 90.9% |
| F1 Macro | 0.909 |
| F1 Weighted | 0.909 |

Per-class breakdown:

| Fault Class | Precision | Recall | F1 |
|---|---|---|---|
| Evaporator Fouling | 1.000 | 1.000 | 1.000 |
| Refrigerant Overcharge | 1.000 | 1.000 | 1.000 |
| Condenser Fouling | 0.943 | 1.000 | 0.971 |
| Liquid Line Restriction | 1.000 | 0.909 | 0.952 |
| Normal | 0.750 | 0.818 | 0.783 |
| Refrigerant Undercharge | 0.774 | 0.727 | 0.750 |

The model separates most fault types cleanly. The remaining confusion is between `normal` and `refrigerant_undercharge`, which makes physical sense — mild undercharge produces telemetry patterns close to normal operation before symptoms become pronounced.

## Architecture

```
Raw HVAC Telemetry (LBNL RTU dataset)
        ↓
Ingestion + Schema Validation
        ↓
Sliding-Window Feature Engineering
(mean, std, min, max, median, quantiles, slope, delta, autocorrelation, oscillation energy)
        ↓
LightGBM Classifier (class-weight balanced)
        ↓
   ┌────────────┬─────────────────┐
   ↓            ↓                 ↓
FastAPI      Diagnostics      MLflow
/predict_window  Fault playbook   Experiment tracking
/batch_predict   + explainer      + model registry
/explain
   ↓
Streamlit Dashboard (fault timeline, ticket detail, HTML/PDF export)
```

## Quick Start

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
Download `Simulated RTU data sets.zip` from the LBNL RTU dataset page and extract it into `data/raw/Simulated RTU/`

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

## Project Structure

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

**Last Updated**: July 2026
**Version**: 1.0.0
