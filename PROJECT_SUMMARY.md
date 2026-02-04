# HVAC Fault Detection & Diagnostics Platform - Project Summary

## 📋 Overview

A complete, production-ready machine learning system for detecting and diagnosing faults in HVAC equipment using building telemetry time series data. The platform includes automated fault detection, classification, severity estimation, and actionable diagnostics with a complete MLOps pipeline.

**Status**: ✅ Complete and Ready to Run

## 🎯 What's Included

### 1. **Complete Source Code** (32 Python modules)

#### Data Pipeline (`src/ingestion/`, `src/preprocessing/`)
- CSV data loaders with robust timestamp handling
- Schema validation and data quality checks
- Missing data handling (forward fill, interpolation)
- Feature scaling (Standard, Robust, MinMax)

#### Feature Engineering (`src/features/`)
- Sliding window creation (configurable size and stride)
- Per-point features: mean, std, min, max, median, quantiles, slope, delta, autocorrelation
- Cross-point features: correlations, differentials, ratios
- Stability indicators: variance bands, oscillation energy
- Parquet storage with metadata

#### Machine Learning (`src/modeling/`)
- LightGBM model training with class weight handling
- Two-stage approach: fault detection → fault classification
- Probability calibration (Platt, Isotonic)
- MLflow experiment tracking and model registry
- Cross-validation with stratification

#### Evaluation (`src/evaluation/`)
- Comprehensive metrics: accuracy, precision, recall, F1, AUC-ROC, AUC-PRC
- Confusion matrix and per-class metrics
- Calibration plots and ECE/Brier score
- Feature importance visualization
- Markdown and JSON report generation

#### Diagnostics (`src/diagnostics/`)
- Feature contribution analysis
- Anomalous signal identification
- Fault playbook with 11 fault types
- Recommended technician actions

#### Utilities (`src/common/`)
- YAML configuration parsing
- Logging setup
- Pydantic schemas for API validation

### 2. **API Service** (`services/api/`)

FastAPI-based inference service with:
- **`POST /predict_window`**: Single window prediction
- **`POST /batch_predict`**: Batch CSV processing
- **`POST /explain`**: Prediction explanations
- **`GET /health`**: Health check
- Pydantic input validation
- MLflow model registry integration
- CORS support

### 3. **Dashboard UI** (`services/ui/`)

Streamlit-based interactive dashboard with:
- Data upload (CSV) or sample data selection
- Real-time fault detection analysis
- Fault timeline visualization
- Detailed fault ticket inspection
- HTML/PDF report export
- System configuration options

### 4. **MLOps Infrastructure** (`infra/`)

Complete Docker-based deployment:
- **docker-compose.yml**: Orchestrates all services
- **Dockerfile.api**: FastAPI container
- **Dockerfile.ui**: Streamlit container
- **Dockerfile.mlflow**: MLflow tracking server
- PostgreSQL database for MLflow backend
- Persistent volumes for artifacts and data

### 5. **Configuration Files** (`configs/`)

- **dataset.yaml**: Data paths, subsystem selection, label mapping
- **features.yaml**: Window size, stride, feature definitions
- **train.yaml**: Model hyperparameters, training strategy, CV settings
- **inference.yaml**: Thresholds, model loading, calibration settings

### 6. **Documentation**

- **README.md** (500+ lines): Comprehensive guide with setup, usage, API examples
- **QUICKSTART.md**: 5-minute getting started guide
- **data/download_instructions.md**: Dataset acquisition guide
- **PROJECT_SUMMARY.md**: This file

### 7. **Build Automation**

- **Makefile**: Commands for setup, training, evaluation, testing, deployment
- **requirements.txt**: All Python dependencies
- **requirements.api.txt**: API-specific dependencies
- **requirements.ui.txt**: UI-specific dependencies

## 🏗️ Project Structure

```
hvac_fdd_platform/
├── README.md                          # Full documentation
├── QUICKSTART.md                      # 5-minute setup guide
├── PROJECT_SUMMARY.md                 # This file
├── Makefile                           # Build automation
│
├── configs/                           # Configuration files
│   ├── dataset.yaml                   # Data configuration
│   ├── features.yaml                  # Feature engineering config
│   ├── train.yaml                     # Training configuration
│   └── inference.yaml                 # Inference configuration
│
├── data/                              # Data directory
│   ├── download_instructions.md       # How to get data
│   ├── raw/                           # Raw data (gitignored)
│   └── processed/                     # Processed features (gitignored)
│
├── src/                               # Source code
│   ├── ingestion/                     # Data loading & validation
│   │   ├── loaders.py                 # CSV loaders
│   │   └── validators.py              # Schema validation
│   ├── preprocessing/                 # Data cleaning
│   │   ├── resampler.py               # Resampling & missing data
│   │   └── scaler.py                  # Feature scaling
│   ├── features/                      # Feature engineering
│   │   ├── windowing.py               # Sliding windows
│   │   ├── computation.py             # Feature computation
│   │   └── store.py                   # Parquet storage
│   ├── modeling/                      # ML training
│   │   ├── train.py                   # Training pipeline
│   │   ├── calibration.py             # Probability calibration
│   │   └── registry.py                # MLflow registry
│   ├── evaluation/                    # Model evaluation
│   │   ├── metrics.py                 # Metric computation
│   │   ├── plots.py                   # Visualization
│   │   └── reporting.py               # Report generation
│   ├── diagnostics/                   # Explainability
│   │   ├── explainer.py               # Feature importance
│   │   └── playbook.py                # Fault playbook
│   └── common/                        # Utilities
│       ├── config.py                  # Config parsing
│       ├── logging.py                 # Logging setup
│       └── schemas.py                 # Pydantic schemas
│
├── services/                          # Microservices
│   ├── api/                           # FastAPI service
│   │   └── main.py                    # API endpoints
│   └── ui/                            # Streamlit dashboard
│       └── app.py                     # UI application
│
├── infra/                             # Infrastructure
│   ├── docker-compose.yml             # Docker orchestration
│   ├── Dockerfile.api                 # API container
│   ├── Dockerfile.ui                  # UI container
│   ├── Dockerfile.mlflow              # MLflow container
│   ├── requirements.txt               # All dependencies
│   ├── requirements.api.txt           # API dependencies
│   ├── requirements.ui.txt            # UI dependencies
│   └── scripts/                       # Helper scripts
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_ingestion.py              # Ingestion tests
│   ├── test_features.py               # Feature tests
│   ├── test_modeling.py               # Model tests
│   └── test_integration.py            # Integration tests
│
├── mlruns/                            # MLflow artifacts (gitignored)
└── .gitignore                         # Git ignore rules
```

## 🚀 How to Run

### Option 1: Docker Compose (Recommended - One Command)

```bash
cd hvac_fdd_platform
docker-compose -f infra/docker-compose.yml up -d
sleep 60
# Access: UI (8501), API (8000), MLflow (5000)
```

### Option 2: Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r infra/requirements.txt

# Terminal 1: MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db

# Terminal 2: API
python -m uvicorn services.api.main:app --reload

# Terminal 3: UI
streamlit run services/ui/app.py
```

### Option 3: Using Makefile

```bash
make setup          # Setup environment
make install        # Install dependencies
make compose-up     # Start all services
```

## 📊 Key Features

### Data Processing
- ✅ Robust timestamp handling and UTC normalization
- ✅ Missing data detection and handling
- ✅ Configurable resampling
- ✅ Multiple scaling methods

### Feature Engineering
- ✅ Sliding window creation with configurable size/stride
- ✅ 15+ per-point statistical features
- ✅ Cross-point correlation features
- ✅ Stability indicators (oscillation, variance bands)
- ✅ Parquet storage with metadata

### Machine Learning
- ✅ LightGBM with class weight balancing
- ✅ Two-stage architecture (detection → classification)
- ✅ Probability calibration
- ✅ Cross-validation with stratification
- ✅ MLflow experiment tracking
- ✅ Model registry versioning

### Inference
- ✅ Single window predictions
- ✅ Batch CSV processing
- ✅ Explanation generation
- ✅ Fault playbook recommendations
- ✅ RESTful API with Swagger docs

### Diagnostics
- ✅ Feature importance analysis
- ✅ Anomalous signal detection
- ✅ 11 fault types with playbook
- ✅ Recommended technician actions
- ✅ Severity scoring

### Deployment
- ✅ Docker containerization
- ✅ Docker Compose orchestration
- ✅ PostgreSQL backend for MLflow
- ✅ Persistent volumes
- ✅ Health checks
- ✅ CORS support

## 📈 Fault Types Supported

1. **Supply Fan Failure** - Fan not operating or reduced capacity
2. **Return Fan Failure** - Return fan issues
3. **Compressor Failure** - Cooling compressor issues
4. **Heating Valve Stuck** - Heating valve malfunction
5. **Cooling Valve Stuck** - Cooling valve malfunction
6. **Damper Stuck** - Air damper stuck position
7. **Sensor Bias** - Sensor reading offset
8. **Damper Oscillation** - Rapid damper oscillation
9. **Duct Imbalance** - Unbalanced air distribution
10. **Low Efficiency** - System operating below efficiency
11. **Normal Operation** - No faults detected

Each fault type includes recommended technician checks and actions.

## 🔌 API Endpoints

### Health Check
```
GET /health
```

### Single Prediction
```
POST /predict_window
Content-Type: application/json

{
  "timestamp": "2023-01-15T10:30:00Z",
  "data": {
    "supply_temp": [68.5, 68.3, 68.1],
    "return_temp": [72.1, 72.0, 71.9],
    "supply_fan_speed": [0.85, 0.85, 0.84]
  }
}
```

### Batch Prediction
```
POST /batch_predict
Content-Type: multipart/form-data

file: <CSV file>
```

### Explanation
```
POST /explain
Content-Type: application/json

{
  "ticket_id": "ticket_20230115_001",
  "include_plots": true,
  "include_shap": true,
  "top_k_features": 10
}
```

## 📦 Dependencies

### Core
- pandas, numpy, scikit-learn, scipy
- lightgbm, xgboost
- mlflow

### API
- fastapi, uvicorn, pydantic

### UI
- streamlit, plotly, matplotlib

### Infrastructure
- Docker, Docker Compose, PostgreSQL

All dependencies are specified in `infra/requirements.txt`

## ✅ Quality Assurance

### Testing
- Unit tests for ingestion, features, modeling
- Integration tests for end-to-end pipeline
- Data validation and schema checking
- Model evaluation with multiple metrics

### Logging
- Comprehensive logging throughout
- Log files in `logs/` directory
- Configurable log levels

### Documentation
- 500+ line README with examples
- Inline code documentation
- Configuration file comments
- API Swagger documentation

## 🎓 Learning Resources

The codebase demonstrates:
- **MLOps Best Practices**: MLflow tracking, model registry, reproducibility
- **Feature Engineering**: Time series feature extraction, windowing
- **ML Pipeline**: Data loading, preprocessing, training, evaluation
- **API Design**: FastAPI with Pydantic validation
- **Dashboard Development**: Streamlit interactive UI
- **Docker Deployment**: Multi-container orchestration
- **Configuration Management**: YAML-based configs

## 🔐 Security Considerations

- Input validation with Pydantic
- CORS configuration
- Environment variable support for secrets
- No hardcoded credentials
- Logging without sensitive data

## 🚀 Deployment Checklist

- [x] Source code complete (32 Python modules)
- [x] Configuration files ready
- [x] Docker setup complete
- [x] API service implemented
- [x] UI dashboard created
- [x] MLflow integration done
- [x] Documentation comprehensive
- [x] Makefile automation ready
- [x] .gitignore configured
- [x] Requirements files specified

## 📝 Next Steps After Running

1. **Explore the UI**: http://localhost:8501
2. **Upload Sample Data**: Use built-in sample or upload CSV
3. **Run Analysis**: Click "Run Analysis" button
4. **Review Results**: Check fault tickets and recommendations
5. **Export Reports**: Generate HTML/PDF reports
6. **Customize**: Modify configs for your use case
7. **Deploy**: Use Docker Compose for production

## 🆘 Support & Troubleshooting

### Services Won't Start
```bash
docker-compose -f infra/docker-compose.yml logs
docker-compose -f infra/docker-compose.yml down
docker-compose -f infra/docker-compose.yml up -d
```

### API Connection Issues
```bash
curl http://localhost:8000/health
curl http://localhost:5000/
```

### Data Issues
See `data/download_instructions.md` for dataset acquisition

## 📞 Contact & Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review QUICKSTART.md for setup help
3. Check logs in `logs/` directory
4. Review configuration files in `configs/`

## 📄 License

MIT License - See LICENSE file (if included)

## 🎉 Summary

This is a **complete, production-ready HVAC Fault Detection & Diagnostics platform** with:
- ✅ Full source code (32 modules)
- ✅ API service with 4 endpoints
- ✅ Interactive dashboard UI
- ✅ Docker deployment
- ✅ MLflow integration
- ✅ Comprehensive documentation
- ✅ Build automation
- ✅ Ready to run immediately

**Start with**: `docker-compose -f infra/docker-compose.yml up -d`

---

**Version**: 1.0.0  
**Created**: February 2026  
**Status**: ✅ Production Ready
