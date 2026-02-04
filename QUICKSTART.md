# Quick Start Guide - HVAC FDD Platform

## 🚀 Get Started in 5 Minutes

### Option 1: Docker Compose (Recommended - Easiest)

**Prerequisites**: Docker and Docker Compose installed

```bash
# Navigate to project directory
cd hvac_fdd_platform

# Start all services
docker-compose -f infra/docker-compose.yml up -d

# Wait for services to initialize (30-60 seconds)
sleep 60

# Access the services:
# - Streamlit UI: http://localhost:8501
# - FastAPI Docs: http://localhost:8000/docs
# - MLflow Tracking: http://localhost:5000
```

**That's it!** The entire platform is now running.

### Option 2: Local Development Setup

**Prerequisites**: Python 3.9+, pip

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r infra/requirements.txt

# 3. Start MLflow (in terminal 1)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# 4. Start API (in terminal 2)
python -m uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Start UI (in terminal 3)
streamlit run services/ui/app.py

# Access:
# - UI: http://localhost:8501
# - API Docs: http://localhost:8000/docs
# - MLflow: http://localhost:5000
```

## 📊 Using the Platform

### 1. Upload Data

1. Open Streamlit UI: http://localhost:8501
2. Choose "Upload CSV" or "Use Sample Data"
3. Configure window size and stride
4. Click "🚀 Run Analysis"

### 2. View Results

- See fault detection results in real-time
- Click on faults to view detailed tickets
- Review recommended technician actions

### 3. Export Reports

- Generate HTML reports
- Download PDF reports
- Share with team members

## 📥 Getting Data

### Option A: Use Sample Data (Recommended for Testing)

The platform includes sample HVAC data. Just select "Use Sample Data" in the UI.

### Option B: Download Real LBNL Dataset

```bash
# Visit: https://faultdetection.lbl.gov/data/
# Download RTU CSV files
# Place in: data/raw/

# Or use the download script:
python scripts/download_data.py --subsystem rtu
```

## 🔧 API Usage Examples

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict_window \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2023-01-15T10:30:00Z",
    "data": {
      "supply_temp": [68.5, 68.3, 68.1],
      "return_temp": [72.1, 72.0, 71.9],
      "supply_fan_speed": [0.85, 0.85, 0.84]
    }
  }'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/batch_predict \
  -F "file=@data.csv"
```

### API Documentation

Visit: http://localhost:8000/docs

## 📋 Configuration

Edit these files to customize:

- `configs/dataset.yaml` - Data paths and subsystem selection
- `configs/features.yaml` - Feature engineering parameters
- `configs/train.yaml` - Model training settings
- `configs/inference.yaml` - Inference thresholds

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_ingestion.py -v
```

## 📚 Project Structure

```
hvac_fdd_platform/
├── README.md                 # Full documentation
├── QUICKSTART.md            # This file
├── configs/                 # Configuration files
├── data/                    # Data directory
├── src/                     # Source code
│   ├── ingestion/          # Data loading
│   ├── preprocessing/      # Data cleaning
│   ├── features/           # Feature engineering
│   ├── modeling/           # ML models
│   ├── evaluation/         # Model evaluation
│   ├── diagnostics/        # Explainability
│   └── common/             # Utilities
├── services/
│   ├── api/                # FastAPI service
│   └── ui/                 # Streamlit dashboard
├── infra/                  # Docker & deployment
├── tests/                  # Test suite
└── Makefile               # Build automation
```

## 🐛 Troubleshooting

### Services won't start

```bash
# Check Docker
docker ps

# View logs
docker-compose -f infra/docker-compose.yml logs

# Rebuild
docker-compose -f infra/docker-compose.yml build --no-cache
```

### API returns 503

```bash
# Wait for MLflow to start
sleep 30

# Check MLflow health
curl http://localhost:5000/
```

### UI can't connect to API

```bash
# Verify API is running
curl http://localhost:8000/health

# Check network connectivity
docker-compose -f infra/docker-compose.yml ps
```

## 🎯 Next Steps

1. **Explore the UI**: Upload sample data and run analysis
2. **Review Results**: Check fault tickets and recommendations
3. **Customize**: Modify configurations for your use case
4. **Deploy**: Use Docker Compose for production deployment

## 📖 Documentation

- Full documentation: See `README.md`
- API documentation: http://localhost:8000/docs
- Dataset info: See `data/download_instructions.md`

## 🆘 Support

- Check `README.md` for detailed documentation
- Review `data/download_instructions.md` for data setup
- See `configs/` for configuration options
- Check logs in `logs/` directory

## ✅ Verification Checklist

- [ ] Docker Compose running: `docker ps`
- [ ] MLflow accessible: http://localhost:5000
- [ ] API running: http://localhost:8000/health
- [ ] UI accessible: http://localhost:8501
- [ ] Sample data loads in UI
- [ ] Analysis runs successfully
- [ ] Fault tickets display correctly

## 🎉 You're Ready!

The HVAC FDD Platform is now running. Start by uploading data and running your first analysis!

---

**Questions?** Refer to README.md for comprehensive documentation.
