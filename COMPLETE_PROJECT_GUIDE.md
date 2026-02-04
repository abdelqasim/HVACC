# 🗺️ HVAC FDD Platform: Complete File Map & Usage Guide

This document provides a full inventory of every file created for the HVAC Fault Detection & Diagnostics (FDD) project. It explains which files you should interact with and which ones are internal logic.

---

## 🚀 1. The "Must-Use" Files (Start Here)
These are the primary entry points for training, running, and configuring your project.

| File | Purpose | Action |
| :--- | :--- | :--- |
| **`run_training.py`** | **The Training Engine.** Processes raw data and trains the ML model. | **RUN THIS** to train your model. |
| **`services/ui/app.py`** | **The Dashboard.** The visual interface for users. | **RUN THIS** to see the results. |
| **`services/api/main.py`** | **The API.** The backend that serves predictions. | **RUN THIS** to power the UI. |
| **`configs/*.yaml`** | **Settings.** Controls window sizes, thresholds, and paths. | **EDIT THESE** to customize behavior. |
| **`QUICKSTART.md`** | **Fast Setup.** 5-minute guide to get everything running. | **READ THIS** first. |

---

## 📂 2. Complete File Inventory

### 🛠️ Core Execution & Automation
- `run_training.py`: End-to-end training pipeline (Data -> Features -> Model).
- `Makefile`: Shortcuts for common commands (`make setup`, `make train`, etc.).
- `requirements.txt`: List of all Python libraries needed.

### ⚙️ Configuration (`configs/`)
- `dataset.yaml`: Defines where data is and how labels are mapped.
- `features.yaml`: Defines what mathematical features to extract from sensors.
- `train.yaml`: Hyperparameters for the LightGBM model.
- `inference.yaml`: Settings for the live API (thresholds, etc.).

### 🧠 Source Code (`src/`) - *Internal Logic*
*You generally don't need to edit these unless you want to change how the math works.*
- `src/ingestion/`: Loaders for CSV files and data validation.
- `src/preprocessing/`: Resampling and scaling sensor data.
- `src/features/`: The "brain" that turns raw sensor data into ML features.
- `src/modeling/`: Training logic, model registry, and probability calibration.
- `src/evaluation/`: Generates metrics, confusion matrices, and plots.
- `src/diagnostics/`: Explainability (SHAP) and the **Fault Playbook**.
- `src/common/`: Shared utilities like logging and data schemas.

### 🌐 Services (`services/`)
- `services/api/`: FastAPI implementation for production inference.
- `services/ui/`: Streamlit code for the interactive dashboard.

### 🐳 Infrastructure (`infra/`)
- `docker-compose.yml`: Orchestrates the whole stack in containers.
- `Dockerfile.*`: Build instructions for API, UI, and MLflow.

---

## 🛑 3. What to Use vs. What to Ignore

### ✅ USE THESE (Active Management)
1.  **`run_training.py`**: Use this every time you get new data and want to update the model.
2.  **`configs/`**: Use these to tune the sensitivity of the fault detection.
3.  **`data/raw/`**: Put your new CSV files here.
4.  **`models/`**: This is where your "brain" (the model) lives. Keep it safe.

### ⚠️ IGNORE THESE (Internal Only)
1.  **`src/` folder**: These are the "gears" inside the machine. Don't change them unless you are an experienced developer.
2.  **`__init__.py` files**: These are just for Python to recognize folders as packages.
3.  **`mlruns/`**: This is an automatic folder created by MLflow to track experiments.
4.  **`tests/`**: These are for automated quality checks.

---

## 📈 Summary of Workflow
1.  **Prepare**: Put data in `data/raw/`.
2.  **Train**: Run `python run_training.py`.
3.  **Serve**: Run `python -m uvicorn services.api.main:app`.
4.  **Visualize**: Run `streamlit run services/ui/app.py`.

---
**Everything is now ready and organized for long-term management!**
