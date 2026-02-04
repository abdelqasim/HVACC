.PHONY: help setup install download-data train evaluate test api ui compose-up compose-down clean

help:
	@echo "HVAC FDD Platform - Available Commands"
	@echo "======================================"
	@echo "make setup              - Setup development environment"
	@echo "make install            - Install dependencies"
	@echo "make download-data      - Download LBNL dataset"
	@echo "make train              - Train ML models"
	@echo "make evaluate           - Evaluate trained models"
	@echo "make test               - Run tests"
	@echo "make api                - Start FastAPI service"
	@echo "make ui                 - Start Streamlit UI"
	@echo "make compose-up         - Start all services with Docker Compose"
	@echo "make compose-down       - Stop all services"
	@echo "make clean              - Clean up generated files"

setup:
	@echo "Setting up development environment..."
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r infra/requirements.txt
	@echo "Setup complete! Activate with: source venv/bin/activate"

install:
	@echo "Installing dependencies..."
	pip install -r infra/requirements.txt

download-data:
	@echo "Downloading HVAC FDD dataset..."
	@echo "Please visit: https://faultdetection.lbl.gov/data/"
	@echo "Download RTU data and place in data/raw/"

train:
	@echo "Training ML models..."
	python -m src.modeling.train --config configs/train.yaml

evaluate:
	@echo "Evaluating models..."
	python -m src.evaluation.evaluate --config configs/train.yaml

test:
	@echo "Running tests..."
	pytest tests/ -v

api:
	@echo "Starting FastAPI service..."
	python -m uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	@echo "Starting Streamlit UI..."
	streamlit run services/ui/app.py

compose-up:
	@echo "Starting all services with Docker Compose..."
	docker-compose -f infra/docker-compose.yml up -d
	@echo "Services starting..."
	@echo "MLflow: http://localhost:5000"
	@echo "API: http://localhost:8000/docs"
	@echo "UI: http://localhost:8501"
	@sleep 30
	@echo "Services should be ready now!"

compose-down:
	@echo "Stopping all services..."
	docker-compose -f infra/docker-compose.yml down

compose-logs:
	@echo "Showing Docker Compose logs..."
	docker-compose -f infra/docker-compose.yml logs -f

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist build *.egg-info
	@echo "Cleanup complete!"

.DEFAULT_GOAL := help
