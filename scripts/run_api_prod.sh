#!/bin/zsh
set -euo pipefail
exec .venv/bin/python -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000
