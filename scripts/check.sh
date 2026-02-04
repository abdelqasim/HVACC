#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate
/bin/zsh tests/api/smoke.sh
