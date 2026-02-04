#!/bin/zsh
set -euo pipefail
source .venv/bin/activate
/bin/zsh tests/api/smoke.sh
/bin/zsh tests/api/batch_smoke.sh