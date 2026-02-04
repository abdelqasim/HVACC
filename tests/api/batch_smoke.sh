#!/bin/zsh
set -euo pipefail

base="${BASE_URL:-http://127.0.0.1:8000}"
CSV="${CSV:-tests/fixtures/sample_telemetry.csv}"

echo "Health..."
curl -sS "$base/health" | python -m json.tool >/dev/null
echo "OK health"

if [[ ! -f "$CSV" ]]; then
  echo "Missing $CSV"
  exit 1
fi

echo "Batch predict should return JSON..."
curl -sS -X POST "$base/batch_predict" \
  -F "file=@${CSV};type=text/csv" \
  > /tmp/batch.json

python -m json.tool /tmp/batch.json >/dev/null
echo "OK batch returned valid JSON"

python - <<'PY'
import json
x=json.load(open("/tmp/batch.json"))
print("total_windows:", x["total_windows"])
print("faults_detected:", x["faults_detected"])
print("tickets_returned:", len(x["tickets"]))
PY
