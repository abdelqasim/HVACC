set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

base="http://127.0.0.1:8000"

NORMAL_JSON="$REPO_ROOT/tests/api/predict_window_normal.json"
UNDER_JSON="$REPO_ROOT/tests/api/predict_window_undercharge.json"
RTU_JSON="$REPO_ROOT/tests/api/predict_window_rtu.json"

for f in "$NORMAL_JSON" "$UNDER_JSON" "$RTU_JSON"; do
  [[ -f "$f" ]] || { echo "Missing file: $f"; exit 1; }
done

echo "Health..."
curl -sS "$base/health" | python -m json.tool >/dev/null

echo "Normal should be normal..."
curl -sS -X POST "$base/predict_window" -H "Content-Type: application/json" \
  --data-binary @"$NORMAL_JSON" \
| python -c 'import json,sys; x=json.load(sys.stdin); assert x["fault_present"] is False, x; assert x["fault_type"]=="normal", x; print("OK normal")'

echo "Undercharge response should have required keys..."
curl -sS -X POST "$base/predict_window" -H "Content-Type: application/json" \
  --data-binary @"$UNDER_JSON" \
| python -c 'import json,sys; x=json.load(sys.stdin); req=("fault_present","fault_type","fault_confidence"); miss=[k for k in req if k not in x]; assert not miss, {"missing":miss,"resp":x}; print("OK undercharge response shape")'

echo "RTU request should return valid JSON..."
curl -sS -X POST "$base/predict_window" -H "Content-Type: application/json" \
  --data-binary @"$RTU_JSON" \
| python -m json.tool >/dev/null

echo "ALL SMOKE TESTS PASSED"
