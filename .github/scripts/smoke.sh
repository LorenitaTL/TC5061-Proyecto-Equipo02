#!/usr/bin/env bash
# Smoke test of the system started with `docker compose up -d --wait` (jobs compose and
# deploy of the ci-cd workflow):
#   1. GET /health answers 200 with model_loaded: true.
#   2. POST /predict answers 200 with a valid prediction.
# The payload is generated inside the api container (it already has mi_bici installed),
# so the runner only needs curl and python3. Its stderr is not hidden: if the container or
# the import fails, the cause appears in the job log.
#
#   API_URL=http://127.0.0.1:8000 bash .github/scripts/smoke.sh
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"

echo "GET $API_URL/health"
body="$(curl --fail --silent --show-error "$API_URL/health")"
echo "$body"
echo "$body" | python3 -c "
import json, sys
r = json.load(sys.stdin)
assert r.get('model_loaded') is True, 'model_loaded is not true'
"

echo "POST $API_URL/predict"
payload="$(docker compose exec -T api python -c "
import json, random
from mi_bici.operations.load_payloads import make_payload
print(json.dumps(make_payload(random.Random(42))))
")"
body="$(curl --fail --silent --show-error -X POST "$API_URL/predict" \
  -H 'Content-Type: application/json' -d "$payload")"
echo "$body"
echo "$body" | python3 -c "
import json, sys
r = json.load(sys.stdin)
assert 0 <= r['risk_probability'] <= 1, r
assert r['risk_label'] in ('alto', 'bajo'), r
assert r['model_name'] and r['model_version'], r
"

echo "Smoke test passed"
