# Rent a vast.ai GPU, run a command in a container image, wait for a
# success/failure marker in the logs, destroy the instance. Exits 0 only
# if the success marker appeared.
#
# Requires VAST_API_KEY (or a key configured for the vastai CLI).

usage() {
  cat <<'EOF'
Usage: vast-smoke-test --image IMAGE [options] [-- CMD...]

Options:
  --image IMAGE       container image reference (required)
  --query QUERY       vastai offer query
                      (default: "gpu_name=RTX_4090 rentable=true verified=true num_gpus=1 reliability>0.98 inet_down>500")
  --login LOGIN       docker login args for private registries, e.g. "-u user -p token ghcr.io"
  --disk GB           disk size in GB (default: 50)
  --timeout SECONDS   overall timeout (default: 1800; image pull alone can
                      take >10 minutes on slow hosts)
  --success-re RE     success marker regex (default: "RESULT: OK")
  --failure-re RE     failure marker regex (default: "RESULT: FAIL")
  --offers N          how many cheapest offers to try (default: 3)
  -- CMD...           command to run in the container
                      (default: python -m server.main --smoke-test)
EOF
}

QUERY="gpu_name=RTX_4090 rentable=true verified=true num_gpus=1 reliability>0.98 inet_down>500"
LOGIN=""
DISK=50
TIMEOUT=1800
SUCCESS_RE="RESULT: OK"
FAILURE_RE="RESULT: FAIL"
OFFERS=3
IMAGE=""
CMD=(python -m server.main --smoke-test)

while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --query) QUERY="$2"; shift 2 ;;
    --login) LOGIN="$2"; shift 2 ;;
    --disk) DISK="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --success-re) SUCCESS_RE="$2"; shift 2 ;;
    --failure-re) FAILURE_RE="$2"; shift 2 ;;
    --offers) OFFERS="$2"; shift 2 ;;
    --) shift; CMD=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$IMAGE" ] || { echo "--image is required" >&2; usage >&2; exit 2; }

vast() {
  if [ -n "${VAST_API_KEY:-}" ]; then
    vastai "$@" --api-key "$VAST_API_KEY"
  else
    vastai "$@"
  fi
}

echo "[vast-smoke-test] query: $QUERY"
mapfile -t OFFER_IDS < <(vast search offers "$QUERY" -o 'dph' --raw \
  | jq -r '.[].id' | head -n "$OFFERS")
[ "${#OFFER_IDS[@]}" -gt 0 ] || { echo "[vast-smoke-test] no offers match" >&2; exit 1; }

INSTANCE=""
# shellcheck disable=SC2329  # invoked via trap
cleanup() {
  if [ -n "$INSTANCE" ]; then
    echo "[vast-smoke-test] destroying instance $INSTANCE"
    vast destroy instance "$INSTANCE" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for OFFER in "${OFFER_IDS[@]}"; do
  echo "[vast-smoke-test] trying offer $OFFER"
  CREATE_ARGS=(create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --cancel-unavail --raw)
  [ -n "$LOGIN" ] && CREATE_ARGS+=(--login "$LOGIN")
  CREATE_ARGS+=(--args "${CMD[@]}")
  if OUT=$(vast "${CREATE_ARGS[@]}" 2>&1) && INSTANCE=$(echo "$OUT" | jq -r '.new_contract') && [ -n "$INSTANCE" ] && [ "$INSTANCE" != "null" ]; then
    echo "[vast-smoke-test] created instance $INSTANCE on offer $OFFER"
    break
  fi
  echo "[vast-smoke-test] offer $OFFER unavailable: $(echo "$OUT" | tail -1)"
  INSTANCE=""
done
[ -n "$INSTANCE" ] || { echo "[vast-smoke-test] could not rent any offer" >&2; exit 1; }

DEADLINE=$(( $(date +%s) + TIMEOUT ))
LOGFILE=$(mktemp)
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 15
  STATUS=$(vast show instances --raw | jq -r ".[] | select(.id == $INSTANCE) | .actual_status // \"unknown\"")
  vast logs "$INSTANCE" --tail 1000 >"$LOGFILE" 2>/dev/null || true
  if grep -qE "$SUCCESS_RE" "$LOGFILE"; then
    echo "[vast-smoke-test] === container logs ==="
    grep -E "^\[" "$LOGFILE" || tail -50 "$LOGFILE"
    echo "[vast-smoke-test] SUCCESS (matched: $SUCCESS_RE)"
    exit 0
  fi
  if grep -qE "$FAILURE_RE" "$LOGFILE"; then
    echo "[vast-smoke-test] === container logs ==="
    tail -100 "$LOGFILE"
    echo "[vast-smoke-test] FAILURE (matched: $FAILURE_RE)" >&2
    exit 1
  fi
  echo "[vast-smoke-test] waiting (instance $INSTANCE is $STATUS, $(( DEADLINE - $(date +%s) ))s left)"
done

echo "[vast-smoke-test] TIMEOUT after ${TIMEOUT}s; last logs:" >&2
tail -100 "$LOGFILE" >&2
exit 1
