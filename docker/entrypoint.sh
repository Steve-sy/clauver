#!/usr/bin/env bash
set -euo pipefail

cd /app

AGENT_MODE="${AGENT_MODE:-default}"

case "$AGENT_MODE" in
  default)
    TARGET="agent.py"
    ;;
  cloud)
    TARGET="agent-cloud.py"
    ;;
  local)
    TARGET="agent-local.py"
    ;;
  *)
    echo "[entrypoint] Invalid AGENT_MODE: $AGENT_MODE (expected: default|cloud|local)"
    exit 1
    ;;
esac

if [[ $# -gt 0 ]]; then
  echo "[entrypoint] Executing custom command: $*"
  exec "$@"
else
  echo "[entrypoint] Starting Clauver agent ($AGENT_MODE) → python $TARGET dev ..."
  exec python "$TARGET" dev
fi