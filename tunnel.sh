#!/bin/bash
# Opens an SSH tunnel from localhost:8000 to the Pod's engine.
set -euo pipefail

POD_HOST=${POD_HOST:-194.68.245.175}
POD_PORT=${POD_PORT:-22133}
LOCAL_PORT=${LOCAL_PORT:-8000}

# Kill existing tunnel on the same local port (best-effort)
EXISTING=$(netstat -ano 2>&1 | grep ":${LOCAL_PORT}\b" | grep LISTENING | awk '{print $NF}' | head -1 || true)
if [ -n "$EXISTING" ]; then
  taskkill //F //PID "$EXISTING" >/dev/null 2>&1 || true
  sleep 1
fi

# Open new tunnel in background
ssh -i ~/.ssh/id_ed25519 \
    -p $POD_PORT \
    -L ${LOCAL_PORT}:localhost:8000 \
    -N -f \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    root@$POD_HOST

sleep 1
HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${LOCAL_PORT}/health)
if [ "$HTTP" = "200" ]; then
  echo "Tunnel up: http://127.0.0.1:${LOCAL_PORT}/health → 200 OK"
else
  echo "Tunnel may not be working: HTTP $HTTP"
  exit 1
fi
