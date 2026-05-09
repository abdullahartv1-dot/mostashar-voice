#!/bin/bash
# Open SSH tunnel from localhost:8080 to Pod's streaming server
set -euo pipefail
POD_HOST=${POD_HOST:-103.207.149.153}
POD_PORT=${POD_PORT:?set POD_PORT}
LOCAL_PORT=${LOCAL_PORT:-8080}

EXISTING=$(netstat -ano 2>&1 | grep ":${LOCAL_PORT}\b" | grep LISTENING | awk '{print $NF}' | head -1 || true)
if [ -n "$EXISTING" ]; then
  taskkill //F //PID "$EXISTING" >/dev/null 2>&1 || true
  sleep 1
fi

ssh -i ~/.ssh/id_ed25519 -p $POD_PORT \
    -L ${LOCAL_PORT}:localhost:8080 \
    -N -f \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    root@$POD_HOST

sleep 1
HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${LOCAL_PORT}/health || echo 000)
if [ "$HTTP" = "200" ]; then
  echo "Tunnel up: http://127.0.0.1:${LOCAL_PORT}/health → 200 OK"
  echo "Open http://127.0.0.1:${LOCAL_PORT}/ in browser to test"
  curl -s http://127.0.0.1:${LOCAL_PORT}/health | python -m json.tool 2>/dev/null || true
else
  echo "Server not ready yet (HTTP $HTTP) — model still loading. Wait ~30s and retry curl."
fi
