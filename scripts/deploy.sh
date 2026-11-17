#!/usr/bin/env bash
# Redeploy the agent on the VPS: pull, build, swap the container, wait until it's
# healthy, and roll back to the previous image automatically if it isn't.
#
# Run on the server:  bash /opt/portfolio-agent/scripts/deploy.sh
set -euo pipefail

cd /opt/portfolio-agent
git pull --ff-only
tag=$(git rev-parse --short HEAD)

# The IMAGE ID of what's running now (not its tag: `latest` gets reused below).
previous=$(docker inspect portfolio-agent --format '{{.Image}}' 2>/dev/null || true)

docker build -t "portfolio-agent:$tag" -t portfolio-agent:latest .

start() {
  docker run -d --name portfolio-agent --restart unless-stopped \
    --network copi-edge --env-file .env \
    --memory 512m --cpus 1 \
    --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
    "$1" >/dev/null
}

wait_healthy() {
  for _ in $(seq 1 25); do
    [ "$(docker inspect portfolio-agent --format '{{.State.Health.Status}}')" = healthy ] && return 0
    sleep 3
  done
  return 1
}

# A few seconds of downtime while the container is swapped: fine for a portfolio.
docker rm -f portfolio-agent >/dev/null 2>&1 || true
start "portfolio-agent:$tag"

if wait_healthy; then
  echo "deployed $tag (healthy)"
  exit 0
fi

echo "!! $tag did not become healthy. Last logs:"
docker logs portfolio-agent --tail 30 || true
docker rm -f portfolio-agent >/dev/null
if [ -n "$previous" ]; then
  start "$previous"
  wait_healthy && echo "rolled back to the previous image ($previous)"
fi
exit 1
