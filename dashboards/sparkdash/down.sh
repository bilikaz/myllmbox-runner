#!/usr/bin/env bash
# Tear sparkDash down. (sparkDash runs only on the head, so there's nothing to stop on other boxes.)
docker rm -f mbx-dashboard >/dev/null 2>&1 || true
echo "sparkDash down"
