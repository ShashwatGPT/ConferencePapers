#!/usr/bin/env bash
set -e

# Kill any process currently holding port 8000
if lsof -ti :8000 &>/dev/null; then
  echo "Killing process(es) on port 8000..."
  lsof -ti :8000 | xargs kill -9
  sleep 1
fi

cd "$(dirname "$0")/backend"
conda run -n conferencepapers uvicorn main:app --reload --port 8000