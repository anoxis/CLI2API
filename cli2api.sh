#!/bin/bash
# Ensure PATH includes common CLI locations
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# Change to project directory
cd "$(dirname "$0")"

# Start uvicorn with logging to file
exec python3 -m uvicorn cli2api.main:app --host 0.0.0.0 --port 8000 "$@" 2>&1 | tee -a /tmp/cli2api.log
