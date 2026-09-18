#!/usr/bin/env bash
# ==============================================================================
# run_dashboard.sh
# One-click launcher for the vLLM on Google Cloud Platform Dashboard
# ==============================================================================

set -e

PORT=${PORT:-8080}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "========================================================================"
echo "⚡ Starting vLLM on Google Cloud Platform Dashboard"
echo "========================================================================"

# Check if port is in use; find next available if needed
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; do
    echo "Notice: Port ${PORT} is in use, trying port $((PORT+1))..."
    PORT=$((PORT+1))
done

export PORT="${PORT}"

echo "Server running at: http://localhost:${PORT}"
echo "Press Ctrl+C to stop the dashboard server."
echo "========================================================================"

# Try opening in browser on macOS
if command -v open >/dev/null 2>&1; then
    (sleep 1 && open "http://localhost:${PORT}") &
fi

exec python3 dashboard_server.py
