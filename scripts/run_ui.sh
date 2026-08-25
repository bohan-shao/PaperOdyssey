#!/usr/bin/env bash
# Start the Streamlit UI (Q2)
set -euo pipefail
cd "$(dirname "$0")/.."
exec streamlit run ui/app.py --server.port "${PORT:-8501}"
