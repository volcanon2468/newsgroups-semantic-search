#!/bin/bash

if [ ! -f /app/data/models/gmm_model.joblib ]; then
    echo "[Entrypoint] No pre-built data found. Running setup pipeline..."
    python -m scripts.setup_data
fi

echo "[Entrypoint] Data ready. Starting server..."
exec "$@"
