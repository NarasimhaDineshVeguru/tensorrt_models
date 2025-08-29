#!/bin/bash

# This script starts the FastAPI application with Gunicorn for production use

# Set the number of workers based on available CPU cores (2-4 workers per core is recommended)
# You can adjust this value based on your server's resources
WORKERS=$(nproc)
WORKERS=$((WORKERS * 2))

# Minimum workers
if [ $WORKERS -lt 2 ]; then
    WORKERS=2
fi

echo "Starting server with $WORKERS workers"

# Start Gunicorn with Uvicorn workers
exec gunicorn sentence_similarity_server:app \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    --timeout 120 \
    --log-level info
