#!/bin/bash
# =============================================================================
# Disk I/O Prediction Accuracy Evaluation Script
# =============================================================================
# Evaluates Disk I/O (WriteIO only) prediction accuracy using trained models.
#
# Usage (via Docker):
#   docker compose up -d
#   docker exec -it OpAdviser.sh -c "sh /root/OpAdviser/eval_io.sh"
#
# Or manually inside the container:
#   sh /root/OpAdviser/eval_io.sh
# =============================================================================

set -e

# Auto-detect the correct directory
if [ -d "/root/OpAdviser/scripts" ]; then
    OPADVISER_DIR="/root/OpAdviser"
elif [ -d "/root/Tuning/scripts" ]; then
    OPADVISER_DIR="/root/Tuning"
elif [ -d "/app/scripts" ]; then
    OPADVISER_DIR="/app"
else
    OPADVISER_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

cd "$OPADVISER_DIR"

# Set environment variables
export PYTHONPATH="$OPADVISER_DIR"
export MYSQL_SOCK=/var/run/mysqld/mysqld.sock

# Run evaluation
python scripts/eval_io_accuracy.py

