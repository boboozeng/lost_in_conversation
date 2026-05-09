#!/bin/bash

if [ ! -f .env ]; then
    echo ".env file not found"
    exit 1
fi

set -a
source .env
set +a

echo "Loaded environment variables:"

grep -v '^#' .env | grep '=' | cut -d '=' -f 1 | while read var; do
    if [ -n "${!var}" ]; then
        echo "  ✓ $var"
    else
        echo "  ✗ $var"
    fi
done