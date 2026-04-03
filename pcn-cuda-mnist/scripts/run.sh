#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== PCN MNIST Trainer ==="

# GPU check
if [ ! -e /dev/nvidia0 ]; then
    echo "ERROR: /dev/nvidia0 not found. Is the NVIDIA driver loaded?"
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi failed. Check driver installation."
    exit 1
fi

echo "GPU found:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Set library path for bundled .so files
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"

cd "$SCRIPT_DIR"
exec ./pcn-mnist "$@"
