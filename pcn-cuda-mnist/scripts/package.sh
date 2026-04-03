#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
DIST_DIR="$PROJECT_DIR/dist"
BINARY="$BUILD_DIR/pcn-mnist"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY. Build first."
    exit 1
fi

echo "Packaging pcn-mnist..."

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/lib"

# Collect non-system shared library dependencies
# Exclude GPU/driver libs that must come from the target system
ldd "$BINARY" | while read -r line; do
    lib_path=$(echo "$line" | awk '{print $3}')
    lib_name=$(echo "$line" | awk '{print $1}')

    # Skip system and GPU-specific libraries
    if echo "$lib_name" | grep -qE '^(libGL\.|libGLX\.|libnvidia|linux-vdso|libdl\.|libpthread\.|libm\.|libc\.|librt\.|ld-linux)'; then
        continue
    fi

    if [ -f "$lib_path" ]; then
        cp "$lib_path" "$DIST_DIR/lib/"
        echo "  Bundled: $lib_name"
    fi
done

# Copy binary and config
cp "$BINARY" "$DIST_DIR/"
cp "$PROJECT_DIR/config.yaml" "$DIST_DIR/"
cp "$SCRIPT_DIR/run.sh" "$DIST_DIR/"
chmod +x "$DIST_DIR/run.sh"

# Write README
cat > "$DIST_DIR/README.txt" << 'EOF'
Requirements: Linux, NVIDIA driver 525+, CUDA SM 7.5 compatible GPU
Run: ./run.sh
EOF

# Create tarball
cd "$PROJECT_DIR"
tar -czf dist/pcn-mnist-bundle.tar.gz -C dist \
    pcn-mnist config.yaml run.sh lib/ README.txt

echo "Package created: dist/pcn-mnist-bundle.tar.gz"
