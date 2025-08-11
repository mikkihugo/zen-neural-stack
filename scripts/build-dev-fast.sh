#!/bin/bash

# Fast Development Build Script
# Target: <15s builds for rapid iteration
# Usage: ./scripts/build-dev-fast.sh [target]

set -e

echo "🚀 Fast Development Build - Target: <15s compile time"
echo "================================================="

# Set environment variables for maximum build speed
export CARGO_INCREMENTAL=1
export CARGO_PROFILE_DEV_CODEGEN_UNITS=256
export CARGO_PROFILE_DEV_OPT_LEVEL=0
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_TARGET_DIR="target/fast-dev"

# Use minimal feature set for fastest builds
FEATURES="dev-fast"

# Allow override of features
if [ ! -z "$1" ]; then
    FEATURES="$1"
fi

echo "📦 Building with features: $FEATURES"
echo "🎯 Target directory: $CARGO_TARGET_DIR"

# Time the build
start_time=$(date +%s)

# Build with minimal features and maximum parallelism
cargo build \
    --profile fast-dev \
    --features "$FEATURES" \
    --jobs $(nproc) \
    2>&1 | tee build-fast.log

end_time=$(date +%s)
build_time=$((end_time - start_time))

echo ""
echo "⏱️  Build completed in ${build_time}s"

# Alert if build took too long
if [ $build_time -gt 15 ]; then
    echo "⚠️  Warning: Build took ${build_time}s (target: <15s)"
    echo "💡 Consider using 'cargo check' for syntax validation only"
else
    echo "✅ Build time within target (<15s)"
fi

echo ""
echo "💡 Pro tips for even faster builds:"
echo "   - Use 'cargo check' for syntax validation only"
echo "   - Use 'cargo build --no-deps' if only your code changed"
echo "   - Use 'cargo watch' for automatic rebuilds on file changes"