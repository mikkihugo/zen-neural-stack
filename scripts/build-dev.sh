#!/bin/bash

# Standard Development Build Script  
# Target: <30s builds with good debugging experience
# Usage: ./scripts/build-dev.sh [additional-features]

set -e

echo "🛠️  Development Build - Target: <30s compile time"
echo "================================================"

# Set environment variables for balanced build speed and debugging
export CARGO_INCREMENTAL=1
export CARGO_PROFILE_DEV_CODEGEN_UNITS=256
export CARGO_TARGET_DIR="target/dev"

# Default features for development (lightweight but functional)
FEATURES="default"

# Allow additional features to be specified
if [ ! -z "$1" ]; then
    FEATURES="$FEATURES,$1"
fi

echo "📦 Building with features: $FEATURES"
echo "🎯 Target directory: $CARGO_TARGET_DIR"

# Time the build
start_time=$(date +%s)

# Build with development profile
cargo build \
    --profile dev \
    --features "$FEATURES" \
    --jobs $(nproc) \
    2>&1 | tee build-dev.log

end_time=$(date +%s)
build_time=$((end_time - start_time))

echo ""
echo "⏱️  Build completed in ${build_time}s"

# Performance feedback
if [ $build_time -gt 30 ]; then
    echo "⚠️  Warning: Build took ${build_time}s (target: <30s)"
    echo "💡 Try using './scripts/build-dev-fast.sh' for faster iteration"
elif [ $build_time -gt 20 ]; then
    echo "⚠️  Build approaching target limit (${build_time}s/30s)"
else
    echo "✅ Build time within target (<30s)"
fi

# Show incremental build time estimate
echo ""
echo "📊 Next incremental build estimated: <10s"
echo "🔄 Run 'cargo build' again to test incremental compilation"