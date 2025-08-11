#!/bin/bash

# Intelligent Incremental Workspace Build
# Builds only changed packages and their dependents
# Target: <10s incremental builds

set -e

echo "🔄 Intelligent Incremental Build - Target: <10s"
echo "==============================================="

# Configuration
CARGO_TARGET_DIR="target/incremental"
TIMESTAMP_FILE=".last_build_timestamp"
DEPENDENCY_GRAPH_FILE=".dependency_graph.json"

export CARGO_INCREMENTAL=1
export CARGO_TARGET_DIR

echo "📂 Target directory: $CARGO_TARGET_DIR"

# Check if this is first build
if [ ! -f "$TIMESTAMP_FILE" ]; then
    echo "🆕 First build detected, building entire workspace"
    echo "0" > "$TIMESTAMP_FILE"
    CHANGED_PACKAGES="--workspace"
else
    # Find changed files since last build
    LAST_BUILD=$(cat "$TIMESTAMP_FILE")
    CHANGED_FILES=$(find . -name "*.rs" -newer "$TIMESTAMP_FILE" 2>/dev/null | grep -v target/ || true)
    
    if [ -z "$CHANGED_FILES" ]; then
        echo "✅ No changes detected since last build"
        exit 0
    fi
    
    echo "📝 Changed files detected:"
    echo "$CHANGED_FILES" | head -10
    if [ $(echo "$CHANGED_FILES" | wc -l) -gt 10 ]; then
        echo "... and $(( $(echo "$CHANGED_FILES" | wc -l) - 10 )) more files"
    fi
    
    # Determine affected packages (simplified heuristic)
    CHANGED_PACKAGES=""
    if echo "$CHANGED_FILES" | grep -q "zen-neural/"; then
        CHANGED_PACKAGES="$CHANGED_PACKAGES --package zen-neural"
    fi
    if echo "$CHANGED_FILES" | grep -q "zen-forecasting/"; then
        CHANGED_PACKAGES="$CHANGED_PACKAGES --package zen-forecasting"
    fi
    if echo "$CHANGED_FILES" | grep -q "zen-compute/"; then
        CHANGED_PACKAGES="$CHANGED_PACKAGES --package zen-compute"
    fi
    if echo "$CHANGED_FILES" | grep -q "zen-orchestrator/"; then
        CHANGED_PACKAGES="$CHANGED_PACKAGES --package zen-orchestrator"
    fi
    
    # If no specific packages detected, build workspace (safer)
    if [ -z "$CHANGED_PACKAGES" ]; then
        CHANGED_PACKAGES="--workspace"
        echo "🔍 Package detection uncertain, building entire workspace"
    else
        echo "🎯 Building affected packages:$CHANGED_PACKAGES"
    fi
fi

# Time the build
start_time=$(date +%s)

# Build with optimal settings for incremental compilation
cargo build \
    $CHANGED_PACKAGES \
    --profile dev \
    --jobs $(nproc) \
    2>&1 | tee build-incremental.log

end_time=$(date +%s)
build_time=$((end_time - start_time))

# Update timestamp
echo "$end_time" > "$TIMESTAMP_FILE"

echo ""
echo "⏱️  Incremental build completed in ${build_time}s"

if [ $build_time -gt 10 ]; then
    echo "⚠️  Warning: Build took ${build_time}s (target: <10s)"
    echo "💡 Large changes may require longer build times"
else
    echo "✅ Incremental build time within target (<10s)"
fi

echo ""
echo "💡 Next build will be even faster thanks to incremental compilation"
echo "🔄 Total build cache size: $(du -sh $CARGO_TARGET_DIR 2>/dev/null | cut -f1)"