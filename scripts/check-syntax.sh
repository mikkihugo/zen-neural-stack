#!/bin/bash

# Ultra-Fast Syntax Check Script
# Target: <5s syntax and type checking
# Usage: ./scripts/check-syntax.sh [workspace-member]

set -e

echo "🔍 Fast Syntax Check - Target: <5s validation"
echo "=============================================="

# Set environment for fastest checking
export CARGO_INCREMENTAL=1
export CARGO_TARGET_DIR="target/check"

# Determine target
if [ ! -z "$1" ]; then
    TARGET="--package $1"
    echo "🎯 Checking package: $1"
else
    TARGET="--workspace"
    echo "🎯 Checking entire workspace"
fi

echo "📂 Target directory: $CARGO_TARGET_DIR"

# Time the check
start_time=$(date +%s)

# Fast syntax and type checking only
cargo check \
    $TARGET \
    --jobs $(nproc) \
    --message-format short \
    2>&1 | tee check.log

end_time=$(date +%s)
check_time=$((end_time - start_time))

echo ""
echo "⏱️  Check completed in ${check_time}s"

if [ $check_time -gt 5 ]; then
    echo "⚠️  Warning: Check took ${check_time}s (target: <5s)"
    echo "💡 Consider checking individual packages for faster feedback"
else
    echo "✅ Check time within target (<5s)"
fi

echo ""
echo "💡 Pro tips:"
echo "   - Use 'cargo clippy' for linting (may be slower)"
echo "   - Use 'cargo check --package <name>' for single package"
echo "   - Run with 'watch' for continuous checking: watch -n 1 ./scripts/check-syntax.sh"