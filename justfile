# Zen Neural Stack Development Commands

# Default recipe
default:
    @just --list

# Build all packages
build:
    cargo build --workspace --all-features

# Build optimized release
build-release:
    cargo build --workspace --release --all-features

# Run all tests
test:
    cargo test --workspace --all-features

# Run benchmarks
bench:
    cargo bench --workspace

# Build WASM packages
build-wasm:
    #!/usr/bin/env bash
    for crate in zen-neural zen-forecasting zen-compute zen-orchestrator; do
        echo "Building WASM for $crate..."
        cd $crate
        wasm-pack build --target web --out-dir ../wasm-dist/$crate
        cd ..
    done

# Check all code
check:
    cargo check --workspace --all-features
    cargo clippy --workspace --all-features -- -D warnings
    cargo fmt --check

# Fix code formatting and lints
fix:
    cargo fmt
    cargo clippy --workspace --all-features --fix --allow-dirty

# Clean all build artifacts
clean:
    cargo clean
    rm -rf wasm-dist/

# Development setup
setup:
    rustup component add clippy rustfmt
    cargo install wasm-pack just

# Quick development cycle
dev: check test

# Performance profiling
profile target:
    cargo build --release --bin {{target}}
    perf record --call-graph dwarf target/release/{{target}}
    perf report

# Documentation
docs:
    cargo doc --workspace --all-features --no-deps --open

# Security audit
audit:
    cargo audit

# Update dependencies
update:
    cargo update