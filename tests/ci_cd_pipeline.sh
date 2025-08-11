#!/bin/bash
# CI/CD Pipeline for Zen Neural Stack Integration Testing
# This script provides automated testing for continuous integration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_ROOT="/home/mhugo/code/claude-code-zen/zen-neural-stack"
TEST_TIMEOUT=${TEST_TIMEOUT:-600}  # 10 minutes default timeout
ENABLE_GPU_TESTS=${ENABLE_GPU_TESTS:-false}
ENABLE_MEMORY_TESTS=${ENABLE_MEMORY_TESTS:-true}
ENABLE_PERF_TESTS=${ENABLE_PERF_TESTS:-true}
PARALLEL_JOBS=${PARALLEL_JOBS:-$(nproc)}

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "=================================="
    echo "$1"
    echo "=================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${YELLOW}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check Rust installation
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust."
        exit 1
    fi
    
    # Check minimum Rust version (1.75+)
    rust_version=$(rustc --version | cut -d' ' -f2)
    print_success "Rust version: $rust_version"
    
    # Check available memory
    available_memory=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
    print_success "Available memory: ${available_memory}GB"
    
    # Check CPU cores
    print_success "CPU cores: $PARALLEL_JOBS"
}

setup_test_environment() {
    print_step "Setting up test environment..."
    
    cd "$WORKSPACE_ROOT"
    
    # Set environment variables for tests
    export RUST_BACKTRACE=1
    export RUST_LOG=info
    export ZEN_TEST_MODE=ci
    
    if [[ "$ENABLE_GPU_TESTS" == "true" ]]; then
        export ZEN_TEST_GPU=1
        print_success "GPU tests enabled"
    fi
    
    if [[ "$ENABLE_MEMORY_TESTS" == "true" ]]; then
        export ZEN_TEST_MEMORY=1
        print_success "Memory stress tests enabled"
    fi
    
    if [[ "$ENABLE_PERF_TESTS" == "true" ]]; then
        export ZEN_TEST_PERF=1
        print_success "Performance benchmarks enabled"
    fi
    
    # Create test output directory
    mkdir -p test_output
    
    print_success "Test environment configured"
}

build_workspace() {
    print_step "Building workspace..."
    
    # Clean previous builds
    cargo clean
    
    # Build in release mode for performance tests
    timeout "${TEST_TIMEOUT}" cargo build --release --workspace
    
    # Build test dependencies
    timeout "${TEST_TIMEOUT}" cargo test --no-run --workspace
    
    print_success "Workspace build completed"
}

run_unit_tests() {
    print_step "Running unit tests..."
    
    timeout "${TEST_TIMEOUT}" cargo test --workspace --lib -- --test-threads="$PARALLEL_JOBS" \
        --format=json > test_output/unit_test_results.json
    
    # Parse results
    passed_tests=$(grep '"type":"test"' test_output/unit_test_results.json | grep '"event":"ok"' | wc -l)
    failed_tests=$(grep '"type":"test"' test_output/unit_test_results.json | grep '"event":"failed"' | wc -l)
    
    print_success "Unit tests: $passed_tests passed, $failed_tests failed"
    
    if [[ $failed_tests -gt 0 ]]; then
        print_error "Unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    print_step "Running integration tests..."
    
    # Compile integration test runner
    cargo build --release --bin integration_test_runner
    
    # Run with timeout
    timeout "${TEST_TIMEOUT}" ./target/release/integration_test_runner > test_output/integration_test_results.log 2>&1
    
    # Check results
    if [[ -f "/tmp/zen_neural_test_status.txt" ]]; then
        status=$(cat /tmp/zen_neural_test_status.txt)
        if [[ "$status" == "PASS" ]]; then
            print_success "Integration tests passed"
        else
            print_error "Integration tests failed"
            echo "Integration test log:"
            cat test_output/integration_test_results.log
            return 1
        fi
    else
        print_error "Integration test status file not found"
        return 1
    fi
}

run_performance_benchmarks() {
    if [[ "$ENABLE_PERF_TESTS" != "true" ]]; then
        print_step "Skipping performance benchmarks (disabled)"
        return 0
    fi
    
    print_step "Running performance benchmarks..."
    
    # Run benchmarks in release mode
    timeout "${TEST_TIMEOUT}" cargo bench --workspace > test_output/benchmark_results.txt 2>&1
    
    # Parse benchmark results
    if grep -q "test result: ok" test_output/benchmark_results.txt; then
        print_success "Performance benchmarks completed"
    else
        print_error "Performance benchmarks failed"
        echo "Benchmark results:"
        cat test_output/benchmark_results.txt
        return 1
    fi
}

run_memory_safety_tests() {
    if [[ "$ENABLE_MEMORY_TESTS" != "true" ]]; then
        print_step "Skipping memory safety tests (disabled)"
        return 0
    fi
    
    print_step "Running memory safety tests..."
    
    # Run with memory sanitizer if available
    if command -v valgrind &> /dev/null; then
        print_step "Running valgrind memory check..."
        valgrind --leak-check=full --error-exitcode=1 \
            ./target/release/integration_test_runner > test_output/valgrind_results.txt 2>&1
        
        if [[ $? -eq 0 ]]; then
            print_success "Valgrind memory check passed"
        else
            print_error "Memory leaks detected"
            cat test_output/valgrind_results.txt
            return 1
        fi
    else
        print_step "Valgrind not available, skipping memory leak detection"
    fi
    
    # Run AddressSanitizer tests if supported
    if cargo --version | grep -q "nightly"; then
        print_step "Running AddressSanitizer tests..."
        export RUSTFLAGS="-Z sanitizer=address"
        cargo test --workspace --target x86_64-unknown-linux-gnu > test_output/asan_results.txt 2>&1
        unset RUSTFLAGS
        
        if [[ $? -eq 0 ]]; then
            print_success "AddressSanitizer tests passed"
        else
            print_error "AddressSanitizer detected issues"
            cat test_output/asan_results.txt
            return 1
        fi
    fi
}

generate_test_report() {
    print_step "Generating test report..."
    
    cat > test_output/test_report.md << EOF
# Zen Neural Stack Integration Test Report

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Configuration
- Rust Version: $(rustc --version)
- Test Timeout: ${TEST_TIMEOUT}s
- Parallel Jobs: $PARALLEL_JOBS
- GPU Tests: $ENABLE_GPU_TESTS
- Memory Tests: $ENABLE_MEMORY_TESTS
- Performance Tests: $ENABLE_PERF_TESTS

## Results Summary

### Unit Tests
$(if [[ -f test_output/unit_test_results.json ]]; then
    passed=$(grep '"event":"ok"' test_output/unit_test_results.json | wc -l)
    failed=$(grep '"event":"failed"' test_output/unit_test_results.json | wc -l)
    echo "- Passed: $passed"
    echo "- Failed: $failed"
else
    echo "- No unit test results found"
fi)

### Integration Tests
$(if [[ -f /tmp/zen_neural_integration_test_results.json ]]; then
    cat /tmp/zen_neural_integration_test_results.json | jq -r '"- Success Rate: " + (.success_rate * 100 | tostring) + "%"'
    cat /tmp/zen_neural_integration_test_results.json | jq -r '"- Duration: " + (.duration_seconds | tostring) + "s"'
else
    echo "- No integration test results found"
fi)

### Performance Benchmarks
$(if [[ -f test_output/benchmark_results.txt ]]; then
    echo "- Completed successfully"
else
    echo "- Not run or failed"
fi)

## Files Generated
- test_output/unit_test_results.json
- test_output/integration_test_results.log
- test_output/benchmark_results.txt
- /tmp/zen_neural_integration_test_results.json
- /tmp/zen_neural_test_status.txt

## Next Steps
1. Review any failed tests in the logs above
2. Check performance targets are met
3. Validate memory usage improvements
4. Ensure no regressions in existing functionality
EOF

    print_success "Test report generated: test_output/test_report.md"
}

cleanup() {
    print_step "Cleaning up..."
    
    # Archive test results
    tar -czf "test_results_$(date +%Y%m%d_%H%M%S).tar.gz" test_output/
    
    print_success "Test artifacts archived"
}

main() {
    print_header "ZEN NEURAL STACK CI/CD PIPELINE"
    
    # Trap to ensure cleanup
    trap cleanup EXIT
    
    check_dependencies
    setup_test_environment
    
    print_header "BUILD PHASE"
    build_workspace
    
    print_header "UNIT TEST PHASE"
    run_unit_tests
    
    print_header "INTEGRATION TEST PHASE"
    run_integration_tests
    
    print_header "PERFORMANCE TEST PHASE"
    run_performance_benchmarks
    
    print_header "MEMORY SAFETY PHASE"
    run_memory_safety_tests
    
    print_header "REPORTING PHASE"
    generate_test_report
    
    print_header "PIPELINE COMPLETED SUCCESSFULLY"
    print_success "All tests passed! 🎉"
    
    # Display final status
    if [[ -f /tmp/zen_neural_test_status.txt ]]; then
        final_status=$(cat /tmp/zen_neural_test_status.txt)
        if [[ "$final_status" == "PASS" ]]; then
            echo -e "${GREEN}FINAL STATUS: PASS ✅${NC}"
            exit 0
        else
            echo -e "${RED}FINAL STATUS: FAIL ❌${NC}"
            exit 1
        fi
    else
        echo -e "${RED}FINAL STATUS: UNKNOWN ❓${NC}"
        exit 1
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi