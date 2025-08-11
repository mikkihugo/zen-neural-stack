//! Zen Neural Stack Integration Testing Suite
//!
//! This module provides comprehensive integration testing for the zen-neural-stack
//! neural architecture migration from JavaScript to Rust. It validates that all
//! components work together seamlessly and achieve the performance targets.
//!
//! ## Test Categories
//!
//! - **Integration Tests**: Component-to-component interaction validation
//! - **Performance Tests**: Benchmark validation and regression detection
//! - **Regression Tests**: Prevent future breakages
//! - **End-to-End Tests**: Complete workflow validation
//!
//! ## Performance Targets (from Phase 1 specification)
//!
//! - Memory usage reduction: 70%
//! - Training speed improvement: 20-50x
//! - Inference speed improvement: 50-100x
//! - Test coverage: >90% for integration tests
//!
//! ## Usage
//!
//! ```bash
//! # Run all integration tests
//! cargo test --test integration
//!
//! # Run performance validation
//! cargo test --test performance --release
//!
//! # Run specific component integration
//! cargo test --test dnn_training_integration
//! ```

pub mod integration;
pub mod performance;
pub mod regression; 
pub mod e2e;

// Common testing utilities
pub mod common;

use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global test configuration and state
pub struct IntegrationTestConfig {
    /// Whether to run GPU tests (if available)
    pub enable_gpu_tests: bool,
    /// Whether to run memory-intensive tests
    pub enable_memory_stress_tests: bool,
    /// Whether to run long-running performance tests
    pub enable_performance_benchmarks: bool,
    /// Test timeout duration
    pub test_timeout: Duration,
    /// Maximum memory usage for tests (in MB)
    pub max_memory_mb: usize,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            enable_gpu_tests: std::env::var("ZEN_TEST_GPU").is_ok(),
            enable_memory_stress_tests: std::env::var("ZEN_TEST_MEMORY").is_ok(),
            enable_performance_benchmarks: std::env::var("ZEN_TEST_PERF").is_ok(),
            test_timeout: Duration::from_secs(300), // 5 minutes
            max_memory_mb: 1024, // 1GB
        }
    }
}

/// Test execution statistics
#[derive(Debug, Default)]
pub struct TestStats {
    pub total_tests: AtomicUsize,
    pub passed_tests: AtomicUsize,
    pub failed_tests: AtomicUsize,
    pub skipped_tests: AtomicUsize,
    pub total_duration: std::sync::Mutex<Duration>,
}

impl TestStats {
    pub fn record_test_result(&self, passed: bool, duration: Duration) {
        if passed {
            self.passed_tests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_tests.fetch_add(1, Ordering::Relaxed);
        }
        self.total_tests.fetch_add(1, Ordering::Relaxed);
        
        if let Ok(mut total_duration) = self.total_duration.lock() {
            *total_duration += duration;
        }
    }
    
    pub fn record_skipped(&self) {
        self.skipped_tests.fetch_add(1, Ordering::Relaxed);
        self.total_tests.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_summary(&self) -> TestSummary {
        TestSummary {
            total: self.total_tests.load(Ordering::Relaxed),
            passed: self.passed_tests.load(Ordering::Relaxed),
            failed: self.failed_tests.load(Ordering::Relaxed),
            skipped: self.skipped_tests.load(Ordering::Relaxed),
            duration: self.total_duration.lock()
                .map(|d| *d)
                .unwrap_or(Duration::ZERO),
        }
    }
}

#[derive(Debug)]
pub struct TestSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub duration: Duration,
}

impl TestSummary {
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / (self.total - self.skipped) as f64
        }
    }
    
    pub fn print_summary(&self) {
        println!("\n=== ZEN NEURAL STACK INTEGRATION TEST SUMMARY ===");
        println!("Total Tests: {}", self.total);
        println!("Passed: {} ({:.1}%)", self.passed, 
                 self.passed as f64 / self.total as f64 * 100.0);
        println!("Failed: {} ({:.1}%)", self.failed,
                 self.failed as f64 / self.total as f64 * 100.0);
        println!("Skipped: {} ({:.1}%)", self.skipped,
                 self.skipped as f64 / self.total as f64 * 100.0);
        println!("Success Rate: {:.1}%", self.success_rate() * 100.0);
        println!("Total Duration: {:.2}s", self.duration.as_secs_f64());
        println!("================================================\n");
    }
}

/// Global test state for coordination across test modules
pub static TEST_STATS: once_cell::sync::Lazy<Arc<TestStats>> = 
    once_cell::sync::Lazy::new(|| Arc::new(TestStats::default()));

/// Macro for running tests with automatic statistics tracking
#[macro_export]
macro_rules! integration_test {
    ($test_name:expr, $test_fn:expr) => {{
        let start = std::time::Instant::now();
        let result = std::panic::catch_unwind(|| $test_fn);
        let duration = start.elapsed();
        
        match result {
            Ok(Ok(())) => {
                println!("✅ {}: PASSED ({:.2}s)", $test_name, duration.as_secs_f64());
                $crate::TEST_STATS.record_test_result(true, duration);
                Ok(())
            }
            Ok(Err(e)) => {
                println!("❌ {}: FAILED ({:.2}s) - {}", $test_name, duration.as_secs_f64(), e);
                $crate::TEST_STATS.record_test_result(false, duration);
                Err(format!("Test failed: {}", e))
            }
            Err(_) => {
                println!("💥 {}: PANICKED ({:.2}s)", $test_name, duration.as_secs_f64());
                $crate::TEST_STATS.record_test_result(false, duration);
                Err("Test panicked".to_string())
            }
        }
    }};
}

/// Utility function to skip tests conditionally
#[macro_export]
macro_rules! skip_test_if {
    ($condition:expr, $reason:expr) => {
        if $condition {
            println!("⏭️  Skipping test: {}", $reason);
            $crate::TEST_STATS.record_skipped();
            return;
        }
    };
}