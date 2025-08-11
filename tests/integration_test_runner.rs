//! Comprehensive Integration Test Runner
//!
//! This is the main entry point for running all integration tests
//! in the zen-neural-stack system. It coordinates all test suites
//! and provides comprehensive reporting.

use zen_neural_stack_tests::*;

use std::time::Instant;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ZEN NEURAL STACK INTEGRATION TEST SUITE");
    println!("==========================================\n");
    
    let start_time = Instant::now();
    let config = IntegrationTestConfig::default();
    
    // Print test configuration
    print_test_config(&config);
    
    let mut all_passed = true;
    
    // Run integration tests
    println!("Phase 1: Integration Tests");
    println!("--------------------------");
    if let Err(e) = integration::run_all_integration_tests() {
        println!("❌ Integration tests failed: {}", e);
        all_passed = false;
    }
    
    // Run performance tests
    println!("\nPhase 2: Performance Tests");
    println!("--------------------------");
    match performance::run_all_performance_tests() {
        Ok(report) => {
            println!("{}", report.generate_report());
            if !report.targets_achieved() {
                println!("❌ Performance targets not achieved");
                all_passed = false;
            }
        }
        Err(e) => {
            println!("❌ Performance tests failed: {}", e);
            all_passed = false;
        }
    }
    
    // Run regression tests
    println!("\nPhase 3: Regression Tests");
    println!("-------------------------");
    if let Err(e) = regression::run_all_regression_tests() {
        println!("❌ Regression tests failed: {}", e);
        all_passed = false;
    }
    
    // Run end-to-end tests
    println!("\nPhase 4: End-to-End Tests");
    println!("-------------------------");
    if let Err(e) = e2e::run_all_e2e_tests() {
        println!("❌ End-to-end tests failed: {}", e);
        all_passed = false;
    }
    
    // Final summary
    let total_duration = start_time.elapsed();
    let final_stats = TEST_STATS.get_summary();
    
    println!("\n" + "=".repeat(50).as_str());
    println!("FINAL TEST SUITE RESULTS");
    println!("=".repeat(50));
    
    final_stats.print_summary();
    
    println!("Total Execution Time: {:.2}s", total_duration.as_secs_f64());
    
    if all_passed && final_stats.success_rate() >= 0.95 {
        println!("🎉 ALL TESTS PASSED - ZEN NEURAL STACK INTEGRATION VALIDATED!");
        
        // Store success metrics for CI/CD
        store_test_results(&final_stats, total_duration, true)?;
        
        Ok(())
    } else {
        println!("💥 TESTS FAILED - INTEGRATION ISSUES DETECTED");
        
        // Store failure metrics for CI/CD
        store_test_results(&final_stats, total_duration, false)?;
        
        std::process::exit(1);
    }
}

fn print_test_config(config: &IntegrationTestConfig) {
    println!("Test Configuration:");
    println!("  GPU Tests: {}", if config.enable_gpu_tests { "Enabled" } else { "Disabled" });
    println!("  Memory Stress Tests: {}", if config.enable_memory_stress_tests { "Enabled" } else { "Disabled" });
    println!("  Performance Benchmarks: {}", if config.enable_performance_benchmarks { "Enabled" } else { "Disabled" });
    println!("  Test Timeout: {:?}", config.test_timeout);
    println!("  Max Memory: {} MB", config.max_memory_mb);
    
    println!("\nEnvironment Variables:");
    println!("  ZEN_TEST_GPU: {}", env::var("ZEN_TEST_GPU").unwrap_or_else(|_| "Not set".to_string()));
    println!("  ZEN_TEST_MEMORY: {}", env::var("ZEN_TEST_MEMORY").unwrap_or_else(|_| "Not set".to_string()));
    println!("  ZEN_TEST_PERF: {}", env::var("ZEN_TEST_PERF").unwrap_or_else(|_| "Not set".to_string()));
    println!();
}

fn store_test_results(
    stats: &TestSummary,
    duration: std::time::Duration,
    success: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;
    use serde_json::json;
    
    let results = json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "success": success,
        "total_tests": stats.total,
        "passed_tests": stats.passed,
        "failed_tests": stats.failed,
        "skipped_tests": stats.skipped,
        "success_rate": stats.success_rate(),
        "duration_seconds": duration.as_secs_f64(),
        "targets": {
            "memory_reduction": 0.70,
            "training_speedup_min": 20.0,
            "inference_speedup_min": 50.0
        }
    });
    
    // Store in temporary directory for CI/CD pickup
    let results_path = std::env::temp_dir().join("zen_neural_integration_test_results.json");
    let mut file = File::create(&results_path)?;
    file.write_all(results.to_string().as_bytes())?;
    
    println!("📊 Test results stored: {}", results_path.display());
    
    // Also create a simple status file for easy CI/CD checking
    let status_path = std::env::temp_dir().join("zen_neural_test_status.txt");
    let mut status_file = File::create(&status_path)?;
    status_file.write_all(if success { b"PASS" } else { b"FAIL" })?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn run_full_test_suite() {
        // This test runs the entire integration test suite
        // It's designed to be run in CI/CD environments
        main().expect("Full test suite should pass");
    }
}