//! Integration tests for zen-neural-stack components
//!
//! These tests validate that different components of the neural stack
//! work together correctly, including DNN implementation, training 
//! infrastructure, SIMD operations, and memory management.

use crate::common::*;
use crate::integration_test;
use crate::skip_test_if;

pub mod dnn_training_integration;
pub mod simd_memory_integration;
pub mod gnn_dnn_compatibility; 
pub mod storage_integration;
pub mod gpu_integration;

/// Run all integration tests
pub fn run_all_integration_tests() -> Result<(), String> {
    println!("\n🔧 STARTING INTEGRATION TESTS\n");
    
    let config = crate::IntegrationTestConfig::default();
    
    // Core integration tests
    integration_test!("DNN + Training Integration", || dnn_training_integration::test_dnn_training_integration())?;
    integration_test!("SIMD + Memory Integration", || simd_memory_integration::test_simd_memory_integration())?;
    integration_test!("GNN + DNN Compatibility", || gnn_dnn_compatibility::test_gnn_dnn_compatibility())?;
    integration_test!("Storage Integration", || storage_integration::test_storage_integration())?;
    
    // GPU tests (if enabled)
    if config.enable_gpu_tests {
        integration_test!("GPU Integration", || gpu_integration::test_gpu_integration())?;
    } else {
        println!("⏭️  Skipping GPU tests (ZEN_TEST_GPU not set)");
        crate::TEST_STATS.record_skipped();
    }
    
    println!("\n✅ INTEGRATION TESTS COMPLETED\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn run_integration_test_suite() {
        let result = run_all_integration_tests();
        
        // Print final statistics
        let summary = crate::TEST_STATS.get_summary();
        summary.print_summary();
        
        // Assert that all tests passed
        if let Err(e) = result {
            panic!("Integration tests failed: {}", e);
        }
        
        // Assert minimum success rate
        assert!(summary.success_rate() >= 0.95, 
                "Success rate too low: {:.1}%", summary.success_rate() * 100.0);
    }
}