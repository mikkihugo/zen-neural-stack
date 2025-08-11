//! Regression testing suite for zen-neural-stack
//!
//! This module contains tests that prevent future breakages by validating
//! that existing functionality continues to work correctly after changes.

use crate::common::*;
use crate::integration_test;

pub mod gnn_functionality_preservation;
pub mod training_accuracy_preservation;
pub mod memory_safety_validation;
pub mod api_compatibility;
pub mod cross_platform_compatibility;

/// Run all regression tests
pub fn run_all_regression_tests() -> Result<(), String> {
    println!("\n🔒 STARTING REGRESSION TESTS\n");
    
    // GNN functionality preservation
    integration_test!("GNN Functionality Preservation", || {
        gnn_functionality_preservation::test_gnn_preserved_functionality()
    })?;
    
    // Training accuracy preservation
    integration_test!("Training Accuracy Preservation", || {
        training_accuracy_preservation::test_training_accuracy_preserved()
    })?;
    
    // Memory safety validation
    integration_test!("Memory Safety Validation", || {
        memory_safety_validation::test_memory_safety()
    })?;
    
    // API compatibility
    integration_test!("API Compatibility", || {
        api_compatibility::test_api_compatibility()
    })?;
    
    // Cross-platform compatibility
    integration_test!("Cross-Platform Compatibility", || {
        cross_platform_compatibility::test_cross_platform_compatibility()
    })?;
    
    println!("\n✅ REGRESSION TESTS COMPLETED\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn run_regression_test_suite() {
        let result = run_all_regression_tests();
        
        if let Err(e) = result {
            panic!("Regression tests failed: {}", e);
        }
        
        println!("🔒 All regression tests passed - no functionality was broken");
    }
}