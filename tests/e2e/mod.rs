//! End-to-End workflow testing
//!
//! These tests validate complete workflows from data loading through
//! model training to inference and export, ensuring the entire
//! zen-neural-stack system works as a cohesive unit.

use crate::common::*;
use crate::integration_test;

pub mod complete_training_workflow;
pub mod multi_model_orchestration;
pub mod production_deployment_scenarios;
pub mod error_handling_and_recovery;

/// Run all end-to-end tests
pub fn run_all_e2e_tests() -> Result<(), String> {
    println!("\n🎯 STARTING END-TO-END TESTS\n");
    
    // Complete training workflows
    integration_test!("Complete Training Workflow", || {
        complete_training_workflow::test_complete_training_workflow()
    })?;
    
    // Multi-model orchestration
    integration_test!("Multi-Model Orchestration", || {
        multi_model_orchestration::test_multi_model_orchestration()
    })?;
    
    // Production deployment scenarios
    integration_test!("Production Deployment Scenarios", || {
        production_deployment_scenarios::test_production_deployment()
    })?;
    
    // Error handling and recovery
    integration_test!("Error Handling and Recovery", || {
        error_handling_and_recovery::test_error_handling_recovery()
    })?;
    
    println!("\n✅ END-TO-END TESTS COMPLETED\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn run_e2e_test_suite() {
        let result = run_all_e2e_tests();
        
        if let Err(e) = result {
            panic!("End-to-end tests failed: {}", e);
        }
        
        println!("🎯 All end-to-end workflows validated successfully");
    }
}