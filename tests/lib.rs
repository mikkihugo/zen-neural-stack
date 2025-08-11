//! Zen Neural Stack Integration Testing Library
//!
//! This library provides comprehensive integration testing for the zen-neural-stack
//! neural architecture migration. It validates component integration, performance
//! targets, regression prevention, and end-to-end workflows.

pub mod common;
pub mod integration;
pub mod performance;
pub mod regression;
pub mod e2e;

// Re-export the main testing framework
pub use common::*;

// External dependencies needed for tests
use std::sync::Arc;
use std::time::Duration;

// Re-export testing macros
pub use integration_test;
pub use skip_test_if;

// Dependencies for serialization (for test result storage)
use serde::{Deserialize, Serialize};

// Add external crates that tests need
pub use ndarray;
pub use serde_json;
pub use chrono;
pub use once_cell;

/// Test suite version for tracking compatibility
pub const TEST_SUITE_VERSION: &str = "1.0.0-alpha.1";

/// Initialize the test framework
pub fn init() {
    // Set up logging for tests
    env_logger::init();
    
    // Initialize test statistics
    lazy_static::initialize(&TEST_STATS);
    
    println!("🧪 Zen Neural Stack Test Suite v{} initialized", TEST_SUITE_VERSION);
}