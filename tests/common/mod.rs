//! Common testing utilities for zen-neural-stack integration tests
//!
//! This module provides shared utilities, fixtures, and helper functions
//! used across all integration test suites.

use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::sync::Arc;
use ndarray::{Array2, Array1};
use serde_json::Value;

pub mod fixtures;
pub mod assertions;
pub mod memory_monitor;
pub mod performance_monitor;

/// Test data generator for neural networks
pub struct TestDataGenerator {
    pub seed: u64,
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_layers: Vec<usize>,
}

impl TestDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            input_size: 10,
            output_size: 3,
            hidden_layers: vec![20, 15],
        }
    }
    
    pub fn with_dimensions(mut self, input: usize, hidden: Vec<usize>, output: usize) -> Self {
        self.input_size = input;
        self.hidden_layers = hidden;
        self.output_size = output;
        self
    }
    
    /// Generate test input data
    pub fn generate_inputs(&self, batch_size: usize) -> Array2<f64> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut data = Array2::zeros((batch_size, self.input_size));
        
        for mut row in data.outer_iter_mut() {
            for val in row.iter_mut() {
                *val = rng.gen_range(-1.0..1.0);
            }
        }
        
        data
    }
    
    /// Generate test target data
    pub fn generate_targets(&self, batch_size: usize) -> Array2<f64> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(self.seed + 1);
        let mut data = Array2::zeros((batch_size, self.output_size));
        
        for mut row in data.outer_iter_mut() {
            for val in row.iter_mut() {
                *val = rng.gen_range(0.0..1.0);
            }
        }
        
        data
    }
    
    /// Generate XOR training data (classic neural network test)
    pub fn generate_xor_data() -> (Array2<f64>, Array2<f64>) {
        let inputs = Array2::from_vec(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ]);
        
        let targets = Array2::from_vec(vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
        ]);
        
        (inputs, targets)
    }
    
    /// Generate sine wave prediction data
    pub fn generate_sine_wave_data(&self, length: usize) -> (Array2<f64>, Array2<f64>) {
        let mut inputs = Array2::zeros((length - 1, 1));
        let mut targets = Array2::zeros((length - 1, 1));
        
        for i in 0..length - 1 {
            let x = i as f64 * 0.1;
            inputs[[i, 0]] = (x * std::f64::consts::PI).sin();
            targets[[i, 0]] = ((x + 0.1) * std::f64::consts::PI).sin();
        }
        
        (inputs, targets)
    }
}

/// Performance measurement utilities
pub struct PerformanceMeter {
    start_time: Instant,
    memory_start: usize,
    name: String,
}

impl PerformanceMeter {
    pub fn new(name: &str) -> Self {
        Self {
            start_time: Instant::now(),
            memory_start: get_current_memory_usage(),
            name: name.to_string(),
        }
    }
    
    pub fn stop(self) -> PerformanceResult {
        let duration = self.start_time.elapsed();
        let memory_end = get_current_memory_usage();
        let memory_delta = memory_end.saturating_sub(self.memory_start);
        
        PerformanceResult {
            name: self.name,
            duration,
            memory_used: memory_delta,
            peak_memory: memory_end,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceResult {
    pub name: String,
    pub duration: Duration,
    pub memory_used: usize, // bytes
    pub peak_memory: usize, // bytes
}

impl PerformanceResult {
    pub fn print(&self) {
        println!(
            "⚡ {} - Duration: {:.2}ms, Memory: {:.2}MB, Peak: {:.2}MB",
            self.name,
            self.duration.as_secs_f64() * 1000.0,
            self.memory_used as f64 / 1024.0 / 1024.0,
            self.peak_memory as f64 / 1024.0 / 1024.0
        );
    }
    
    /// Compare against a baseline performance result
    pub fn compare_to(&self, baseline: &PerformanceResult) -> PerformanceComparison {
        let speed_improvement = if baseline.duration.as_nanos() == 0 {
            f64::INFINITY
        } else {
            baseline.duration.as_secs_f64() / self.duration.as_secs_f64()
        };
        
        let memory_reduction = if baseline.memory_used == 0 {
            0.0
        } else {
            1.0 - (self.memory_used as f64 / baseline.memory_used as f64)
        };
        
        PerformanceComparison {
            speed_improvement,
            memory_reduction,
            baseline: baseline.clone(),
            current: self.clone(),
        }
    }
}

#[derive(Debug)]
pub struct PerformanceComparison {
    pub speed_improvement: f64, // multiplier (e.g., 2.0 = 2x faster)
    pub memory_reduction: f64,  // fraction (e.g., 0.7 = 70% less memory)
    pub baseline: PerformanceResult,
    pub current: PerformanceResult,
}

impl PerformanceComparison {
    pub fn meets_targets(&self, min_speed_improvement: f64, min_memory_reduction: f64) -> bool {
        self.speed_improvement >= min_speed_improvement && 
        self.memory_reduction >= min_memory_reduction
    }
    
    pub fn print_comparison(&self) {
        println!("\n=== PERFORMANCE COMPARISON ===");
        println!("Test: {}", self.current.name);
        println!("Speed Improvement: {:.1}x", self.speed_improvement);
        println!("Memory Reduction: {:.1}%", self.memory_reduction * 100.0);
        println!("Baseline: {:.2}ms, {:.2}MB", 
                self.baseline.duration.as_secs_f64() * 1000.0,
                self.baseline.memory_used as f64 / 1024.0 / 1024.0);
        println!("Current:  {:.2}ms, {:.2}MB",
                self.current.duration.as_secs_f64() * 1000.0, 
                self.current.memory_used as f64 / 1024.0 / 1024.0);
        println!("==============================\n");
    }
}

/// Get current memory usage (approximation)
pub fn get_current_memory_usage() -> usize {
    // This is a simplified implementation
    // In a real scenario, you'd use platform-specific APIs
    use std::alloc::{GlobalAlloc, Layout, System};
    
    // For testing purposes, return a mock value
    // In production, integrate with jemalloc or similar
    1024 * 1024 // 1MB baseline
}

/// Test environment setup
pub struct TestEnvironment {
    pub temp_dir: PathBuf,
    pub config: crate::IntegrationTestConfig,
}

impl TestEnvironment {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir = std::env::temp_dir().join("zen_neural_tests");
        std::fs::create_dir_all(&temp_dir)?;
        
        Ok(Self {
            temp_dir,
            config: crate::IntegrationTestConfig::default(),
        })
    }
    
    pub fn cleanup(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.temp_dir.exists() {
            std::fs::remove_dir_all(&self.temp_dir)?;
        }
        Ok(())
    }
}

impl Drop for TestEnvironment {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// Assertion helpers
pub mod assert {
    use super::*;
    
    /// Assert that two floating point arrays are approximately equal
    pub fn arrays_approx_equal(a: &Array2<f64>, b: &Array2<f64>, tolerance: f64) -> Result<(), String> {
        if a.shape() != b.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", a.shape(), b.shape()));
        }
        
        for (i, (a_val, b_val)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (a_val - b_val).abs();
            if diff > tolerance {
                return Err(format!(
                    "Values differ at index {}: {} vs {} (diff: {}, tolerance: {})",
                    i, a_val, b_val, diff, tolerance
                ));
            }
        }
        
        Ok(())
    }
    
    /// Assert performance improvement
    pub fn performance_improvement(
        result: &PerformanceComparison,
        min_speed: f64,
        min_memory: f64
    ) -> Result<(), String> {
        if !result.meets_targets(min_speed, min_memory) {
            return Err(format!(
                "Performance targets not met. Speed: {:.1}x (target: {:.1}x), Memory: {:.1}% (target: {:.1}%)",
                result.speed_improvement, min_speed,
                result.memory_reduction * 100.0, min_memory * 100.0
            ));
        }
        Ok(())
    }
    
    /// Assert that a value is within expected bounds
    pub fn within_bounds<T: PartialOrd + std::fmt::Display>(
        value: T,
        min: T,
        max: T
    ) -> Result<(), String> {
        if value < min || value > max {
            return Err(format!("Value {} is outside bounds [{}, {}]", value, min, max));
        }
        Ok(())
    }
}