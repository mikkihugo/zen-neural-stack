//! Performance validation and regression testing
//!
//! This module contains tests that validate the performance targets
//! for the zen-neural-stack migration and detect performance regressions.

use crate::common::*;
use crate::integration_test;
use std::collections::HashMap;

pub mod benchmark_suite;
pub mod memory_performance; 
pub mod training_speed;
pub mod inference_speed;
pub mod regression_detection;

/// Performance targets from Phase 1 specification
pub struct PerformanceTargets {
    /// Memory usage reduction target (70%)
    pub memory_reduction: f64,
    /// Training speed improvement target (20-50x)
    pub training_speed_min: f64,
    pub training_speed_max: f64,
    /// Inference speed improvement target (50-100x)  
    pub inference_speed_min: f64,
    pub inference_speed_max: f64,
    /// Maximum acceptable training time for standard datasets
    pub max_training_time_ms: u64,
    /// Maximum acceptable inference time per sample
    pub max_inference_time_ms: u64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            memory_reduction: 0.70,
            training_speed_min: 20.0,
            training_speed_max: 50.0,
            inference_speed_min: 50.0,
            inference_speed_max: 100.0,
            max_training_time_ms: 5000,
            max_inference_time_ms: 10,
        }
    }
}

/// Performance test results aggregator
#[derive(Debug, Default)]
pub struct PerformanceReport {
    pub memory_tests: Vec<PerformanceResult>,
    pub training_tests: Vec<PerformanceResult>,
    pub inference_tests: Vec<PerformanceResult>,
    pub comparisons: Vec<PerformanceComparison>,
    pub targets: PerformanceTargets,
}

impl PerformanceReport {
    pub fn new() -> Self {
        Self {
            targets: PerformanceTargets::default(),
            ..Default::default()
        }
    }
    
    pub fn add_memory_result(&mut self, result: PerformanceResult) {
        self.memory_tests.push(result);
    }
    
    pub fn add_training_result(&mut self, result: PerformanceResult) {
        self.training_tests.push(result);
    }
    
    pub fn add_inference_result(&mut self, result: PerformanceResult) {
        self.inference_tests.push(result);
    }
    
    pub fn add_comparison(&mut self, comparison: PerformanceComparison) {
        self.comparisons.push(comparison);
    }
    
    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("\n=== ZEN NEURAL STACK PERFORMANCE REPORT ===\n\n");
        
        // Memory performance summary
        report.push_str("📊 MEMORY PERFORMANCE:\n");
        let avg_memory_reduction = self.comparisons.iter()
            .map(|c| c.memory_reduction)
            .fold(0.0, |acc, x| acc + x) / self.comparisons.len().max(1) as f64;
        
        report.push_str(&format!("  Average Memory Reduction: {:.1}% (target: {:.1}%)\n", 
                                avg_memory_reduction * 100.0, 
                                self.targets.memory_reduction * 100.0));
        
        // Training speed summary
        report.push_str("\n⚡ TRAINING SPEED:\n");
        let avg_training_speedup = self.comparisons.iter()
            .filter(|c| c.current.name.contains("Training"))
            .map(|c| c.speed_improvement)
            .fold(0.0, |acc, x| acc + x) / self.comparisons.len().max(1) as f64;
        
        report.push_str(&format!("  Average Training Speedup: {:.1}x (target: {:.1}x-{:.1}x)\n",
                                avg_training_speedup,
                                self.targets.training_speed_min,
                                self.targets.training_speed_max));
        
        // Inference speed summary
        report.push_str("\n🚀 INFERENCE SPEED:\n");
        let avg_inference_speedup = self.comparisons.iter()
            .filter(|c| c.current.name.contains("Inference"))
            .map(|c| c.speed_improvement)
            .fold(0.0, |acc, x| acc + x) / self.comparisons.len().max(1) as f64;
        
        report.push_str(&format!("  Average Inference Speedup: {:.1}x (target: {:.1}x-{:.1}x)\n",
                                avg_inference_speedup,
                                self.targets.inference_speed_min,
                                self.targets.inference_speed_max));
        
        // Target achievement summary
        report.push_str("\n🎯 TARGET ACHIEVEMENT:\n");
        let memory_achieved = avg_memory_reduction >= self.targets.memory_reduction;
        let training_achieved = avg_training_speedup >= self.targets.training_speed_min;
        let inference_achieved = avg_inference_speedup >= self.targets.inference_speed_min;
        
        report.push_str(&format!("  ✅ Memory Reduction: {}\n", 
                                if memory_achieved { "ACHIEVED" } else { "MISSED" }));
        report.push_str(&format!("  ✅ Training Speed: {}\n",
                                if training_achieved { "ACHIEVED" } else { "MISSED" }));
        report.push_str(&format!("  ✅ Inference Speed: {}\n",
                                if inference_achieved { "ACHIEVED" } else { "MISSED" }));
        
        let overall_success = memory_achieved && training_achieved && inference_achieved;
        report.push_str(&format!("\n🏆 OVERALL: {}\n",
                                if overall_success { "ALL TARGETS ACHIEVED!" } else { "TARGETS MISSED" }));
        
        // Detailed results
        report.push_str("\n📈 DETAILED RESULTS:\n");
        for comparison in &self.comparisons {
            report.push_str(&format!("  {} - {:.1}x speed, {:.1}% memory reduction\n",
                                    comparison.current.name,
                                    comparison.speed_improvement,
                                    comparison.memory_reduction * 100.0));
        }
        
        report.push_str("\n============================================\n");
        report
    }
    
    /// Check if all performance targets are met
    pub fn targets_achieved(&self) -> bool {
        let avg_memory_reduction = self.comparisons.iter()
            .map(|c| c.memory_reduction)
            .fold(0.0, |acc, x| acc + x) / self.comparisons.len().max(1) as f64;
        
        let avg_training_speedup = self.comparisons.iter()
            .filter(|c| c.current.name.contains("Training"))
            .map(|c| c.speed_improvement)
            .fold(0.0, |acc, x| acc + x) / self.comparisons.len().max(1) as f64;
        
        let avg_inference_speedup = self.comparisons.iter()
            .filter(|c| c.current.name.contains("Inference"))
            .map(|c| c.speed_improvement)
            .fold(0.0, |acc, x| acc + x) / self.comparisons.len().max(1) as f64;
        
        avg_memory_reduction >= self.targets.memory_reduction &&
        avg_training_speedup >= self.targets.training_speed_min &&
        avg_inference_speedup >= self.targets.inference_speed_min
    }
}

/// Run all performance validation tests
pub fn run_all_performance_tests() -> Result<PerformanceReport, String> {
    println!("\n🚀 STARTING PERFORMANCE TESTS\n");
    
    let mut report = PerformanceReport::new();
    
    // Memory performance tests
    integration_test!("Memory Performance Validation", || {
        let results = memory_performance::run_memory_tests()?;
        for result in results.results {
            report.add_memory_result(result);
        }
        for comparison in results.comparisons {
            report.add_comparison(comparison);
        }
        Ok(())
    })?;
    
    // Training speed tests
    integration_test!("Training Speed Validation", || {
        let results = training_speed::run_training_speed_tests()?;
        for result in results.results {
            report.add_training_result(result);
        }
        for comparison in results.comparisons {
            report.add_comparison(comparison);
        }
        Ok(())
    })?;
    
    // Inference speed tests
    integration_test!("Inference Speed Validation", || {
        let results = inference_speed::run_inference_speed_tests()?;
        for result in results.results {
            report.add_inference_result(result);
        }
        for comparison in results.comparisons {
            report.add_comparison(comparison);
        }
        Ok(())
    })?;
    
    // Regression detection
    integration_test!("Performance Regression Detection", || {
        regression_detection::check_for_regressions(&report)?;
        Ok(())
    })?;
    
    println!("\n✅ PERFORMANCE TESTS COMPLETED\n");
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn run_performance_test_suite() {
        let report = run_all_performance_tests()
            .expect("Performance tests should complete successfully");
        
        // Print comprehensive report
        println!("{}", report.generate_report());
        
        // Assert that performance targets are achieved
        assert!(report.targets_achieved(), 
                "Performance targets not met - see report above");
        
        // Store results for regression detection
        store_performance_baseline(&report)
            .expect("Should be able to store performance baseline");
    }
}

/// Store performance results as baseline for future regression testing
fn store_performance_baseline(report: &PerformanceReport) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;
    
    let baseline_path = std::env::temp_dir().join("zen_neural_performance_baseline.json");
    let baseline_data = serde_json::to_string_pretty(&report)?;
    
    let mut file = File::create(baseline_path)?;
    file.write_all(baseline_data.as_bytes())?;
    
    println!("📊 Performance baseline stored for regression testing");
    Ok(())
}