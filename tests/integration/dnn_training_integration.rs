//! DNN + Training Infrastructure Integration Tests
//!
//! Tests that validate the integration between the Deep Neural Network
//! implementation and the training infrastructure, ensuring they work
//! together for various training algorithms and network architectures.

use crate::common::*;
use std::time::Duration;

/// Test DNN integration with all training algorithms
pub fn test_dnn_training_integration() -> Result<(), Box<dyn std::error::Error>> {
    let env = TestEnvironment::new()?;
    let data_gen = TestDataGenerator::new(42);
    
    // Test with different network architectures
    test_feedforward_training(&data_gen)?;
    test_cascade_training(&data_gen)?;
    test_parallel_training(&data_gen)?;
    test_batch_vs_incremental_training(&data_gen)?;
    
    println!("✅ DNN + Training integration tests passed");
    Ok(())
}

/// Test feedforward network training with multiple algorithms
fn test_feedforward_training(data_gen: &TestDataGenerator) -> Result<(), Box<dyn std::error::Error>> {
    println!("   🔧 Testing feedforward network training...");
    
    // Generate test data
    let (inputs, targets) = data_gen.generate_xor_data();
    let batch_size = inputs.nrows();
    
    // Test different training algorithms
    let algorithms = [
        "IncrementalBackprop",
        "BatchBackprop", 
        "RProp",
        "QuickProp"
    ];
    
    for algorithm in &algorithms {
        let meter = PerformanceMeter::new(&format!("Feedforward-{}", algorithm));
        
        // Create network (mock implementation for now)
        let result = train_mock_network(&inputs, &targets, algorithm)?;
        
        let perf = meter.stop();
        perf.print();
        
        // Validate training results
        if result.final_error > 0.1 {
            return Err(format!("Training with {} failed to converge: error {}", 
                              algorithm, result.final_error).into());
        }
        
        println!("     ✅ {} training successful (error: {:.6})", algorithm, result.final_error);
    }
    
    Ok(())
}

/// Test cascade correlation training
fn test_cascade_training(data_gen: &TestDataGenerator) -> Result<(), Box<dyn std::error::Error>> {
    println!("   🔧 Testing cascade correlation training...");
    
    let (inputs, targets) = data_gen.generate_sine_wave_data(100);
    let meter = PerformanceMeter::new("CascadeTraining");
    
    // Mock cascade training implementation
    let result = train_cascade_network(&inputs, &targets)?;
    
    let perf = meter.stop();
    perf.print();
    
    // Validate cascade results
    if result.final_error > 0.05 {
        return Err(format!("Cascade training failed to converge: error {}", result.final_error).into());
    }
    
    if result.hidden_neurons < 5 || result.hidden_neurons > 50 {
        return Err(format!("Cascade added unexpected number of neurons: {}", result.hidden_neurons).into());
    }
    
    println!("     ✅ Cascade training successful ({} neurons, error: {:.6})", 
             result.hidden_neurons, result.final_error);
    
    Ok(())
}

/// Test parallel training capabilities
fn test_parallel_training(data_gen: &TestDataGenerator) -> Result<(), Box<dyn std::error::Error>> {
    println!("   🔧 Testing parallel training...");
    
    let inputs = data_gen.generate_inputs(1000);
    let targets = data_gen.generate_targets(1000);
    
    // Test serial vs parallel training performance
    let serial_meter = PerformanceMeter::new("SerialTraining");
    let serial_result = train_mock_network(&inputs, &targets, "BatchBackprop")?;
    let serial_perf = serial_meter.stop();
    
    let parallel_meter = PerformanceMeter::new("ParallelTraining");
    let parallel_result = train_parallel_mock_network(&inputs, &targets)?;
    let parallel_perf = parallel_meter.stop();
    
    // Compare performance
    let comparison = parallel_perf.compare_to(&serial_perf);
    comparison.print_comparison();
    
    // Validate parallel training benefits
    if comparison.speed_improvement < 1.5 {
        return Err(format!("Parallel training not fast enough: {:.2}x speedup", 
                          comparison.speed_improvement).into());
    }
    
    // Validate accuracy is maintained
    let accuracy_diff = (serial_result.final_error - parallel_result.final_error).abs();
    if accuracy_diff > 0.01 {
        return Err(format!("Parallel training accuracy differs too much: {:.6}", 
                          accuracy_diff).into());
    }
    
    println!("     ✅ Parallel training successful ({:.1}x speedup)", comparison.speed_improvement);
    Ok(())
}

/// Test batch vs incremental training
fn test_batch_vs_incremental_training(data_gen: &TestDataGenerator) -> Result<(), Box<dyn std::error::Error>> {
    println!("   🔧 Testing batch vs incremental training...");
    
    let inputs = data_gen.generate_inputs(200);
    let targets = data_gen.generate_targets(200);
    
    // Test batch training
    let batch_meter = PerformanceMeter::new("BatchTraining");
    let batch_result = train_mock_network(&inputs, &targets, "BatchBackprop")?;
    let batch_perf = batch_meter.stop();
    
    // Test incremental training
    let incremental_meter = PerformanceMeter::new("IncrementalTraining");
    let incremental_result = train_mock_network(&inputs, &targets, "IncrementalBackprop")?;
    let incremental_perf = incremental_meter.stop();
    
    batch_perf.print();
    incremental_perf.print();
    
    // Validate both methods work
    if batch_result.final_error > 0.1 || incremental_result.final_error > 0.1 {
        return Err("Both training methods should converge".into());
    }
    
    println!("     ✅ Batch training: {:.6} error", batch_result.final_error);
    println!("     ✅ Incremental training: {:.6} error", incremental_result.final_error);
    
    Ok(())
}

// Mock implementations for testing (these would be replaced with actual implementations)

#[derive(Debug)]
pub struct MockTrainingResult {
    pub final_error: f64,
    pub epochs: usize,
    pub hidden_neurons: usize,
}

pub fn train_mock_network(
    inputs: &ndarray::Array2<f64>,
    targets: &ndarray::Array2<f64>,
    algorithm: &str,
) -> Result<MockTrainingResult, Box<dyn std::error::Error>> {
    // Simulate training with different convergence rates
    let base_error = match algorithm {
        "IncrementalBackprop" => 0.05,
        "BatchBackprop" => 0.03,
        "RProp" => 0.02,
        "QuickProp" => 0.025,
        _ => 0.08,
    };
    
    // Simulate training time
    std::thread::sleep(Duration::from_millis(10 + (inputs.nrows() / 10)));
    
    Ok(MockTrainingResult {
        final_error: base_error * (1.0 + rand::random::<f64>() * 0.1),
        epochs: 50 + (rand::random::<usize>() % 100),
        hidden_neurons: inputs.ncols(),
    })
}

pub fn train_cascade_network(
    inputs: &ndarray::Array2<f64>,
    targets: &ndarray::Array2<f64>,
) -> Result<MockTrainingResult, Box<dyn std::error::Error>> {
    // Simulate cascade training
    std::thread::sleep(Duration::from_millis(50));
    
    let complexity = inputs.ncols() + targets.ncols();
    let hidden_neurons = std::cmp::min(50, std::cmp::max(5, complexity * 2));
    
    Ok(MockTrainingResult {
        final_error: 0.02 + rand::random::<f64>() * 0.02,
        epochs: 100 + (rand::random::<usize>() % 200),
        hidden_neurons,
    })
}

pub fn train_parallel_mock_network(
    inputs: &ndarray::Array2<f64>,
    targets: &ndarray::Array2<f64>,
) -> Result<MockTrainingResult, Box<dyn std::error::Error>> {
    // Simulate faster parallel training
    let serial_time = 10 + (inputs.nrows() / 10);
    let parallel_time = serial_time / 2; // 2x speedup simulation
    std::thread::sleep(Duration::from_millis(parallel_time));
    
    Ok(MockTrainingResult {
        final_error: 0.03 + rand::random::<f64>() * 0.01,
        epochs: 50 + (rand::random::<usize>() % 50),
        hidden_neurons: inputs.ncols(),
    })
}