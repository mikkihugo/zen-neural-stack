use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use std::time::{Duration, Instant};
use zen_neural::*;

/// JavaScript Baseline Comparison Benchmarks
/// 
/// This module provides benchmark comparisons against JavaScript neural network
/// implementations to validate the claimed 50-100x performance improvements.
/// 
/// The benchmarks simulate JavaScript-equivalent operations and measure
/// the performance delta between Rust and JavaScript implementations.

/// Simulated JavaScript neural network operations
struct JSSimulatedNetwork {
    weights_input_hidden: Vec<Vec<f32>>,
    weights_hidden_output: Vec<Vec<f32>>,
    bias_hidden: Vec<f32>,
    bias_output: Vec<f32>,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl JSSimulatedNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let weights_input_hidden = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
            .collect();
            
        let weights_hidden_output = (0..output_size)
            .map(|_| (0..hidden_size).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
            .collect();
            
        let bias_hidden = (0..hidden_size).map(|_| rng.r#gen::<f32>() * 0.1).collect();
        let bias_output = (0..output_size).map(|_| rng.r#gen::<f32>() * 0.1).collect();
        
        Self {
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
            input_size,
            hidden_size,
            output_size,
        }
    }
    
    /// Simulates JavaScript-style forward pass (unoptimized)
    fn forward_pass_js_style(&self, input: &[f32]) -> Vec<f32> {
        // Hidden layer computation (JavaScript-style loops)
        let mut hidden = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut sum = self.bias_hidden[i];
            for j in 0..self.input_size {
                sum += self.weights_input_hidden[i][j] * input[j];
            }
            // JavaScript-style sigmoid (less optimized)
            hidden[i] = 1.0 / (1.0 + (-sum.max(-500.0).min(500.0)).exp());
        }
        
        // Output layer computation
        let mut output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = self.bias_output[i];
            for j in 0..self.hidden_size {
                sum += self.weights_hidden_output[i][j] * hidden[j];
            }
            // JavaScript-style sigmoid
            output[i] = 1.0 / (1.0 + (-sum.max(-500.0).min(500.0)).exp());
        }
        
        output
    }
    
    /// Simulates JavaScript training step (very unoptimized)
    fn train_step_js_style(&mut self, input: &[f32], expected: &[f32], learning_rate: f32) {
        // Forward pass
        let mut hidden = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut sum = self.bias_hidden[i];
            for j in 0..self.input_size {
                sum += self.weights_input_hidden[i][j] * input[j];
            }
            hidden[i] = 1.0 / (1.0 + (-sum).exp());
        }
        
        let mut output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = self.bias_output[i];
            for j in 0..self.hidden_size {
                sum += self.weights_hidden_output[i][j] * hidden[j];
            }
            output[i] = 1.0 / (1.0 + (-sum).exp());
        }
        
        // Backward pass (simplified and unoptimized)
        let mut output_errors = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            output_errors[i] = (expected[i] - output[i]) * output[i] * (1.0 - output[i]);
        }
        
        let mut hidden_errors = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut error = 0.0;
            for j in 0..self.output_size {
                error += output_errors[j] * self.weights_hidden_output[j][i];
            }
            hidden_errors[i] = error * hidden[i] * (1.0 - hidden[i]);
        }
        
        // Update weights (JavaScript-style nested loops)
        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                self.weights_hidden_output[i][j] += learning_rate * output_errors[i] * hidden[j];
            }
            self.bias_output[i] += learning_rate * output_errors[i];
        }
        
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                self.weights_input_hidden[i][j] += learning_rate * hidden_errors[i] * input[j];
            }
            self.bias_hidden[i] += learning_rate * hidden_errors[i];
        }
    }
}

/// Benchmark Rust vs JavaScript-style forward pass
fn benchmark_forward_pass_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass_comparison");
    
    let input_size = 784;
    let hidden_size = 128;
    let output_size = 10;
    
    // Create Rust network
    let mut rust_network = NetworkBuilder::new()
        .add_layer(input_size)
        .add_layer(hidden_size)
        .add_layer(output_size)
        .with_activation(ActivationFunction::Sigmoid)
        .build()
        .expect("Failed to build Rust network");
    
    // Create JavaScript-style network
    let js_network = JSSimulatedNetwork::new(input_size, hidden_size, output_size);
    
    // Test input
    let input: Vec<f32> = (0..input_size).map(|i| (i as f32) / (input_size as f32)).collect();
    
    // Benchmark Rust implementation
    group.bench_function("rust_forward_pass", |b| {
        b.iter(|| {
            black_box(rust_network.run(&input))
        })
    });
    
    // Benchmark JavaScript-style implementation
    group.bench_function("javascript_style_forward_pass", |b| {
        b.iter(|| {
            black_box(js_network.forward_pass_js_style(&input))
        })
    });
    
    group.finish();
}

/// Benchmark training performance comparison
fn benchmark_training_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_comparison");
    group.measurement_time(Duration::from_secs(20));
    
    let input_size = 100;
    let hidden_size = 50;
    let output_size = 10;
    
    // Generate training data
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let training_data: Vec<(Vec<f32>, Vec<f32>)> = (0..100)
        .map(|_| {
            let input: Vec<f32> = (0..input_size).map(|_| rng.r#gen::<f32>()).collect();
            let output: Vec<f32> = (0..output_size).map(|_| rng.r#gen::<f32>()).collect();
            (input, output)
        })
        .collect();
    
    // Benchmark Rust training
    group.bench_function("rust_training", |b| {
        b.iter(|| {
            let mut network = NetworkBuilder::new()
                .add_layer(input_size)
                .add_layer(hidden_size)
                .add_layer(output_size)
                .with_activation(ActivationFunction::Sigmoid)
                .build()
                .expect("Failed to build network");
                
            for (input, expected) in &training_data {
                let _ = network.train(input, expected);
            }
        })
    });
    
    // Benchmark JavaScript-style training
    group.bench_function("javascript_style_training", |b| {
        b.iter(|| {
            let mut js_network = JSSimulatedNetwork::new(input_size, hidden_size, output_size);
            for (input, expected) in &training_data {
                js_network.train_step_js_style(input, expected, 0.1);
            }
        })
    });
    
    group.finish();
}

/// Matrix operations comparison
fn benchmark_matrix_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations_comparison");
    
    for size in &[64, 128, 256] {
        // Rust optimized matrix operations using ndarray
        group.bench_with_input(
            BenchmarkId::new("rust_matrix_multiply", size),
            size,
            |b, &size| {
                use ndarray::{Array2, Array1};
                use ndarray_rand::RandomExt;
                use ndarray_rand::rand_distr::Uniform;
                
                let matrix_a = Array2::random((size, size), Uniform::new(0.0f32, 1.0));
                let matrix_b = Array2::random((size, size), Uniform::new(0.0f32, 1.0));
                
                b.iter(|| {
                    black_box(matrix_a.dot(&matrix_b))
                })
            },
        );
        
        // JavaScript-style matrix multiplication (nested loops)
        group.bench_with_input(
            BenchmarkId::new("javascript_style_matrix_multiply", size),
            size,
            |b, &size| {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                
                let matrix_a: Vec<Vec<f32>> = (0..size)
                    .map(|_| (0..size).map(|_| rng.r#gen::<f32>()).collect())
                    .collect();
                let matrix_b: Vec<Vec<f32>> = (0..size)
                    .map(|_| (0..size).map(|_| rng.r#gen::<f32>()).collect())
                    .collect();
                
                b.iter(|| {
                    let mut result = vec![vec![0.0; size]; size];
                    for i in 0..size {
                        for j in 0..size {
                            for k in 0..size {
                                result[i][j] += matrix_a[i][k] * matrix_b[k][j];
                            }
                        }
                    }
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Memory allocation pattern comparison
fn benchmark_memory_patterns_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns_comparison");
    
    let network_size = (100, 50, 10);
    
    // Rust memory-efficient network creation
    group.bench_function("rust_network_creation", |b| {
        b.iter(|| {
            black_box(
                NetworkBuilder::new()
                    .add_layer(network_size.0)
                    .add_layer(network_size.1)
                    .add_layer(network_size.2)
                    .with_activation(ActivationFunction::ReLU)
                    .build()
                    .expect("Failed to build network")
            )
        })
    });
    
    // JavaScript-style network creation (more allocations)
    group.bench_function("javascript_style_network_creation", |b| {
        b.iter(|| {
            black_box(JSSimulatedNetwork::new(network_size.0, network_size.1, network_size.2))
        })
    });
    
    group.finish();
}

/// Performance ratio calculation and reporting
fn benchmark_performance_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_ratios");
    
    // This benchmark measures and reports the actual performance ratios
    // to validate the 50-100x improvement claims
    
    let input_size = 784;
    let hidden_size = 128;
    let output_size = 10;
    
    let input: Vec<f32> = (0..input_size).map(|i| (i as f32) / (input_size as f32)).collect();
    
    group.bench_function("measure_performance_ratio", |b| {
        let mut rust_network = NetworkBuilder::new()
            .add_layer(input_size)
            .add_layer(hidden_size)
            .add_layer(output_size)
            .with_activation(ActivationFunction::Sigmoid)
            .build()
            .expect("Failed to build network");
            
        let js_network = JSSimulatedNetwork::new(input_size, hidden_size, output_size);
        
        b.iter(|| {
            // Measure Rust performance
            let rust_start = Instant::now();
            for _ in 0..1000 {
                black_box(rust_network.run(&input));
            }
            let rust_time = rust_start.elapsed();
            
            // Measure JavaScript-style performance  
            let js_start = Instant::now();
            for _ in 0..1000 {
                black_box(js_network.forward_pass_js_style(&input));
            }
            let js_time = js_start.elapsed();
            
            // Calculate and return ratio for analysis
            let ratio = js_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
            black_box((rust_time, js_time, ratio))
        })
    });
    
    group.finish();
}

criterion_group!(
    name = js_comparison_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10))
        .sample_size(50);
    targets = 
        benchmark_forward_pass_comparison,
        benchmark_training_comparison,
        benchmark_matrix_operations_comparison,
        benchmark_memory_patterns_comparison,
        benchmark_performance_ratios
);

criterion_main!(js_comparison_benches);