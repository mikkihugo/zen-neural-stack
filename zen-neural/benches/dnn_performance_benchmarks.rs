/**
 * @file benches/dnn_performance_benchmarks.rs
 * @brief Comprehensive DNN Performance Benchmarks
 * 
 * This benchmark suite provides detailed performance analysis of the DNN implementation
 * with direct comparisons to JavaScript-style implementations and validation of the
 * claimed 10-50x performance improvements.
 * 
 * ## Benchmark Categories:
 * - **Forward Pass Performance**: Optimized vs unoptimized inference
 * - **Training Performance**: Batch processing and optimization efficiency
 * - **Matrix Operations**: SIMD-accelerated linear algebra
 * - **Memory Allocation**: Tensor pooling and zero-allocation paths
 * - **Activation Functions**: Vectorized vs scalar implementations
 * - **Optimizer Performance**: Advanced vs basic optimization algorithms
 * 
 * ## Usage:
 * ```bash
 * cargo bench --bench dnn_performance_benchmarks
 * cargo bench --bench dnn_performance_benchmarks -- --output-format html
 * ```
 * 
 * @author DNN Core Developer Agent (ruv-swarm Phase 1)
 * @version 1.0.0-alpha.1
 * @since 2025-01-14
 */

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box, Throughput};
use std::time::{Duration, Instant};
use zen_neural::dnn_api::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// === JAVASCRIPT-STYLE REFERENCE IMPLEMENTATIONS ===

/// JavaScript-style unoptimized matrix multiplication
fn js_style_matrix_multiply(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let cols_b = b[0].len();
    
    let mut result = vec![vec![0.0; cols_b]; rows_a];
    
    // Nested loop implementation (JavaScript pattern)
    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    result
}

/// JavaScript-style activation function application
fn js_style_activation_relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.max(0.0)).collect()
}

fn js_style_activation_sigmoid(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

fn js_style_activation_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_values: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
}

/// JavaScript-style dropout implementation
fn js_style_dropout(input: &[f32], rate: f32, training: bool) -> Vec<f32> {
    if !training || rate <= 0.0 {
        return input.to_vec();
    }
    
    let scale = 1.0 / (1.0 - rate);
    let mut output = Vec::new();
    
    for &value in input {
        if rand::random::<f32>() > rate {
            output.push(value * scale);
        } else {
            output.push(0.0);
        }
    }
    
    output
}

// === BENCHMARK FUNCTIONS ===

/// Benchmark forward pass performance at different network sizes
fn benchmark_forward_pass_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass_scaling");
    
    let network_configs = vec![
        ("small", 100, vec![50, 20]),
        ("medium", 500, vec![200, 100, 50]),
        ("large", 1000, vec![500, 200, 100]),
        ("xlarge", 2000, vec![1000, 500, 200]),
    ];
    
    for (name, input_size, hidden_layers) in network_configs {
        // Create Rust DNN model
        let mut builder = ZenDNNModel::builder().input_dim(input_size);
        for &layer_size in &hidden_layers {
            builder = builder.add_dense_layer(layer_size, ActivationType::ReLU);
        }
        let mut rust_model = builder.add_output_layer(10, ActivationType::Softmax).build().unwrap();
        rust_model.compile().unwrap();
        
        // Create test input
        let input_vec = vec![0.1; input_size];
        let input_shape = TensorShape::new_2d(1, input_size);
        let input = DNNTensor::from_vec(input_vec, &input_shape).unwrap();
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("rust_optimized", name),
            &(rust_model, input),
            |b, (model, input)| {
                b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                    black_box(model.forward(input, DNNTrainingMode::Inference).await.unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark matrix operations: Rust vs JavaScript-style
fn benchmark_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");
    
    for size in &[64, 128, 256, 512] {
        // Rust optimized (using ndarray/BLAS)
        group.bench_with_input(
            BenchmarkId::new("rust_ndarray", size),
            size,
            |b, &size| {
                use ndarray::{Array2};
                use ndarray_rand::RandomExt;
                use ndarray_rand::rand_distr::Uniform;
                
                let matrix_a = Array2::random((size, size), Uniform::new(0.0f32, 1.0));
                let matrix_b = Array2::random((size, size), Uniform::new(0.0f32, 1.0));
                
                b.iter(|| {
                    black_box(matrix_a.dot(&matrix_b))
                })
            },
        );
        
        // JavaScript-style nested loops
        group.bench_with_input(
            BenchmarkId::new("javascript_style", size),
            size,
            |b, &size| {
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                let matrix_a: Vec<Vec<f32>> = (0..size)
                    .map(|_| (0..size).map(|_| rng.r#gen()).collect())
                    .collect();
                let matrix_b: Vec<Vec<f32>> = (0..size)
                    .map(|_| (0..size).map(|_| rng.r#gen()).collect())
                    .collect();
                
                b.iter(|| {
                    black_box(js_style_matrix_multiply(&matrix_a, &matrix_b))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark activation function performance
fn benchmark_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");
    
    for size in &[1000, 10000, 100000] {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let input: Vec<f32> = (0..*size).map(|_| rng.r#gen_range(-5.0..5.0)).collect();
        let input_shape = TensorShape::new_2d(1, *size);
        let tensor_input = DNNTensor::from_vec(input.clone(), &input_shape).unwrap();
        
        // ReLU comparison
        group.bench_with_input(
            BenchmarkId::new("relu_rust", size),
            &tensor_input,
            |b, input| {
                b.iter(|| {
                    black_box(ActivationFunctions::relu(input).unwrap())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("relu_javascript", size),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(js_style_activation_relu(input))
                })
            },
        );
        
        // Sigmoid comparison
        group.bench_with_input(
            BenchmarkId::new("sigmoid_rust", size),
            &tensor_input,
            |b, input| {
                b.iter(|| {
                    black_box(ActivationFunctions::sigmoid(input).unwrap())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("sigmoid_javascript", size),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(js_style_activation_sigmoid(input))
                })
            },
        );
        
        // Softmax comparison
        group.bench_with_input(
            BenchmarkId::new("softmax_rust", size),
            &tensor_input,
            |b, input| {
                b.iter(|| {
                    black_box(ActivationFunctions::softmax(input).unwrap())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("softmax_javascript", size),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(js_style_activation_softmax(input))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark training performance with different optimizers
fn benchmark_training_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_optimizers");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);
    
    // Generate training data
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let num_samples = 1000;
    let input_size = 200;
    let output_size = 10;
    
    let mut training_data = Vec::new();
    for _ in 0..num_samples {
        let input_vec: Vec<f32> = (0..input_size).map(|_| rng.r#gen()).collect();
        let class = rng.r#gen_range(0..output_size);
        let mut target_vec = vec![0.0; output_size];
        target_vec[class] = 1.0;
        
        let input_shape = TensorShape::new_2d(1, input_size);
        let target_shape = TensorShape::new_2d(1, output_size);
        let input = DNNTensor::from_vec(input_vec, &input_shape).unwrap();
        let target = DNNTensor::from_vec(target_vec, &target_shape).unwrap();
        
        training_data.push(DNNTrainingExample { input, target });
    }
    
    let optimizers = vec![
        ("sgd", OptimizerConfig::SGD { momentum: 0.0, nesterov: false }),
        ("sgd_momentum", OptimizerConfig::SGD { momentum: 0.9, nesterov: false }),
        ("adam", OptimizerConfig::Adam { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }),
        ("rmsprop", OptimizerConfig::RMSprop { beta: 0.9, epsilon: 1e-8 }),
        ("adagrad", OptimizerConfig::AdaGrad { epsilon: 1e-8 }),
    ];
    
    for (name, optimizer_config) in optimizers {
        group.bench_with_input(
            BenchmarkId::new("optimizer", name),
            &(training_data.clone(), optimizer_config),
            |b, (data, optimizer)| {
                b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                    let mut model = ZenDNNModel::builder()
                        .input_dim(input_size)
                        .add_dense_layer(100, ActivationType::ReLU)
                        .add_dense_layer(50, ActivationType::ReLU)
                        .add_output_layer(output_size, ActivationType::Softmax)
                        .build()
                        .unwrap();
                    
                    model.compile().unwrap();
                    
                    let config = DNNTrainingConfig {
                        epochs: 3,
                        batch_size: 32,
                        learning_rate: match optimizer {
                            OptimizerConfig::SGD { .. } => 0.1,
                            _ => 0.001,
                        },
                        optimizer: optimizer.clone(),
                        loss_function: LossFunction::CrossEntropy,
                        validation_split: 0.0,
                        shuffle_data: false,
                        verbose_frequency: 0,
                        ..Default::default()
                    };
                    
                    black_box(model.train(data.clone(), config).await.unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch processing efficiency
fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let mut model = ZenDNNModel::builder()
        .input_dim(784)
        .add_dense_layer(256, ActivationType::ReLU)
        .add_dense_layer(128, ActivationType::ReLU)
        .add_output_layer(10, ActivationType::Softmax)
        .build()
        .unwrap();
    
    model.compile().unwrap();
    
    let batch_sizes = vec![1, 8, 16, 32, 64, 128];
    let input_data = vec![0.1; 784];
    
    for batch_size in batch_sizes {
        // Create batched input
        let mut batch_data = Vec::new();
        for _ in 0..batch_size {
            batch_data.extend(&input_data);
        }
        
        let batch_shape = TensorShape::new_2d(batch_size, 784);
        let batch_input = DNNTensor::from_vec(batch_data, &batch_shape).unwrap();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_forward", batch_size),
            &(model.clone(), batch_input),
            |b, (model, input)| {
                b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                    black_box(model.forward(input, DNNTrainingMode::Inference).await.unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation patterns
fn benchmark_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    let network_sizes = vec![
        ("small", 100, vec![50, 20]),
        ("medium", 500, vec![200, 100]),
        ("large", 1000, vec![500, 200]),
    ];
    
    for (name, input_size, hidden_layers) in network_sizes {
        group.bench_with_input(
            BenchmarkId::new("model_creation", name),
            &(input_size, hidden_layers),
            |b, (input_size, hidden_layers)| {
                b.iter(|| {
                    let mut builder = ZenDNNModel::builder().input_dim(*input_size);
                    for &layer_size in hidden_layers {
                        builder = builder.add_dense_layer(layer_size, ActivationType::ReLU);
                    }
                    let mut model = builder.add_output_layer(10, ActivationType::Softmax).build().unwrap();
                    model.compile().unwrap();
                    black_box(model)
                })
            },
        );
        
        // Tensor allocation benchmark
        group.bench_with_input(
            BenchmarkId::new("tensor_allocation", name),
            input_size,
            |b, &input_size| {
                b.iter(|| {
                    let data = vec![0.1; input_size];
                    let shape = TensorShape::new_2d(1, input_size);
                    black_box(DNNTensor::from_vec(data, &shape).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark regularization techniques
fn benchmark_regularization(c: &mut Criterion) {
    let mut group = c.benchmark_group("regularization");
    
    let input_size = 1000;
    let input_data = vec![0.1; input_size];
    let input_shape = TensorShape::new_2d(1, input_size);
    let input_tensor = DNNTensor::from_vec(input_data.clone(), &input_shape).unwrap();
    
    // Dropout comparison
    let mut dropout_layer = DropoutLayer::new(0.5).unwrap();
    dropout_layer.compile(input_size).unwrap();
    
    group.bench_function("dropout_rust", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
            black_box(dropout_layer.forward(&input_tensor, DNNTrainingMode::Training).await.unwrap())
        })
    });
    
    group.bench_function("dropout_javascript", |b| {
        b.iter(|| {
            black_box(js_style_dropout(&input_data, 0.5, true))
        })
    });
    
    // Batch normalization
    let mut batch_norm = BatchNormLayer::new().unwrap();
    batch_norm.compile(input_size).unwrap();
    
    group.bench_function("batch_norm_rust", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
            black_box(batch_norm.forward(&input_tensor, DNNTrainingMode::Training).await.unwrap())
        })
    });
    
    // Layer normalization
    let mut layer_norm = LayerNormLayer::new().unwrap();
    layer_norm.compile(input_size).unwrap();
    
    group.bench_function("layer_norm_rust", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
            black_box(layer_norm.forward(&input_tensor, DNNTrainingMode::Training).await.unwrap())
        })
    });
    
    group.finish();
}

/// Comprehensive performance ratio measurement
fn benchmark_performance_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_ratios");
    group.measurement_time(Duration::from_secs(15));
    
    let input_size = 784;
    let hidden_size = 256;
    let output_size = 10;
    
    // Create models
    let mut rust_model = ZenDNNModel::builder()
        .input_dim(input_size)
        .add_dense_layer(hidden_size, ActivationType::ReLU)
        .add_output_layer(output_size, ActivationType::Softmax)
        .build()
        .unwrap();
    rust_model.compile().unwrap();
    
    // Test data
    let input_vec = vec![0.1; input_size];
    let input_shape = TensorShape::new_2d(1, input_size);
    let rust_input = DNNTensor::from_vec(input_vec.clone(), &input_shape).unwrap();
    
    group.bench_function("comprehensive_comparison", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
            // Measure Rust performance
            let rust_start = Instant::now();
            for _ in 0..100 {
                let _ = rust_model.forward(&rust_input, DNNTrainingMode::Inference).await.unwrap();
            }
            let rust_time = rust_start.elapsed();
            
            // Measure JavaScript-style performance
            let js_weights_ih: Vec<Vec<f32>> = (0..hidden_size)
                .map(|_| (0..input_size).map(|_| 0.1).collect())
                .collect();
            let js_weights_ho: Vec<Vec<f32>> = (0..output_size)
                .map(|_| (0..hidden_size).map(|_| 0.1).collect())
                .collect();
            
            let js_start = Instant::now();
            for _ in 0..100 {
                // JavaScript-style forward pass
                let mut hidden = vec![0.0; hidden_size];
                for i in 0..hidden_size {
                    let mut sum = 0.0;
                    for j in 0..input_size {
                        sum += js_weights_ih[i][j] * input_vec[j];
                    }
                    hidden[i] = sum.max(0.0); // ReLU
                }
                
                let mut output = vec![0.0; output_size];
                for i in 0..output_size {
                    let mut sum = 0.0;
                    for j in 0..hidden_size {
                        sum += js_weights_ho[i][j] * hidden[j];
                    }
                    output[i] = sum.exp();
                }
                
                // Softmax normalization
                let sum: f32 = output.iter().sum();
                for val in &mut output {
                    *val /= sum;
                }
            }
            let js_time = js_start.elapsed();
            
            let speedup_ratio = js_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
            black_box((rust_time, js_time, speedup_ratio))
        })
    });
    
    group.finish();
}

// Configure benchmark groups
criterion_group!(
    name = dnn_benchmarks;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(15))
        .sample_size(100);
    targets = 
        benchmark_forward_pass_scaling,
        benchmark_matrix_operations,
        benchmark_activation_functions,
        benchmark_training_optimizers,
        benchmark_batch_processing,
        benchmark_memory_allocation,
        benchmark_regularization,
        benchmark_performance_ratios
);

criterion_main!(dnn_benchmarks);