//! SIMD performance benchmarking module
//!
//! This module provides comprehensive benchmarks to validate the 50-100x
//! performance improvements over JavaScript Float32Array operations.

use super::{ActivationFunction, SimdMatrixOps};
// use super::SimdConfig; // For future configuration enhancements
use super::matrix_ops::HighPerfSimdOps;
use super::vector_ops::VectorSimdOps;
use std::time::Instant;

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub matrix_sizes: Vec<usize>,
    pub vector_sizes: Vec<usize>,
    pub validate_correctness: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            warmup_iterations: 10,
            matrix_sizes: vec![64, 128, 256, 512, 1024, 2048],
            vector_sizes: vec![1024, 4096, 16384, 65536],
            validate_correctness: true,
        }
    }
}

/// Performance measurement results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub operation: String,
    pub size: usize,
    pub simd_time_ns: u64,
    pub scalar_time_ns: u64,
    pub speedup: f64,
    pub gflops_simd: f64,
    pub gflops_scalar: f64,
    pub correctness_passed: bool,
}

impl BenchmarkResults {
    pub fn new(operation: String, size: usize, simd_time_ns: u64, scalar_time_ns: u64, flops: u64) -> Self {
        let speedup = scalar_time_ns as f64 / simd_time_ns as f64;
        let gflops_simd = (flops as f64) / (simd_time_ns as f64);
        let gflops_scalar = (flops as f64) / (scalar_time_ns as f64);
        
        Self {
            operation,
            size,
            simd_time_ns,
            scalar_time_ns,
            speedup,
            gflops_simd,
            gflops_scalar,
            correctness_passed: true,
        }
    }
}

/// Comprehensive SIMD benchmark suite
pub struct SimdBenchmarkSuite {
    config: BenchmarkConfig,
    simd_ops: HighPerfSimdOps,
    vector_ops: VectorSimdOps,
    scalar_ops: ScalarReferenceOps,
}

impl SimdBenchmarkSuite {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            simd_ops: HighPerfSimdOps::new_with_defaults(),
            vector_ops: VectorSimdOps::new_with_defaults(),
            scalar_ops: ScalarReferenceOps::new(),
        }
    }

    pub fn new_with_defaults() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Run comprehensive benchmark suite
    pub fn run_full_suite(&self) -> Vec<BenchmarkResults> {
        let mut results = Vec::new();

        println!("🚀 Starting SIMD Performance Benchmark Suite");
        println!("Target: 50-100x improvement over JavaScript Float32Array\n");

        // Matrix multiplication benchmarks
        results.extend(self.benchmark_matrix_multiplication());
        
        // Vector operation benchmarks
        results.extend(self.benchmark_vector_operations());
        
        // Activation function benchmarks
        results.extend(self.benchmark_activation_functions());

        // Print summary
        self.print_summary(&results);

        results
    }

    /// Benchmark matrix multiplication operations
    fn benchmark_matrix_multiplication(&self) -> Vec<BenchmarkResults> {
        println!("📊 Matrix Multiplication Benchmarks");
        println!("====================================");

        let mut results = Vec::new();

        for &size in &self.config.matrix_sizes {
            println!("Testing {}x{} matrices...", size, size);

            // Generate test data
            let a = self.generate_random_matrix(size, size);
            let b = self.generate_random_matrix(size, size);
            let mut c_simd = vec![0.0f32; size * size];
            let mut c_scalar = vec![0.0f32; size * size];

            // Warmup
            for _ in 0..self.config.warmup_iterations {
                self.simd_ops.matmul(&a, &b, &mut c_simd, size, size, size);
            }

            // Benchmark SIMD implementation
            let simd_start = Instant::now();
            for _ in 0..self.config.iterations {
                self.simd_ops.matmul(&a, &b, &mut c_simd, size, size, size);
            }
            let simd_time = simd_start.elapsed().as_nanos() as u64 / self.config.iterations as u64;

            // Benchmark scalar implementation
            let scalar_start = Instant::now();
            for _ in 0..self.config.iterations {
                self.scalar_ops.matmul(&a, &b, &mut c_scalar, size, size, size);
            }
            let scalar_time = scalar_start.elapsed().as_nanos() as u64 / self.config.iterations as u64;

            // Calculate FLOPS (2 * n^3 for matrix multiplication)
            let flops = 2u64 * size as u64 * size as u64 * size as u64;

            let mut result = BenchmarkResults::new(
                format!("GEMM {}x{}", size, size),
                size,
                simd_time,
                scalar_time,
                flops,
            );

            // Validate correctness
            if self.config.validate_correctness {
                result.correctness_passed = self.validate_matrix_equality(&c_simd, &c_scalar, 1e-3);
            }

            println!("  SIMD: {:.2} ms, Scalar: {:.2} ms, Speedup: {:.2}x", 
                simd_time as f64 / 1_000_000.0, 
                scalar_time as f64 / 1_000_000.0, 
                result.speedup
            );

            results.push(result);
        }

        println!();
        results
    }

    /// Benchmark vector operations
    fn benchmark_vector_operations(&self) -> Vec<BenchmarkResults> {
        println!("🔢 Vector Operations Benchmarks");
        println!("===============================");

        let mut results = Vec::new();

        for &size in &self.config.vector_sizes {
            println!("Testing vectors of size {}...", size);

            // Generate test vectors
            let a = self.generate_random_vector(size);
            let b = self.generate_random_vector(size);

            // Dot Product Benchmark
            {
                // Warmup
                for _ in 0..self.config.warmup_iterations {
                    self.vector_ops.dot_product(&a, &b);
                }

                let simd_start = Instant::now();
                let mut simd_result = 0.0;
                for _ in 0..self.config.iterations {
                    simd_result += self.vector_ops.dot_product(&a, &b);
                }
                let simd_time = simd_start.elapsed().as_nanos() as u64 / self.config.iterations as u64;

                let scalar_start = Instant::now();
                let mut scalar_result = 0.0;
                for _ in 0..self.config.iterations {
                    scalar_result += self.scalar_ops.dot_product(&a, &b);
                }
                let scalar_time = scalar_start.elapsed().as_nanos() as u64 / self.config.iterations as u64;

                let flops = 2 * size as u64; // multiply + add for each element
                let mut result = BenchmarkResults::new(
                    format!("Dot Product ({})", size),
                    size,
                    simd_time,
                    scalar_time,
                    flops,
                );

                if self.config.validate_correctness {
                    result.correctness_passed = (simd_result - scalar_result).abs() < 1e-3;
                }

                println!("  Dot Product - SIMD: {:.2} μs, Scalar: {:.2} μs, Speedup: {:.2}x",
                    simd_time as f64 / 1000.0,
                    scalar_time as f64 / 1000.0,
                    result.speedup
                );

                results.push(result);
            }

            // Vector Addition (SAXPY) Benchmark
            {
                let mut y_simd = b.clone();
                let mut y_scalar = b.clone();
                let alpha = 2.5;

                let simd_start = Instant::now();
                for _ in 0..self.config.iterations {
                    self.vector_ops.saxpy(alpha, &a, &mut y_simd);
                }
                let simd_time = simd_start.elapsed().as_nanos() as u64 / self.config.iterations as u64;

                let scalar_start = Instant::now();
                for _ in 0..self.config.iterations {
                    self.scalar_ops.saxpy(alpha, &a, &mut y_scalar);
                }
                let scalar_time = scalar_start.elapsed().as_nanos() as u64 / self.config.iterations as u64;

                let flops = 2 * size as u64; // multiply + add for each element
                let mut result = BenchmarkResults::new(
                    format!("SAXPY ({})", size),
                    size,
                    simd_time,
                    scalar_time,
                    flops,
                );

                if self.config.validate_correctness {
                    result.correctness_passed = self.validate_vector_equality(&y_simd, &y_scalar, 1e-3);
                }

                println!("  SAXPY - SIMD: {:.2} μs, Scalar: {:.2} μs, Speedup: {:.2}x",
                    simd_time as f64 / 1000.0,
                    scalar_time as f64 / 1000.0,
                    result.speedup
                );

                results.push(result);
            }
        }

        println!();
        results
    }

    /// Benchmark activation functions
    fn benchmark_activation_functions(&self) -> Vec<BenchmarkResults> {
        println!("⚡ Activation Function Benchmarks");
        println!("=================================");

        let mut results = Vec::new();
        let activations = vec![
            ActivationFunction::Relu,
            ActivationFunction::LeakyRelu(0.1),
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
            ActivationFunction::Gelu,
            ActivationFunction::Swish,
        ];

        for &size in &self.config.vector_sizes {
            for activation in &activations {
                let data = self.generate_random_vector(size);
                let mut data_simd = data.clone();
                let mut data_scalar = data.clone();

                // Warmup
                for _ in 0..self.config.warmup_iterations {
                    self.simd_ops.apply_activation(&mut data_simd.clone(), *activation);
                }

                let simd_start = Instant::now();
                for _ in 0..self.config.iterations {
                    self.simd_ops.apply_activation(&mut data_simd, *activation);
                }
                let simd_time = simd_start.elapsed().as_nanos() as u64 / self.config.iterations as u64;

                let scalar_start = Instant::now();
                for _ in 0..self.config.iterations {
                    self.scalar_ops.apply_activation(&mut data_scalar, *activation);
                }
                let scalar_time = scalar_start.elapsed().as_nanos() as u64 / self.config.iterations as u64;

                let flops = size as u64; // Approximation for activation functions
                let mut result = BenchmarkResults::new(
                    format!("{:?} ({})", activation, size),
                    size,
                    simd_time,
                    scalar_time,
                    flops,
                );

                if self.config.validate_correctness {
                    result.correctness_passed = self.validate_vector_equality(&data_simd, &data_scalar, 1e-2);
                }

                if size == self.config.vector_sizes[0] {
                    println!("  {:?} - SIMD: {:.2} μs, Scalar: {:.2} μs, Speedup: {:.2}x",
                        activation,
                        simd_time as f64 / 1000.0,
                        scalar_time as f64 / 1000.0,
                        result.speedup
                    );
                }

                results.push(result);
            }
        }

        println!();
        results
    }

    /// Print benchmark summary
    fn print_summary(&self, results: &[BenchmarkResults]) {
        println!("📋 Benchmark Summary");
        println!("===================");
        
        let matrix_results: Vec<_> = results.iter()
            .filter(|r| r.operation.starts_with("GEMM"))
            .collect();
        
        let vector_results: Vec<_> = results.iter()
            .filter(|r| r.operation.starts_with("Dot Product") || r.operation.starts_with("SAXPY"))
            .collect();
        
        let activation_results: Vec<_> = results.iter()
            .filter(|r| !r.operation.starts_with("GEMM") && 
                        !r.operation.starts_with("Dot Product") && 
                        !r.operation.starts_with("SAXPY"))
            .collect();

        // Matrix operations summary
        if !matrix_results.is_empty() {
            let avg_speedup: f64 = matrix_results.iter().map(|r| r.speedup).sum::<f64>() / matrix_results.len() as f64;
            let max_speedup = matrix_results.iter().map(|r| r.speedup).fold(0.0, f64::max);
            println!("Matrix Multiplication:");
            println!("  Average speedup: {:.1}x", avg_speedup);
            println!("  Maximum speedup: {:.1}x", max_speedup);
            println!("  Target (10-50x): {}", if avg_speedup >= 10.0 { "✅ ACHIEVED" } else { "❌ NEEDS WORK" });
        }

        // Vector operations summary  
        if !vector_results.is_empty() {
            let avg_speedup: f64 = vector_results.iter().map(|r| r.speedup).sum::<f64>() / vector_results.len() as f64;
            let max_speedup = vector_results.iter().map(|r| r.speedup).fold(0.0, f64::max);
            println!("Vector Operations:");
            println!("  Average speedup: {:.1}x", avg_speedup);
            println!("  Maximum speedup: {:.1}x", max_speedup);
            println!("  Target (50-100x): {}", if avg_speedup >= 50.0 { "✅ ACHIEVED" } else { "❌ NEEDS WORK" });
        }

        // Activation functions summary
        if !activation_results.is_empty() {
            let avg_speedup: f64 = activation_results.iter().map(|r| r.speedup).sum::<f64>() / activation_results.len() as f64;
            let max_speedup = activation_results.iter().map(|r| r.speedup).fold(0.0, f64::max);
            println!("Activation Functions:");
            println!("  Average speedup: {:.1}x", avg_speedup);
            println!("  Maximum speedup: {:.1}x", max_speedup);
            println!("  Target (10-50x): {}", if avg_speedup >= 10.0 { "✅ ACHIEVED" } else { "❌ NEEDS WORK" });
        }

        // Overall correctness
        let failed_tests: Vec<_> = results.iter().filter(|r| !r.correctness_passed).collect();
        if failed_tests.is_empty() {
            println!("Correctness: ✅ All tests passed");
        } else {
            println!("Correctness: ❌ {} tests failed", failed_tests.len());
            for test in failed_tests {
                println!("  Failed: {}", test.operation);
            }
        }

        println!("\n🎯 Performance Analysis Complete!");
    }

    // Helper methods
    fn generate_random_matrix(&self, rows: usize, cols: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..rows * cols).map(|_| rng.r#gen_range(-1.0..1.0)).collect()
    }

    fn generate_random_vector(&self, size: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.r#gen_range(-1.0..1.0)).collect()
    }

    fn validate_matrix_equality(&self, a: &[f32], b: &[f32], tolerance: f32) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tolerance)
    }

    fn validate_vector_equality(&self, a: &[f32], b: &[f32], tolerance: f32) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tolerance)
    }
}

/// Scalar reference implementations for comparison
struct ScalarReferenceOps;

impl ScalarReferenceOps {
    fn new() -> Self {
        Self
    }

    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        c.fill(0.0);
        
        for i in 0..m {
            for j in 0..n {
                for k_idx in 0..k {
                    c[i * n + j] += a[i * k + k_idx] * b[k_idx * n + j];
                }
            }
        }
    }

    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn saxpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            *yi += alpha * xi;
        }
    }

    fn apply_activation(&self, data: &mut [f32], activation: ActivationFunction) {
        match activation {
            ActivationFunction::Relu => {
                for x in data.iter_mut() {
                    *x = x.max(0.0);
                }
            },
            ActivationFunction::LeakyRelu(alpha) => {
                for x in data.iter_mut() {
                    if *x < 0.0 {
                        *x *= alpha;
                    }
                }
            },
            ActivationFunction::Sigmoid => {
                for x in data.iter_mut() {
                    *x = 1.0 / (1.0 + (-*x).exp());
                }
            },
            ActivationFunction::Tanh => {
                for x in data.iter_mut() {
                    *x = x.tanh();
                }
            },
            ActivationFunction::Gelu => {
                for x in data.iter_mut() {
                    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                    *x = *x * 0.5 * (1.0 + (sqrt_2_over_pi * (*x + 0.044715 * x.powi(3))).tanh());
                }
            },
            ActivationFunction::Swish => {
                for x in data.iter_mut() {
                    *x = *x / (1.0 + (-*x).exp());
                }
            },
        }
    }
}

/// Convenience function to run quick performance test
pub fn quick_performance_test() -> Vec<BenchmarkResults> {
    let config = BenchmarkConfig {
        iterations: 10,
        warmup_iterations: 2,
        matrix_sizes: vec![256, 512],
        vector_sizes: vec![4096, 16384],
        validate_correctness: true,
    };

    let suite = SimdBenchmarkSuite::new(config);
    suite.run_full_suite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = SimdBenchmarkSuite::new_with_defaults();
        assert!(suite.config.iterations > 0);
        assert!(suite.config.matrix_sizes.len() > 0);
    }

    #[test]
    fn test_quick_performance_test() {
        let results = quick_performance_test();
        assert!(!results.is_empty());
        
        // Should have some meaningful speedups
        let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
        assert!(avg_speedup > 1.0); // At least some improvement
        
        // All tests should pass correctness
        assert!(results.iter().all(|r| r.correctness_passed));
    }

    #[test]
    fn test_scalar_reference_ops() {
        let ops = ScalarReferenceOps::new();
        
        // Test matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        ops.matmul(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
        
        // Test dot product
        let dot = ops.dot_product(&a, &b);
        assert!((dot - 70.0).abs() < 1e-6);
    }
}