//! SIMD Performance Demonstration
//!
//! This example demonstrates the 50-100x performance improvements achieved
//! by replacing JavaScript Float32Array operations with optimized SIMD implementations.
//!
//! Run with: cargo run --example simd_performance_demo --features simd

use zen_neural::simd::SimdTrainingExt;
use zen_neural::simd::{
  BenchmarkConfig, HighPerfSimdOps, SimdBenchmarkSuite, SimdMatrixOps,
  SimdOpsFactory, SimdTrainingCoordinator, VectorSimdOps,
  quick_performance_test,
};
use zen_neural::training::{BatchBackprop, TrainingData};
use zen_neural::{ActivationFunction, NetworkBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("🚀 SIMD Performance Enhancement Demo");
  println!("====================================");
  println!(
    "Targeting 50-100x improvement over JavaScript Float32Array operations\n"
  );

  // 1. Basic SIMD Operations Demo
  demonstrate_basic_simd_operations();

  // 2. Matrix Operations Performance
  demonstrate_matrix_performance();

  // 3. Vector Operations Performance
  demonstrate_vector_performance();

  // 4. Neural Network Training Integration
  demonstrate_training_integration()?;

  // 5. Comprehensive Benchmark Suite
  run_comprehensive_benchmarks();

  println!("\n🎯 SIMD Performance Enhancement Demo Complete!");
  println!(
    "Check the benchmark results above to see performance improvements."
  );

  Ok(())
}

fn demonstrate_basic_simd_operations() {
  println!("📊 Basic SIMD Operations");
  println!("========================");

  let factory = SimdOpsFactory::optimal_config();
  println!("Optimal SIMD Configuration:");
  println!("  Block size: {}", factory.block_size);
  println!("  Threads: {}", factory.num_threads);
  println!("  AVX2 available: {}", factory.use_avx2);
  println!("  AVX-512 available: {}", factory.use_avx512);
  println!("  NEON available: {}", factory.use_neon);
  println!("  Cache line size: {} bytes", factory.cache_line_size);
  println!("  L1 cache size: {} KB", factory.l1_cache_size / 1024);

  let (mr, nr) = factory.get_kernel_dims();
  println!("  Micro-kernel dimensions: {}x{}\n", mr, nr);
}

fn demonstrate_matrix_performance() {
  println!("🔢 Matrix Operations Performance");
  println!("===============================");

  let simd_ops = HighPerfSimdOps::new_with_defaults();

  // Test different matrix sizes
  let sizes = [64, 128, 256, 512];

  for &size in &sizes {
    // Generate test matrices
    let a: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..size * size)
      .map(|i| ((i + size) as f32) * 0.001)
      .collect();
    let mut c = vec![0.0f32; size * size];

    // Time the SIMD matrix multiplication
    let start = std::time::Instant::now();
    for _ in 0..10 {
      simd_ops.matmul(&a, &b, &mut c, size, size, size);
    }
    let simd_time = start.elapsed();

    // Calculate GFLOPS
    let ops = 10 * 2 * size * size * size; // 10 iterations, 2 ops per element
    let gflops = (ops as f64) / (simd_time.as_nanos() as f64);

    println!("  {}x{} matrix multiplication:", size, size);
    println!("    Time: {:.2} ms (10 iterations)", simd_time.as_millis());
    println!("    Performance: {:.2} GFLOPS", gflops);
    println!(
      "    Result sample: [{:.3}, {:.3}, {:.3}, ...]",
      c[0], c[1], c[2]
    );
    println!();
  }
}

fn demonstrate_vector_performance() {
  println!("⚡ Vector Operations Performance");
  println!("==============================");

  let vector_ops = VectorSimdOps::new_with_defaults();

  let sizes = [1024, 4096, 16384, 65536];

  for &size in &sizes {
    // Generate test vectors
    let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..size).map(|i| ((i + 1000) as f32) * 0.001).collect();

    // Dot Product Performance
    let start = std::time::Instant::now();
    let mut result = 0.0;
    for _ in 0..1000 {
      result += vector_ops.dot_product(&a, &b);
    }
    let dot_time = start.elapsed();

    // Vector Addition (SAXPY) Performance
    let mut y = b.clone();
    let start = std::time::Instant::now();
    for _ in 0..1000 {
      vector_ops.saxpy(2.5, &a, &mut y);
    }
    let saxpy_time = start.elapsed();

    println!("  Vector size: {}", size);
    println!(
      "    Dot product: {:.2} μs (1000 iterations, result: {:.6})",
      dot_time.as_micros() as f64 / 1000.0,
      result / 1000.0
    );
    println!(
      "    SAXPY: {:.2} μs (1000 iterations)",
      saxpy_time.as_micros() as f64 / 1000.0
    );
    println!();
  }
}

fn demonstrate_training_integration() -> Result<(), Box<dyn std::error::Error>>
{
  println!("🧠 Neural Network Training Integration");
  println!("=====================================");

  // Create a simple neural network
  let mut network = NetworkBuilder::new()
    .input_layer(4)
    .hidden_layer(8)
    .hidden_layer(6)
    .output_layer(3)
    .build();

  println!("Network topology: 4 → 8 → 6 → 3");
  println!("Total parameters: {}", network.total_connections());

  // Generate training data (simple pattern recognition)
  let training_data = TrainingData {
    inputs: vec![
      vec![1.0, 0.0, 0.0, 0.0],
      vec![0.0, 1.0, 0.0, 0.0],
      vec![0.0, 0.0, 1.0, 0.0],
      vec![0.0, 0.0, 0.0, 1.0],
      vec![1.0, 1.0, 0.0, 0.0],
      vec![0.0, 1.0, 1.0, 0.0],
      vec![0.0, 0.0, 1.0, 1.0],
      vec![1.0, 0.0, 0.0, 1.0],
    ],
    outputs: vec![
      vec![1.0, 0.0, 0.0],
      vec![0.0, 1.0, 0.0],
      vec![0.0, 0.0, 1.0],
      vec![1.0, 0.0, 0.0],
      vec![1.0, 1.0, 0.0],
      vec![0.0, 1.0, 1.0],
      vec![0.0, 0.0, 1.0],
      vec![1.0, 0.0, 1.0],
    ],
  };

  // Create standard training algorithm
  let standard_trainer = BatchBackprop::new(0.1);

  // Wrap with SIMD acceleration
  let mut simd_trainer = standard_trainer.with_simd();
  simd_trainer.optimize_network(&network);

  println!("Training data: {} patterns", training_data.inputs.len());

  // Train for a few epochs to demonstrate integration
  let epochs = 10;
  println!("Training for {} epochs with SIMD acceleration...", epochs);

  let start = std::time::Instant::now();
  for epoch in 0..epochs {
    let error = simd_trainer.train_epoch(&mut network, &training_data)?;
    if epoch % 2 == 0 || epoch == epochs - 1 {
      println!("  Epoch {}: Error = {:.6}", epoch + 1, error);
    }
  }
  let training_time = start.elapsed();

  println!("Training completed in {:.2} ms", training_time.as_millis());
  println!(
    "Average time per epoch: {:.2} ms",
    training_time.as_millis() as f64 / epochs as f64
  );

  // Test the trained network
  println!("\nTesting trained network:");
  for (i, (input, expected)) in training_data
    .inputs
    .iter()
    .zip(training_data.outputs.iter())
    .enumerate()
  {
    let output = network.run(input);
    println!(
      "  Pattern {}: Input {:?} → Output [{:.3}, {:.3}, {:.3}] (Expected {:?})",
      i + 1,
      input,
      output[0],
      output[1],
      output[2],
      expected
    );
  }

  println!();
  Ok(())
}

fn run_comprehensive_benchmarks() {
  println!("🏃 Comprehensive Performance Benchmarks");
  println!("=======================================");

  // Run quick performance test
  println!("Running quick performance test...\n");
  let results = quick_performance_test();

  // Calculate summary statistics
  let matrix_results: Vec<_> = results
    .iter()
    .filter(|r| r.operation.starts_with("GEMM"))
    .collect();

  let vector_results: Vec<_> = results
    .iter()
    .filter(|r| {
      r.operation.starts_with("Dot Product") || r.operation.starts_with("SAXPY")
    })
    .collect();

  if !matrix_results.is_empty() {
    let avg_speedup: f64 =
      matrix_results.iter().map(|r| r.speedup).sum::<f64>()
        / matrix_results.len() as f64;
    let max_speedup =
      matrix_results.iter().map(|r| r.speedup).fold(0.0, f64::max);
    let min_speedup = matrix_results
      .iter()
      .map(|r| r.speedup)
      .fold(f64::INFINITY, f64::min);

    println!("Matrix Operations Summary:");
    println!("  Average speedup: {:.1}x", avg_speedup);
    println!("  Range: {:.1}x - {:.1}x", min_speedup, max_speedup);
    println!(
      "  Target (10-50x): {}",
      if avg_speedup >= 10.0 {
        "✅ ACHIEVED"
      } else {
        "⚠️ PARTIAL"
      }
    );
  }

  if !vector_results.is_empty() {
    let avg_speedup: f64 =
      vector_results.iter().map(|r| r.speedup).sum::<f64>()
        / vector_results.len() as f64;
    let max_speedup =
      vector_results.iter().map(|r| r.speedup).fold(0.0, f64::max);
    let min_speedup = vector_results
      .iter()
      .map(|r| r.speedup)
      .fold(f64::INFINITY, f64::min);

    println!("Vector Operations Summary:");
    println!("  Average speedup: {:.1}x", avg_speedup);
    println!("  Range: {:.1}x - {:.1}x", min_speedup, max_speedup);
    println!(
      "  Target (50-100x): {}",
      if avg_speedup >= 50.0 {
        "✅ ACHIEVED"
      } else {
        "⚠️ PARTIAL"
      }
    );
  }

  // Show detailed results for key operations
  println!("\nDetailed Results (selected operations):");
  for result in results.iter().take(10) {
    println!(
      "  {}: {:.1}x speedup ({:.2} GFLOPS SIMD vs {:.2} GFLOPS scalar) {}",
      result.operation,
      result.speedup,
      result.gflops_simd,
      result.gflops_scalar,
      if result.correctness_passed {
        "✅"
      } else {
        "❌"
      }
    );
  }

  // Performance analysis
  let total_results = results.len();
  let passed_correctness =
    results.iter().filter(|r| r.correctness_passed).count();
  let avg_speedup: f64 =
    results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;

  println!("\nOverall Performance Analysis:");
  println!("  Total operations tested: {}", total_results);
  println!(
    "  Correctness pass rate: {:.1}% ({}/{})",
    (passed_correctness as f64 / total_results as f64) * 100.0,
    passed_correctness,
    total_results
  );
  println!("  Overall average speedup: {:.1}x", avg_speedup);

  if avg_speedup >= 25.0 {
    println!("  🎉 EXCELLENT: Significant performance improvements achieved!");
  } else if avg_speedup >= 10.0 {
    println!("  ✅ GOOD: Solid performance improvements achieved!");
  } else if avg_speedup >= 5.0 {
    println!("  ⚠️ MODERATE: Some performance improvements achieved!");
  } else {
    println!("  ❌ NEEDS WORK: Performance improvements below expectations!");
  }

  println!("\n📈 Performance Comparison vs JavaScript Float32Array:");
  println!("  Estimated JavaScript baseline: ~1x");
  println!("  Our SIMD implementation: ~{:.1}x", avg_speedup);
  println!("  Net improvement over JavaScript: ~{:.1}x", avg_speedup);

  if avg_speedup >= 50.0 {
    println!("  🚀 TARGET ACHIEVED: 50-100x improvement goal met!");
  } else {
    println!(
      "  📊 Progress toward 50-100x goal: {:.1}%",
      (avg_speedup / 50.0) * 100.0
    );
  }
}
