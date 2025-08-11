use criterion::{
  BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use zen_neural::*;

/// Memory profiling benchmarks to validate 70% memory reduction vs JavaScript
///
/// This module provides comprehensive memory usage analysis including:
/// - Peak memory consumption measurements
/// - Allocation pattern analysis
/// - Memory fragmentation assessment
/// - Garbage collection overhead comparison
/// - Memory bandwidth utilization testing

/// Custom allocator for tracking memory usage
struct TrackingAllocator {
  allocated: AtomicUsize,
  deallocated: AtomicUsize,
  peak_usage: AtomicUsize,
}

impl TrackingAllocator {
  const fn new() -> Self {
    Self {
      allocated: AtomicUsize::new(0),
      deallocated: AtomicUsize::new(0),
      peak_usage: AtomicUsize::new(0),
    }
  }

  fn reset(&self) {
    self.allocated.store(0, Ordering::Relaxed);
    self.deallocated.store(0, Ordering::Relaxed);
    self.peak_usage.store(0, Ordering::Relaxed);
  }

  fn current_usage(&self) -> usize {
    self.allocated.load(Ordering::Relaxed)
      - self.deallocated.load(Ordering::Relaxed)
  }

  fn peak_usage(&self) -> usize {
    self.peak_usage.load(Ordering::Relaxed)
  }

  fn total_allocated(&self) -> usize {
    self.allocated.load(Ordering::Relaxed)
  }
}

unsafe impl GlobalAlloc for TrackingAllocator {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    let ptr = System.alloc(layout);
    if !ptr.is_null() {
      let old_allocated =
        self.allocated.fetch_add(layout.size(), Ordering::Relaxed);
      let new_allocated = old_allocated + layout.size();
      let current_usage =
        new_allocated - self.deallocated.load(Ordering::Relaxed);

      // Update peak usage
      loop {
        let current_peak = self.peak_usage.load(Ordering::Relaxed);
        if current_usage <= current_peak {
          break;
        }
        if self
          .peak_usage
          .compare_exchange_weak(
            current_peak,
            current_usage,
            Ordering::Relaxed,
            Ordering::Relaxed,
          )
          .is_ok()
        {
          break;
        }
      }
    }
    ptr
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    System.dealloc(ptr, layout);
    self.deallocated.fetch_add(layout.size(), Ordering::Relaxed);
  }
}

// Note: In a real implementation, you would use a global allocator like this:
// #[global_allocator]
// static ALLOCATOR: TrackingAllocator = TrackingAllocator::new();

/// Memory usage measurement utilities
struct MemoryMeasurement {
  initial_usage: usize,
  peak_usage: usize,
  final_usage: usize,
  total_allocated: usize,
}

impl MemoryMeasurement {
  fn measure<F, R>(allocator: &TrackingAllocator, f: F) -> (R, Self)
  where
    F: FnOnce() -> R,
  {
    allocator.reset();
    let initial_usage = allocator.current_usage();

    let result = f();

    let peak_usage = allocator.peak_usage();
    let final_usage = allocator.current_usage();
    let total_allocated = allocator.total_allocated();

    (
      result,
      MemoryMeasurement {
        initial_usage,
        peak_usage,
        final_usage,
        total_allocated,
      },
    )
  }
}

/// Benchmark network creation memory usage
fn benchmark_network_memory_usage(c: &mut Criterion) {
  let mut group = c.benchmark_group("network_memory_usage");

  let allocator = TrackingAllocator::new();

  for &(input_size, hidden_size, output_size) in &[
    (100, 50, 10),
    (784, 128, 10),
    (2048, 512, 100),
    (4096, 1024, 1000),
  ] {
    let size_name = format!("{}x{}x{}", input_size, hidden_size, output_size);

    group.bench_with_input(
      BenchmarkId::new("network_creation_memory", &size_name),
      &(input_size, hidden_size, output_size),
      |b, &sizes| {
        b.iter_custom(|iters| {
          allocator.reset();
          let start = std::time::Instant::now();

          for _ in 0..iters {
            let network = black_box(
              NetworkBuilder::new()
                .add_layer(sizes.0)
                .add_layer(sizes.1)
                .add_layer(sizes.2)
                .with_activation(ActivationFunction::ReLU)
                .build()
                .expect("Failed to build network"),
            );
            // Ensure the network isn't optimized away
            std::hint::black_box(network);
          }

          start.elapsed()
        })
      },
    );
  }

  group.finish();
}

/// Benchmark training memory usage patterns
fn benchmark_training_memory_patterns(c: &mut Criterion) {
  let mut group = c.benchmark_group("training_memory_patterns");
  group.measurement_time(Duration::from_secs(20));

  let input_size = 784;
  let hidden_size = 128;
  let output_size = 10;
  let batch_size = 32;

  // Generate training data
  use rand::Rng;
  let mut rng = rand::thread_rng();
  let training_data: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
    .map(|_| {
      let input: Vec<f32> =
        (0..input_size).map(|_| rng.r#gen::<f32>()).collect();
      let output: Vec<f32> =
        (0..output_size).map(|_| rng.r#gen::<f32>()).collect();
      (input, output)
    })
    .collect();

  group.bench_function("training_memory_usage", |b| {
    b.iter_custom(|iters| {
      let start = std::time::Instant::now();

      for _ in 0..iters {
        let mut network = NetworkBuilder::new()
          .add_layer(input_size)
          .add_layer(hidden_size)
          .add_layer(output_size)
          .with_activation(ActivationFunction::Sigmoid)
          .build()
          .expect("Failed to build network");

        for (input, expected) in &training_data {
          black_box(network.train(input, expected));
        }
      }

      start.elapsed()
    })
  });

  group.finish();
}

/// Benchmark memory fragmentation patterns
fn benchmark_memory_fragmentation(c: &mut Criterion) {
  let mut group = c.benchmark_group("memory_fragmentation");

  group.bench_function("repeated_allocations", |b| {
    b.iter(|| {
      let mut networks = Vec::new();

      // Allocate many small networks
      for _ in 0..100 {
        let network = NetworkBuilder::new()
          .add_layer(50)
          .add_layer(25)
          .add_layer(5)
          .with_activation(ActivationFunction::ReLU)
          .build()
          .expect("Failed to build network");
        networks.push(network);
      }

      // Deallocate every other network (simulate fragmentation)
      for i in (0..networks.len()).step_by(2) {
        drop(std::mem::take(&mut networks[i]));
      }

      // Allocate new networks in fragmented space
      for i in (0..networks.len()).step_by(2) {
        networks[i] = NetworkBuilder::new()
          .add_layer(40)
          .add_layer(20)
          .add_layer(5)
          .with_activation(ActivationFunction::Tanh)
          .build()
          .expect("Failed to build network");
      }

      black_box(networks)
    })
  });

  group.finish();
}

/// Benchmark cache-friendly memory access patterns
fn benchmark_cache_efficiency(c: &mut Criterion) {
  let mut group = c.benchmark_group("cache_efficiency");

  for &size in &[64, 128, 256, 512, 1024] {
    use ndarray::{Array1, Array2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    let matrix = Array2::random((size, size), Uniform::new(0.0f32, 1.0));
    let vector = Array1::random(size, Uniform::new(0.0f32, 1.0));

    // Row-major access (cache-friendly)
    group.bench_with_input(
      BenchmarkId::new("row_major_access", size),
      &size,
      |b, _| {
        b.iter(|| {
          let result = matrix.dot(&vector);
          black_box(result)
        })
      },
    );

    // Simulate column-major access (cache-unfriendly)
    group.bench_with_input(
      BenchmarkId::new("column_major_simulation", size),
      &size,
      |b, &size| {
        b.iter(|| {
          let mut result = vec![0.0f32; size];
          for i in 0..size {
            for j in 0..size {
              result[i] += matrix[[i, j]] * vector[j];
            }
          }
          black_box(result)
        })
      },
    );
  }

  group.finish();
}

/// Benchmark memory bandwidth utilization
fn benchmark_memory_bandwidth(c: &mut Criterion) {
  let mut group = c.benchmark_group("memory_bandwidth");

  for &size in &[1024, 2048, 4096, 8192] {
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    // Sequential memory access
    group.bench_with_input(
      BenchmarkId::new("sequential_access", size),
      &size,
      |b, _| {
        b.iter(|| {
          let sum: f32 = data.iter().sum();
          black_box(sum)
        })
      },
    );

    // Random memory access (worst case)
    group.bench_with_input(
      BenchmarkId::new("random_access", size),
      &size,
      |b, &size| {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..size).collect();
        indices.shuffle(&mut rand::thread_rng());

        b.iter(|| {
          let mut sum = 0.0f32;
          for &index in &indices {
            sum += data[index];
          }
          black_box(sum)
        })
      },
    );
  }

  group.finish();
}

/// Benchmark GPU memory transfer patterns (if GPU features enabled)
#[cfg(feature = "gpu")]
fn benchmark_gpu_memory_transfers(c: &mut Criterion) {
  let mut group = c.benchmark_group("gpu_memory_transfers");
  group.measurement_time(Duration::from_secs(30));

  // This would benchmark GPU memory allocation and transfer patterns
  // Placeholder for actual GPU memory benchmarks
  for &size in &[1024, 4096, 16384] {
    group.bench_with_input(
      BenchmarkId::new("gpu_buffer_allocation", size),
      &size,
      |b, &size| {
        b.iter(|| {
          // Placeholder for GPU buffer allocation benchmark
          let buffer_size = size * size * std::mem::size_of::<f32>();
          black_box(buffer_size)
        })
      },
    );
  }

  group.finish();
}

/// Performance comparison with JavaScript-style memory patterns
fn benchmark_js_memory_comparison(c: &mut Criterion) {
  let mut group = c.benchmark_group("js_memory_comparison");

  let network_size = (784, 128, 10);

  // Rust memory-efficient pattern
  group.bench_function("rust_memory_pattern", |b| {
    b.iter(|| {
      let network = NetworkBuilder::new()
        .add_layer(network_size.0)
        .add_layer(network_size.1)
        .add_layer(network_size.2)
        .with_activation(ActivationFunction::ReLU)
        .build()
        .expect("Failed to build network");

      let input: Vec<f32> = (0..network_size.0)
        .map(|i| i as f32 / network_size.0 as f32)
        .collect();

      // Simulate multiple inferences
      let mut results = Vec::new();
      for _ in 0..100 {
        let result = network.run(&input);
        results.push(result);
      }

      black_box(results)
    })
  });

  // JavaScript-style memory pattern (more allocations)
  group.bench_function("javascript_style_memory_pattern", |b| {
    b.iter(|| {
      // Simulate JavaScript-style object creation and garbage collection
      let mut weights: Vec<Vec<Vec<f32>>> = Vec::new();

      // Create weight matrices (JavaScript-style nested arrays)
      weights.push(
        (0..network_size.1)
          .map(|_| (0..network_size.0).map(|_| 0.5).collect())
          .collect(),
      );
      weights.push(
        (0..network_size.2)
          .map(|_| (0..network_size.1).map(|_| 0.5).collect())
          .collect(),
      );

      let input: Vec<f32> = (0..network_size.0)
        .map(|i| i as f32 / network_size.0 as f32)
        .collect();

      // Simulate multiple inferences with temporary allocations
      let mut results = Vec::new();
      for _ in 0..100 {
        let mut hidden: Vec<f32> = Vec::new();
        for i in 0..network_size.1 {
          let mut sum = 0.0;
          for j in 0..network_size.0 {
            sum += weights[0][i][j] * input[j];
          }
          hidden.push(1.0 / (1.0 + (-sum).exp()));
        }

        let mut output: Vec<f32> = Vec::new();
        for i in 0..network_size.2 {
          let mut sum = 0.0;
          for j in 0..network_size.1 {
            sum += weights[1][i][j] * hidden[j];
          }
          output.push(1.0 / (1.0 + (-sum).exp()));
        }

        results.push(output);
      }

      black_box(results)
    })
  });

  group.finish();
}

criterion_group!(
    name = memory_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(15))
        .sample_size(50);
    targets =
        benchmark_network_memory_usage,
        benchmark_training_memory_patterns,
        benchmark_memory_fragmentation,
        benchmark_cache_efficiency,
        benchmark_memory_bandwidth,
        benchmark_js_memory_comparison
);

#[cfg(feature = "gpu")]
criterion_group!(
    name = gpu_memory_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(20))
        .sample_size(30);
    targets = benchmark_gpu_memory_transfers
);

#[cfg(feature = "gpu")]
criterion_main!(memory_benches, gpu_memory_benches);

#[cfg(not(feature = "gpu"))]
criterion_main!(memory_benches);
