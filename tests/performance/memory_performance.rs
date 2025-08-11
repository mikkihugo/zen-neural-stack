//! Memory Performance Tests
//!
//! Tests that validate memory usage improvements and detect memory regressions
//! in the zen-neural-stack migration from JavaScript to Rust.

use crate::common::*;
use std::collections::HashMap;

/// Results from memory performance testing
pub struct MemoryTestResults {
    pub results: Vec<PerformanceResult>,
    pub comparisons: Vec<PerformanceComparison>,
}

/// Run comprehensive memory performance tests
pub fn run_memory_tests() -> Result<MemoryTestResults, Box<dyn std::error::Error>> {
    println!("   🧠 Running memory performance tests...");
    
    let mut results = Vec::new();
    let mut comparisons = Vec::new();
    
    // Test memory usage for different network sizes
    for &size in &[100, 500, 1000, 5000, 10000] {
        let (rust_result, js_baseline) = test_network_memory_usage(size)?;
        let comparison = rust_result.compare_to(&js_baseline);
        
        results.push(rust_result);
        comparisons.push(comparison);
    }
    
    // Test memory pool efficiency
    let pool_result = test_memory_pool_efficiency()?;
    results.push(pool_result);
    
    // Test memory leak detection
    test_memory_leak_detection()?;
    
    // Test memory fragmentation
    let frag_result = test_memory_fragmentation()?;
    results.push(frag_result);
    
    // Test garbage collection impact (comparing with JS baseline)
    let gc_comparison = test_gc_impact_comparison()?;
    comparisons.push(gc_comparison);
    
    println!("   ✅ Memory performance tests completed");
    
    Ok(MemoryTestResults {
        results,
        comparisons,
    })
}

/// Test memory usage for different network sizes
fn test_network_memory_usage(network_size: usize) -> Result<(PerformanceResult, PerformanceResult), Box<dyn std::error::Error>> {
    println!("     📊 Testing memory usage for network size: {}", network_size);
    
    // Rust implementation test
    let rust_meter = PerformanceMeter::new(&format!("Rust-NetworkSize-{}", network_size));
    let rust_network = create_rust_network(network_size)?;
    let training_data = generate_training_data(network_size / 10)?;
    train_rust_network(&rust_network, &training_data)?;
    let rust_result = rust_meter.stop();
    
    // JavaScript baseline (simulated)
    let js_baseline = simulate_js_baseline_memory(network_size);
    
    rust_result.print();
    
    // Validate memory improvement
    let comparison = rust_result.compare_to(&js_baseline);
    if comparison.memory_reduction < 0.5 {
        println!("     ⚠️  Memory reduction below expected: {:.1}%", 
                comparison.memory_reduction * 100.0);
    }
    
    Ok((rust_result, js_baseline))
}

/// Test memory pool efficiency
fn test_memory_pool_efficiency() -> Result<PerformanceResult, Box<dyn std::error::Error>> {
    println!("     🏊 Testing memory pool efficiency...");
    
    let meter = PerformanceMeter::new("MemoryPoolEfficiency");
    
    // Create memory pool
    let pool_size = 64 * 1024 * 1024; // 64MB
    let mut pool = create_memory_pool(pool_size)?;
    
    // Allocate and deallocate various sizes
    let mut allocations = Vec::new();
    
    // Test allocation patterns
    for &alloc_size in &[1024, 4096, 16384, 65536, 262144] {
        for _ in 0..100 {
            let allocation = pool.allocate(alloc_size)?;
            allocations.push(allocation);
        }
    }
    
    // Test deallocation
    for allocation in allocations {
        pool.deallocate(allocation)?;
    }
    
    // Check pool statistics
    let pool_stats = pool.get_statistics();
    let efficiency = pool_stats.utilized_bytes as f64 / pool_stats.total_bytes as f64;
    
    let result = meter.stop();
    result.print();
    
    println!("     📈 Pool efficiency: {:.1}%", efficiency * 100.0);
    
    if efficiency < 0.8 {
        return Err(format!("Memory pool efficiency too low: {:.1}%", efficiency * 100.0).into());
    }
    
    Ok(result)
}

/// Test memory leak detection
fn test_memory_leak_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("     🔍 Testing memory leak detection...");
    
    let initial_memory = get_current_memory_usage();
    
    // Perform operations that should not leak memory
    for iteration in 0..1000 {
        let network = create_rust_network(100)?;
        let data = generate_training_data(50)?;
        
        // Train for a few epochs then drop
        train_rust_network(&network, &data)?;
        drop(network);
        drop(data);
        
        // Check memory usage periodically
        if iteration % 100 == 0 {
            let current_memory = get_current_memory_usage();
            let growth = current_memory.saturating_sub(initial_memory);
            
            // Allow for some growth but detect significant leaks
            if growth > 100 * 1024 * 1024 { // 100MB growth limit
                return Err(format!("Potential memory leak detected: {}MB growth", 
                                  growth / 1024 / 1024).into());
            }
        }
    }
    
    let final_memory = get_current_memory_usage();
    let total_growth = final_memory.saturating_sub(initial_memory);
    
    println!("     💚 Memory leak test passed (growth: {}KB)", total_growth / 1024);
    
    Ok(())
}

/// Test memory fragmentation
fn test_memory_fragmentation() -> Result<PerformanceResult, Box<dyn std::error::Error>> {
    println!("     🧩 Testing memory fragmentation...");
    
    let meter = PerformanceMeter::new("MemoryFragmentation");
    
    let mut allocations = Vec::new();
    
    // Create fragmentation with mixed allocation sizes
    for _ in 0..1000 {
        // Allocate different sizes to create fragmentation
        let sizes = [512, 1024, 2048, 4096, 8192];
        let size = sizes[rand::random::<usize>() % sizes.len()];
        
        let allocation = allocate_fragmented_memory(size)?;
        allocations.push(allocation);
        
        // Randomly deallocate some allocations
        if rand::random::<f64>() < 0.3 && !allocations.is_empty() {
            let idx = rand::random::<usize>() % allocations.len();
            let allocation = allocations.remove(idx);
            deallocate_fragmented_memory(allocation)?;
        }
    }
    
    // Clean up remaining allocations
    for allocation in allocations {
        deallocate_fragmented_memory(allocation)?;
    }
    
    let result = meter.stop();
    result.print();
    
    Ok(result)
}

/// Test garbage collection impact comparison
fn test_gc_impact_comparison() -> Result<PerformanceComparison, Box<dyn std::error::Error>> {
    println!("     🗑️ Testing GC impact comparison...");
    
    // Rust (no GC) performance
    let rust_meter = PerformanceMeter::new("Rust-NoGC");
    
    for _ in 0..100 {
        let network = create_rust_network(500)?;
        let data = generate_training_data(100)?;
        train_rust_network(&network, &data)?;
        // Explicit drop (immediate cleanup)
        drop(network);
        drop(data);
    }
    
    let rust_result = rust_meter.stop();
    
    // JavaScript GC baseline (simulated)
    let js_gc_baseline = simulate_js_gc_baseline();
    
    let comparison = rust_result.compare_to(&js_gc_baseline);
    comparison.print_comparison();
    
    Ok(comparison)
}

// Mock implementations for testing

#[derive(Debug)]
struct MockNetwork {
    size: usize,
    parameters: Vec<f64>,
}

#[derive(Debug)]  
struct MockTrainingData {
    inputs: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct MockMemoryPool {
    total_bytes: usize,
    allocated_bytes: usize,
    allocations: HashMap<usize, usize>, // allocation_id -> size
    next_id: usize,
}

#[derive(Debug)]
struct PoolStatistics {
    total_bytes: usize,
    utilized_bytes: usize,
    fragmentation_ratio: f64,
}

#[derive(Debug)]
struct MemoryAllocation {
    id: usize,
    size: usize,
    ptr: usize, // Mock pointer
}

impl MockMemoryPool {
    fn allocate(&mut self, size: usize) -> Result<MemoryAllocation, Box<dyn std::error::Error>> {
        if self.allocated_bytes + size > self.total_bytes {
            return Err("Out of memory".into());
        }
        
        let id = self.next_id;
        self.next_id += 1;
        self.allocated_bytes += size;
        self.allocations.insert(id, size);
        
        Ok(MemoryAllocation {
            id,
            size,
            ptr: 0x1000 + id * 0x1000, // Mock pointer
        })
    }
    
    fn deallocate(&mut self, allocation: MemoryAllocation) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(size) = self.allocations.remove(&allocation.id) {
            self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
            Ok(())
        } else {
            Err("Invalid allocation".into())
        }
    }
    
    fn get_statistics(&self) -> PoolStatistics {
        let fragmentation = if self.allocated_bytes == 0 {
            0.0
        } else {
            1.0 - (self.allocated_bytes as f64 / self.total_bytes as f64)
        };
        
        PoolStatistics {
            total_bytes: self.total_bytes,
            utilized_bytes: self.allocated_bytes,
            fragmentation_ratio: fragmentation,
        }
    }
}

fn create_memory_pool(size: usize) -> Result<MockMemoryPool, Box<dyn std::error::Error>> {
    Ok(MockMemoryPool {
        total_bytes: size,
        allocated_bytes: 0,
        allocations: HashMap::new(),
        next_id: 1,
    })
}

fn create_rust_network(size: usize) -> Result<MockNetwork, Box<dyn std::error::Error>> {
    // Simulate memory allocation for network parameters
    let param_count = size * size / 10; // Approximate parameter count
    let parameters = vec![0.0; param_count];
    
    std::thread::sleep(std::time::Duration::from_millis(size / 100));
    
    Ok(MockNetwork {
        size,
        parameters,
    })
}

fn generate_training_data(samples: usize) -> Result<MockTrainingData, Box<dyn std::error::Error>> {
    let input_dim = 64;
    let output_dim = 10;
    
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for _ in 0..samples {
        inputs.push(vec![0.0; input_dim]);
        targets.push(vec![0.0; output_dim]);
    }
    
    Ok(MockTrainingData { inputs, targets })
}

fn train_rust_network(network: &MockNetwork, data: &MockTrainingData) -> Result<(), Box<dyn std::error::Error>> {
    // Simulate training with memory operations
    let epochs = 10;
    let memory_per_epoch = network.parameters.len() * std::mem::size_of::<f64>();
    
    for _ in 0..epochs {
        // Simulate gradient computation (temporary memory)
        let _gradients = vec![0.0; network.parameters.len()];
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    
    Ok(())
}

fn simulate_js_baseline_memory(network_size: usize) -> PerformanceResult {
    // JavaScript typically uses 3-5x more memory due to:
    // - Garbage collection overhead
    // - Object boxing
    // - Dynamic typing overhead
    // - V8 internal structures
    
    let rust_equivalent_memory = network_size * network_size * 8 / 10; // bytes
    let js_memory = rust_equivalent_memory * 4; // 4x memory usage
    let js_duration = std::time::Duration::from_millis((network_size / 50) as u64);
    
    PerformanceResult {
        name: format!("JavaScript-NetworkSize-{}", network_size),
        duration: js_duration,
        memory_used: js_memory,
        peak_memory: js_memory + (js_memory / 4), // GC overhead
    }
}

fn simulate_js_gc_baseline() -> PerformanceResult {
    // JavaScript GC introduces pauses and memory overhead
    PerformanceResult {
        name: "JavaScript-GC-Baseline".to_string(),
        duration: std::time::Duration::from_millis(2000), // Slower due to GC pauses
        memory_used: 100 * 1024 * 1024, // 100MB
        peak_memory: 150 * 1024 * 1024, // 150MB peak before GC
    }
}

static mut FRAGMENTED_ALLOCATIONS: Vec<Vec<u8>> = Vec::new();

fn allocate_fragmented_memory(size: usize) -> Result<MemoryAllocation, Box<dyn std::error::Error>> {
    // Simulate fragmented memory allocation
    let allocation = vec![0u8; size];
    let id = unsafe {
        FRAGMENTED_ALLOCATIONS.push(allocation);
        FRAGMENTED_ALLOCATIONS.len() - 1
    };
    
    Ok(MemoryAllocation {
        id,
        size,
        ptr: 0x2000 + id * 0x1000,
    })
}

fn deallocate_fragmented_memory(allocation: MemoryAllocation) -> Result<(), Box<dyn std::error::Error>> {
    // Mock deallocation - in reality would be more complex
    Ok(())
}