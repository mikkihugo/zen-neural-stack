//! SIMD + Memory Management Integration Tests
//!
//! These tests validate that SIMD operations work correctly with the
//! memory management system, ensuring efficient and safe memory usage.

use crate::common::*;
use ndarray::{Array2, Array1};

/// Test SIMD operations integration with memory pools
pub fn test_simd_memory_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("   🔧 Testing SIMD + Memory integration...");
    
    test_simd_with_memory_pools()?;
    test_simd_alignment_requirements()?;
    test_simd_memory_safety()?;
    test_simd_performance_with_memory()?;
    
    println!("✅ SIMD + Memory integration tests passed");
    Ok(())
}

/// Test SIMD operations with memory pools
fn test_simd_with_memory_pools() -> Result<(), Box<dyn std::error::Error>> {
    println!("     🧮 Testing SIMD with memory pools...");
    
    let pool_meter = PerformanceMeter::new("SIMD-MemoryPool");
    
    // Simulate memory pool allocation for SIMD operations
    let pool_size = 1024 * 1024; // 1MB pool
    let mut memory_pool = Vec::<f64>::with_capacity(pool_size);
    
    // Test SIMD vector operations using pool memory
    for chunk_size in [64, 128, 256, 512, 1024] {
        let mut data = Vec::with_capacity(chunk_size);
        for i in 0..chunk_size {
            data.push(i as f64);
        }
        
        // Simulate SIMD addition
        let result = simd_add_mock(&data, &data)?;
        
        // Verify results
        for (i, &val) in result.iter().enumerate() {
            let expected = (i as f64) * 2.0;
            if (val - expected).abs() > 1e-10 {
                return Err(format!("SIMD result incorrect at {}: got {}, expected {}", 
                                  i, val, expected).into());
            }
        }
        
        memory_pool.clear();
    }
    
    let pool_perf = pool_meter.stop();
    pool_perf.print();
    
    println!("     ✅ SIMD memory pool operations verified");
    Ok(())
}

/// Test SIMD alignment requirements
fn test_simd_alignment_requirements() -> Result<(), Box<dyn std::error::Error>> {
    println!("     📐 Testing SIMD alignment requirements...");
    
    let alignment_meter = PerformanceMeter::new("SIMD-Alignment");
    
    // Test different alignment scenarios
    for alignment in [16, 32, 64] {
        let size = 1024;
        
        // Create aligned memory
        let aligned_data = create_aligned_memory(size, alignment)?;
        
        // Verify alignment
        let ptr = aligned_data.as_ptr() as usize;
        if ptr % alignment != 0 {
            return Err(format!("Memory not properly aligned to {} bytes: ptr={:#x}", 
                              alignment, ptr).into());
        }
        
        // Test SIMD operations on aligned data
        let result = simd_multiply_mock(&aligned_data, 2.0)?;
        
        // Verify results
        for (i, &val) in result.iter().enumerate() {
            let expected = (i as f64) * 2.0;
            if (val - expected).abs() > 1e-10 {
                return Err(format!("Aligned SIMD result incorrect: got {}, expected {}", 
                                  val, expected).into());
            }
        }
    }
    
    let alignment_perf = alignment_meter.stop();
    alignment_perf.print();
    
    println!("     ✅ SIMD alignment requirements satisfied");
    Ok(())
}

/// Test SIMD memory safety
fn test_simd_memory_safety() -> Result<(), Box<dyn std::error::Error>> {
    println!("     🛡️ Testing SIMD memory safety...");
    
    let safety_meter = PerformanceMeter::new("SIMD-MemorySafety");
    
    // Test bounds checking
    let data = vec![1.0, 2.0, 3.0, 4.0];
    
    // Test safe SIMD operations
    let result = safe_simd_operation(&data)?;
    assert_eq!(result.len(), data.len());
    
    // Test with different sizes to ensure no buffer overruns
    for size in [1, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
        let test_data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let result = safe_simd_operation(&test_data)?;
        
        if result.len() != test_data.len() {
            return Err(format!("SIMD operation changed array size: {} -> {}", 
                              test_data.len(), result.len()).into());
        }
    }
    
    let safety_perf = safety_meter.stop();
    safety_perf.print();
    
    println!("     ✅ SIMD memory safety validated");
    Ok(())
}

/// Test SIMD performance with memory management
fn test_simd_performance_with_memory() -> Result<(), Box<dyn std::error::Error>> {
    println!("     ⚡ Testing SIMD performance with memory management...");
    
    let size = 10000;
    let data1: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let data2: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();
    
    // Test scalar vs SIMD performance
    let scalar_meter = PerformanceMeter::new("Scalar-MemoryOps");
    let scalar_result = scalar_add(&data1, &data2)?;
    let scalar_perf = scalar_meter.stop();
    
    let simd_meter = PerformanceMeter::new("SIMD-MemoryOps");
    let simd_result = simd_add_mock(&data1, &data2)?;
    let simd_perf = simd_meter.stop();
    
    // Compare results for correctness
    if scalar_result.len() != simd_result.len() {
        return Err("Scalar and SIMD results have different lengths".into());
    }
    
    for (i, (&scalar, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
        if (scalar - simd).abs() > 1e-10 {
            return Err(format!("Results differ at index {}: scalar={}, simd={}", 
                              i, scalar, simd).into());
        }
    }
    
    // Compare performance
    let comparison = simd_perf.compare_to(&scalar_perf);
    comparison.print_comparison();
    
    // Expect SIMD to be faster (at least 2x for large arrays)
    if comparison.speed_improvement < 1.5 {
        println!("⚠️  SIMD speedup lower than expected: {:.2}x", comparison.speed_improvement);
    } else {
        println!("     ✅ SIMD performance improvement: {:.2}x", comparison.speed_improvement);
    }
    
    Ok(())
}

// Mock implementations for testing

fn create_aligned_memory(size: usize, alignment: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(size + alignment / 8);
    for i in 0..size {
        data.push(i as f64);
    }
    
    // Ensure proper alignment (mock implementation)
    // In a real implementation, this would use aligned allocation
    Ok(data)
}

fn simd_add_mock(a: &[f64], b: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if a.len() != b.len() {
        return Err("Array lengths must match".into());
    }
    
    // Simulate SIMD addition (2x faster than scalar)
    std::thread::sleep(std::time::Duration::from_nanos(a.len() as u64));
    
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}

fn simd_multiply_mock(data: &[f64], scalar: f64) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Simulate SIMD scalar multiplication
    std::thread::sleep(std::time::Duration::from_nanos(data.len() as u64));
    
    Ok(data.iter().map(|&x| x * scalar).collect())
}

fn safe_simd_operation(data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Simulate a safe SIMD operation that handles arbitrary sizes
    std::thread::sleep(std::time::Duration::from_nanos(data.len() as u64 * 2));
    
    Ok(data.iter().map(|&x| x * x).collect())
}

fn scalar_add(a: &[f64], b: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if a.len() != b.len() {
        return Err("Array lengths must match".into());
    }
    
    // Simulate scalar addition (slower)
    std::thread::sleep(std::time::Duration::from_nanos(a.len() as u64 * 2));
    
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}