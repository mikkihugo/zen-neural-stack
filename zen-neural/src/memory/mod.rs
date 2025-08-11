/**
 * @file zen-neural/src/memory/mod.rs
 * @brief Zero-Allocation Memory Management System for Zen Neural Stack
 * 
 * This module implements a comprehensive memory management system designed to achieve
 * 70% memory reduction compared to JavaScript neural network implementations. It provides
 * zero-allocation hot paths for inference and training, thread-safe memory pools, and
 * cache-friendly data structures optimized for SIMD operations.
 * 
 * ## Architecture Overview
 * 
 * The memory system is built around three core principles:
 * 
 * ### 1. Zero-Allocation Hot Paths
 * - Pre-allocated buffers for all neural network operations
 * - In-place computation wherever possible
 * - Stack-allocated small tensors for minimal heap pressure
 * - Reusable intermediate computation buffers
 * 
 * ### 2. Memory Pool Management
 * - Thread-safe pools with size classes for efficient allocation
 * - Automatic pool sizing based on workload patterns
 * - Memory pool statistics and monitoring
 * - Cross-thread memory sharing with safety guarantees
 * 
 * ### 3. Cache-Friendly Data Layout
 * - Memory layout optimized for SIMD access patterns
 * - Cache-line aligned data structures
 * - Spatial locality optimization for graph traversal
 * - Prefetch-friendly sequential access patterns
 * 
 * ## Performance Targets
 * 
 * - **70% memory reduction** vs JavaScript implementations
 * - **Zero allocations** in inference and training hot paths
 * - **Sub-microsecond** memory pool allocation/deallocation
 * - **95% cache hit rate** for frequently accessed tensors
 * - **Thread-safe** memory sharing with minimal contention
 * 
 * ## Integration with Other Components
 * 
 * - **SIMD Operations**: Memory layout optimized for vectorized operations
 * - **Training System**: Efficient gradient accumulation and parameter updates
 * - **DNN Core**: Tensor memory management for forward/backward passes
 * - **GPU Backend**: Pinned memory allocation for fast CPU-GPU transfers
 * 
 * @author Memory Management Expert Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2024-08-11
 * 
 * @see ../gnn/mod.rs Graph Neural Network implementation
 * @see ../training/mod.rs Training infrastructure
 * @see ../gpu/mod.rs GPU acceleration backend
 */

use std::sync::{Arc, RwLock, Mutex};
use std::collections::HashMap;
use std::alloc::{GlobalAlloc, Layout};
use std::ptr::NonNull;
use std::marker::PhantomData;
use std::mem::{size_of, align_of};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::ZenNeuralError;

// === MODULE DECLARATIONS ===

/// Memory pool implementations with size classes and thread safety
pub mod pools;

/// Zero-allocation tensor operations and buffer management
pub mod tensors;

/// Memory profiling, statistics, and debugging utilities
pub mod profiler;

/// Cache-optimized data structures for neural network operations
pub mod layout;

/// RAII memory management patterns and safety guarantees
pub mod safety;

/// Custom allocators for specialized workloads
pub mod allocators;

// === RE-EXPORTS ===

pub use pools::{
    MemoryPool, ThreadSafePool, SizeClassPool, PoolConfig,
    TensorPool, GradientPool, ActivationPool
};

pub use tensors::{
    ZeroAllocTensor, PreAllocatedBuffer, StackTensor,
    TensorArena, BufferManager
};

pub use profiler::{
    MemoryProfiler, MemoryStats, AllocationTracker,
    ProfileReport, MemoryUsage
};

pub use layout::{
    CacheAlignedArray, SimdOptimizedLayout, GraphMemoryLayout,
    MemoryHierarchy, CacheConfig
};

pub use safety::{
    SafeMemoryRegion, BoundsChecker, MemoryGuard,
    LifetimeManager, OwnershipTracker
};

pub use allocators::{
    NeuralAllocator, PoolAllocator, ArenaAllocator,
    JemallocIntegration, CustomAllocator
};

// === CORE MEMORY MANAGEMENT SYSTEM ===

/**
 * Main memory management system for Zen Neural Stack.
 * 
 * This is the central coordinator for all memory operations, providing:
 * - Global memory pool management
 * - Zero-allocation operation coordination
 * - Memory usage monitoring and optimization
 * - Integration with neural network components
 * 
 * ## Usage Pattern
 * 
 * ```rust
 * // Initialize memory system with configuration
 * let memory_config = MemoryConfig {
 *     total_pool_size: 1024 * 1024 * 1024, // 1GB pool
 *     tensor_pool_ratio: 0.6,              // 60% for tensors
 *     gradient_pool_ratio: 0.3,            // 30% for gradients
 *     activation_pool_ratio: 0.1,          // 10% for activations
 *     enable_profiling: true,
 *     cache_line_size: 64,
 * };
 * 
 * let memory_system = ZenMemorySystem::new(memory_config)?;
 * 
 * // Use for neural network operations
 * let tensor = memory_system.allocate_tensor([1024, 768], TensorType::Float32)?;
 * let result = neural_network.forward(&tensor, &memory_system)?;
 * ```
 */
#[derive(Debug)]
pub struct ZenMemorySystem {
    /// Configuration parameters
    config: MemoryConfig,
    
    /// Thread-safe tensor memory pool
    tensor_pool: Arc<ThreadSafePool<f32>>,
    
    /// Gradient accumulation memory pool
    gradient_pool: Arc<ThreadSafePool<f32>>,
    
    /// Activation buffer memory pool
    activation_pool: Arc<ThreadSafePool<f32>>,
    
    /// Memory usage profiler and statistics
    profiler: Arc<Mutex<MemoryProfiler>>,
    
    /// Buffer managers for different tensor types
    buffer_managers: HashMap<TensorType, Arc<BufferManager>>,
    
    /// Memory layout optimizer
    layout_optimizer: Arc<RwLock<MemoryHierarchy>>,
    
    /// Safety and bounds checking system
    safety_system: Arc<SafeMemoryRegion>,
}

/**
 * Configuration for the memory management system.
 * 
 * Provides comprehensive control over memory allocation strategies,
 * pool sizes, and performance optimizations.
 */
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryConfig {
    /// Total memory pool size in bytes (default: 1GB)
    pub total_pool_size: usize,
    
    /// Ratio of memory dedicated to tensor storage (0.0-1.0)
    pub tensor_pool_ratio: f32,
    
    /// Ratio of memory dedicated to gradient storage (0.0-1.0)
    pub gradient_pool_ratio: f32,
    
    /// Ratio of memory dedicated to activation storage (0.0-1.0)
    pub activation_pool_ratio: f32,
    
    /// Enable detailed memory profiling (impacts performance)
    pub enable_profiling: bool,
    
    /// CPU cache line size for alignment optimization
    pub cache_line_size: usize,
    
    /// Maximum number of size classes in memory pools
    pub max_size_classes: usize,
    
    /// Minimum allocation size in bytes
    pub min_allocation_size: usize,
    
    /// Maximum allocation size in bytes
    pub max_allocation_size: usize,
    
    /// Enable automatic memory defragmentation
    pub enable_defragmentation: bool,
    
    /// Memory alignment requirements for SIMD operations
    pub simd_alignment: usize,
    
    /// Pre-allocation strategy for common tensor sizes
    pub preallocation_strategy: PreAllocationStrategy,
}

/// Tensor data types supported by the memory system
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorType {
    /// 32-bit floating point (primary neural network type)
    Float32,
    /// 64-bit floating point (high precision)
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 16-bit half-precision float (GPU optimized)
    Half,
    /// 8-bit unsigned integer (quantized networks)
    UInt8,
}

/// Pre-allocation strategies for common use patterns
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreAllocationStrategy {
    /// Allocate based on historical usage patterns
    Adaptive,
    /// Pre-allocate common neural network tensor sizes
    NeuralNetwork,
    /// Pre-allocate for graph neural network workloads
    GraphNeural,
    /// Conservative pre-allocation for memory-constrained environments
    Conservative,
    /// Aggressive pre-allocation for high-performance scenarios
    Aggressive,
    /// Custom pre-allocation pattern
    Custom(Vec<TensorShape>),
}

/// Tensor shape specification for pre-allocation
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape {
    /// Tensor dimensions
    pub dims: Vec<usize>,
    /// Data type
    pub dtype: TensorType,
    /// Expected usage frequency (0.0-1.0)
    pub frequency: f32,
}

/// Memory operation result type
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Memory management error types
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },
    
    #[error("Invalid allocation size: {size} bytes (min: {min}, max: {max})")]
    InvalidSize { size: usize, min: usize, max: usize },
    
    #[error("Pool exhausted: {pool_type} pool has no available slots")]
    PoolExhausted { pool_type: String },
    
    #[error("Memory alignment error: address {address:p} not aligned to {alignment} bytes")]
    AlignmentError { address: usize, alignment: usize },
    
    #[error("Bounds violation: access at offset {offset} exceeds buffer size {size}")]
    BoundsViolation { offset: usize, size: usize },
    
    #[error("Thread safety error: {message}")]
    ThreadSafetyError { message: String },
    
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    #[error("Profiler error: {message}")]
    ProfilerError { message: String },
    
    #[error("General zen-neural error: {0}")]
    ZenNeuralError(#[from] ZenNeuralError),
}

impl Default for MemoryConfig {
    /// Default memory configuration optimized for neural network workloads
    fn default() -> Self {
        Self {
            total_pool_size: 1024 * 1024 * 1024, // 1GB
            tensor_pool_ratio: 0.6,               // 60% for tensors
            gradient_pool_ratio: 0.3,             // 30% for gradients
            activation_pool_ratio: 0.1,           // 10% for activations
            enable_profiling: false,              // Disabled by default for performance
            cache_line_size: 64,                  // Common x86_64 cache line size
            max_size_classes: 32,                 // Balance between fragmentation and overhead
            min_allocation_size: 64,              // Minimum meaningful allocation
            max_allocation_size: 128 * 1024 * 1024, // 128MB max single allocation
            enable_defragmentation: true,         // Enable automatic defrag
            simd_alignment: 32,                   // AVX/AVX2 alignment requirement
            preallocation_strategy: PreAllocationStrategy::NeuralNetwork,
        }
    }
}

impl TensorType {
    /// Get the byte size of this tensor type
    pub fn byte_size(&self) -> usize {
        match self {
            TensorType::Float32 => 4,
            TensorType::Float64 => 8,
            TensorType::Int32 => 4,
            TensorType::Int64 => 8,
            TensorType::Half => 2,
            TensorType::UInt8 => 1,
        }
    }
    
    /// Get the alignment requirement for this tensor type
    pub fn alignment(&self) -> usize {
        match self {
            TensorType::Float32 => 4,
            TensorType::Float64 => 8,
            TensorType::Int32 => 4,
            TensorType::Int64 => 8,
            TensorType::Half => 2,
            TensorType::UInt8 => 1,
        }
    }
    
    /// Check if this type supports SIMD operations
    pub fn supports_simd(&self) -> bool {
        matches!(self, TensorType::Float32 | TensorType::Float64 | TensorType::Int32)
    }
}

impl TensorShape {
    /// Create a new tensor shape
    pub fn new(dims: Vec<usize>, dtype: TensorType, frequency: f32) -> Self {
        Self { dims, dtype, frequency }
    }
    
    /// Calculate total number of elements
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
    
    /// Calculate total memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.num_elements() * self.dtype.byte_size()
    }
    
    /// Check if this shape is compatible with SIMD operations
    pub fn is_simd_compatible(&self) -> bool {
        self.dtype.supports_simd() && self.num_elements() % 8 == 0
    }
}

impl ZenMemorySystem {
    /// Create a new memory management system
    pub fn new(config: MemoryConfig) -> MemoryResult<Self> {
        // Validate configuration
        Self::validate_config(&config)?;
        
        // Calculate pool sizes
        let tensor_pool_size = (config.total_pool_size as f32 * config.tensor_pool_ratio) as usize;
        let gradient_pool_size = (config.total_pool_size as f32 * config.gradient_pool_ratio) as usize;
        let activation_pool_size = (config.total_pool_size as f32 * config.activation_pool_ratio) as usize;
        
        // Initialize memory pools
        let tensor_pool = Arc::new(ThreadSafePool::new(
            tensor_pool_size,
            config.max_size_classes,
            config.cache_line_size,
        )?);
        
        let gradient_pool = Arc::new(ThreadSafePool::new(
            gradient_pool_size,
            config.max_size_classes,
            config.cache_line_size,
        )?);
        
        let activation_pool = Arc::new(ThreadSafePool::new(
            activation_pool_size,
            config.max_size_classes,
            config.cache_line_size,
        )?);
        
        // Initialize profiler
        let profiler = Arc::new(Mutex::new(MemoryProfiler::new(
            config.enable_profiling
        )?));
        
        // Initialize buffer managers for each tensor type
        let mut buffer_managers = HashMap::new();
        for &tensor_type in &[TensorType::Float32, TensorType::Float64, TensorType::Int32] {
            let manager = Arc::new(BufferManager::new(
                tensor_type,
                config.simd_alignment,
                tensor_pool.clone(),
            )?);
            buffer_managers.insert(tensor_type, manager);
        }
        
        // Initialize memory layout optimizer
        let layout_optimizer = Arc::new(RwLock::new(MemoryHierarchy::new(
            config.cache_line_size
        )?));
        
        // Initialize safety system
        let safety_system = Arc::new(SafeMemoryRegion::new(
            config.total_pool_size
        )?);
        
        let system = Self {
            config,
            tensor_pool,
            gradient_pool,
            activation_pool,
            profiler,
            buffer_managers,
            layout_optimizer,
            safety_system,
        };
        
        // Perform pre-allocation based on strategy
        system.perform_preallocation()?;
        
        Ok(system)
    }
    
    /// Create a memory system with default configuration
    pub fn default() -> MemoryResult<Self> {
        Self::new(MemoryConfig::default())
    }
    
    /// Allocate a tensor with specified shape and type
    pub fn allocate_tensor<T>(&self, shape: &[usize], tensor_type: TensorType) -> MemoryResult<ZeroAllocTensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Calculate memory requirements
        let num_elements: usize = shape.iter().product();
        let element_size = tensor_type.byte_size();
        let total_size = num_elements * element_size;
        
        // Validate allocation size
        if total_size < self.config.min_allocation_size || total_size > self.config.max_allocation_size {
            return Err(MemoryError::InvalidSize {
                size: total_size,
                min: self.config.min_allocation_size,
                max: self.config.max_allocation_size,
            });
        }
        
        // Get appropriate buffer manager
        let buffer_manager = self.buffer_managers.get(&tensor_type)
            .ok_or_else(|| MemoryError::ConfigError {
                message: format!("No buffer manager for tensor type: {:?}", tensor_type)
            })?;
        
        // Allocate from appropriate pool
        let buffer = buffer_manager.allocate(total_size)?;
        
        // Update profiler
        if self.config.enable_profiling {
            if let Ok(mut profiler) = self.profiler.lock() {
                profiler.record_allocation(total_size, tensor_type);
            }
        }
        
        // Create zero-allocation tensor
        Ok(ZeroAllocTensor::new(buffer, shape.to_vec(), tensor_type)?)
    }
    
    /// Allocate gradient buffer for training
    pub fn allocate_gradient_buffer(&self, size: usize) -> MemoryResult<PreAllocatedBuffer<f32>> {
        let buffer = self.gradient_pool.allocate(size * size_of::<f32>())?;
        
        if self.config.enable_profiling {
            if let Ok(mut profiler) = self.profiler.lock() {
                profiler.record_gradient_allocation(size * size_of::<f32>());
            }
        }
        
        Ok(PreAllocatedBuffer::new(buffer, size)?)
    }
    
    /// Allocate activation buffer for forward pass
    pub fn allocate_activation_buffer(&self, size: usize) -> MemoryResult<PreAllocatedBuffer<f32>> {
        let buffer = self.activation_pool.allocate(size * size_of::<f32>())?;
        
        if self.config.enable_profiling {
            if let Ok(mut profiler) = self.profiler.lock() {
                profiler.record_activation_allocation(size * size_of::<f32>());
            }
        }
        
        Ok(PreAllocatedBuffer::new(buffer, size)?)
    }
    
    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryResult<MemoryStats> {
        let tensor_stats = self.tensor_pool.get_stats();
        let gradient_stats = self.gradient_pool.get_stats();
        let activation_stats = self.activation_pool.get_stats();
        
        let profiler_stats = if self.config.enable_profiling {
            self.profiler.lock()
                .map_err(|_| MemoryError::ThreadSafetyError {
                    message: "Failed to lock profiler".to_string()
                })?
                .get_stats()
        } else {
            MemoryStats::default()
        };
        
        Ok(MemoryStats {
            total_allocated: tensor_stats.total_allocated + gradient_stats.total_allocated + activation_stats.total_allocated,
            total_available: tensor_stats.total_available + gradient_stats.total_available + activation_stats.total_available,
            peak_usage: profiler_stats.peak_usage,
            allocation_count: profiler_stats.allocation_count,
            deallocation_count: profiler_stats.deallocation_count,
            fragmentation_ratio: Self::calculate_fragmentation_ratio(&tensor_stats, &gradient_stats, &activation_stats),
            pool_efficiency: Self::calculate_pool_efficiency(&tensor_stats, &gradient_stats, &activation_stats),
            cache_hit_rate: profiler_stats.cache_hit_rate,
        })
    }
    
    /// Optimize memory layout for better cache performance
    pub fn optimize_memory_layout(&self) -> MemoryResult<()> {
        let mut optimizer = self.layout_optimizer.write()
            .map_err(|_| MemoryError::ThreadSafetyError {
                message: "Failed to lock layout optimizer".to_string()
            })?;
        
        optimizer.optimize_layout()?
        Ok(())
    }
    
    /// Defragment memory pools
    pub fn defragment(&self) -> MemoryResult<()> {
        if !self.config.enable_defragmentation {
            return Ok(());
        }
        
        self.tensor_pool.defragment()?;
        self.gradient_pool.defragment()?;
        self.activation_pool.defragment()?;
        
        Ok(())
    }
    
    /// Generate comprehensive memory report
    pub fn generate_memory_report(&self) -> MemoryResult<ProfileReport> {
        let stats = self.get_memory_stats()?;
        
        let profiler_report = if self.config.enable_profiling {
            self.profiler.lock()
                .map_err(|_| MemoryError::ThreadSafetyError {
                    message: "Failed to lock profiler".to_string()
                })?
                .generate_report()
        } else {
            ProfileReport::default()
        };
        
        Ok(ProfileReport {
            memory_stats: stats,
            config: self.config.clone(),
            pool_reports: vec![
                self.tensor_pool.get_detailed_stats(),
                self.gradient_pool.get_detailed_stats(),
                self.activation_pool.get_detailed_stats(),
            ],
            layout_analysis: self.layout_optimizer.read()
                .map_err(|_| MemoryError::ThreadSafetyError {
                    message: "Failed to lock layout optimizer".to_string()
                })?
                .analyze_layout(),
            ..profiler_report
        })
    }
    
    // === PRIVATE HELPER METHODS ===
    
    /// Validate memory configuration parameters
    fn validate_config(config: &MemoryConfig) -> MemoryResult<()> {
        let total_ratio = config.tensor_pool_ratio + config.gradient_pool_ratio + config.activation_pool_ratio;
        if (total_ratio - 1.0).abs() > 1e-6 {
            return Err(MemoryError::ConfigError {
                message: format!("Pool ratios must sum to 1.0, got {}", total_ratio)
            });
        }
        
        if config.total_pool_size == 0 {
            return Err(MemoryError::ConfigError {
                message: "Total pool size must be greater than 0".to_string()
            });
        }
        
        if config.cache_line_size == 0 || !config.cache_line_size.is_power_of_two() {
            return Err(MemoryError::ConfigError {
                message: format!("Cache line size must be a power of 2, got {}", config.cache_line_size)
            });
        }
        
        if config.simd_alignment == 0 || !config.simd_alignment.is_power_of_two() {
            return Err(MemoryError::ConfigError {
                message: format!("SIMD alignment must be a power of 2, got {}", config.simd_alignment)
            });
        }
        
        Ok(())
    }
    
    /// Perform pre-allocation based on configured strategy
    fn perform_preallocation(&self) -> MemoryResult<()> {
        match self.config.preallocation_strategy {
            PreAllocationStrategy::NeuralNetwork => {
                // Pre-allocate common neural network tensor sizes
                let common_shapes = vec![
                    TensorShape::new(vec![32, 768], TensorType::Float32, 0.8),   // Transformer embeddings
                    TensorShape::new(vec![1024, 256], TensorType::Float32, 0.6), // Dense layer weights
                    TensorShape::new(vec![64, 128], TensorType::Float32, 0.7),   // Small batch processing
                    TensorShape::new(vec![512, 512], TensorType::Float32, 0.5),  // Square weight matrices
                ];
                
                for shape in common_shapes {
                    self.preallocate_tensor_shape(&shape)?;
                }
            },
            PreAllocationStrategy::GraphNeural => {
                // Pre-allocate for graph neural network workloads
                let graph_shapes = vec![
                    TensorShape::new(vec![1000, 128], TensorType::Float32, 0.9), // Node embeddings
                    TensorShape::new(vec![2000, 64], TensorType::Float32, 0.8),  // Edge features
                    TensorShape::new(vec![500, 256], TensorType::Float32, 0.7),  // Message buffers
                ];
                
                for shape in graph_shapes {
                    self.preallocate_tensor_shape(&shape)?;
                }
            },
            PreAllocationStrategy::Custom(ref shapes) => {
                for shape in shapes {
                    self.preallocate_tensor_shape(shape)?;
                }
            },
            PreAllocationStrategy::Conservative | PreAllocationStrategy::Aggressive | PreAllocationStrategy::Adaptive => {
                // These strategies will be implemented based on runtime patterns
                // For now, just pre-allocate a minimal set
                let basic_shape = TensorShape::new(vec![256, 256], TensorType::Float32, 1.0);
                self.preallocate_tensor_shape(&basic_shape)?;
            }
        }
        
        Ok(())
    }
    
    /// Pre-allocate memory for a specific tensor shape
    fn preallocate_tensor_shape(&self, shape: &TensorShape) -> MemoryResult<()> {
        let buffer_manager = self.buffer_managers.get(&shape.dtype)
            .ok_or_else(|| MemoryError::ConfigError {
                message: format!("No buffer manager for tensor type: {:?}", shape.dtype)
            })?;
        
        let size = shape.memory_size();
        buffer_manager.preallocate(size, (shape.frequency * 10.0) as usize)?;
        
        Ok(())
    }
    
    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation_ratio(
        tensor_stats: &pools::PoolStats,
        gradient_stats: &pools::PoolStats,
        activation_stats: &pools::PoolStats,
    ) -> f32 {
        let total_allocated = tensor_stats.total_allocated + gradient_stats.total_allocated + activation_stats.total_allocated;
        let total_requested = tensor_stats.total_requested + gradient_stats.total_requested + activation_stats.total_requested;
        
        if total_requested > 0 {
            (total_allocated - total_requested) as f32 / total_requested as f32
        } else {
            0.0
        }
    }
    
    /// Calculate memory pool efficiency
    fn calculate_pool_efficiency(
        tensor_stats: &pools::PoolStats,
        gradient_stats: &pools::PoolStats,
        activation_stats: &pools::PoolStats,
    ) -> f32 {
        let total_capacity = tensor_stats.total_capacity + gradient_stats.total_capacity + activation_stats.total_capacity;
        let total_allocated = tensor_stats.total_allocated + gradient_stats.total_allocated + activation_stats.total_allocated;
        
        if total_capacity > 0 {
            total_allocated as f32 / total_capacity as f32
        } else {
            0.0
        }
    }
}

// === THREAD SAFETY AND SYNCHRONIZATION ===

unsafe impl Send for ZenMemorySystem {}
unsafe impl Sync for ZenMemorySystem {}

// === INTEGRATION TRAITS ===

/// Trait for types that can be managed by the memory system
pub trait MemoryManaged: Sized + Send + Sync {
    /// Get the memory size of this type
    fn memory_size(&self) -> usize;
    
    /// Get the alignment requirement for this type
    fn alignment() -> usize;
    
    /// Check if this type supports zero-copy operations
    fn supports_zero_copy() -> bool {
        false
    }
}

/// Trait for neural network components that use the memory system
pub trait NeuralMemoryUser {
    /// Initialize memory resources for this component
    fn initialize_memory(&mut self, memory_system: &ZenMemorySystem) -> MemoryResult<()>;
    
    /// Get memory usage for this component
    fn get_memory_usage(&self) -> MemoryUsage;
    
    /// Optimize memory usage for this component
    fn optimize_memory(&mut self, memory_system: &ZenMemorySystem) -> MemoryResult<()>;
}

// === BASIC TYPE IMPLEMENTATIONS ===

impl MemoryManaged for f32 {
    fn memory_size(&self) -> usize { size_of::<f32>() }
    fn alignment() -> usize { align_of::<f32>() }
    fn supports_zero_copy() -> bool { true }
}

impl MemoryManaged for f64 {
    fn memory_size(&self) -> usize { size_of::<f64>() }
    fn alignment() -> usize { align_of::<f64>() }
    fn supports_zero_copy() -> bool { true }
}

impl MemoryManaged for i32 {
    fn memory_size(&self) -> usize { size_of::<i32>() }
    fn alignment() -> usize { align_of::<i32>() }
    fn supports_zero_copy() -> bool { true }
}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.total_pool_size, 1024 * 1024 * 1024);
        assert_eq!(config.tensor_pool_ratio, 0.6);
        assert_eq!(config.gradient_pool_ratio, 0.3);
        assert_eq!(config.activation_pool_ratio, 0.1);
        assert!((config.tensor_pool_ratio + config.gradient_pool_ratio + config.activation_pool_ratio - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_tensor_type_properties() {
        assert_eq!(TensorType::Float32.byte_size(), 4);
        assert_eq!(TensorType::Float64.byte_size(), 8);
        assert_eq!(TensorType::Half.byte_size(), 2);
        assert_eq!(TensorType::UInt8.byte_size(), 1);
        
        assert!(TensorType::Float32.supports_simd());
        assert!(!TensorType::UInt8.supports_simd());
    }
    
    #[test]
    fn test_tensor_shape_calculations() {
        let shape = TensorShape::new(vec![10, 20, 30], TensorType::Float32, 0.8);
        assert_eq!(shape.num_elements(), 6000);
        assert_eq!(shape.memory_size(), 24000); // 6000 * 4 bytes
        assert!(!shape.is_simd_compatible()); // 6000 is not divisible by 8
        
        let simd_shape = TensorShape::new(vec![8, 16], TensorType::Float32, 0.5);
        assert!(simd_shape.is_simd_compatible()); // 128 is divisible by 8
    }
    
    #[tokio::test]
    async fn test_memory_system_creation() {
        let config = MemoryConfig::default();
        let memory_system = ZenMemorySystem::new(config);
        assert!(memory_system.is_ok());
        
        let system = memory_system.unwrap();
        let stats = system.get_memory_stats();
        assert!(stats.is_ok());
    }
    
    #[tokio::test]
    async fn test_tensor_allocation() {
        let memory_system = ZenMemorySystem::default().unwrap();
        
        let tensor: Result<ZeroAllocTensor<f32>, _> = memory_system.allocate_tensor(&[10, 20], TensorType::Float32);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape(), &[10, 20]);
        assert_eq!(tensor.len(), 200);
    }
    
    #[test]
    fn test_config_validation() {
        // Invalid pool ratios
        let invalid_config = MemoryConfig {
            tensor_pool_ratio: 0.5,
            gradient_pool_ratio: 0.5,
            activation_pool_ratio: 0.2, // Sum > 1.0
            ..MemoryConfig::default()
        };
        
        assert!(ZenMemorySystem::validate_config(&invalid_config).is_err());
        
        // Invalid cache line size
        let invalid_cache_config = MemoryConfig {
            cache_line_size: 63, // Not power of 2
            ..MemoryConfig::default()
        };
        
        assert!(ZenMemorySystem::validate_config(&invalid_cache_config).is_err());
    }
}

