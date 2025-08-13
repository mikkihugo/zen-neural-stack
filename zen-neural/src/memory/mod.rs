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
// Removed unused imports: GlobalAlloc, Layout, NonNull, PhantomData
use std::mem::{size_of, align_of};

// Conditional import for parallel processing - only warn about unused when parallel feature is disabled
#[cfg_attr(not(feature = "parallel"), allow(unused_imports))]
#[cfg(feature = "parallel")]
#[allow(unused_imports)] // False positive: used by parallel iterators when parallel feature is enabled
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

/// Memory layout strategies for optimal performance
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// Simple contiguous layout
    Contiguous,
    /// Contiguous with specific alignment (in bytes)
    ContiguousAligned(usize),
    /// Strided layout with specific stride
    Strided(usize),
    /// Structure of Arrays layout for SIMD
    StructureOfArrays,
    /// Array of Structures layout
    ArrayOfStructures,
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq)] // Can't derive Eq/Hash due to f32 field
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
    
    #[error("Memory alignment error: address {address:#x} not aligned to {alignment} bytes")] // Fixed pointer formatting
    AlignmentError { address: usize, alignment: usize },
    
    #[error("Bounds violation: access at offset {offset} exceeds buffer size {size}")]
    BoundsViolation { offset: usize, size: usize },
    
    #[error("Thread safety error: {message}")]
    ThreadSafetyError { message: String },
    
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    #[error("Profiler error: {message}")]
    ProfilerError { message: String },
    
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    
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
    
    /// Get the name of the tensor type as a string
    pub fn name(&self) -> &'static str {
        match self {
            TensorType::Float32 => "float32",
            TensorType::Float64 => "float64",
            TensorType::Int32 => "int32",
            TensorType::Int64 => "int64",
            TensorType::Half => "float16",
            TensorType::UInt8 => "uint8",
        }
    }
    
    /// Check if the type is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, TensorType::Float32 | TensorType::Float64 | TensorType::Half)
    }
    
    /// Check if the type is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(self, TensorType::Int32 | TensorType::Int64 | TensorType::UInt8)
    }
    
    /// Check if the type is signed
    pub fn is_signed(&self) -> bool {
        matches!(self, TensorType::Float32 | TensorType::Float64 | TensorType::Half | TensorType::Int32 | TensorType::Int64)
    }
    
    /// Get the maximum value for this type (if applicable)
    pub fn max_value(&self) -> Option<f64> {
        match self {
            TensorType::Float32 => Some(f32::MAX as f64),
            TensorType::Float64 => Some(f64::MAX),
            TensorType::Int32 => Some(i32::MAX as f64),
            TensorType::Int64 => Some(i64::MAX as f64),
            TensorType::Half => Some(65504.0), // Half precision max
            TensorType::UInt8 => Some(u8::MAX as f64),
        }
    }
    
    /// Get the minimum value for this type (if applicable)
    pub fn min_value(&self) -> Option<f64> {
        match self {
            TensorType::Float32 => Some(f32::MIN as f64),
            TensorType::Float64 => Some(f64::MIN),
            TensorType::Int32 => Some(i32::MIN as f64),
            TensorType::Int64 => Some(i64::MIN as f64),
            TensorType::Half => Some(-65504.0), // Half precision min
            TensorType::UInt8 => Some(0.0),
        }
    }
    
    /// Check if this type can be cast to another type safely
    pub fn can_cast_to(&self, target: TensorType) -> bool {
        match (self, target) {
            // Same type is always safe
            (a, b) if *a == b => true,
            
            // Widening conversions are generally safe
            (TensorType::UInt8, _) => true, // uint8 can cast to anything
            (TensorType::Half, TensorType::Float32 | TensorType::Float64) => true,
            (TensorType::Float32, TensorType::Float64) => true,
            (TensorType::Int32, TensorType::Int64 | TensorType::Float64) => true,
            
            // Integer to float conversions
            (TensorType::Int32, TensorType::Float32) => true, // May lose precision but safe
            (TensorType::Int64, TensorType::Float64) => true,
            
            // Narrowing conversions may lose data
            _ => false,
        }
    }
    
    /// Get SIMD vector width for this type (number of elements per vector)
    pub fn simd_width(&self) -> Option<usize> {
        if !self.supports_simd() {
            return None;
        }
        
        match self {
            TensorType::Float32 => Some(8), // 256-bit SIMD / 32-bit float = 8 elements
            TensorType::Float64 => Some(4), // 256-bit SIMD / 64-bit float = 4 elements  
            TensorType::Int32 => Some(8),   // 256-bit SIMD / 32-bit int = 8 elements
            _ => None,
        }
    }
    
    /// Get the zero value for this type as a byte representation
    pub fn zero_bytes(&self) -> Vec<u8> {
        match self {
            TensorType::Float32 => (0.0f32).to_le_bytes().to_vec(),
            TensorType::Float64 => (0.0f64).to_le_bytes().to_vec(),
            TensorType::Int32 => (0i32).to_le_bytes().to_vec(),
            TensorType::Int64 => (0i64).to_le_bytes().to_vec(),
            TensorType::Half => (0u16).to_le_bytes().to_vec(), // Half precision zero
            TensorType::UInt8 => vec![0u8],
        }
    }
    
    /// Get the preferred layout for efficient memory access
    pub fn preferred_layout(&self) -> MemoryLayout {
        match self {
            TensorType::Float32 | TensorType::Int32 => MemoryLayout::ContiguousAligned(32),
            TensorType::Float64 | TensorType::Int64 => MemoryLayout::ContiguousAligned(64),
            TensorType::Half => MemoryLayout::ContiguousAligned(16),
            TensorType::UInt8 => MemoryLayout::Contiguous,
        }
    }
    
    /// Check if the type is suitable for neural network computations
    pub fn is_neural_compatible(&self) -> bool {
        match self {
            TensorType::Float32 | TensorType::Half => true, // Primary neural types
            TensorType::Float64 => true, // High precision research
            TensorType::UInt8 => true,   // Quantized networks
            TensorType::Int32 | TensorType::Int64 => false, // Generally not used for weights/activations
        }
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
    
    /// Calculate memory size with alignment padding
    pub fn aligned_memory_size(&self) -> usize {
        let base_size = self.memory_size();
        let alignment = self.dtype.alignment();
        
        // Round up to nearest alignment boundary
        (base_size + alignment - 1) & !(alignment - 1)
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    
    /// Check if the shape is suitable for SIMD operations
    pub fn is_simd_compatible(&self) -> bool {
        if !self.dtype.supports_simd() {
            return false;
        }
        
        // Last dimension should be divisible by SIMD width for optimal performance
        if let Some(simd_width) = self.dtype.simd_width() {
            self.dims.last().map_or(false, |&last_dim| last_dim % simd_width == 0)
        } else {
            false
        }
    }
    
    /// Check if the shape represents a matrix (2D tensor)
    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }
    
    /// Check if the shape represents a vector (1D tensor)
    pub fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }
    
    /// Check if the shape represents a scalar (0D tensor)
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty() || (self.dims.len() == 1 && self.dims[0] == 1)
    }
    
    /// Get matrix dimensions if this is a 2D tensor
    pub fn matrix_dims(&self) -> Option<(usize, usize)> {
        if self.is_matrix() {
            Some((self.dims[0], self.dims[1]))
        } else {
            None
        }
    }
    
    /// Create a reshaped version of this tensor shape (same number of elements)
    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<TensorShape, MemoryError> {
        let new_elements: usize = new_dims.iter().product();
        
        if new_elements != self.num_elements() {
            return Err(MemoryError::InvalidArguments(
                format!("Cannot reshape tensor with {} elements to shape with {} elements",
                    self.num_elements(), new_elements)
            ));
        }
        
        Ok(TensorShape {
            dims: new_dims,
            dtype: self.dtype,
            frequency: self.frequency,
        })
    }
    
    /// Calculate strides for this tensor shape in row-major order
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.dims.len()];
        
        if !self.dims.is_empty() {
            // Calculate strides from right to left
            for i in (0..self.dims.len() - 1).rev() {
                strides[i] = strides[i + 1] * self.dims[i + 1];
            }
        }
        
        strides
    }
    
    /// Calculate the linear index from multi-dimensional indices
    pub fn linear_index(&self, indices: &[usize]) -> Result<usize, MemoryError> {
        if indices.len() != self.dims.len() {
            return Err(MemoryError::InvalidArguments(
                format!("Expected {} indices, got {}", self.dims.len(), indices.len())
            ));
        }
        
        // Check bounds
        for (i, (&idx, &dim)) in indices.iter().zip(self.dims.iter()).enumerate() {
            if idx >= dim {
                return Err(MemoryError::InvalidArguments(
                    format!("Index {} out of bounds for dimension {} (size {})", idx, i, dim)
                ));
            }
        }
        
        let strides = self.strides();
        let linear_idx = indices.iter()
            .zip(strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum();
            
        Ok(linear_idx)
    }
    
    /// Convert linear index back to multi-dimensional indices
    pub fn multi_index(&self, linear_index: usize) -> Result<Vec<usize>, MemoryError> {
        if linear_index >= self.num_elements() {
            return Err(MemoryError::InvalidArguments(
                format!("Linear index {} out of bounds for tensor with {} elements",
                    linear_index, self.num_elements())
            ));
        }
        
        let mut indices = vec![0; self.dims.len()];
        let mut remaining = linear_index;
        let strides = self.strides();
        
        for (i, &stride) in strides.iter().enumerate() {
            indices[i] = remaining / stride;
            remaining %= stride;
        }
        
        Ok(indices)
    }
    
    /// Check if this shape is broadcast compatible with another shape
    pub fn is_broadcast_compatible(&self, other: &TensorShape) -> bool {
        // Tensors are broadcast compatible if their types match and dimensions are compatible
        if self.dtype != other.dtype {
            return false;
        }
        
        let max_dims = self.dims.len().max(other.dims.len());
        
        for i in 0..max_dims {
            let self_dim = if i < self.dims.len() {
                self.dims[self.dims.len() - 1 - i]
            } else {
                1
            };
            
            let other_dim = if i < other.dims.len() {
                other.dims[other.dims.len() - 1 - i]
            } else {
                1
            };
            
            // Dimensions must be equal or one of them must be 1
            if self_dim != other_dim && self_dim != 1 && other_dim != 1 {
                return false;
            }
        }
        
        true
    }
    
    /// Calculate the resulting shape after broadcasting with another shape
    pub fn broadcast_with(&self, other: &TensorShape) -> Result<TensorShape, MemoryError> {
        if !self.is_broadcast_compatible(other) {
            return Err(MemoryError::InvalidArguments(
                "Shapes are not broadcast compatible".to_string()
            ));
        }
        
        let max_dims = self.dims.len().max(other.dims.len());
        let mut result_dims = vec![1; max_dims];
        
        for i in 0..max_dims {
            let self_dim = if i < self.dims.len() {
                self.dims[self.dims.len() - 1 - i]
            } else {
                1
            };
            
            let other_dim = if i < other.dims.len() {
                other.dims[other.dims.len() - 1 - i]
            } else {
                1
            };
            
            result_dims[max_dims - 1 - i] = self_dim.max(other_dim);
        }
        
        Ok(TensorShape {
            dims: result_dims,
            dtype: self.dtype,
            frequency: self.frequency.max(other.frequency),
        })
    }
    
    /// Get the preferred memory layout for this shape
    pub fn preferred_layout(&self) -> MemoryLayout {
        if self.is_simd_compatible() {
            MemoryLayout::ContiguousAligned(self.dtype.alignment())
        } else {
            self.dtype.preferred_layout()
        }
    }
    
    /// Estimate memory access efficiency for this shape (0.0 to 1.0)
    pub fn memory_efficiency_score(&self) -> f32 {
        let mut score = 1.0f32;
        
        // Penalize non-SIMD compatible shapes
        if !self.is_simd_compatible() && self.dtype.supports_simd() {
            score *= 0.7;
        }
        
        // Favor power-of-2 dimensions for better cache behavior
        let power_of_2_bonus: f32 = self.dims.iter()
            .map(|&dim| if dim.is_power_of_two() { 1.1 } else { 1.0 })
            .product();
        score *= power_of_2_bonus.min(1.5); // Cap the bonus
        
        // Penalize very large or very small tensors
        let num_elements = self.num_elements();
        if num_elements < 64 {
            score *= 0.8; // Small tensors are less efficient due to overhead
        } else if num_elements > 1_000_000 {
            score *= 0.9; // Very large tensors may cause cache misses
        }
        
        score.min(1.0)
    }
}

// Custom Eq implementation ignoring f32 for comparison
impl Eq for TensorShape {}

// Custom Hash implementation for TensorShape
impl std::hash::Hash for TensorShape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dims.hash(state);
        self.dtype.hash(state);
        // Skip f32 frequency for hash stability
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
        // Create both typed pools and a raw byte pool for BufferManager
        let tensor_pool = Arc::new(ThreadSafePool::<f32>::new(
            tensor_pool_size,
            config.max_size_classes,
            config.cache_line_size,
        )?); // Typed pool for f32 tensors
        
        let raw_pool = Arc::new(ThreadSafePool::<u8>::new(
            tensor_pool_size / 2, // Share the allocation between typed and raw pools
            config.max_size_classes,
            config.cache_line_size,
        )?); // Raw byte pool for BufferManager
        
        let gradient_pool = Arc::new(ThreadSafePool::<f32>::new(
            gradient_pool_size,
            config.max_size_classes,
            config.cache_line_size,
        )?); // Explicitly specify f32 type
        
        let activation_pool = Arc::new(ThreadSafePool::<f32>::new(
            activation_pool_size,
            config.max_size_classes,
            config.cache_line_size,
        )?); // Explicitly specify f32 type
        
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
                raw_pool.clone(),
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
            if let Ok(profiler) = self.profiler.lock() {
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
            if let Ok(profiler) = self.profiler.lock() {
                profiler.record_gradient_allocation(size * size_of::<f32>());
            }
        }
        
        Ok(PreAllocatedBuffer::new(buffer, size)?)
    }
    
    /// Allocate activation buffer for forward pass
    pub fn allocate_activation_buffer(&self, size: usize) -> MemoryResult<PreAllocatedBuffer<f32>> {
        let buffer = self.activation_pool.allocate(size * size_of::<f32>())?;
        
        if self.config.enable_profiling {
            if let Ok(profiler) = self.profiler.lock() {
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
            average_allocation_size: profiler_stats.average_allocation_size,
            utilization_efficiency: Self::calculate_pool_efficiency(&tensor_stats, &gradient_stats, &activation_stats),
            total_time_us: profiler_stats.total_time_us,
            cache_hit_rate: profiler_stats.cache_hit_rate,
        })
    }
    
    /// Optimize memory layout for better cache performance
    pub fn optimize_memory_layout(&self) -> MemoryResult<()> {
        let mut optimizer = self.layout_optimizer.write()
            .map_err(|_| MemoryError::ThreadSafetyError {
                message: "Failed to lock layout optimizer".to_string()
            })?;
        
        optimizer.optimize_layout()?;
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
            pool_reports: {
                let mut reports = Vec::new();
                reports.extend(self.tensor_pool.get_detailed_stats());
                reports.extend(self.gradient_pool.get_detailed_stats());
                reports.extend(self.activation_pool.get_detailed_stats());
                reports
            },
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
    fn test_tensor_type_extended_operations() {
        // Test type classification methods
        assert!(TensorType::Float32.is_float());
        assert!(TensorType::Half.is_float());
        assert!(!TensorType::Int32.is_float());
        
        assert!(TensorType::Int32.is_integer());
        assert!(TensorType::UInt8.is_integer());
        assert!(!TensorType::Float32.is_integer());
        
        assert!(TensorType::Float32.is_signed());
        assert!(TensorType::Int64.is_signed());
        assert!(!TensorType::UInt8.is_signed());
        
        // Test type names
        assert_eq!(TensorType::Float32.name(), "float32");
        assert_eq!(TensorType::Half.name(), "float16");
        assert_eq!(TensorType::UInt8.name(), "uint8");
        
        // Test value ranges
        assert!(TensorType::Float32.max_value().unwrap() > 3e38);
        assert!(TensorType::Int32.max_value().unwrap() > 2e9);
        assert_eq!(TensorType::UInt8.max_value().unwrap(), 255.0);
        assert_eq!(TensorType::UInt8.min_value().unwrap(), 0.0);
        
        // Test SIMD widths
        assert_eq!(TensorType::Float32.simd_width(), Some(8));
        assert_eq!(TensorType::Float64.simd_width(), Some(4));
        assert_eq!(TensorType::Half.simd_width(), None);
        
        // Test neural compatibility
        assert!(TensorType::Float32.is_neural_compatible());
        assert!(TensorType::Half.is_neural_compatible());
        assert!(TensorType::UInt8.is_neural_compatible()); // Quantized networks
        assert!(!TensorType::Int64.is_neural_compatible());
        
        // Test type casting compatibility
        assert!(TensorType::UInt8.can_cast_to(TensorType::Float32));
        assert!(TensorType::Float32.can_cast_to(TensorType::Float64));
        assert!(TensorType::Half.can_cast_to(TensorType::Float32));
        assert!(!TensorType::Float64.can_cast_to(TensorType::Float32)); // Narrowing
    }
    
    #[test]
    fn test_tensor_shape_extended_operations() {
        let shape = TensorShape::new(vec![10, 20, 30], TensorType::Float32, 0.8);
        
        // Test dimension queries
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.num_elements(), 6000);
        assert_eq!(shape.memory_size(), 24000); // 6000 * 4 bytes
        assert!(shape.aligned_memory_size() >= shape.memory_size());
        
        // Test shape classification
        assert!(!shape.is_scalar());
        assert!(!shape.is_vector());
        assert!(!shape.is_matrix());
        
        let matrix_shape = TensorShape::new(vec![5, 10], TensorType::Float32, 0.5);
        assert!(matrix_shape.is_matrix());
        assert_eq!(matrix_shape.matrix_dims(), Some((5, 10)));
        
        let vector_shape = TensorShape::new(vec![100], TensorType::Float32, 0.9);
        assert!(vector_shape.is_vector());
        
        // Test SIMD compatibility (last dimension should be multiple of SIMD width)
        let simd_shape = TensorShape::new(vec![10, 16], TensorType::Float32, 0.7); // 16 % 8 = 0
        assert!(simd_shape.is_simd_compatible());
        
        let non_simd_shape = TensorShape::new(vec![10, 15], TensorType::Float32, 0.7); // 15 % 8 != 0
        assert!(!non_simd_shape.is_simd_compatible());
        
        // Test strides calculation
        let strides = shape.strides();
        assert_eq!(strides, vec![600, 30, 1]); // [20*30, 30, 1]
        
        // Test linear indexing
        let linear_idx = shape.linear_index(&[1, 2, 3]).unwrap();
        assert_eq!(linear_idx, 1 * 600 + 2 * 30 + 3 * 1);
        assert_eq!(linear_idx, 663);
        
        // Test multi-index conversion
        let multi_idx = shape.multi_index(linear_idx).unwrap();
        assert_eq!(multi_idx, vec![1, 2, 3]);
    }
    
    #[test] 
    fn test_tensor_shape_reshaping() {
        let original = TensorShape::new(vec![4, 6], TensorType::Float32, 0.5);
        
        // Valid reshape (same number of elements)
        let reshaped = original.reshape(vec![2, 12]).unwrap();
        assert_eq!(reshaped.dims, vec![2, 12]);
        assert_eq!(reshaped.num_elements(), 24);
        
        let flattened = original.reshape(vec![24]).unwrap();
        assert_eq!(flattened.dims, vec![24]);
        assert!(flattened.is_vector());
        
        // Invalid reshape (different number of elements)
        let invalid_reshape = original.reshape(vec![5, 5]);
        assert!(invalid_reshape.is_err());
    }
    
    #[test]
    fn test_tensor_shape_broadcasting() {
        let shape1 = TensorShape::new(vec![3, 1], TensorType::Float32, 0.5);
        let shape2 = TensorShape::new(vec![1, 4], TensorType::Float32, 0.7);
        
        // Compatible for broadcasting
        assert!(shape1.is_broadcast_compatible(&shape2));
        
        // Result should be [3, 4]
        let broadcast_result = shape1.broadcast_with(&shape2).unwrap();
        assert_eq!(broadcast_result.dims, vec![3, 4]);
        assert_eq!(broadcast_result.frequency, 0.7); // Max frequency
        
        // Different types should not be compatible
        let shape3 = TensorShape::new(vec![1, 4], TensorType::Int32, 0.6);
        assert!(!shape1.is_broadcast_compatible(&shape3));
        
        // Incompatible dimensions
        let shape4 = TensorShape::new(vec![3, 5], TensorType::Float32, 0.4);
        assert!(!shape1.is_broadcast_compatible(&shape4));
    }
    
    #[test]
    fn test_memory_efficiency_scoring() {
        // SIMD-compatible shape should score high
        let simd_shape = TensorShape::new(vec![64, 128], TensorType::Float32, 0.8);
        let simd_score = simd_shape.memory_efficiency_score();
        assert!(simd_score > 0.8);
        
        // Non-SIMD shape should score lower
        let non_simd_shape = TensorShape::new(vec![63, 127], TensorType::Float32, 0.8);
        let non_simd_score = non_simd_shape.memory_efficiency_score();
        assert!(non_simd_score < simd_score);
        
        // Power-of-2 dimensions should get bonus
        let power_of_2_shape = TensorShape::new(vec![64, 128, 256], TensorType::Float32, 0.5);
        let power_score = power_of_2_shape.memory_efficiency_score();
        assert!(power_score > 0.9);
        
        // Very small tensor should be penalized
        let small_shape = TensorShape::new(vec![2, 3], TensorType::Float32, 0.5);
        let small_score = small_shape.memory_efficiency_score();
        assert!(small_score < 0.9);
    }
    
    #[test]
    fn test_memory_layout_preferences() {
        let float32_shape = TensorShape::new(vec![100, 200], TensorType::Float32, 0.5);
        match float32_shape.preferred_layout() {
            MemoryLayout::ContiguousAligned(alignment) => assert!(alignment >= 4),
            _ => panic!("Expected aligned layout for Float32"),
        }
        
        let uint8_shape = TensorShape::new(vec![1000], TensorType::UInt8, 0.3);
        match uint8_shape.preferred_layout() {
            MemoryLayout::Contiguous => {}, // Expected
            _ => panic!("Expected contiguous layout for UInt8"),
        }
    }
    
    #[test]
    fn test_tensor_type_zero_bytes() {
        let float32_zero = TensorType::Float32.zero_bytes();
        assert_eq!(float32_zero, (0.0f32).to_le_bytes().to_vec());
        
        let int32_zero = TensorType::Int32.zero_bytes();
        assert_eq!(int32_zero, (0i32).to_le_bytes().to_vec());
        
        let uint8_zero = TensorType::UInt8.zero_bytes();
        assert_eq!(uint8_zero, vec![0u8]);
    }
    
    #[test]
    fn test_bounds_checking() {
        let shape = TensorShape::new(vec![3, 4, 5], TensorType::Float32, 0.5);
        
        // Valid indices
        assert!(shape.linear_index(&[0, 0, 0]).is_ok());
        assert!(shape.linear_index(&[2, 3, 4]).is_ok());
        
        // Out of bounds indices
        assert!(shape.linear_index(&[3, 0, 0]).is_err()); // First dim out of bounds
        assert!(shape.linear_index(&[0, 4, 0]).is_err()); // Second dim out of bounds
        assert!(shape.linear_index(&[0, 0, 5]).is_err()); // Third dim out of bounds
        
        // Wrong number of indices
        assert!(shape.linear_index(&[0, 0]).is_err()); // Too few
        assert!(shape.linear_index(&[0, 0, 0, 0]).is_err()); // Too many
        
        // Test multi_index bounds
        assert!(shape.multi_index(60).is_err()); // 60 >= 3*4*5
        assert!(shape.multi_index(0).is_ok());
        assert!(shape.multi_index(59).is_ok()); // Last valid index
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

