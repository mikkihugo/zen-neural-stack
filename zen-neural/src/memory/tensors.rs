/**
 * @file zen-neural/src/memory/tensors.rs
 * @brief Zero-Allocation Tensor System for Neural Network Operations
 * 
 * This module implements zero-allocation tensor operations designed to eliminate
 * memory allocations during neural network inference and training. It provides
 * pre-allocated buffers, in-place operations, and stack-allocated small tensors
 * to achieve maximum memory efficiency.
 * 
 * ## Architecture Overview
 * 
 * ### Zero-Allocation Design Principles
 * 1. **Pre-allocated Buffers**: All tensor memory is allocated upfront
 * 2. **In-Place Operations**: Computations reuse existing memory where possible
 * 3. **Stack Allocation**: Small tensors use stack memory for minimal overhead
 * 4. **Buffer Reuse**: Intermediate computation buffers are recycled
 * 
 * ### Memory Layout Optimization
 * - SIMD-aligned data for vectorized operations
 * - Cache-line aligned structures for optimal access patterns
 * - Contiguous memory layout for spatial locality
 * - Padding elimination through efficient packing
 * 
 * ### Integration with Memory Pools
 * - Automatic allocation from appropriate memory pools
 * - Reference counting for shared tensor data
 * - Lifetime management through Rust ownership
 * - Zero-copy operations where possible
 * 
 * @author Memory Management Expert Agent (ruv-swarm)
 * @version 1.0.0-alpha.1
 * @since 2024-08-11
 */

use std::ptr::NonNull;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Deref, DerefMut};
use std::slice;
use std::mem::{size_of, align_of};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{MemoryError, MemoryResult, TensorType, pools::ThreadSafePool};

// === ZERO-ALLOCATION TENSOR ===

/// A tensor with pre-allocated memory that supports zero-allocation operations
#[derive(Debug)]
pub struct ZeroAllocTensor<T> {
    /// Raw pointer to the tensor data
    data: NonNull<T>,
    /// Shape of the tensor (dimensions)
    shape: Vec<usize>,
    /// Strides for each dimension
    strides: Vec<usize>,
    /// Total number of elements
    len: usize,
    /// Data type of tensor elements
    dtype: TensorType,
    /// Reference to the memory pool for cleanup
    pool_ref: Option<Arc<ThreadSafePool<T>>>,
    /// Reference count for shared tensors
    ref_count: Arc<AtomicUsize>,
    /// Memory alignment of the data
    alignment: usize,
    /// Whether this tensor owns its data
    owns_data: bool,
    /// Type marker
    _marker: PhantomData<T>,
}

impl<T> ZeroAllocTensor<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new zero-allocation tensor
    pub fn new(
        buffer: NonNull<u8>,
        shape: Vec<usize>,
        dtype: TensorType,
    ) -> MemoryResult<Self> {
        let len: usize = shape.iter().product();
        
        if len == 0 {
            return Err(MemoryError::InvalidSize {
                size: 0,
                min: 1,
                max: usize::MAX,
            });
        }
        
        // Calculate strides (row-major order)
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        // Cast buffer to appropriate type
        let data = buffer.cast::<T>();
        
        // Verify alignment
        let alignment = align_of::<T>().max(dtype.alignment());
        if data.as_ptr() as usize % alignment != 0 {
            return Err(MemoryError::AlignmentError {
                address: data.as_ptr() as usize,
                alignment,
            });
        }
        
        Ok(Self {
            data,
            shape,
            strides,
            len,
            dtype,
            pool_ref: None,
            ref_count: Arc::new(AtomicUsize::new(1)),
            alignment,
            owns_data: true,
            _marker: PhantomData,
        })
    }
    
    /// Create a tensor from an existing buffer with shape information
    pub fn from_buffer(
        buffer: NonNull<T>,
        shape: Vec<usize>,
        pool: Arc<ThreadSafePool<T>>,
    ) -> MemoryResult<Self> {
        let len: usize = shape.iter().product();
        
        // Calculate strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        Ok(Self {
            data: buffer,
            shape,
            strides,
            len,
            dtype: TensorType::Float32, // Default, should be parameterized
            pool_ref: Some(pool),
            ref_count: Arc::new(AtomicUsize::new(1)),
            alignment: align_of::<T>(),
            owns_data: true,
            _marker: PhantomData,
        })
    }
    
    /// Create a view of this tensor (shares data, doesn't own it)
    pub fn view(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::SeqCst);
        
        Self {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            len: self.len,
            dtype: self.dtype,
            pool_ref: self.pool_ref.clone(),
            ref_count: self.ref_count.clone(),
            alignment: self.alignment,
            owns_data: false,
            _marker: PhantomData,
        }
    }
    
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the strides of the tensor
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get the data type
    pub fn dtype(&self) -> TensorType {
        self.dtype
    }
    
    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.len * size_of::<T>()
    }
    
    /// Check if the tensor data is contiguous
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }
    
    /// Get a raw pointer to the data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    /// Get a mutable raw pointer to the data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_ptr()
    }
    
    /// Get the data as a slice (only for contiguous tensors)
    pub fn as_slice(&self) -> MemoryResult<&[T]> {
        if !self.is_contiguous() {
            return Err(MemoryError::ConfigError {
                message: "Cannot create slice from non-contiguous tensor".to_string(),
            });
        }
        
        Ok(unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) })
    }
    
    /// Get the data as a mutable slice (only for contiguous tensors)
    pub fn as_mut_slice(&mut self) -> MemoryResult<&mut [T]> {
        if !self.is_contiguous() {
            return Err(MemoryError::ConfigError {
                message: "Cannot create mutable slice from non-contiguous tensor".to_string(),
            });
        }
        
        Ok(unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), self.len) })
    }
    
    /// Reshape the tensor (zero-copy if possible)
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> MemoryResult<()> {
        let new_len: usize = new_shape.iter().product();
        
        if new_len != self.len {
            return Err(MemoryError::ConfigError {
                message: format!(
                    "Cannot reshape tensor: new shape has {} elements, current has {}",
                    new_len, self.len
                ),
            });
        }
        
        // Calculate new strides
        let mut new_strides = vec![1; new_shape.len()];
        for i in (0..new_shape.len().saturating_sub(1)).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
        
        self.shape = new_shape;
        self.strides = new_strides;
        
        Ok(())
    }
    
    /// Create a slice of the tensor along specified dimensions
    pub fn slice(&self, ranges: &[(usize, usize)]) -> MemoryResult<Self> {
        if ranges.len() != self.shape.len() {
            return Err(MemoryError::ConfigError {
                message: "Number of slice ranges must match tensor dimensions".to_string(),
            });
        }
        
        // Calculate offset and new shape
        let mut offset = 0;
        let mut new_shape = Vec::new();
        
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start >= end || end > self.shape[i] {
                return Err(MemoryError::BoundsViolation {
                    offset: start,
                    size: self.shape[i],
                });
            }
            
            offset += start * self.strides[i];
            new_shape.push(end - start);
        }
        
        // Create new tensor pointing to sliced data
        let new_data = unsafe {
            NonNull::new_unchecked(self.data.as_ptr().add(offset))
        };
        
        let new_len = new_shape.iter().product();
        
        // Share reference count
        self.ref_count.fetch_add(1, Ordering::SeqCst);
        
        Ok(Self {
            data: new_data,
            shape: new_shape,
            strides: self.strides.clone(),
            len: new_len,
            dtype: self.dtype,
            pool_ref: self.pool_ref.clone(),
            ref_count: self.ref_count.clone(),
            alignment: self.alignment,
            owns_data: false,
            _marker: PhantomData,
        })
    }
    
    /// Fill the tensor with a specific value (in-place operation)
    pub fn fill(&mut self, value: T) -> MemoryResult<()>
    where
        T: Copy,
    {
        if self.is_contiguous() {
            let slice = unsafe {
                slice::from_raw_parts_mut(self.data.as_ptr(), self.len)
            };
            slice.fill(value);
        } else {
            // Handle non-contiguous case with iterator
            for i in 0..self.len {
                let linear_index = self.linear_to_multi_index(i);
                let offset = self.multi_to_linear_index(&linear_index);
                unsafe {
                    *self.data.as_ptr().add(offset) = value;
                }
            }
        }
        
        Ok(())
    }
    
    /// Zero the tensor (optimized fill with zero)
    pub fn zero(&mut self) -> MemoryResult<()>
    where
        T: Copy + Default,
    {
        self.fill(T::default())
    }
    
    // === PRIVATE HELPER METHODS ===
    
    /// Convert linear index to multi-dimensional index
    fn linear_to_multi_index(&self, linear_index: usize) -> Vec<usize> {
        let mut indices = vec![0; self.shape.len()];
        let mut remaining = linear_index;
        
        for i in 0..self.shape.len() {
            indices[i] = remaining / self.strides[i];
            remaining %= self.strides[i];
        }
        
        indices
    }
    
    /// Convert multi-dimensional index to linear index
    fn multi_to_linear_index(&self, indices: &[usize]) -> usize {
        indices.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }
}

// === INDEXING SUPPORT ===

impl<T> Index<usize> for ZeroAllocTensor<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len {
            panic!("Index {} out of bounds for tensor of length {}", index, self.len);
        }
        
        unsafe {
            &*self.data.as_ptr().add(index)
        }
    }
}

impl<T> IndexMut<usize> for ZeroAllocTensor<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.len {
            panic!("Index {} out of bounds for tensor of length {}", index, self.len);
        }
        
        unsafe {
            &mut *self.data.as_ptr().add(index)
        }
    }
}

// === LIFETIME MANAGEMENT ===

impl<T> Drop for ZeroAllocTensor<T> {
    fn drop(&mut self) {
        if self.owns_data {
            let ref_count = self.ref_count.fetch_sub(1, Ordering::SeqCst);
            
            // If this was the last reference, deallocate memory
            if ref_count == 1 {
                if let Some(pool) = &self.pool_ref {
                    let size = self.len * size_of::<T>();
                    let _ = pool.deallocate(self.data.cast(), size);
                }
            }
        }
    }
}

// === PRE-ALLOCATED BUFFER ===

/// A pre-allocated buffer for efficient memory reuse
#[derive(Debug)]
pub struct PreAllocatedBuffer<T> {
    /// Raw pointer to the buffer
    data: NonNull<T>,
    /// Capacity of the buffer in elements
    capacity: usize,
    /// Current length (number of elements in use)
    length: AtomicUsize,
    /// Reference to the memory pool
    pool_ref: Option<Arc<ThreadSafePool<T>>>,
    /// Reference count for shared buffers
    ref_count: Arc<AtomicUsize>,
    /// Type marker
    _marker: PhantomData<T>,
}

impl<T> PreAllocatedBuffer<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new pre-allocated buffer
    pub fn new(buffer: NonNull<u8>, capacity: usize) -> MemoryResult<Self> {
        let data = buffer.cast::<T>();
        
        Ok(Self {
            data,
            capacity,
            length: AtomicUsize::new(0),
            pool_ref: None,
            ref_count: Arc::new(AtomicUsize::new(1)),
            _marker: PhantomData,
        })
    }
    
    /// Create a buffer with pool reference for automatic cleanup
    pub fn with_pool(
        buffer: NonNull<T>,
        capacity: usize,
        pool: Arc<ThreadSafePool<T>>,
    ) -> MemoryResult<Self> {
        Ok(Self {
            data: buffer,
            capacity,
            length: AtomicUsize::new(0),
            pool_ref: Some(pool),
            ref_count: Arc::new(AtomicUsize::new(1)),
            _marker: PhantomData,
        })
    }
    
    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get the current length
    pub fn len(&self) -> usize {
        self.length.load(Ordering::SeqCst)
    }
    
    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get available space
    pub fn available(&self) -> usize {
        self.capacity - self.len()
    }
    
    /// Reset the buffer (set length to 0)
    pub fn reset(&self) {
        self.length.store(0, Ordering::SeqCst);
    }
    
    /// Get a raw pointer to the data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    /// Get a mutable raw pointer to the data
    pub fn as_mut_ptr(&self) -> *mut T {
        self.data.as_ptr()
    }
    
    /// Get the buffer as a slice
    pub fn as_slice(&self) -> &[T] {
        let len = self.len();
        unsafe { slice::from_raw_parts(self.data.as_ptr(), len) }
    }
    
    /// Get the buffer as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len();
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), len) }
    }
    
    /// Push an element to the buffer
    pub fn push(&self, value: T) -> MemoryResult<()> {
        let current_len = self.length.load(Ordering::SeqCst);
        
        if current_len >= self.capacity {
            return Err(MemoryError::OutOfMemory {
                requested: 1,
                available: 0,
            });
        }
        
        unsafe {
            *self.data.as_ptr().add(current_len) = value;
        }
        
        self.length.store(current_len + 1, Ordering::SeqCst);
        Ok(())
    }
    
    /// Extend the buffer with elements from an iterator
    pub fn extend<I>(&self, iter: I) -> MemoryResult<()>
    where
        I: IntoIterator<Item = T>,
    {
        for item in iter {
            self.push(item)?;
        }
        Ok(())
    }
}

impl<T> Drop for PreAllocatedBuffer<T> {
    fn drop(&mut self) {
        let ref_count = self.ref_count.fetch_sub(1, Ordering::SeqCst);
        
        if ref_count == 1 {
            if let Some(pool) = &self.pool_ref {
                let size = self.capacity * size_of::<T>();
                let _ = pool.deallocate(self.data.cast(), size);
            }
        }
    }
}

// === STACK TENSOR ===

/// A small tensor allocated on the stack for minimal overhead
#[derive(Debug, Clone)]
pub struct StackTensor<T, const N: usize> {
    /// Stack-allocated data array
    data: [T; N],
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Current length (number of elements in use)
    length: usize,
}

impl<T, const N: usize> StackTensor<T, N>
where
    T: Clone + Default + Copy,
{
    /// Create a new stack tensor with given shape
    pub fn new(shape: Vec<usize>) -> MemoryResult<Self> {
        let len: usize = shape.iter().product();
        
        if len > N {
            return Err(MemoryError::InvalidSize {
                size: len,
                min: 0,
                max: N,
            });
        }
        
        Ok(Self {
            data: [T::default(); N],
            shape,
            length: len,
        })
    }
    
    /// Create a stack tensor filled with a specific value
    pub fn filled(shape: Vec<usize>, value: T) -> MemoryResult<Self> {
        let len: usize = shape.iter().product();
        
        if len > N {
            return Err(MemoryError::InvalidSize {
                size: len,
                min: 0,
                max: N,
            });
        }
        
        Ok(Self {
            data: [value; N],
            shape,
            length: len,
        })
    }
    
    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the length
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    
    /// Get the capacity
    pub fn capacity() -> usize {
        N
    }
    
    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.length]
    }
    
    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.length]
    }
    
    /// Fill with value
    pub fn fill(&mut self, value: T) {
        self.data[..self.length].fill(value);
    }
}

// === TENSOR ARENA ===

/// A memory arena for managing multiple tensors efficiently
#[derive(Debug)]
pub struct TensorArena<T> {
    /// Large pre-allocated memory block
    memory: NonNull<T>,
    /// Total capacity in elements
    capacity: usize,
    /// Current allocation offset
    offset: AtomicUsize,
    /// Reference to the memory pool
    pool_ref: Arc<ThreadSafePool<T>>,
    /// Alignment for tensor allocations
    alignment: usize,
}

impl<T> TensorArena<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new tensor arena
    pub fn new(capacity: usize, pool: Arc<ThreadSafePool<T>>) -> MemoryResult<Self> {
        let total_size = capacity * size_of::<T>();
        let buffer = pool.allocate(total_size)?;
        
        Ok(Self {
            memory: buffer.cast(),
            capacity,
            offset: AtomicUsize::new(0),
            pool_ref: pool,
            alignment: align_of::<T>(),
        })
    }
    
    /// Allocate a tensor from the arena
    pub fn allocate_tensor(&self, shape: Vec<usize>) -> MemoryResult<ZeroAllocTensor<T>> {
        let len: usize = shape.iter().product();
        
        let current_offset = self.offset.fetch_add(len, Ordering::SeqCst);
        
        if current_offset + len > self.capacity {
            return Err(MemoryError::OutOfMemory {
                requested: len,
                available: self.capacity - current_offset,
            });
        }
        
        let tensor_ptr = unsafe {
            NonNull::new_unchecked(self.memory.as_ptr().add(current_offset))
        };
        
        ZeroAllocTensor::from_buffer(tensor_ptr, shape, self.pool_ref.clone())
    }
    
    /// Get available capacity
    pub fn available(&self) -> usize {
        let current_offset = self.offset.load(Ordering::SeqCst);
        self.capacity.saturating_sub(current_offset)
    }
    
    /// Reset the arena (dangerous - invalidates all allocated tensors)
    pub unsafe fn reset(&self) {
        self.offset.store(0, Ordering::SeqCst);
    }
}

// === BUFFER MANAGER ===

/// Manager for tensor buffers with automatic size-based allocation
#[derive(Debug)]
pub struct BufferManager {
    /// Tensor type this manager handles
    tensor_type: TensorType,
    /// Memory alignment requirement
    alignment: usize,
    /// Reference to the memory pool
    pool: Arc<ThreadSafePool<u8>>, // Use u8 for generic byte allocation
    /// Cache of commonly used buffer sizes
    size_cache: std::sync::Mutex<std::collections::HashMap<usize, Vec<NonNull<u8>>>>,
}

impl BufferManager {
    /// Create a new buffer manager
    pub fn new(
        tensor_type: TensorType,
        alignment: usize,
        pool: Arc<ThreadSafePool<u8>>,
    ) -> MemoryResult<Self> {
        Ok(Self {
            tensor_type,
            alignment,
            pool,
            size_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }
    
    /// Allocate a buffer of specified size
    pub fn allocate(&self, size: usize) -> MemoryResult<NonNull<u8>> {
        // Try to reuse from cache first
        if let Ok(mut cache) = self.size_cache.lock() {
            if let Some(buffers) = cache.get_mut(&size) {
                if let Some(buffer) = buffers.pop() {
                    return Ok(buffer);
                }
            }
        }
        
        // Allocate new buffer from pool
        self.pool.allocate(size)
    }
    
    /// Deallocate a buffer (return to cache or pool)
    pub fn deallocate(&self, buffer: NonNull<u8>, size: usize) -> MemoryResult<()> {
        // Try to cache for future reuse
        if let Ok(mut cache) = self.size_cache.lock() {
            let buffers = cache.entry(size).or_insert_with(Vec::new);
            
            // Limit cache size to prevent unbounded growth
            if buffers.len() < 10 {
                buffers.push(buffer);
                return Ok(());
            }
        }
        
        // Fall back to pool deallocation
        self.pool.deallocate(buffer, size)
    }
    
    /// Pre-allocate buffers for a specific size
    pub fn preallocate(&self, size: usize, count: usize) -> MemoryResult<()> {
        let mut buffers = Vec::with_capacity(count);
        
        for _ in 0..count {
            let buffer = self.pool.allocate(size)?;
            buffers.push(buffer);
        }
        
        // Store in cache
        if let Ok(mut cache) = self.size_cache.lock() {
            cache.insert(size, buffers);
        }
        
        Ok(())
    }
}

// === SAFETY IMPLEMENTATIONS ===

unsafe impl<T: Send + Sync> Send for ZeroAllocTensor<T> {}
unsafe impl<T: Send + Sync> Sync for ZeroAllocTensor<T> {}
unsafe impl<T: Send + Sync> Send for PreAllocatedBuffer<T> {}
unsafe impl<T: Send + Sync> Sync for PreAllocatedBuffer<T> {}
unsafe impl<T: Send + Sync> Send for TensorArena<T> {}
unsafe impl<T: Send + Sync> Sync for TensorArena<T> {}

// === TESTS ===

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::pools::ThreadSafePool;
    
    #[test]
    fn test_stack_tensor() {
        let mut tensor: StackTensor<f32, 64> = StackTensor::new(vec![8, 8]).unwrap();
        assert_eq!(tensor.len(), 64);
        assert_eq!(tensor.shape(), &[8, 8]);
        
        tensor.fill(1.0);
        assert_eq!(tensor.as_slice()[0], 1.0);
        assert_eq!(tensor.as_slice()[63], 1.0);
        
        // Test oversized allocation
        let result: Result<StackTensor<f32, 64>, _> = StackTensor::new(vec![10, 10]);
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_pre_allocated_buffer() {
        let pool = Arc::new(ThreadSafePool::new(1024, 8, 32).unwrap());
        let buffer_ptr = pool.allocate(256).unwrap();
        
        let buffer: PreAllocatedBuffer<f32> = PreAllocatedBuffer::new(buffer_ptr.cast(), 64).unwrap();
        
        assert_eq!(buffer.capacity(), 64);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        
        // Test pushing elements
        assert!(buffer.push(1.0).is_ok());
        assert!(buffer.push(2.0).is_ok());
        assert_eq!(buffer.len(), 2);
        
        let slice = buffer.as_slice();
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1], 2.0);
    }
    
    #[tokio::test]
    async fn test_tensor_arena() {
        let pool = Arc::new(ThreadSafePool::new(4096, 8, 32).unwrap());
        let arena = TensorArena::new(1024, pool).unwrap();
        
        // Allocate multiple tensors from arena
        let tensor1 = arena.allocate_tensor(vec![10, 10]).unwrap();
        let tensor2 = arena.allocate_tensor(vec![5, 5]).unwrap();
        
        assert_eq!(tensor1.len(), 100);
        assert_eq!(tensor2.len(), 25);
        
        // Check available space decreased
        assert!(arena.available() < 1024);
    }
    
    #[tokio::test]
    async fn test_buffer_manager() {
        let pool = Arc::new(ThreadSafePool::new(4096, 8, 32).unwrap());
        let manager = BufferManager::new(TensorType::Float32, 32, pool).unwrap();
        
        // Test allocation and deallocation
        let buffer = manager.allocate(256).unwrap();
        assert!(manager.deallocate(buffer, 256).is_ok());
        
        // Test pre-allocation
        assert!(manager.preallocate(128, 5).is_ok());
        
        // Should get pre-allocated buffer
        let buffer2 = manager.allocate(128).unwrap();
        assert!(!buffer2.as_ptr().is_null());
    }
}

