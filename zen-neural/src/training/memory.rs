//! Memory optimization utilities for high-performance training
//!
//! This module provides memory pools, buffer management, and zero-allocation
//! utilities to minimize garbage collection pressure and maximize training
//! performance in memory-constrained environments.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use num_traits::Float;

use crate::TrainingError;

/// Training memory manager with object pools and buffer reuse
pub struct TrainingMemoryManager<T: Float> {
    /// Pool of gradient buffers
    gradient_pools: Arc<Mutex<Vec<Vec<T>>>>,
    /// Pool of activation buffers
    activation_pools: Arc<Mutex<Vec<Vec<T>>>>,
    /// Pool of temporary computation buffers
    temp_pools: Arc<Mutex<Vec<Vec<T>>>>,
    /// Buffer size tracking
    buffer_sizes: HashMap<BufferType, usize>,
    /// Memory usage statistics
    memory_stats: MemoryStats,
    /// Pool configuration
    config: MemoryPoolConfig,
}

impl<T: Float + Clone + Default + Send + Sync> TrainingMemoryManager<T> {
    /// Create a new memory manager with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryPoolConfig::default())
    }
    
    /// Create a new memory manager with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        Self {
            gradient_pools: Arc::new(Mutex::new(Vec::new())),
            activation_pools: Arc::new(Mutex::new(Vec::new())),
            temp_pools: Arc::new(Mutex::new(Vec::new())),
            buffer_sizes: HashMap::new(),
            memory_stats: MemoryStats::new(),
            config,
        }
    }
    
    /// Get a gradient buffer from the pool or create a new one
    pub fn get_gradient_buffer(&mut self, size: usize) -> Result<ManagedBuffer<T>, TrainingError> {
        self.get_buffer_from_pool(&self.gradient_pools.clone(), BufferType::Gradient, size)
    }
    
    /// Get an activation buffer from the pool or create a new one
    pub fn get_activation_buffer(&mut self, size: usize) -> Result<ManagedBuffer<T>, TrainingError> {
        self.get_buffer_from_pool(&self.activation_pools.clone(), BufferType::Activation, size)
    }
    
    /// Get a temporary buffer from the pool or create a new one
    pub fn get_temp_buffer(&mut self, size: usize) -> Result<ManagedBuffer<T>, TrainingError> {
        self.get_buffer_from_pool(&self.temp_pools.clone(), BufferType::Temporary, size)
    }
    
    /// Return a buffer to the appropriate pool for reuse
    pub fn return_buffer(&mut self, buffer: ManagedBuffer<T>) {
        let pool = match buffer.buffer_type {
            BufferType::Gradient => &self.gradient_pools,
            BufferType::Activation => &self.activation_pools,
            BufferType::Temporary => &self.temp_pools,
        };
        
        if let Ok(mut pool_guard) = pool.lock() {
            // Clear the buffer and return to pool if within size limits
            let mut vec_buffer = buffer.into_inner();
            
            if pool_guard.len() < self.config.max_pool_size {
                vec_buffer.clear();
                vec_buffer.resize(vec_buffer.capacity(), T::default());
                pool_guard.push(vec_buffer);
                self.memory_stats.buffers_reused += 1;
            } else {
                // Pool is full, let buffer be dropped
                self.memory_stats.buffers_dropped += 1;
            }
        }
    }
    
    /// Pre-allocate buffers for a training session
    pub fn preallocate_buffers(
        &mut self,
        gradient_size: usize,
        activation_size: usize,
        temp_size: usize,
        count: usize,
    ) -> Result<(), TrainingError> {
        // Preallocate gradient buffers
        if let Ok(mut pool) = self.gradient_pools.lock() {
            for _ in 0..count {
                pool.push(vec![T::default(); gradient_size]);
            }
        }
        
        // Preallocate activation buffers
        if let Ok(mut pool) = self.activation_pools.lock() {
            for _ in 0..count {
                pool.push(vec![T::default(); activation_size]);
            }
        }
        
        // Preallocate temporary buffers
        if let Ok(mut pool) = self.temp_pools.lock() {
            for _ in 0..count {
                pool.push(vec![T::default(); temp_size]);
            }
        }
        
        self.memory_stats.buffers_preallocated += count * 3;
        Ok(())
    }
    
    /// Clear all buffer pools
    pub fn clear_pools(&mut self) {
        if let Ok(mut pool) = self.gradient_pools.lock() {
            pool.clear();
        }
        if let Ok(mut pool) = self.activation_pools.lock() {
            pool.clear();
        }
        if let Ok(mut pool) = self.temp_pools.lock() {
            pool.clear();
        }
        
        self.memory_stats.pools_cleared += 1;
    }
    
    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> &MemoryStats {
        &self.memory_stats
    }
    
    /// Reset memory statistics
    pub fn reset_stats(&mut self) {
        self.memory_stats = MemoryStats::new();
    }
    
    /// Get estimated memory usage in bytes
    pub fn get_estimated_memory_usage(&self) -> usize {
        let mut total_size = 0;
        
        if let Ok(pool) = self.gradient_pools.lock() {
            total_size += pool.iter().map(|v| v.capacity() * std::mem::size_of::<T>()).sum::<usize>();
        }
        if let Ok(pool) = self.activation_pools.lock() {
            total_size += pool.iter().map(|v| v.capacity() * std::mem::size_of::<T>()).sum::<usize>();
        }
        if let Ok(pool) = self.temp_pools.lock() {
            total_size += pool.iter().map(|v| v.capacity() * std::mem::size_of::<T>()).sum::<usize>();
        }
        
        total_size
    }
    
    /// Internal method to get buffer from pool
    fn get_buffer_from_pool(
        &mut self,
        pool: &Arc<Mutex<Vec<Vec<T>>>>,
        buffer_type: BufferType,
        size: usize,
    ) -> Result<ManagedBuffer<T>, TrainingError> {
        let buffer = if let Ok(mut pool_guard) = pool.lock() {
            if let Some(mut vec_buffer) = pool_guard.pop() {
                // Resize buffer if needed
                if vec_buffer.len() != size {
                    vec_buffer.resize(size, T::default());
                }
                self.memory_stats.buffers_reused += 1;
                vec_buffer
            } else {
                // Create new buffer
                self.memory_stats.buffers_created += 1;
                vec![T::default(); size]
            }
        } else {
            return Err(TrainingError::TrainingFailed(
                "Failed to acquire buffer pool lock".to_string()
            ));
        };
        
        Ok(ManagedBuffer::new(buffer, buffer_type, self.config.auto_return))
    }
}

/// Managed buffer that can be automatically returned to pool
pub struct ManagedBuffer<T: Float> {
    buffer: Vec<T>,
    buffer_type: BufferType,
    auto_return: bool,
}

impl<T: Float + Clone + Default> ManagedBuffer<T> {
    fn new(buffer: Vec<T>, buffer_type: BufferType, auto_return: bool) -> Self {
        Self {
            buffer,
            buffer_type,
            auto_return,
        }
    }
    
    /// Get mutable reference to the underlying buffer
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buffer
    }
    
    /// Get immutable reference to the underlying buffer
    pub fn as_slice(&self) -> &[T] {
        &self.buffer
    }
    
    /// Get the size of the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Fill buffer with a specific value
    pub fn fill(&mut self, value: T) {
        self.buffer.fill(value);
    }
    
    /// Clear buffer (set all elements to default)
    pub fn clear(&mut self) {
        self.buffer.fill(T::default());
    }
    
    /// Resize the buffer
    pub fn resize(&mut self, new_size: usize) {
        self.buffer.resize(new_size, T::default());
    }
    
    /// Extract the underlying vector (consuming the managed buffer)
    pub fn into_inner(self) -> Vec<T> {
        self.buffer
    }
    
    /// Get a reference to the underlying vector
    pub fn as_vec(&self) -> &Vec<T> {
        &self.buffer
    }
    
    /// Get a mutable reference to the underlying vector
    pub fn as_vec_mut(&mut self) -> &mut Vec<T> {
        &mut self.buffer
    }
}

impl<T: Float + Clone + Default> std::ops::Index<usize> for ManagedBuffer<T> {
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer[index]
    }
}

impl<T: Float + Clone + Default> std::ops::IndexMut<usize> for ManagedBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buffer[index]
    }
}

/// Buffer types for different use cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferType {
    Gradient,
    Activation,
    Temporary,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum number of buffers to keep in each pool
    pub max_pool_size: usize,
    /// Whether buffers should be automatically returned to pool on drop
    pub auto_return: bool,
    /// Initial pool sizes
    pub initial_pool_sizes: HashMap<BufferType, usize>,
    /// Whether to preallocate buffers on startup
    pub preallocate_on_init: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        let mut initial_sizes = HashMap::new();
        initial_sizes.insert(BufferType::Gradient, 10);
        initial_sizes.insert(BufferType::Activation, 10);
        initial_sizes.insert(BufferType::Temporary, 5);
        
        Self {
            max_pool_size: 50,
            auto_return: true,
            initial_pool_sizes: initial_sizes,
            preallocate_on_init: false,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub buffers_created: usize,
    pub buffers_reused: usize,
    pub buffers_dropped: usize,
    pub buffers_preallocated: usize,
    pub pools_cleared: usize,
    pub start_time: Instant,
}

impl MemoryStats {
    pub fn new() -> Self {
        Self {
            buffers_created: 0,
            buffers_reused: 0,
            buffers_dropped: 0,
            buffers_preallocated: 0,
            pools_cleared: 0,
            start_time: Instant::now(),
        }
    }
    
    /// Get buffer reuse ratio (reused / total)
    pub fn get_reuse_ratio(&self) -> f64 {
        let total = self.buffers_created + self.buffers_reused;
        if total == 0 {
            0.0
        } else {
            self.buffers_reused as f64 / total as f64
        }
    }
    
    /// Get total runtime
    pub fn get_runtime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
    
    /// Get buffers per second creation rate
    pub fn get_creation_rate(&self) -> f64 {
        let runtime_secs = self.get_runtime().as_secs_f64();
        if runtime_secs == 0.0 {
            0.0
        } else {
            self.buffers_created as f64 / runtime_secs
        }
    }
}

/// Zero-allocation vector operations
pub struct ZeroAllocOps;

impl ZeroAllocOps {
    /// Add two vectors in-place (no allocation)
    pub fn add_inplace<T: Float + Clone>(target: &mut [T], source: &[T]) -> Result<(), TrainingError> {
        if target.len() != source.len() {
            return Err(TrainingError::TrainingFailed("Vector size mismatch".to_string()));
        }
        
        for (t, &s) in target.iter_mut().zip(source.iter()) {
            *t = *t + s;
        }
        
        Ok(())
    }
    
    /// Subtract two vectors in-place (no allocation)
    pub fn sub_inplace<T: Float + Clone>(target: &mut [T], source: &[T]) -> Result<(), TrainingError> {
        if target.len() != source.len() {
            return Err(TrainingError::TrainingFailed("Vector size mismatch".to_string()));
        }
        
        for (t, &s) in target.iter_mut().zip(source.iter()) {
            *t = *t - s;
        }
        
        Ok(())
    }
    
    /// Multiply vector by scalar in-place (no allocation)
    pub fn scale_inplace<T: Float + Clone>(target: &mut [T], scalar: T) {
        for t in target.iter_mut() {
            *t = *t * scalar;
        }
    }
    
    /// Compute dot product without allocation
    pub fn dot_product<T: Float + Clone>(a: &[T], b: &[T]) -> Result<T, TrainingError> {
        if a.len() != b.len() {
            return Err(TrainingError::TrainingFailed("Vector size mismatch".to_string()));
        }
        
        let mut result = T::zero();
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            result = result + ai * bi;
        }
        
        Ok(result)
    }
    
    /// Compute L2 norm without allocation
    pub fn l2_norm<T: Float + Clone>(vector: &[T]) -> T {
        let sum_of_squares = vector.iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x);
        sum_of_squares.sqrt()
    }
    
    /// Normalize vector in-place
    pub fn normalize_inplace<T: Float + Clone>(vector: &mut [T]) -> Result<(), TrainingError> {
        let norm = Self::l2_norm(vector);
        if norm == T::zero() {
            return Err(TrainingError::TrainingFailed("Cannot normalize zero vector".to_string()));
        }
        
        Self::scale_inplace(vector, T::one() / norm);
        Ok(())
    }
    
    /// Copy vector without allocation (panics if sizes don't match)
    pub fn copy_noalloc<T: Float + Clone>(target: &mut [T], source: &[T]) -> Result<(), TrainingError> {
        if target.len() != source.len() {
            return Err(TrainingError::TrainingFailed("Vector size mismatch".to_string()));
        }
        
        target.copy_from_slice(source);
        Ok(())
    }
    
    /// Accumulate gradients in-place with averaging
    pub fn accumulate_gradients<T: Float + Clone>(
        accumulated: &mut [T],
        gradients: &[T],
        count: usize,
    ) -> Result<(), TrainingError> {
        if accumulated.len() != gradients.len() {
            return Err(TrainingError::TrainingFailed("Gradient size mismatch".to_string()));
        }
        
        if count == 0 {
            return Err(TrainingError::TrainingFailed("Cannot accumulate with zero count".to_string()));
        }
        
        let weight = T::one() / T::from(count).unwrap();
        for (acc, &grad) in accumulated.iter_mut().zip(gradients.iter()) {
            *acc = *acc + weight * grad;
        }
        
        Ok(())
    }
}

/// Memory-efficient batch processor
pub struct BatchProcessor<T: Float> {
    batch_buffers: VecDeque<Vec<T>>,
    current_batch: Vec<T>,
    batch_size: usize,
    element_size: usize,
}

impl<T: Float + Clone + Default> BatchProcessor<T> {
    pub fn new(batch_size: usize, element_size: usize) -> Self {
        Self {
            batch_buffers: VecDeque::new(),
            current_batch: Vec::with_capacity(batch_size * element_size),
            batch_size,
            element_size,
        }
    }
    
    /// Add element to current batch
    pub fn add_element(&mut self, element: &[T]) -> Result<(), TrainingError> {
        if element.len() != self.element_size {
            return Err(TrainingError::TrainingFailed("Element size mismatch".to_string()));
        }
        
        self.current_batch.extend_from_slice(element);
        
        // If batch is full, move to buffer queue
        if self.current_batch.len() >= self.batch_size * self.element_size {
            let mut full_batch = Vec::with_capacity(self.batch_size * self.element_size);
            std::mem::swap(&mut full_batch, &mut self.current_batch);
            self.batch_buffers.push_back(full_batch);
        }
        
        Ok(())
    }
    
    /// Get next complete batch
    pub fn get_next_batch(&mut self) -> Option<Vec<T>> {
        self.batch_buffers.pop_front()
    }
    
    /// Get remaining elements in current (incomplete) batch
    pub fn get_remaining(&mut self) -> Option<Vec<T>> {
        if self.current_batch.is_empty() {
            None
        } else {
            let mut remaining = Vec::new();
            std::mem::swap(&mut remaining, &mut self.current_batch);
            Some(remaining)
        }
    }
    
    /// Get number of complete batches ready
    pub fn batch_count(&self) -> usize {
        self.batch_buffers.len()
    }
    
    /// Clear all batches
    pub fn clear(&mut self) {
        self.batch_buffers.clear();
        self.current_batch.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_manager() {
        let mut manager = TrainingMemoryManager::<f32>::new();
        
        // Test buffer allocation and reuse
        let buffer1 = manager.get_gradient_buffer(10).unwrap();
        assert_eq!(buffer1.len(), 10);
        
        manager.return_buffer(buffer1);
        
        let buffer2 = manager.get_gradient_buffer(10).unwrap();
        assert_eq!(buffer2.len(), 10);
        
        let stats = manager.get_memory_stats();
        assert_eq!(stats.buffers_created, 1);
        assert_eq!(stats.buffers_reused, 1);
    }
    
    #[test]
    fn test_managed_buffer() {
        let buffer_vec = vec![1.0f32, 2.0, 3.0];
        let mut buffer = ManagedBuffer::new(buffer_vec, BufferType::Gradient, false);
        
        assert_eq!(buffer.len(), 3);
        assert!(!buffer.is_empty());
        assert_eq!(buffer[0], 1.0);
        
        buffer.fill(0.0);
        assert_eq!(buffer[0], 0.0);
        assert_eq!(buffer[1], 0.0);
        assert_eq!(buffer[2], 0.0);
    }
    
    #[test]
    fn test_zero_alloc_ops() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b = vec![0.5, 1.0, 1.5];
        
        ZeroAllocOps::add_inplace(&mut a, &b).unwrap();
        assert_eq!(a, vec![1.5, 3.0, 4.5]);
        
        ZeroAllocOps::scale_inplace(&mut a, 2.0);
        assert_eq!(a, vec![3.0, 6.0, 9.0]);
        
        let dot = ZeroAllocOps::dot_product(&a, &b).unwrap();
        assert!((dot - 15.0).abs() < 1e-6); // 3*0.5 + 6*1.0 + 9*1.5 = 22.5
        
        let norm = ZeroAllocOps::l2_norm(&a);
        assert!((norm - (9.0 + 36.0 + 81.0_f32).sqrt()).abs() < 1e-6);
    }
    
    #[test]
    fn test_batch_processor() {
        let mut processor = BatchProcessor::<f32>::new(2, 3);
        
        processor.add_element(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(processor.batch_count(), 0);
        
        processor.add_element(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(processor.batch_count(), 1);
        
        let batch = processor.get_next_batch().unwrap();
        assert_eq!(batch.len(), 6);
        assert_eq!(batch, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
    
    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats::new();
        assert_eq!(stats.get_reuse_ratio(), 0.0);
        
        let mut stats = MemoryStats::new();
        stats.buffers_created = 5;
        stats.buffers_reused = 15;
        
        assert!((stats.get_reuse_ratio() - 0.75).abs() < 1e-6);
    }
    
    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.max_pool_size, 50);
        assert!(config.auto_return);
        assert_eq!(config.initial_pool_sizes.len(), 3);
    }
}