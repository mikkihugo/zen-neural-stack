//! Memory layout optimization for SIMD operations
//!
//! This module provides memory-efficient data structures and layouts
//! specifically optimized for SIMD operations and cache performance:
//!
//! - SIMD-aligned memory allocation
//! - Cache-friendly data layouts (AoS vs SoA)
//! - Memory pool management
//! - Zero-copy operations where possible
//! - Prefetching strategies

use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::marker::PhantomData;

/// SIMD-aligned memory allocator
pub struct SimdAllocator {
    alignment: usize,
}

impl SimdAllocator {
    /// Create a new SIMD allocator with the specified alignment
    pub fn new(alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be a power of 2");
        Self { alignment }
    }

    /// Create allocator with optimal alignment for current architecture
    pub fn optimal() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                Self::new(64) // 512-bit alignment
            } else if is_x86_feature_detected!("avx2") {
                Self::new(32) // 256-bit alignment
            } else {
                Self::new(16) // 128-bit alignment
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::new(16) // 128-bit NEON alignment
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::new(16) // Conservative default
        }
    }

    /// Allocate aligned memory for the given number of elements
    pub fn allocate<T>(&self, count: usize) -> Result<SimdBuffer<T>, std::alloc::AllocError> {
        let size = count * std::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, self.alignment)
            .map_err(|_| std::alloc::AllocError)?;

        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(std::alloc::AllocError);
        }

        Ok(SimdBuffer {
            ptr: NonNull::new(ptr as *mut T).unwrap(),
            capacity: count,
            layout,
            _phantom: PhantomData,
        })
    }

    /// Allocate and initialize with data
    pub fn allocate_with_data<T: Copy>(&self, data: &[T]) -> Result<SimdBuffer<T>, std::alloc::AllocError> {
        let mut buffer = self.allocate(data.len())?;
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), buffer.as_mut_ptr(), data.len());
        }
        Ok(buffer)
    }
}

/// SIMD-aligned buffer with automatic deallocation
pub struct SimdBuffer<T> {
    ptr: NonNull<T>,
    capacity: usize,
    layout: Layout,
    _phantom: PhantomData<T>,
}

impl<T> SimdBuffer<T> {
    /// Get a slice view of the buffer
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) }
    }

    /// Get a mutable slice view of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get buffer capacity
    pub fn len(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is properly aligned for SIMD
    pub fn is_simd_aligned(&self) -> bool {
        let alignment = if cfg!(target_arch = "x86_64") {
            if is_x86_feature_detected!("avx512f") { 64 }
            else if is_x86_feature_detected!("avx2") { 32 }
            else { 16 }
        } else {
            16
        };

        self.ptr.as_ptr() as usize % alignment == 0
    }
}

impl<T> Drop for SimdBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

unsafe impl<T> Send for SimdBuffer<T> where T: Send {}
unsafe impl<T> Sync for SimdBuffer<T> where T: Sync {}

/// Structure of Arrays (SoA) layout for better SIMD performance
#[derive(Debug, Clone)]
pub struct SoAMatrix<T> {
    /// Data stored in column-major order for better SIMD access
    data: Vec<T>,
    rows: usize,
    cols: usize,
    /// Padding to ensure each column is SIMD-aligned
    col_stride: usize,
}

impl<T: Copy + Default> SoAMatrix<T> {
    /// Create a new SoA matrix with SIMD-friendly layout
    pub fn new(rows: usize, cols: usize) -> Self {
        let simd_width = Self::get_simd_width();
        let col_stride = ((rows + simd_width - 1) / simd_width) * simd_width;
        let total_size = col_stride * cols;
        
        let mut data = Vec::with_capacity(total_size);
        data.resize(total_size, T::default());

        Self {
            data,
            rows,
            cols,
            col_stride,
        }
    }

    /// Create from row-major data (converts to column-major)
    pub fn from_row_major(data: &[T], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        
        let mut matrix = Self::new(rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                matrix.set(i, j, data[i * cols + j]);
            }
        }
        
        matrix
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows && col < self.cols);
        self.data[col * self.col_stride + row]
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.cols);
        self.data[col * self.col_stride + row] = value;
    }

    /// Get a column slice (SIMD-friendly access)
    pub fn column(&self, col: usize) -> &[T] {
        assert!(col < self.cols);
        let start = col * self.col_stride;
        &self.data[start..start + self.rows]
    }

    /// Get a mutable column slice
    pub fn column_mut(&mut self, col: usize) -> &mut [T] {
        assert!(col < self.cols);
        let start = col * self.col_stride;
        &mut self.data[start..start + self.rows]
    }

    /// Get pointer to column data for SIMD operations
    pub fn column_ptr(&self, col: usize) -> *const T {
        assert!(col < self.cols);
        self.data.as_ptr().wrapping_add(col * self.col_stride)
    }

    /// Get mutable pointer to column data
    pub fn column_ptr_mut(&mut self, col: usize) -> *mut T {
        assert!(col < self.cols);
        self.data.as_mut_ptr().wrapping_add(col * self.col_stride)
    }

    /// Convert back to row-major layout
    pub fn to_row_major(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.rows * self.cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.push(self.get(i, j));
            }
        }
        
        result
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get SIMD width for current architecture
    fn get_simd_width() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                16 // 512 bits / 32 bits per f32
            } else if is_x86_feature_detected!("avx2") {
                8  // 256 bits / 32 bits per f32
            } else {
                4  // 128 bits / 32 bits per f32
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            4 // 128 bits / 32 bits per f32
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            4 // Conservative default
        }
    }
}

/// Memory pool for reducing allocation overhead
pub struct SimdMemoryPool<T> {
    allocator: SimdAllocator,
    pools: Vec<Vec<SimdBuffer<T>>>,
    size_classes: Vec<usize>,
}

impl<T: Copy + Default> SimdMemoryPool<T> {
    /// Create a new memory pool with size classes
    pub fn new() -> Self {
        let size_classes = vec![
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
        ];
        
        let pools = size_classes.iter().map(|_| Vec::new()).collect();
        
        Self {
            allocator: SimdAllocator::optimal(),
            pools,
            size_classes,
        }
    }

    /// Allocate buffer from pool or create new one
    pub fn allocate(&mut self, size: usize) -> Result<SimdBuffer<T>, std::alloc::AllocError> {
        // Find appropriate size class
        let size_class_idx = self.size_classes
            .iter()
            .position(|&s| s >= size)
            .unwrap_or(self.size_classes.len() - 1);

        // Try to reuse from pool
        if let Some(buffer) = self.pools[size_class_idx].pop() {
            Ok(buffer)
        } else {
            // Allocate new buffer
            let actual_size = self.size_classes[size_class_idx];
            self.allocator.allocate(actual_size)
        }
    }

    /// Return buffer to pool for reuse
    pub fn deallocate(&mut self, buffer: SimdBuffer<T>) {
        let capacity = buffer.len();
        
        // Find appropriate size class
        if let Some(size_class_idx) = self.size_classes.iter().position(|&s| s == capacity) {
            // Clear buffer and return to pool
            // Note: In a real implementation, you'd want to zero out the buffer
            self.pools[size_class_idx].push(buffer);
        }
        // If no matching size class, buffer will be dropped automatically
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        for pool in &mut self.pools {
            pool.clear();
        }
    }
}

impl<T: Copy + Default> Default for SimdMemoryPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Prefetching utilities for improved memory performance
pub struct PrefetchManager {
    strategy: PrefetchStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Prefetch to L1 cache
    L1,
    /// Prefetch to L2 cache  
    L2,
    /// Prefetch to L3 cache
    L3,
    /// Prefetch for non-temporal access (bypass cache)
    NonTemporal,
}

impl PrefetchManager {
    pub fn new(strategy: PrefetchStrategy) -> Self {
        Self { strategy }
    }

    /// Prefetch memory location
    pub fn prefetch<T>(&self, ptr: *const T) {
        match self.strategy {
            PrefetchStrategy::None => {},
            PrefetchStrategy::L1 => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                }
            },
            PrefetchStrategy::L2 => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T1);
                }
            },
            PrefetchStrategy::L3 => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T2);
                }
            },
            PrefetchStrategy::NonTemporal => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_NTA);
                }
            },
        }
    }

    /// Prefetch range of memory
    pub fn prefetch_range<T>(&self, ptr: *const T, count: usize) {
        let cache_line_size = 64; // bytes
        let element_size = std::mem::size_of::<T>();
        let elements_per_line = cache_line_size / element_size;
        
        for i in (0..count).step_by(elements_per_line) {
            unsafe {
                self.prefetch(ptr.add(i));
            }
        }
    }
}

/// Cache-friendly matrix blocking parameters
#[derive(Debug, Clone, Copy)]
pub struct BlockingParams {
    pub mc: usize, // M dimension blocking (L2 cache)
    pub nc: usize, // N dimension blocking (L3 cache)
    pub kc: usize, // K dimension blocking (L1 cache)
}

impl BlockingParams {
    /// Calculate optimal blocking parameters based on cache sizes
    pub fn optimal() -> Self {
        let l1_size = 32 * 1024;   // 32KB L1
        let l2_size = 256 * 1024;  // 256KB L2 
        let l3_size = 8 * 1024 * 1024; // 8MB L3
        
        let element_size = std::mem::size_of::<f32>();
        
        // L1 blocking: fit A slice and B panel
        let kc = (l1_size / (2 * element_size)).min(384);
        
        // L2 blocking: fit A block and B panel
        let mc = (l2_size / element_size / kc).min(512);
        
        // L3 blocking: fit entire C block
        let nc = (l3_size / element_size / mc).min(4096);

        Self { mc, nc, kc }
    }

    /// Get parameters for specific cache configuration
    pub fn for_cache_sizes(l1: usize, l2: usize, l3: usize) -> Self {
        let element_size = std::mem::size_of::<f32>();
        
        let kc = (l1 / (2 * element_size)).min(512);
        let mc = (l2 / element_size / kc).min(1024);
        let nc = (l3 / element_size / mc).min(8192);

        Self { mc, nc, kc }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_allocator() {
        let allocator = SimdAllocator::optimal();
        let buffer = allocator.allocate::<f32>(1024).unwrap();
        
        assert_eq!(buffer.len(), 1024);
        assert!(buffer.is_simd_aligned());
    }

    #[test]
    fn test_soa_matrix() {
        let mut matrix = SoAMatrix::<f32>::new(4, 3);
        
        // Set some values
        matrix.set(1, 2, 42.0);
        assert_eq!(matrix.get(1, 2), 42.0);
        
        // Test dimensions
        assert_eq!(matrix.dimensions(), (4, 3));
        
        // Test column access
        let col = matrix.column(0);
        assert_eq!(col.len(), 4);
    }

    #[test]
    fn test_row_major_conversion() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let matrix = SoAMatrix::from_row_major(&data, 2, 3);
        
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 1), 2.0);
        assert_eq!(matrix.get(1, 2), 6.0);
        
        let recovered = matrix.to_row_major();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = SimdMemoryPool::<f32>::new();
        
        let buffer1 = pool.allocate(128).unwrap();
        let buffer2 = pool.allocate(256).unwrap();
        
        assert!(buffer1.len() >= 128);
        assert!(buffer2.len() >= 256);
        
        pool.deallocate(buffer1);
        pool.deallocate(buffer2);
        
        // Should reuse from pool
        let buffer3 = pool.allocate(128).unwrap();
        assert!(buffer3.len() >= 128);
    }

    #[test]
    fn test_blocking_params() {
        let params = BlockingParams::optimal();
        
        assert!(params.kc > 0);
        assert!(params.mc > 0);
        assert!(params.nc > 0);
        
        // Should be reasonable values
        assert!(params.kc <= 1024);
        assert!(params.mc <= 2048);
        assert!(params.nc <= 16384);
    }

    #[test]
    fn test_prefetch_manager() {
        let manager = PrefetchManager::new(PrefetchStrategy::L1);
        let data = vec![1.0f32; 1000];
        
        // Should not panic
        manager.prefetch(data.as_ptr());
        manager.prefetch_range(data.as_ptr(), data.len());
    }
}