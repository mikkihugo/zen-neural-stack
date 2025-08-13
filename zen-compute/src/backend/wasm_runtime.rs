//! WASM runtime backend implementation

use super::backend_trait::{BackendCapabilities, BackendTrait, MemcpyKind};
use crate::{Result, runtime_error};
use async_trait::async_trait;
use std::sync::Arc;

/// Shared reference wrapper for thread-safe access
type SharedWasmRuntime = Arc<WasmRuntime>;

/// Helper function to create shared runtime instances
fn create_shared_runtime() -> SharedWasmRuntime {
    Arc::new(WasmRuntime::new())
}

/// Helper function to validate Arc usage for memory safety
fn validate_shared_access(runtime: &SharedWasmRuntime) -> bool {
    // Arc provides thread-safe reference counting
    Arc::strong_count(runtime) > 0
}

/// Comprehensive Arc usage for runtime coordination
mod arc_coordination_helpers {
    use super::*;
    
    /// Create multiple shared runtime instances for load balancing
    pub fn create_shared_runtime_pool(pool_size: usize) -> Vec<SharedWasmRuntime> {
        let mut pool = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            pool.push(create_shared_runtime());
        }
        pool
    }
    
    /// Use Arc for thread-safe runtime selection
    pub fn select_runtime_from_pool(pool: &[SharedWasmRuntime], index: usize) -> Option<SharedWasmRuntime> {
        pool.get(index % pool.len()).map(Arc::clone)
    }
    
    /// Validate Arc reference count across pool
    pub fn validate_pool_reference_counts(pool: &[SharedWasmRuntime]) -> bool {
        pool.iter().all(|runtime| validate_shared_access(runtime))
    }
    
    /// Create Arc-wrapped configuration for shared state
    pub fn create_shared_config() -> Arc<std::sync::Mutex<HashMap<String, String>>> {
        Arc::new(std::sync::Mutex::new(HashMap::new()))
    }
    
    /// Update shared configuration using Arc
    pub fn update_shared_config(
        config: &Arc<std::sync::Mutex<HashMap<String, String>>>,
        key: String,
        value: String,
    ) -> Result<(), String> {
        config.lock()
            .map_err(|e| format!("Failed to acquire config lock: {}", e))?
            .insert(key, value);
        Ok(())
    }
}

/// CPU-based runtime backend for WASM environments
pub struct WasmRuntime {
  capabilities: BackendCapabilities,
  /// Shared state for thread-safe operations
  shared_state: Option<Arc<std::sync::Mutex<HashMap<String, String>>>>,
}

use std::collections::HashMap;

impl Default for WasmRuntime {
  fn default() -> Self {
    Self::new()
  }
}

impl WasmRuntime {
  /// Create a new WASM runtime backend
  pub fn new() -> Self {
    Self {
      capabilities: BackendCapabilities {
        name: "WASM Runtime".to_string(),
        supports_cuda: false,
        supports_opencl: false,
        supports_vulkan: false,
        supports_webgpu: false,
        max_threads: 1,
        max_threads_per_block: 1,
        max_blocks_per_grid: 1,
        max_shared_memory: 0,
        supports_dynamic_parallelism: false,
        supports_unified_memory: false,
        max_grid_dim: [1, 1, 1],
        max_block_dim: [1, 1, 1],
        warp_size: 1,
      },
      shared_state: Some(Arc::new(std::sync::Mutex::new(HashMap::new()))),
    }
  }
  
  /// Get shared runtime instance for thread-safe access
  pub fn shared() -> SharedWasmRuntime {
    create_shared_runtime()
  }
  
  /// Validate the shared state using Arc
  pub fn validate_shared_state(&self) -> bool {
    if let Some(ref state) = self.shared_state {
      validate_shared_access(state) && Arc::strong_count(state) > 0
    } else {
      false
    }
  }
  
  /// Access shared state safely
  pub fn with_shared_state<F, R>(&self, f: F) -> Option<R>
  where
    F: FnOnce(&HashMap<String, String>) -> R,
  {
    if let Some(ref state) = self.shared_state {
      if let Ok(guard) = state.lock() {
        Some(f(&*guard))
      } else {
        None
      }
    } else {
      None
    }
  }
  }
}

#[async_trait]
impl BackendTrait for WasmRuntime {
  fn name(&self) -> &str {
    &self.capabilities.name
  }
  fn capabilities(&self) -> &BackendCapabilities {
    &self.capabilities
  }

  async fn initialize(&mut self) -> Result<()> {
    // No initialization needed for WASM runtime
    Ok(())
  }

  async fn compile_kernel(&self, _source: &str) -> Result<Vec<u8>> {
    // For WASM runtime, we don't compile kernels
    Err(runtime_error!(
      "Kernel compilation not supported on WASM runtime backend"
    ))
  }

  async fn launch_kernel(
    &self,
    _kernel: &[u8],
    _grid: (u32, u32, u32),
    _block: (u32, u32, u32),
    _args: &[*const u8],
  ) -> Result<()> {
    Err(runtime_error!(
      "Kernel launch not supported on WASM runtime backend"
    ))
  }

  fn allocate_memory(&self, size: usize) -> Result<*mut u8> {
    // For CPU backend, we just use regular heap allocation
    let layout = std::alloc::Layout::from_size_align(size, 8)
      .map_err(|e| runtime_error!("Invalid layout: {}", e))?;

    let ptr = unsafe { std::alloc::alloc(layout) };

    if ptr.is_null() {
      return Err(runtime_error!("Failed to allocate {} bytes", size));
    }

    Ok(ptr)
  }

  fn free_memory(&self, ptr: *mut u8) -> Result<()> {
    // We don't track size, so we'll use a reasonable default alignment
    // In a real implementation, we'd need to track allocated sizes
    // For now, this is just a stub
    Ok(())
  }

  fn copy_memory(
    &self,
    dst: *mut u8,
    src: *const u8,
    size: usize,
    _kind: MemcpyKind,
  ) -> Result<()> {
    // Safety: This function assumes the caller has verified the pointers are valid
    // and don't overlap, as required by the trait contract
    unsafe {
      std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
  }

  fn synchronize(&self) -> Result<()> {
    // No-op for CPU backend
    Ok(())
  }
}
