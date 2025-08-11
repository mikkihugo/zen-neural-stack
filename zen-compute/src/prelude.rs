//! Prelude module for convenient imports

pub use crate::error::{CudaRustError, Result};
pub use crate::memory::{DeviceBuffer, HostBuffer};
pub use crate::runtime::{
  BackendType, Block, Device, Dim3, Grid, KernelFunction, LaunchConfig,
  Runtime, Stream, ThreadContext, launch_kernel,
};

// Re-export macros - these are automatically available from #[macro_export]
// No need to re-export them here as they're already globally available
