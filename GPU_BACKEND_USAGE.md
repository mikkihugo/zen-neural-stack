# GPU Backend Usage Guide for Zen-Compute

## 🚀 **Overview**

Zen-compute provides comprehensive GPU acceleration through multiple backends:
- **CUDA** - NVIDIA GPU acceleration  
- **OpenCL** - Cross-platform GPU acceleration
- **Vulkan** - Modern cross-platform graphics/compute API
- **WebGPU** - Web-based GPU acceleration

## 🎯 **Quick Start**

### **Default Build (CPU + WebGPU only)**
```bash
cargo build
```

### **Enable CUDA Backend**
```bash
cargo build --features cuda-backend
```

### **Enable OpenCL Backend** 
```bash
cargo build --features opencl-backend
```

### **Enable Vulkan Backend**
```bash
cargo build --features vulkan
```

### **Enable All GPU Backends**
```bash
cargo build --features cuda-backend,opencl-backend,vulkan
```

## 🛠️ **Feature Flags Explained**

| Feature | Description | Requirements |
|---------|-------------|--------------|
| `cuda-backend` | NVIDIA CUDA acceleration | CUDA Toolkit installed |
| `opencl-backend` | OpenCL acceleration | OpenCL drivers installed |
| `vulkan` | Vulkan compute acceleration | Vulkan SDK installed |
| `native-gpu` | Enables both CUDA and OpenCL | Default feature |
| `webgpu-only` | Web-only GPU acceleration | None |

## 🏗️ **Build Requirements**

### **CUDA Backend Requirements**
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Set environment variable (optional)
export CUDA_PATH=/usr/local/cuda
```

### **OpenCL Backend Requirements**  
```bash
# Ubuntu/Debian
sudo apt install opencl-headers ocl-icd-opencl-dev

# macOS (via Homebrew)
brew install opencl-headers
```

### **Vulkan Backend Requirements**
```bash
# Ubuntu/Debian  
sudo apt install vulkan-tools libvulkan-dev

# Set environment variable (optional)
export VULKAN_SDK=/usr/share/vulkan
```

## 📋 **Feature Detection**

The build system automatically detects available GPU backends:

```bash
# Build with verbose GPU detection
cargo build --features cuda-backend --verbose
```

**Detection Output:**
```
cargo:warning=Starting comprehensive CUDA installation search...
cargo:warning=CUDA 12.1 detected at /usr/local/cuda
cargo:warning=OpenCL found at /usr/lib/x86_64-linux-gnu
```

## 🧪 **Testing GPU Backends**

```bash
# Test specific backend
cargo test --features cuda-backend gpu_acceleration_tests

# Test all backends
cargo test --features cuda-backend,opencl-backend,vulkan --test gpu_integration
```

## 🎮 **Runtime Usage Examples**

### **CUDA Acceleration**
```rust
use zen_compute::cuda::CudaBackend;

let mut backend = CudaBackend::new()?;
let result = backend.matrix_multiply(&a, &b)?;
```

### **OpenCL Acceleration** 
```rust
use zen_compute::opencl::OpenCLBackend;

let mut backend = OpenCLBackend::new()?;
let result = backend.vector_add(&a, &b)?;
```

### **WebGPU (Works in Browser)**
```rust
use zen_compute::webgpu::WebGPUBackend;

let mut backend = WebGPUBackend::new().await?;
let result = backend.compute_shader(&data).await?;
```

## 🚨 **Troubleshooting**

### **No GPU Found**
If you see warnings like "CUDA not found", ensure:
1. GPU drivers are installed
2. Development libraries are installed  
3. Environment variables are set (optional)

### **Build Failures**
```bash
# Clean and rebuild
cargo clean
cargo build --features cuda-backend
```

### **Feature Not Working**
```bash
# Verify feature detection
cargo build --features cuda-backend --verbose | grep -i cuda
```

## 🎯 **Production Deployment**

### **Docker with GPU Support**
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y opencl-headers vulkan-tools
COPY . /app
WORKDIR /app
RUN cargo build --release --features cuda-backend,opencl-backend
```

### **WebAssembly Deployment**
```bash
# WASM build (WebGPU only)
wasm-pack build --features webgpu-only --target web
```

## 📊 **Performance Benchmarks**

```bash
# Run GPU benchmarks
cargo bench --features cuda-backend,opencl-backend,vulkan

# Compare backends
cargo run --example benchmark_comparison --features cuda-backend,opencl-backend
```

Expected performance improvements:
- **CUDA**: 10-100x faster than CPU
- **OpenCL**: 5-50x faster than CPU  
- **WebGPU**: 2-20x faster than CPU

## 🔧 **Advanced Configuration**

### **Custom CUDA Path**
```bash
export CUDA_PATH=/opt/cuda-12.1
cargo build --features cuda-backend
```

### **Multiple GPU Selection**
```rust
use zen_compute::cuda::CudaBackend;

let backend = CudaBackend::with_device(1)?; // Use GPU 1
```

### **Memory Management**
```rust
use zen_compute::memory::GpuMemoryPool;

let pool = GpuMemoryPool::new(1_000_000_000); // 1GB pool
```

---

**Need Help?** Open an issue at [zen-neural-stack issues](https://github.com/mikkihugo/zen-neural-stack/issues)