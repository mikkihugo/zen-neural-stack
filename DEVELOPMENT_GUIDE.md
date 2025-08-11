# 🛠️ Zen Neural Stack - Development Guide

## 🚀 **Quick Start Development**

### **Prerequisites**
```bash
# Install Rust (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install development tools
rustup component add clippy rustfmt
cargo install wasm-pack just
```

### **Development Workflow**
```bash
# Clone and setup
git clone <repo-url> zen-neural-stack
cd zen-neural-stack

# Quick development cycle
just dev          # Check + test everything
just build        # Build all packages  
just build-wasm   # Build WASM packages
just docs         # Generate documentation
```

## 📋 **Development Commands**

### **Core Development**
- `just build` - Build all workspace packages
- `just test` - Run comprehensive test suite
- `just check` - Lint, format, and type check
- `just fix` - Auto-fix formatting and simple lints

### **Performance & Optimization**
- `just bench` - Run performance benchmarks
- `just profile <target>` - Profile specific binary
- `just build-release` - Optimized release build

### **WASM & Deployment**
- `just build-wasm` - Build all WASM packages
- `just clean` - Clean all build artifacts

## 🏗️ **Architecture Guidelines**

### **1. THE COLLECTIVE Integration**
Every neural component should support Borg coordination:
```rust
pub struct ZenNetwork {
    collective_id: CollectiveId,
    borg_protocol: BorgProtocol,
    // ... network implementation
}
```

### **2. Async-First Design**
All operations should be async-compatible:
```rust
pub async fn train(&mut self, data: TrainingData) -> Result<Metrics>;
pub async fn infer(&self, input: Tensor) -> Result<Output>;
```

### **3. WASM Compatibility** 
Design for universal deployment:
```rust
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
```

### **4. Zero-Copy Operations**
Minimize allocations in hot paths:
```rust
pub fn forward_view(&self, input: &Tensor) -> TensorView;
```

## 🎯 **Development Priorities**

### **Phase 0 (Current): Strategic Independence**
1. Complete package rebranding
2. Fix internal dependencies  
3. Verify compilation across all targets
4. Establish development workflow

### **Phase 1: Core Neural Implementation**
1. Port 758-line GNN from JavaScript
2. Implement LSTM for time series
3. Add CNN for computer vision
4. Create unified training API

### **Phase 2: THE COLLECTIVE Integration**
1. Borg coordination protocols
2. Neural task routing via DAA
3. Multi-agent neural coordination
4. Performance optimization

## 🔧 **Component-Specific Guidelines**

### **zen-neural (Core Networks)**
- Focus on raw neural network performance
- THE COLLECTIVE coordination integration
- SIMD optimization for math operations
- Memory pool management

### **zen-forecasting (Time Series)**
- Streaming data processing
- Real-time prediction APIs
- Model ensemble management
- Temporal pattern detection

### **zen-compute (GPU/WASM)**
- Multi-backend GPU support
- WASM compilation optimization
- Memory management across backends
- Performance monitoring

### **zen-orchestrator (DAA Coordination)**
- Agent-based neural routing
- Load balancing algorithms
- Fault tolerance and recovery
- Distributed coordination

## 🧪 **Testing Strategy**

### **Unit Tests**
```bash
cargo test --workspace --lib
```

### **Integration Tests**
```bash  
cargo test --workspace --test '*'
```

### **Benchmarks**
```bash
cargo bench --workspace
```

### **WASM Tests**
```bash
wasm-pack test --headless --firefox zen-neural
```

## 📊 **Performance Targets**

### **Compilation**
- Clean build: < 2 minutes
- Incremental build: < 30 seconds
- WASM build: < 1 minute

### **Runtime**
- GNN inference: < 10ms (1000 nodes)
- LSTM training: < 1s (1000 sequences)
- Memory usage: < 100MB baseline

### **WASM**
- Bundle size: < 2MB compressed
- Load time: < 500ms
- Runtime overhead: < 20%

---

**🎯 Goal**: World-class neural computing platform with THE COLLECTIVE intelligence!