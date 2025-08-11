# Claude Code Configuration for Zen Neural Stack

## 🧠 **ZEN NEURAL STACK - STRATEGIC INDEPENDENCE PROJECT**

### **Project Status**: 🚀 **ACTIVE DEVELOPMENT - Phase 0 Strategic Independence**

## 🎯 **PROJECT MISSION**

Create a complete, ownership-controlled neural computing platform by forking and rebranding external dependencies into the **Zen Neural Stack**. This project runs as a **side project** while claude-code-zen continues normal operation.

## 📁 **DIRECTORY STRUCTURE**

```
zen-neural-stack/
├── README.md                    # Project overview
├── AI_ROADMAP.md               # Complete development roadmap
├── zen-neural/                 # Core neural networks (from ruv-fann)
├── zen-forecasting/            # Time series forecasting (from neuro-divergent)
├── zen-compute/                # GPU/WASM acceleration (from cuda-rust-wasm)
├── zen-orchestrator/           # DAA coordination (from ruv-swarm)
└── reference-implementations/  # Our proven JS neural models (758-line GNN)
```

## 🚨 **CRITICAL SAFETY RULES**

### **⚠️ DO NOT BREAK CLAUDE-CODE-ZEN**
- **This is a SIDE PROJECT** - claude-code-zen must continue working
- **No changes** to /home/mhugo/code/claude-code-zen/src/ 
- **Work only inside** /home/mhugo/code/claude-code-zen/zen-neural-stack/
- **Test zen-neural-stack separately** before any integration

### **🔄 DEVELOPMENT APPROACH**
- **Parallel development** - Both systems running simultaneously
- **Gradual integration** - Only after zen-neural-stack is proven stable
- **Rollback ready** - Can abandon zen-neural-stack if issues arise

## 📋 **CURRENT TASKS (Phase 0: Strategic Independence)**

### **✅ COMPLETED**
- [x] Created zen-neural-stack directory structure
- [x] Copied all upstream components (ruv-fann, neuro-divergent, cuda-wasm, ruv-swarm)
- [x] Preserved our proven JS neural models as reference

### **🔄 IN PROGRESS**
- [ ] **Rebrand all package names**: ruv-fann → zen-neural, neuro-divergent → zen-forecasting
- [ ] **Update Rust Edition**: 2021 → 2024 for latest features
- [ ] **Fix all dependencies**: Point to zen-neural-stack components
- [ ] **Compilation verification**: Ensure everything builds

### **⭕ TODO (Priority Order)**
1. **Complete rebranding** of all 4 components
2. **Update internal dependencies** to use zen-* packages
3. **Verify compilation** with `cargo build --all`
4. **Run test suites** with `cargo test --all`
5. **Performance benchmarks** vs original implementations
6. **WASM compilation** for universal deployment

## 🛠️ **DEVELOPMENT COMMANDS**

### **Build Commands**
```bash
cd /home/mhugo/code/claude-code-zen/zen-neural-stack

# Build all components
cargo build --all --release

# Test all components  
cargo test --all

# Individual component builds
cd zen-neural && cargo build --release
cd zen-forecasting && cargo build --release
cd zen-compute && cargo build --release  
cd zen-orchestrator && cargo build --release
```

### **Rebranding Commands**
```bash
# Find all ruv-fann references for rebranding
grep -r "ruv-fann" zen-neural/

# Find all neuro-divergent references  
grep -r "neuro-divergent" zen-forecasting/

# Find all cuda-rust-wasm references
grep -r "cuda-rust-wasm" zen-compute/
```

## 🎯 **SUCCESS CRITERIA**

### **Phase 0 Complete When:**
- ✅ All packages renamed to zen-* branding
- ✅ All components compile with Rust 2024
- ✅ All tests pass
- ✅ Zero external neural dependencies
- ✅ Performance maintained or improved
- ✅ WASM compilation successful

## 🔗 **INTEGRATION STRATEGY**

### **Current State**
- **claude-code-zen**: Uses existing neural infrastructure (KEEP WORKING)
- **zen-neural-stack**: Independent development (NEW PROJECT)

### **Future Integration**  
1. **Prove zen-neural-stack** works independently
2. **Create bridge layer** for gradual migration
3. **Parallel testing** both systems
4. **Gradual cutover** component by component
5. **Complete migration** when zen-neural-stack proven superior

## 📊 **PERFORMANCE TARGETS**

### **Must Maintain or Exceed**
- **Compilation speed**: Same or faster than current
- **Runtime performance**: Same or better neural inference
- **Memory usage**: Same or lower memory footprint  
- **WASM size**: Optimized for browser deployment

### **New Capabilities to Add**
- **THE COLLECTIVE integration**: Native Borg coordination
- **DAA routing**: Intelligent neural task distribution  
- **Hot reloading**: Update models without restart
- **Multi-GPU**: Automatic GPU backend selection

## 🔄 **STATUS TRACKING**

When working on this project:
1. **Update README.md** with progress
2. **Use TodoWrite** to track specific tasks
3. **Document decisions** in AI_ROADMAP.md
4. **Test frequently** to avoid breaking changes
5. **Commit often** for easy rollback

## ⚡ **QUICK START FOR NEW SESSIONS**

When starting a new Claude Code session in zen-neural-stack:

1. **Read AI_ROADMAP.md** - Full project context
2. **Check current TodoWrite** - Active tasks  
3. **Verify builds** - `cargo build --all`
4. **Continue rebranding** - Phase 0 priority
5. **Maintain claude-code-zen safety** - Never break main system

---

**Remember**: This is strategic independence for neural computing. Take time to do it right. Claude-code-zen must continue working throughout this process.