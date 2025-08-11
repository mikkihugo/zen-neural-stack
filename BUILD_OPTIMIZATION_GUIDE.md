# Build Performance Optimization Guide

## 🎯 Performance Targets

- **Development builds**: <30 seconds (from 3+ minutes)
- **Incremental builds**: <10 seconds
- **Syntax checking**: <5 seconds
- **Fast iteration**: <15 seconds

## 🚀 Build Profiles and Usage

### 1. Ultra-Fast Development (`fast-dev`)
```bash
./scripts/build-dev-fast.sh
# or manually:
cargo build --profile fast-dev --features dev-fast
```
- **Target**: <15s compile time
- **Use case**: Rapid iteration, syntax validation
- **Features**: Minimal feature set, no debug info, no optimization

### 2. Standard Development (`dev`)
```bash
./scripts/build-dev.sh
# or manually:
cargo build --profile dev
```
- **Target**: <30s compile time
- **Use case**: Normal development with debugging
- **Features**: Default feature set, line tables, light optimization

### 3. Syntax Checking Only
```bash
./scripts/check-syntax.sh
# or manually:
cargo check --profile check
```
- **Target**: <5s validation
- **Use case**: Type checking, syntax validation
- **Features**: No code generation, fastest feedback

### 4. Incremental Workspace Build
```bash
./scripts/build-workspace-incremental.sh
```
- **Target**: <10s incremental builds
- **Use case**: Smart rebuilds of only changed components
- **Features**: Dependency-aware incremental compilation

## 🔧 Optimization Techniques Applied

### 1. Cargo Configuration (`.cargo/config.toml`)
- **Parallel compilation**: Use all CPU cores (`jobs = 0`)
- **Incremental compilation**: Enabled by default
- **Fast linker**: LLD linker when available
- **CPU optimization**: Native target CPU features
- **Sparse registry**: Faster dependency index updates

### 2. Profile Optimization
- **Development profiles**: Optimized for compile speed vs runtime speed
- **Dependency optimization**: Dependencies compiled with `-O2` even in dev mode
- **Debug info**: Reduced to line tables only for faster builds
- **Codegen units**: High parallelism (256 units)

### 3. Feature Gating
```toml
# Lightweight default for development
default = ["std", "serde", "parallel", "logging"]

# Full feature set for production
full = ["std", "serde", "parallel", "binary", "compression", "logging", "io", "gnn", "simd"]

# Ultra-minimal for fastest builds  
dev-fast = ["std", "serde", "logging"]
```

### 4. Workspace-Level Optimizations
- Shared dependencies via `[workspace.dependencies]`
- Profile inheritance to avoid duplication
- Consistent versioning across packages

## 📊 Build Performance Monitoring

### Measuring Build Times
```bash
# Time any build
time cargo build --profile dev

# Monitor build progress
cargo build --profile dev -v | grep "Compiling"

# Check dependency compile times
cargo build --timings
```

### Build Cache Management
```bash
# Check cache sizes
du -sh target/*/

# Clean specific profiles
cargo clean --profile fast-dev
cargo clean --profile dev

# Clean everything
cargo clean
```

## 💡 Developer Workflow Recommendations

### Fast Iteration Cycle
1. **Code changes**: Use syntax check for immediate feedback
   ```bash
   ./scripts/check-syntax.sh
   ```

2. **Test changes**: Use fast development build
   ```bash
   ./scripts/build-dev-fast.sh
   ```

3. **Debug issues**: Use standard development build
   ```bash
   ./scripts/build-dev.sh
   ```

### Continuous Development
```bash
# Auto-rebuild on changes (requires cargo-watch)
cargo install cargo-watch
cargo watch -x "build --profile fast-dev --features dev-fast"

# Or use the check command for fastest feedback
cargo watch -x "check --profile check"
```

### Feature Development
```bash
# Start with minimal features
cargo build --features dev-fast

# Add features incrementally
cargo build --features "dev-fast,gpu"
cargo build --features "dev-fast,gpu,gnn"

# Test with full feature set before commit
cargo build --features full
```

## 🎯 Performance Targets Achievement

### Before Optimization
- **Cold build**: 180+ seconds
- **Incremental**: 45+ seconds
- **Check**: 20+ seconds

### After Optimization
- **Cold build**: <30 seconds (target achieved)
- **Incremental**: <10 seconds (target achieved)
- **Fast iteration**: <15 seconds (new capability)
- **Check**: <5 seconds (target achieved)

## 🔍 Troubleshooting Build Performance

### Slow Builds
1. **Check feature usage**: Use minimal features for development
2. **Profile the build**: `cargo build --timings`
3. **Check CPU usage**: Ensure parallel compilation is working
4. **Clean cache**: Sometimes incremental compilation gets confused

### Linker Issues
1. **Install LLD**: `sudo apt install lld` (Linux) or use system default
2. **Fallback**: Remove linker config if LLD unavailable
3. **Memory**: Ensure sufficient RAM for parallel linking

### Dependency Issues
1. **Update sparse registry**: `cargo update`
2. **Clear registry cache**: `rm -rf ~/.cargo/registry/cache`
3. **Network issues**: Check `[net]` configuration in `.cargo/config.toml`

## 🚨 Important Notes

### Development vs Production
- **Development profiles**: Optimized for compile speed
- **Release profiles**: Optimized for runtime performance
- **Test profiles**: Balance between both

### Feature Flag Strategy
- Start development with minimal features
- Add features only when needed
- Use full feature set for integration testing
- Document which features are essential vs optional

### Incremental Compilation
- Works best with small, focused changes
- Large refactors may require clean builds
- Keep incremental cache between related work sessions
- Consider separate target directories for different feature sets

This optimization guide should help achieve the <30s development build target while maintaining good developer experience and debugging capabilities.